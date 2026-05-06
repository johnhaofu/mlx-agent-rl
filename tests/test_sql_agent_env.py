"""Unit tests for SQLAgentEnvironment.

Tests cover extract_action parsing (envelope, raw fallback, slop
normalisation), reset/step semantics against a real Spider DB, and
the exec-match reward computation. We exercise live SQLite — no
mocking — because the env's whole reward signal depends on real query
execution and the cost (~1ms per query) is negligible in tests.

Skip the entire module when the Spider data directory isn't present;
the tests can't run without the train_spider.json + database/ folder.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mlx_agent_rl.environments.sql_agent import (
    SQLAgentEnvironment,
    _exec_match,
    _is_order_sensitive,
    _normalise_action,
)


_SPIDER_DATA = Path(__file__).resolve().parents[1] / "data" / "spider" / "spider_data"


pytestmark = pytest.mark.skipif(
    not (_SPIDER_DATA / "train_spider.json").exists(),
    reason="Spider data not present at data/spider/spider_data/",
)


# ----------------------------------------------------------------------
# extract_action
# ----------------------------------------------------------------------


def _env() -> SQLAgentEnvironment:
    return SQLAgentEnvironment(data_dir=_SPIDER_DATA, split="train")


def test_extract_action_envelope_sql():
    env = _env()
    out = "<action>sql[SELECT count(*) FROM head]</action>"
    assert env.extract_action(out) == "sql[SELECT count(*) FROM head]"


def test_extract_action_envelope_answer():
    env = _env()
    out = "<action>answer[SELECT * FROM department]</action>"
    assert env.extract_action(out) == "answer[SELECT * FROM department]"


def test_extract_action_raw_fallback():
    env = _env()
    out = "I'll try sql[SELECT name FROM head WHERE age > 50]"
    assert env.extract_action(out) == "sql[SELECT name FROM head WHERE age > 50]"


def test_extract_action_slop_normalised():
    # ``answer>X`` instead of ``answer[X]`` — observed on HotpotQA.
    env = _env()
    out = "<action>answer>SELECT * FROM dept</action>"
    assert env.extract_action(out) == "answer[SELECT * FROM dept]"


def test_extract_action_chinese_rejected():
    env = _env()
    out = "<action>sql[SELECT name 我 FROM s]</action>"
    assert env.extract_action(out) is None


def test_extract_action_no_match_returns_none():
    env = _env()
    assert env.extract_action("just thinking, no action call") is None


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def test_normalise_action_canonical_passes_through():
    assert _normalise_action("sql[SELECT 1]") == "sql[SELECT 1]"


def test_normalise_action_handles_colon():
    assert _normalise_action("answer: SELECT 1") == "answer[SELECT 1]"


def test_normalise_action_strips_trailing_semicolon():
    # Observed on test set: ``answer[X)];`` with stray ``;`` outside ``]``.
    assert _normalise_action("answer[SELECT 1];") == "answer[SELECT 1]"
    assert _normalise_action("sql[SELECT 1)];") == "sql[SELECT 1)]"


def test_normalise_action_rejects_missing_close_bracket():
    # Observed: ``sql[SELECT … LIKE '%Dell%';`` — no ``]`` at all.
    # Returning None lets rollout apply invalid_action_penalty.
    assert _normalise_action("sql[SELECT product_size FROM t;") is None
    assert _normalise_action("answer[SELECT 1") is None


def test_is_order_sensitive():
    assert _is_order_sensitive("SELECT * FROM t ORDER BY x")
    assert _is_order_sensitive("SELECT * FROM t LIMIT 5")
    assert not _is_order_sensitive("SELECT count(*) FROM t")


# ----------------------------------------------------------------------
# reset / step against a real Spider DB
# ----------------------------------------------------------------------


def test_reset_returns_question_and_schema():
    env = _env()
    obs = env.reset(prompt="", answer=0)
    # First Spider train item is from db_id=department_management.
    assert "Database: department_management" in obs.text
    assert "Schema:" in obs.text
    assert "Question:" in obs.text
    # Schema should look like CREATE TABLE statements.
    assert "CREATE TABLE" in obs.text.upper()


def test_step_sql_returns_rows_no_done():
    env = _env()
    env.reset(prompt="", answer=0)
    obs, reward, done = env.step('sql[SELECT count(*) FROM head]')
    assert reward == 0.0
    assert done is False
    assert "rows shown" in obs.text


def test_step_sql_invalid_returns_error_no_done():
    env = _env()
    env.reset(prompt="", answer=0)
    obs, reward, done = env.step('sql[SELECT * FROM nonexistent_table]')
    assert reward == 0.0
    assert done is False
    assert "SQL error" in obs.text


def test_step_answer_with_gold_gives_reward_1():
    env = _env()
    env.reset(prompt="", answer=0)
    gold = env.gold_sql
    assert gold is not None
    obs, reward, done = env.step(f"answer[{gold}]")
    assert reward == 1.0
    assert done is True
    assert env.last_step_info["match"] == 1.0
    assert env.last_step_info["won"] is True


def test_step_answer_with_wrong_valid_query_gives_partial_credit():
    # Valid SQL that runs without error but returns a different result
    # → partial credit 0.1 (dense signal for "wrote valid SQL"). Replaces
    # the old hard-zero behavior to give GiGPO usable advantage variance
    # on hard/extra questions where most failures are 1-token-off.
    env = _env()
    env.reset(prompt="", answer=0)
    obs, reward, done = env.step("answer[SELECT 1]")
    assert reward == 0.1
    assert done is True
    assert env.last_step_info["match"] == 0.0
    assert env.last_step_info["won"] is False
    assert env.last_step_info["partial"] is True


def test_step_answer_with_invalid_sql_gives_zero():
    # SQL syntax/runtime error → 0.0 reward (not partial), so the agent
    # is still penalised for unparseable answers vs only-wrong answers.
    env = _env()
    env.reset(prompt="", answer=0)
    obs, reward, done = env.step("answer[SELECT * FROM nonexistent_table]")
    assert reward == 0.0
    assert done is True
    assert env.last_step_info["match"] == 0.0
    assert env.last_step_info["partial"] is False
    assert env.last_step_info["had_error"] is True


def test_step_after_done_is_idempotent():
    env = _env()
    env.reset(prompt="", answer=0)
    env.step("answer[SELECT 1]")
    obs, reward, done = env.step("sql[SELECT 1]")
    assert done is True
    assert reward == 0.0


def test_invalid_action_message_text_envelope():
    env = SQLAgentEnvironment(data_dir=_SPIDER_DATA, split="train")
    msg = env.invalid_action_message
    assert "<action>" in msg


def test_stop_strings_present():
    env = _env()
    assert "</action>" in env.stop_strings


# ----------------------------------------------------------------------
# _exec_match low-level
# ----------------------------------------------------------------------


def test_exec_match_self_match_true():
    env = _env()
    env.reset(prompt="", answer=0)
    db = env._db_path
    gold = env.gold_sql
    match, err = _exec_match(gold, gold, db)
    assert match is True
    assert err == ""


def test_exec_match_invalid_pred_returns_false_with_error():
    env = _env()
    env.reset(prompt="", answer=0)
    db = env._db_path
    gold = env.gold_sql
    match, err = _exec_match("SELECT * FROM nope", gold, db)
    assert match is False
    assert err  # non-empty error message


def test_exec_match_different_results_returns_false():
    env = _env()
    env.reset(prompt="", answer=0)
    db = env._db_path
    gold = env.gold_sql
    # SELECT 1 returns one row with one int; almost never equals gold.
    match, err = _exec_match("SELECT 1", gold, db)
    assert match is False


# ----------------------------------------------------------------------
# n_examples / split routing
# ----------------------------------------------------------------------


def test_train_split_has_seven_thousand():
    env = SQLAgentEnvironment(data_dir=_SPIDER_DATA, split="train")
    assert env.n_examples == 7000


def test_validation_split_alias():
    env = SQLAgentEnvironment(data_dir=_SPIDER_DATA, split="validation")
    assert env.n_examples == 1034
    # 'dev' is also accepted.
    env2 = SQLAgentEnvironment(data_dir=_SPIDER_DATA, split="dev")
    assert env2.n_examples == 1034


def test_unknown_split_raises():
    with pytest.raises(ValueError):
        SQLAgentEnvironment(data_dir=_SPIDER_DATA, split="bogus")
