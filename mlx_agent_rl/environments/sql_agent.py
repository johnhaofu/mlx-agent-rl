"""Spider text-to-SQL agent environment — fully local SQLite.

Unlike WebShop / HotpotQA this env doesn't need a sidecar: the dataset is
on disk (``data/spider/spider_data/``) and SQLite ships with Python.
Each episode binds to one (question, db_id, gold_sql) tuple. The model
issues ``sql[…]`` queries to inspect the database and ultimately commits
its solution via ``answer[…]``.

Action verbs
------------
``sql[query]``
    Execute against the per-episode SQLite DB. The first 10 result rows
    (or the SQLite error string) are returned as the next observation.
    Reward stays at 0; episode continues.

``answer[query]``
    Terminal action. The query is executed and compared against the
    gold SQL via execution-match (Spider's "EX" metric). Reward is 1.0
    if the result sets match, else 0.0.

Reward design follows the official Spider EX semantics: result rows are
normalised to multisets of repr-stringified tuples per column-set so
column ordering and equivalent JOIN orderings don't fail us. ORDER BY
queries are special-cased to preserve row order. The gold execution is
cached per episode so step() doesn't re-run it after each invalid attempt.

Action format mirrors the WebShop/HotpotQA envs: ``<action>sql[…]</action>``
with verl-style ``<think>+<action>`` envelope, plus a normaliser that
forgives ``answer>X`` typos (we hit those on HotpotQA at ~78% rate before
fixing).
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Optional

from mlx_agent_rl.environments.base import BaseEnvironment, Observation


_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
# Raw-form fallback (e.g. model emits ``sql[SELECT …]`` without the
# ``<action>`` envelope). Matches the LAST verb[arg] occurrence so any
# echoed schema text earlier in the output doesn't get picked up.
_RAW_ACTION_RE = re.compile(
    r"(sql|answer)\[(.+)\](?!.*?(?:sql|answer)\[)",
    re.DOTALL,
)
# Slop normaliser (``answer>X``, ``sql: SELECT…``). See HotpotQA env for
# rationale — base models often emit these and the env should coerce
# rather than reject so episodes don't silently time out.
_SLOP_VERB_RE = re.compile(
    r"^(sql|answer)\s*[>:\-=]\s*(.+?)\s*$", re.DOTALL
)

_DEFAULT_DATA_DIR = Path(
    os.environ.get(
        "SPIDER_DATA_DIR",
        Path(__file__).parents[2] / "data" / "spider" / "spider_data",
    )
)


def _normalise_action(action: str) -> str | None:
    a = action.strip()
    if not a:
        return None
    # Strip trailing slop OUTSIDE the bracket: ``;``, whitespace.
    # Common pattern: ``answer[SELECT … )];`` — bracketed arg is fine,
    # the trailing ``;`` would otherwise fail step's strict regex.
    a = re.sub(r"[;\s]+$", "", a)
    if re.match(r"^(sql|answer)\[.+\]$", a, re.DOTALL):
        return a
    m = _SLOP_VERB_RE.match(a)
    if m:
        verb, arg = m.group(1), m.group(2).strip()
        if arg:
            return f"{verb}[{arg}]"
    # Malformed (e.g. missing closing ``]``): return None so rollout
    # applies invalid_action_penalty and the agent gets a clear signal,
    # rather than silently passing through to step's strict regex.
    return None


def _is_order_sensitive(sql: str) -> bool:
    """Best-effort detection of queries whose result row order matters.

    ORDER BY makes row order meaningful. LIMIT can too (top-N). For
    everything else we treat the result as an unordered multiset.
    """
    s = sql.lower()
    return " order by " in s or " limit " in s


def _normalise_rows(rows: list, ordered: bool) -> Any:
    """Reduce a SQLite result-set to a hashable shape suitable for ==.

    If ``ordered`` is False, columns within each row are also sorted —
    this lets predictions with different SELECT orderings still match.
    """
    norm_rows = [tuple(repr(c) for c in row) for row in rows]
    if not ordered:
        norm_rows = [tuple(sorted(r)) for r in norm_rows]
        return tuple(sorted(norm_rows))
    return tuple(norm_rows)


def _exec_match(pred_sql: str, gold_sql: str, db_path: Path) -> tuple[bool, str]:
    """Run both queries against ``db_path`` and compare result sets.

    Returns ``(match, error_msg)``. ``error_msg`` is empty on success or
    contains the SQLite exception message if the prediction fails.
    """
    try:
        conn = sqlite3.connect(str(db_path))
        conn.text_factory = str
    except sqlite3.Error as exc:
        return False, f"db connect: {exc}"
    try:
        try:
            pred_rows = conn.execute(pred_sql).fetchall()
        except sqlite3.Error as exc:
            return False, str(exc)
        try:
            gold_rows = conn.execute(gold_sql).fetchall()
        except sqlite3.Error as exc:
            # Gold failing is a dataset issue — treat as not-match but
            # surface the message for debugging rather than crashing.
            return False, f"gold failed: {exc}"
        ordered = _is_order_sensitive(gold_sql)
        return (
            _normalise_rows(pred_rows, ordered) == _normalise_rows(gold_rows, ordered),
            "",
        )
    finally:
        conn.close()


class SQLAgentEnvironment(BaseEnvironment):
    """Spider-flavored text-to-SQL agent.

    Parameters
    ----------
    data_dir:
        Path to the unzipped ``spider_data`` directory.
    split:
        ``train`` (uses ``train_spider.json`` + ``database/``),
        ``validation`` / ``dev`` (uses ``dev.json`` + ``database/``), or
        ``test`` (uses ``test.json`` + ``test_database/``). Yale released
        the test gold queries in 2023 after closing the leaderboard.
    schema_max_chars:
        Cap on the schema block embedded in the initial obs. Some Spider
        DBs have ~30 tables and would dwarf the prompt budget; we
        truncate at this many characters.
    rows_per_query:
        Max rows returned to the model after a ``sql[…]`` execution.
    require_think:
        Reject outputs missing a ``<think>`` block. Default False to be
        forgiving with base models.
    """

    TASK_TYPE = "sql_agent"

    def __init__(
        self,
        data_dir: Path | str | None = None,
        split: str = "train",
        schema_max_chars: int = 4000,
        rows_per_query: int = 10,
        require_think: bool = False,
        use_tools_schema: bool = False,
        partial_credit: float = 0.1,
        format_reward: float = 0.0,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
        # Test split lives in a separate database directory.
        if split == "test":
            self.db_dir = self.data_dir / "test_database"
        else:
            self.db_dir = self.data_dir / "database"
        self.split = split
        self.schema_max_chars = schema_max_chars
        self.rows_per_query = rows_per_query
        self.require_think = require_think
        self.use_tools_schema = use_tools_schema
        self.partial_credit = partial_credit
        # Small bonus paid on every well-formed action (sql[…] or answer[…])
        # to reinforce envelope/syntax correctness during early training.
        # Default 0.0 (off) — invalid_action_penalty (-0.1) already gives
        # the negative signal; format_reward is a knob for runs where you
        # want to explicitly accelerate format lock-in.
        self.format_reward = format_reward
        self._examples = self._load_examples(split)
        self._schema_cache: dict[str, str] = {}
        self._reset_state()

    # ------------------------------------------------------------------
    # Public env interface
    # ------------------------------------------------------------------

    def reset(
        self,
        prompt: str,
        answer: Optional[int] = None,
        **kwargs,
    ) -> Observation:
        idx = int(answer) % len(self._examples) if answer is not None else 0
        ex = self._examples[idx]
        db_id = ex["db_id"]
        db_path = self.db_dir / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            raise FileNotFoundError(f"missing Spider DB: {db_path}")

        self._current = ex
        self._db_path = db_path
        self._gold_sql = ex["query"]
        self._finished = False
        self._steps_done = 0
        self._last_match = False
        self._last_pred_sql: str | None = None
        self._last_error: str | None = None
        self._last_partial = False

        question = ex["question"]
        schema_block = self._render_schema(db_id, db_path)
        obs_text = (
            f"Database: {db_id}\n"
            f"Schema:\n{schema_block}\n\n"
            f"Question: {question}"
        )
        self._last_obs = obs_text
        return Observation(text=obs_text, done=False, anchor=self._anchor())

    def step(self, action: str) -> tuple[Observation, float, bool]:
        if self._finished:
            return (
                Observation(text=self._last_obs, done=True, anchor=self._anchor()),
                0.0,
                True,
            )

        m = re.match(r"^(sql|answer)\[(.+)\]\s*$", action, re.DOTALL)
        if not m:
            obs_text = (
                "Error: action must be 'sql[query]' or 'answer[query]'. "
                "Try again."
            )
            self._last_obs = obs_text
            self._last_error = "parse"
            return (
                Observation(text=obs_text, done=False, anchor=self._anchor()),
                0.0,
                False,
            )
        verb, arg = m.group(1), m.group(2).strip()
        self._steps_done += 1

        if verb == "sql":
            return self._handle_sql(arg)
        return self._handle_answer(arg)

    def extract_action(self, model_output: str) -> str | None:
        if self.require_think and not _THINK_RE.search(model_output):
            return None
        if re.search(r"[\u4e00-\u9fff]", model_output):
            return None
        match = _ACTION_RE.search(model_output)
        if match:
            inner = match.group(1).strip()
            return _normalise_action(inner) if inner else None
        raw = _RAW_ACTION_RE.search(model_output)
        if raw:
            verb = raw.group(1)
            arg = raw.group(2).strip()
            return f"{verb}[{arg}]" if arg else None
        return None

    @property
    def invalid_action_message(self) -> str:
        return (
            "Error: could not parse <action>. Respond with "
            "<think>your reasoning</think>"
            "<action>sql[SELECT …] or answer[final SELECT …]</action>."
        )

    @property
    def last_step_info(self) -> dict:
        return {
            "match": float(self._last_match),
            "won": bool(self._last_match),
            "partial": bool(self._last_partial),
            "n_steps": self._steps_done,
            "had_error": self._last_error is not None,
        }

    @property
    def stop_strings(self) -> list[str]:
        return ["</action>"]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        self._current: dict | None = None
        self._db_path: Path | None = None
        self._gold_sql: str | None = None
        self._finished = False
        self._steps_done = 0
        self._last_match = False
        self._last_pred_sql: str | None = None
        self._last_error: str | None = None
        self._last_partial = False
        self._last_obs: str = ""

    def _load_examples(self, split: str) -> list[dict]:
        if split == "train":
            files = ["train_spider.json"]
        elif split in ("validation", "dev"):
            files = ["dev.json"]
        elif split == "test":
            files = ["test.json"]
        else:
            raise ValueError(
                f"unknown split {split!r}; choose 'train', 'validation', or 'test'"
            )
        items: list[dict] = []
        for fname in files:
            path = self.data_dir / fname
            if not path.exists():
                raise FileNotFoundError(f"missing Spider split file: {path}")
            with open(path) as fh:
                items.extend(json.load(fh))
        return items

    def _render_schema(self, db_id: str, db_path: Path) -> str:
        """Return CREATE TABLE statements for the given DB.

        Cached per ``db_id`` because dev/test re-touch the same 20
        databases many times across rollouts.
        """
        if db_id in self._schema_cache:
            return self._schema_cache[db_id]
        try:
            conn = sqlite3.connect(str(db_path))
            rows = conn.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type='table' AND sql IS NOT NULL ORDER BY name"
            ).fetchall()
        finally:
            conn.close()
        schema = "\n\n".join(r[0].strip() + ";" for r in rows if r[0])
        if len(schema) > self.schema_max_chars:
            schema = schema[: self.schema_max_chars] + "\n-- (truncated)"
        self._schema_cache[db_id] = schema
        return schema

    def _handle_sql(self, query: str) -> tuple[Observation, float, bool]:
        """Read-only execute: return rows or error, no termination."""
        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.text_factory = str
            rows = conn.execute(query).fetchmany(self.rows_per_query + 1)
        except sqlite3.Error as exc:
            obs_text = f"SQL error: {exc}"
            self._last_error = str(exc)
            self._last_obs = obs_text
            return (
                Observation(text=obs_text, done=False, anchor=self._anchor()),
                0.0,
                False,
            )
        finally:
            try:
                conn.close()
            except Exception:
                pass
        truncated = len(rows) > self.rows_per_query
        rows = rows[: self.rows_per_query]
        if not rows:
            preview = "(0 rows)"
        else:
            preview = "\n".join(repr(r) for r in rows)
            if truncated:
                preview += f"\n… (truncated to {self.rows_per_query} rows)"
        obs_text = f"SQL ok ({len(rows)} rows shown):\n{preview}"
        self._last_error = None
        self._last_obs = obs_text
        return (
            Observation(text=obs_text, done=False, anchor=self._anchor()),
            self.format_reward,  # 0.0 by default; small bonus for well-formed sql[…]
            False,
        )

    def _handle_answer(self, query: str) -> tuple[Observation, float, bool]:
        """Terminal: compute exec match, compute reward, mark finished.

        Reward shape (terminal-only, no per-sql exploration credit so the
        agent can't farm partial reward by spamming sql[]):

            EX match                                   → 1.0
            SQL ran without error, result mismatch     → 0.1
            SQL had an error                           → 0.0

        The 0.1 bucket gives a dense signal for "wrote valid SQL" — most
        of our hard/extra failures are 1-token-off (wrong column, missing
        GROUP BY) and would otherwise share the 0 bucket with garbage,
        leaving GiGPO advantages flat across the group.
        """
        match, err = _exec_match(query, self._gold_sql, self._db_path)
        self._finished = True
        self._last_match = match
        self._last_pred_sql = query
        self._last_error = err if err else None
        if match:
            obs_text = "Answer recorded. EX=1 (match)"
            reward = 1.0
            partial = False
        elif err:
            obs_text = f"Answer recorded. EX=0 ({err})"
            reward = 0.0
            partial = False
        else:
            obs_text = "Answer recorded. EX=0 (different result)"
            reward = self.partial_credit
            partial = self.partial_credit > 0
        # format_reward is added on top regardless of correctness — the agent
        # got this far through extract_action + the strict step regex, so the
        # envelope/syntax was well-formed. Constant shifts cancel inside
        # group-relative advantage, so this only differentiates well-formed
        # vs invalid_action_penalty=-0.1 trajectories.
        reward = reward + self.format_reward
        self._last_obs = obs_text
        self._last_partial = partial
        return (
            Observation(text=obs_text, done=True, anchor=self._anchor()),
            reward,
            True,
        )

    def _anchor(self) -> str:
        return self._last_obs

    @property
    def n_examples(self) -> int:
        return len(self._examples)

    @property
    def gold_sql(self) -> str | None:
        return self._gold_sql

    @property
    def db_id(self) -> str | None:
        return self._current["db_id"] if self._current else None
