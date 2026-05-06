"""Unit tests for HotpotQAEnvironment.

The env is HTTP-only — we monkey-patch ``_post`` so tests don't require
a running sidecar.
"""

from __future__ import annotations

from unittest.mock import patch

from mlx_agent_rl.environments.hotpotqa import HotpotQAEnvironment


def _patched_post(responses: list):
    iterator = iter(responses)

    def fake(self, path, body):  # noqa: ARG001
        return next(iterator)

    return fake


# ----------------------------------------------------------------------
# extract_action
# ----------------------------------------------------------------------


def test_extract_action_envelope_search():
    env = HotpotQAEnvironment()
    out = "<action>search[Scott Derrickson nationality]</action>"
    assert env.extract_action(out) == "search[Scott Derrickson nationality]"


def test_extract_action_envelope_answer():
    env = HotpotQAEnvironment()
    out = "<action>answer[yes]</action>"
    assert env.extract_action(out) == "answer[yes]"


def test_extract_action_raw_search_fallback():
    env = HotpotQAEnvironment()
    out = "Let me search. search[Scott Derrickson]"
    assert env.extract_action(out) == "search[Scott Derrickson]"


def test_extract_action_raw_answer_fallback():
    env = HotpotQAEnvironment()
    out = "answer[Doctor Strange]"
    assert env.extract_action(out) == "answer[Doctor Strange]"


def test_extract_action_picks_last_when_multiple():
    env = HotpotQAEnvironment()
    out = "answer[wrong] then search[query] then answer[final]"
    assert env.extract_action(out) == "answer[final]"


def test_extract_action_chinese_rejected():
    env = HotpotQAEnvironment()
    out = "<action>answer[搜索结果]</action>"
    assert env.extract_action(out) is None


def test_extract_action_no_match_returns_none():
    env = HotpotQAEnvironment()
    assert env.extract_action("just thinking, no action call") is None


def test_extract_action_empty_envelope_returns_none():
    env = HotpotQAEnvironment()
    # An empty <action></action> body parses to nothing — reject.
    assert env.extract_action("<action></action>") is None
    # Empty argument inside a verb is forwarded as-is and the sidecar
    # surfaces an "empty_arg" error; we don't filter at extract_action.
    assert env.extract_action("<action>search[]</action>") == "search[]"


def test_extract_action_normalises_answer_with_gt():
    env = HotpotQAEnvironment()
    # The base model often emits `answer>X</action>` — without normalisation
    # the sidecar rejects every turn and the episode times out.
    assert (
        env.extract_action("<action>answer>Eenasul Fateh</action>")
        == "answer[Eenasul Fateh]"
    )
    assert (
        env.extract_action("<action>search>foo bar</action>")
        == "search[foo bar]"
    )


def test_extract_action_normalises_colon_form():
    env = HotpotQAEnvironment()
    assert (
        env.extract_action("<action>search: red jacket</action>")
        == "search[red jacket]"
    )


def test_extract_action_require_think():
    env = HotpotQAEnvironment(require_think=True)
    assert env.extract_action("<action>answer[no]</action>") is None
    assert (
        env.extract_action("<think>x</think><action>answer[no]</action>")
        == "answer[no]"
    )


# ----------------------------------------------------------------------
# reset / step
# ----------------------------------------------------------------------


def test_reset_passes_seed_and_split():
    env = HotpotQAEnvironment(base_url="http://x", split="validation")
    response = {
        "session_id": "s-1",
        "obs": "Question: …\nAvailable passages:\n  - A\n  - B",
        "available_actions": {"tools": ["search", "answer"], "titles": ["A", "B"]},
        "instruction": "Question A?",
        "info": {},
    }
    captured = {}

    def fake(self, path, body):  # noqa: ARG001
        captured.update(body)
        return response

    with patch.object(HotpotQAEnvironment, "_post", fake):
        obs = env.reset(prompt="", answer=42)
    assert captured["seed_idx"] == 42
    assert captured["split"] == "validation"
    assert env.instruction == "Question A?"
    assert env._titles == ["A", "B"]
    assert obs.text.startswith("Question:")


def test_step_search_returns_zero_reward_no_done():
    env = HotpotQAEnvironment(base_url="http://x")
    env._session_id = "s1"
    env._last_obs = "page"
    response = {
        "obs": "Top-3 passages:\n…",
        "reward": 0.0,
        "done": False,
        "available_actions": {"tools": ["search", "answer"], "titles": []},
    }
    with patch.object(HotpotQAEnvironment, "_post", _patched_post([response])):
        obs, reward, done = env.step("search[query]")
    assert reward == 0.0
    assert done is False


def test_step_answer_correct_gives_full_reward():
    env = HotpotQAEnvironment(base_url="http://x")
    env._session_id = "s1"
    env._last_obs = "page"
    response = {
        "obs": "Answer recorded. F1=1.000  EM=1",
        "reward": 1.0,
        "done": True,
        "won": True,
        "f1": 1.0,
        "em": 1.0,
        "reference": "yes",
    }
    with patch.object(HotpotQAEnvironment, "_post", _patched_post([response])):
        obs, reward, done = env.step("answer[yes]")
    assert reward == 1.0
    assert done is True
    assert env.last_step_info["f1"] == 1.0
    assert env.last_step_info["em"] == 1.0
    assert env.last_step_info["won"] is True


def test_step_answer_partial_match_gives_partial_f1():
    env = HotpotQAEnvironment(base_url="http://x")
    env._session_id = "s1"
    env._last_obs = "page"
    response = {
        "obs": "Answer recorded. F1=0.500",
        "reward": 0.5,
        "done": True,
        "won": False,
        "f1": 0.5,
        "em": 0.0,
        "reference": "Doctor Strange",
    }
    with patch.object(HotpotQAEnvironment, "_post", _patched_post([response])):
        obs, reward, done = env.step("answer[Strange]")
    assert reward == 0.5
    assert env.last_step_info["em"] == 0.0
    assert env.last_step_info["won"] is False


def test_step_invalid_action_returns_zero_keeps_obs():
    env = HotpotQAEnvironment(base_url="http://x")
    env._session_id = "s1"
    env._last_obs = "current page"
    response = {"error": "parse", "obs": "", "reward": 0.0, "done": False}
    with patch.object(HotpotQAEnvironment, "_post", _patched_post([response])):
        obs, reward, done = env.step("nonsense")
    assert reward == 0.0
    assert done is False
    assert obs.text == "current page"


def test_anchor_is_observation_text():
    env = HotpotQAEnvironment(base_url="http://x")
    response = {
        "session_id": "s",
        "obs": "page A",
        "available_actions": {"tools": ["search", "answer"], "titles": []},
        "instruction": "Q?",
        "info": {},
    }
    with patch.object(HotpotQAEnvironment, "_post", _patched_post([response])):
        obs = env.reset("")
    assert obs.anchor == "page A"


def test_stop_strings_present():
    env = HotpotQAEnvironment()
    assert "</action>" in env.stop_strings


def test_invalid_action_message_distinguishes_modes():
    env_text = HotpotQAEnvironment(use_tools_schema=False)
    assert "<action>" in env_text.invalid_action_message
    env_json = HotpotQAEnvironment(use_tools_schema=True)
    assert "tool_call" in env_json.invalid_action_message
