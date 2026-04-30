"""Unit tests for WebShopEnvironment.

The env is HTTP-only — we monkey-patch ``_post`` so tests don't need a
running sidecar.
"""

from __future__ import annotations

from unittest.mock import patch

from mlx_agent_rl.environments.webshop import WebShopEnvironment


def _patched_post(responses: list):
    """Return a callable that yields the next pre-canned response on each call."""
    iterator = iter(responses)

    def fake(self, path, body):  # noqa: ARG001
        return next(iterator)

    return fake


def test_extract_action_proper():
    env = WebShopEnvironment()
    out = "<think>think</think><action>search[red jacket]</action>"
    assert env.extract_action(out) == "search[red jacket]"


def test_extract_action_missing_think_rejected_when_required():
    env = WebShopEnvironment(require_think=True)
    assert env.extract_action("<action>click[B0001]</action>") is None


def test_extract_action_chinese_rejected():
    env = WebShopEnvironment()
    out = "<think>用中文</think><action>search[red]</action>"
    assert env.extract_action(out) is None


def test_extract_action_no_action_returns_none():
    env = WebShopEnvironment()
    assert env.extract_action("just thinking, no brackets here") is None


def test_extract_action_empty_action_returns_none():
    env = WebShopEnvironment()
    assert env.extract_action("<think>x</think><action></action>") is None


def test_extract_action_default_allows_no_think():
    env = WebShopEnvironment()  # default require_think=False
    assert env.extract_action("<action>click[B0001]</action>") == "click[B0001]"


def test_extract_action_raw_search_fallback():
    env = WebShopEnvironment()
    out = "search[red jacket size m]"
    assert env.extract_action(out) == "search[red jacket size m]"


def test_extract_action_raw_click_fallback():
    env = WebShopEnvironment()
    out = "I think I should click[Buy Now] now."
    assert env.extract_action(out) == "click[Buy Now]"


def test_extract_action_raw_picks_last_when_multiple():
    env = WebShopEnvironment()
    out = "Admissible: [search[query], click[search]]\nclick[search]"
    assert env.extract_action(out) == "click[search]"


def test_reset_calls_sidecar_with_seed_idx():
    env = WebShopEnvironment(base_url="http://x")
    response = {
        "session_id": "sess-1",
        "obs": "WebShop [SEP] hello",
        "available_actions": {"clickables": ["search"], "has_search_bar": True},
        "instruction": "Find a red shirt",
        "info": {},
    }
    with patch.object(WebShopEnvironment, "_post", _patched_post([response])):
        obs = env.reset(prompt="", answer=42)
    assert obs.text.startswith("WebShop")
    assert env.instruction == "Find a red shirt"
    assert env._session_id == "sess-1"


def test_step_shapes_reward_zero_when_lost():
    env = WebShopEnvironment(base_url="http://x")
    env._session_id = "s1"
    env._last_obs = "page"
    response = {
        "obs": "next page",
        "reward": 0.4,  # raw reward 0.4 (lost)
        "done": True,
        "won": False,
        "available_actions": {"clickables": []},
        "info": {},
    }
    with patch.object(WebShopEnvironment, "_post", _patched_post([response])):
        obs, reward, done = env.step("click[buy now]")
    # verl shaping: not won → reward forced to 0
    assert reward == 0.0
    assert done is True


def test_step_shapes_reward_ten_when_won():
    env = WebShopEnvironment(base_url="http://x")
    env._session_id = "s1"
    env._last_obs = "page"
    response = {
        "obs": "thank you",
        "reward": 1.0,
        "done": True,
        "won": True,
        "available_actions": {"clickables": []},
        "info": {},
    }
    with patch.object(WebShopEnvironment, "_post", _patched_post([response])):
        obs, reward, done = env.step("click[buy now]")
    assert reward == 10.0
    assert done is True


def test_step_invalid_action_returns_zero_reward_no_done():
    env = WebShopEnvironment(base_url="http://x")
    env._session_id = "s1"
    env._last_obs = "page"
    response = {"error": "invalid", "obs": "", "reward": 0.0, "done": False}
    with patch.object(WebShopEnvironment, "_post", _patched_post([response])):
        obs, reward, done = env.step("nonsense")
    assert reward == 0.0
    assert done is False
    # observation falls back to last_obs
    assert obs.text == "page"


def test_anchor_is_observation_text():
    env = WebShopEnvironment(base_url="http://x")
    env._last_obs = "page A"
    response = {
        "session_id": "s",
        "obs": "page A",
        "available_actions": {},
        "instruction": "",
        "info": {},
    }
    with patch.object(WebShopEnvironment, "_post", _patched_post([response])):
        obs = env.reset("")
    assert obs.anchor == "page A"
