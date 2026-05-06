"""Unit tests for NumberLineEnvironment."""

from __future__ import annotations

from mlx_agent_rl.environments.numberline import NumberLineEnvironment


def test_reset_parses_prompt():
    env = NumberLineEnvironment(max_position=5)
    obs = env.reset("start=2|goal=5")
    assert obs.text == "Target number: 5. Current number: 2"
    assert env._goal == 5
    assert env._position == 2


def test_step_plus_moves_closer_zero_reward():
    env = NumberLineEnvironment(max_position=5)
    env.reset("start=2|goal=5")
    obs, reward, done = env.step("+")
    assert env._position == 3
    assert reward == 0.0  # closer
    assert done is False


def test_step_minus_moves_away_negative_reward():
    env = NumberLineEnvironment(max_position=5)
    env.reset("start=2|goal=5")
    obs, reward, done = env.step("-")
    assert env._position == 1
    assert reward == -1.0  # farther
    assert done is False


def test_step_arrives_at_goal_reward_one_done():
    env = NumberLineEnvironment(max_position=5)
    env.reset("start=4|goal=5")
    obs, reward, done = env.step("+")
    assert env._position == 5
    assert reward == 1.0
    assert done is True


def test_step_clipped_at_bounds():
    env = NumberLineEnvironment(max_position=5)
    env.reset("start=0|goal=2")
    # try to go below zero: position stays 0, distance unchanged -> reward -1
    _, reward, _ = env.step("-")
    assert env._position == 0
    assert reward == -1.0


def test_truncation_after_2max_steps():
    env = NumberLineEnvironment(max_position=3)
    env.reset("start=0|goal=3")
    # take 6 wrong steps; should truncate
    for _ in range(6):
        obs, _, done = env.step("-")  # always pinned at 0, can never reach 3
    assert done is True
    assert env._steps == 6  # 2 * max_position


def test_extract_action_tool_call_plus():
    env = NumberLineEnvironment()
    out = '<tool_call>\n{"name":"move","arguments":{"direction":"+"}}\n</tool_call>'
    assert env.extract_action(out) == "+"


def test_extract_action_tool_call_minus():
    env = NumberLineEnvironment()
    out = '<tool_call>{"name":"move","arguments":{"direction":"-"}}</tool_call>'
    assert env.extract_action(out) == "-"


def test_extract_action_unknown_tool_returns_none():
    env = NumberLineEnvironment()
    out = '<tool_call>{"name":"answer","arguments":{"value":3}}</tool_call>'
    assert env.extract_action(out) is None


def test_extract_action_fallback_lone_plus():
    env = NumberLineEnvironment()
    assert env.extract_action("I think we should go +") == "+"


def test_extract_action_fallback_lone_minus():
    env = NumberLineEnvironment()
    assert env.extract_action("Subtract 1 (use -)") == "-"


def test_extract_action_ambiguous_returns_none():
    env = NumberLineEnvironment()
    # contains both '+' and '-' free-floating, no tool_call -> ambiguous
    assert env.extract_action("the answer is +1 or -1") is None


def test_extract_action_no_signal_returns_none():
    env = NumberLineEnvironment()
    assert env.extract_action("hello world, no direction here") is None


def test_get_tools_schema_has_move():
    env = NumberLineEnvironment()
    schema = env.get_tools_schema()
    assert len(schema) == 1
    assert schema[0]["function"]["name"] == "move"
    assert "direction" in schema[0]["function"]["parameters"]["properties"]
