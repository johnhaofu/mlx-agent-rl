"""Tests for the RolloutCollector."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mlx_agent_rl.core.rollout import RolloutCollector
from mlx_agent_rl.data.trajectory import Trajectory
from mlx_agent_rl.environments.calculator import CalculatorEnvironment
from mlx_agent_rl.memory.memory import SlidingMemory


def _make_mock_policy(action_output: str = "<action>answer(4)</action>"):
    """Create a mock policy that always returns the given action string."""
    policy = MagicMock()
    # tokenizer.encode returns a list of ints
    policy.tokenizer.encode.side_effect = lambda text: list(range(len(text.split())))
    # generate_with_log_probs returns (text, log_probs)
    policy.generate_with_log_probs.return_value = (action_output, [-0.5, -0.4, -0.6])
    return policy


def _make_collector(policy, group_size_default=4):
    env = CalculatorEnvironment()
    memory = SlidingMemory(window_size=3)
    return RolloutCollector(
        policy=policy,
        env=env,
        memory=memory,
        max_steps=5,
        max_tokens=64,
        invalid_action_penalty=-0.1,
    )


# ------------------------------------------------------------------
# Basic single-trajectory test
# ------------------------------------------------------------------

def test_single_trajectory_completes():
    """A single rollout should produce a non-empty Trajectory."""
    policy = _make_mock_policy("<action>answer(4)</action>")
    collector = _make_collector(policy)
    prompts = [{"prompt": "What is 2 + 2?", "answer": "4"}]
    trajectories = collector.collect(prompts, group_size=1)

    assert len(trajectories) == 1
    traj = trajectories[0]
    assert isinstance(traj, Trajectory)
    assert traj.total_steps >= 1


def test_trajectory_fields_populated():
    """Steps should have non-empty token lists and a log-prob list."""
    policy = _make_mock_policy("<action>answer(4)</action>")
    collector = _make_collector(policy)
    prompts = [{"prompt": "What is 2 + 2?", "answer": "4"}]
    trajectories = collector.collect(prompts, group_size=1)
    traj = trajectories[0]

    for step in traj.steps:
        assert isinstance(step.prompt_tokens, list)
        assert isinstance(step.action_tokens, list)
        assert isinstance(step.log_probs, list)
        assert isinstance(step.reward, float)
        assert isinstance(step.done, bool)


# ------------------------------------------------------------------
# Group-size tests
# ------------------------------------------------------------------

def test_group_size_produces_correct_number_of_trajectories():
    """group_size=4 should yield 4 trajectories for 1 prompt."""
    policy = _make_mock_policy("<action>answer(4)</action>")
    collector = _make_collector(policy)
    prompts = [{"prompt": "What is 2 + 2?", "answer": "4"}]
    trajectories = collector.collect(prompts, group_size=4)

    assert len(trajectories) == 4


def test_group_size_multiple_prompts():
    """M prompts × group_size should yield M*group_size trajectories."""
    policy = _make_mock_policy("<action>answer(4)</action>")
    collector = _make_collector(policy)
    prompts = [
        {"prompt": "What is 2 + 2?", "answer": "4"},
        {"prompt": "What is 3 + 3?", "answer": "6"},
    ]
    trajectories = collector.collect(prompts, group_size=3)
    assert len(trajectories) == 6


# ------------------------------------------------------------------
# UID sharing within a group
# ------------------------------------------------------------------

def test_same_uid_within_group():
    """All trajectories for a single prompt must share the same uid."""
    policy = _make_mock_policy("<action>answer(4)</action>")
    collector = _make_collector(policy)
    prompts = [{"prompt": "What is 2 + 2?", "answer": "4"}]
    trajectories = collector.collect(prompts, group_size=4)

    uids = {t.uid for t in trajectories}
    assert len(uids) == 1, "All group members should share the same uid"


def test_different_uid_across_prompts():
    """Different prompts should produce different uids."""
    policy = _make_mock_policy("<action>answer(4)</action>")
    collector = _make_collector(policy)
    prompts = [
        {"prompt": "What is 2 + 2?", "answer": "4"},
        {"prompt": "What is 3 + 3?", "answer": "6"},
    ]
    trajectories = collector.collect(prompts, group_size=2)
    uid_group1 = {t.uid for t in trajectories[:2]}
    uid_group2 = {t.uid for t in trajectories[2:]}
    assert uid_group1.isdisjoint(uid_group2)


# ------------------------------------------------------------------
# Invalid action handling
# ------------------------------------------------------------------

def test_invalid_action_gets_penalty():
    """When the model outputs no valid action, the step reward should equal the penalty."""
    policy = _make_mock_policy("this is not a valid action")
    collector = _make_collector(policy)
    prompts = [{"prompt": "What is 2 + 2?", "answer": "4"}]
    trajectories = collector.collect(prompts, group_size=1)

    traj = trajectories[0]
    assert traj.total_steps >= 1
    # All steps should have the penalty reward (no valid action ever fired)
    for step in traj.steps:
        assert step.reward == pytest.approx(-0.1)


# ------------------------------------------------------------------
# Successful episode
# ------------------------------------------------------------------

def test_correct_answer_gives_positive_reward():
    """A correct answer(N) action should yield reward 1.0 and done=True."""
    policy = _make_mock_policy("<action>answer(4)</action>")
    collector = _make_collector(policy)
    prompts = [{"prompt": "What is 2 + 2?", "answer": "4"}]
    trajectories = collector.collect(prompts, group_size=1)

    traj = trajectories[0]
    last_step = traj.steps[-1]
    assert last_step.done is True
    assert last_step.reward == pytest.approx(1.0)
    assert traj.episode_reward == pytest.approx(1.0)
