"""Integration tests: full pipeline with mock policy."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mlx_agent_rl.algorithms.gigpo import GiGPOEstimator
from mlx_agent_rl.algorithms.grpo import GRPOEstimator
from mlx_agent_rl.core.rollout import RolloutCollector
from mlx_agent_rl.data.trajectory import Trajectory
from mlx_agent_rl.environments.calculator import CalculatorEnvironment
from mlx_agent_rl.memory.memory import SlidingMemory


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_mock_policy(action_output: str = "<action>answer(4)</action>"):
    """Mock policy that always returns the same action string."""
    policy = MagicMock()
    # Tokenizer encodes by splitting on spaces (gives stable token count)
    policy.tokenizer.encode.side_effect = lambda text: list(range(max(1, len(text.split()))))
    policy.generate_with_log_probs.return_value = (
        action_output,
        [-0.5, -0.4, -0.6],
        [101, 102, 103],
    )
    return policy


def _build_collector(policy, max_steps: int = 5) -> RolloutCollector:
    env = CalculatorEnvironment()
    memory = SlidingMemory(window_size=3)
    return RolloutCollector(
        policy=policy,
        env=env,
        memory=memory,
        max_steps=max_steps,
        max_tokens=64,
        invalid_action_penalty=-0.1,
    )


PROMPTS = [
    {"prompt": "What is 2 + 2?", "answer": "4"},
    {"prompt": "What is 3 * 3?", "answer": "9"},
]


# ---------------------------------------------------------------------------
# Pipeline smoke tests
# ---------------------------------------------------------------------------


def test_full_pipeline_runs():
    """End-to-end: collect -> GRPO advantages -> shape checks."""
    policy = _make_mock_policy("<action>answer(4)</action>")
    collector = _build_collector(policy)

    group_size = 4
    trajectories = collector.collect(PROMPTS, group_size=group_size)

    # Should have len(PROMPTS) * group_size trajectories
    expected_count = len(PROMPTS) * group_size
    assert len(trajectories) == expected_count

    # All must be Trajectory instances with at least 1 step
    for traj in trajectories:
        assert isinstance(traj, Trajectory)
        assert traj.total_steps >= 1


def test_grpo_advantages_shape():
    """GRPOEstimator.compute output shape must match trajectories."""
    policy = _make_mock_policy("<action>answer(4)</action>")
    collector = _build_collector(policy)
    group_size = 4
    trajectories = collector.collect(PROMPTS, group_size=group_size)

    estimator = GRPOEstimator()
    advantages = estimator.compute(trajectories)

    # Outer list length matches trajectories
    assert len(advantages) == len(trajectories)

    # Inner list length matches each trajectory's total_steps
    for traj, adv_list in zip(trajectories, advantages):
        assert len(adv_list) == traj.total_steps


def test_gigpo_advantages_shape():
    """GiGPOEstimator.compute output shape must match trajectories."""
    policy = _make_mock_policy("<action>answer(4)</action>")
    collector = _build_collector(policy)
    group_size = 4
    trajectories = collector.collect(PROMPTS, group_size=group_size)

    estimator = GiGPOEstimator()
    advantages = estimator.compute(trajectories)

    assert len(advantages) == len(trajectories)
    for traj, adv_list in zip(trajectories, advantages):
        assert len(adv_list) == traj.total_steps


# ---------------------------------------------------------------------------
# Advantage value sanity checks
# ---------------------------------------------------------------------------


def test_grpo_advantages_values_are_floats():
    """All advantage values should be Python floats."""
    policy = _make_mock_policy("<action>answer(4)</action>")
    collector = _build_collector(policy)
    trajectories = collector.collect(PROMPTS, group_size=4)
    advantages = GRPOEstimator().compute(trajectories)

    for adv_list in advantages:
        for v in adv_list:
            assert isinstance(v, float)


def test_gigpo_advantages_values_are_floats():
    policy = _make_mock_policy("<action>answer(4)</action>")
    collector = _build_collector(policy)
    trajectories = collector.collect(PROMPTS, group_size=4)
    advantages = GiGPOEstimator().compute(trajectories)

    for adv_list in advantages:
        for v in adv_list:
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# Group structure tests
# ---------------------------------------------------------------------------


def test_uid_grouping_preserved():
    """Trajectories from the same prompt should share a uid."""
    policy = _make_mock_policy("<action>answer(4)</action>")
    collector = _build_collector(policy)
    group_size = 4
    trajectories = collector.collect(PROMPTS[:1], group_size=group_size)

    uids = [t.uid for t in trajectories]
    assert len(set(uids)) == 1, "All trajectories for one prompt share a uid"


def test_different_prompts_different_uids():
    policy = _make_mock_policy("<action>answer(4)</action>")
    collector = _build_collector(policy)
    group_size = 2
    trajectories = collector.collect(PROMPTS, group_size=group_size)

    uid_set_0 = {t.uid for t in trajectories[:group_size]}
    uid_set_1 = {t.uid for t in trajectories[group_size:]}
    assert uid_set_0.isdisjoint(uid_set_1)


# ---------------------------------------------------------------------------
# Invalid-action mixed pipeline
# ---------------------------------------------------------------------------


def test_invalid_actions_included_in_trajectory():
    """Even when all actions are invalid the pipeline should complete."""
    policy = _make_mock_policy("no action here")
    collector = _build_collector(policy, max_steps=3)
    trajectories = collector.collect(PROMPTS[:1], group_size=2)

    for traj in trajectories:
        assert traj.total_steps >= 1
        # All steps should carry the penalty reward
        for step in traj.steps:
            assert step.reward == pytest.approx(-0.1)

    advantages = GRPOEstimator().compute(trajectories)
    assert len(advantages) == len(trajectories)
    for traj, adv_list in zip(trajectories, advantages):
        assert len(adv_list) == traj.total_steps
