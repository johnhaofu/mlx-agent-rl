import pytest
from mlx_agent_rl.algorithms.dr_grpo import DrGRPOEstimator
from mlx_agent_rl.data.trajectory import Step, Trajectory


def make_step(anchor_obs: str = "s0") -> Step:
    return Step(
        prompt_tokens=[1],
        action_tokens=[2],
        log_probs=[-0.5],
        reward=0.0,
        done=False,
        anchor_obs=anchor_obs,
    )


def make_trajectory(uid: str, episode_reward: float, n_steps: int = 2) -> Trajectory:
    steps = [make_step() for _ in range(n_steps)]
    return Trajectory(steps=steps, episode_reward=episode_reward, uid=uid)


class TestDrGRPOEstimator:
    def test_advantage_equals_reward_minus_mean(self):
        """advantage = reward - mean_reward exactly (no std division)."""
        estimator = DrGRPOEstimator()
        rewards = [3.0, 1.0, 2.0]
        trajs = [make_trajectory("g1", r) for r in rewards]
        mean = sum(rewards) / len(rewards)
        advantages = estimator.compute(trajs)
        for i, r in enumerate(rewards):
            expected = r - mean
            for adv in advantages[i]:
                assert adv == pytest.approx(expected), (
                    f"Traj {i}: expected {expected}, got {adv}"
                )

    def test_same_reward_gives_zero_advantage(self):
        """Identical rewards in a group yield zero advantage."""
        estimator = DrGRPOEstimator()
        trajs = [make_trajectory("g1", 5.0), make_trajectory("g1", 5.0)]
        advantages = estimator.compute(trajs)
        for adv_list in advantages:
            for adv in adv_list:
                assert adv == pytest.approx(0.0)

    def test_no_std_division(self):
        """Dr. GRPO should NOT divide by std — verify via extreme variance case."""
        estimator = DrGRPOEstimator()
        # Rewards 0 and 100 — if std division were applied, advantage would be ~±1
        trajs = [make_trajectory("g1", 0.0), make_trajectory("g1", 100.0)]
        advantages = estimator.compute(trajs)
        # mean = 50; advantage for traj 0 = 0 - 50 = -50
        assert advantages[0][0] == pytest.approx(-50.0)
        assert advantages[1][0] == pytest.approx(50.0)

    def test_multiple_groups_independent(self):
        """Groups are normalized independently."""
        estimator = DrGRPOEstimator()
        trajs = [
            make_trajectory("A", 4.0),
            make_trajectory("A", 2.0),
            make_trajectory("B", 10.0),
            make_trajectory("B", 0.0),
        ]
        advantages = estimator.compute(trajs)
        # Group A mean = 3; advantages = [1, -1]
        assert advantages[0][0] == pytest.approx(1.0)
        assert advantages[1][0] == pytest.approx(-1.0)
        # Group B mean = 5; advantages = [5, -5]
        assert advantages[2][0] == pytest.approx(5.0)
        assert advantages[3][0] == pytest.approx(-5.0)

    def test_all_steps_in_trajectory_same_advantage(self):
        """All steps within a trajectory share the same advantage."""
        estimator = DrGRPOEstimator()
        trajs = [make_trajectory("g1", 3.0, n_steps=4), make_trajectory("g1", 1.0, n_steps=3)]
        advantages = estimator.compute(trajs)
        for adv_list in advantages:
            assert len(set(adv_list)) == 1
