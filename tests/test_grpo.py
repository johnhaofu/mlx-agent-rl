import pytest
from mlx_agent_rl.algorithms.grpo import GRPOEstimator
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


class TestGRPOEstimator:
    def test_same_reward_gives_zero_advantage(self):
        """All trajectories in a group with the same reward should yield ~0 advantage."""
        estimator = GRPOEstimator()
        trajs = [make_trajectory("g1", 1.0), make_trajectory("g1", 1.0)]
        advantages = estimator.compute(trajs)
        for adv_list in advantages:
            for adv in adv_list:
                assert abs(adv) < 1.0, f"Expected near-zero advantage, got {adv}"

    def test_higher_reward_positive_advantage(self):
        """Trajectory with higher reward should receive positive advantage."""
        estimator = GRPOEstimator()
        trajs = [make_trajectory("g1", 2.0), make_trajectory("g1", 0.0)]
        advantages = estimator.compute(trajs)
        # First trajectory has higher reward; all its steps should have positive advantage
        for adv in advantages[0]:
            assert adv > 0, f"Expected positive advantage, got {adv}"
        for adv in advantages[1]:
            assert adv < 0, f"Expected negative advantage, got {adv}"

    def test_multiple_groups_independent(self):
        """Trajectories in different groups should be normalized independently."""
        estimator = GRPOEstimator()
        # Group A: rewards [10, 0]
        # Group B: rewards [1, 0]
        trajs = [
            make_trajectory("A", 10.0),
            make_trajectory("A", 0.0),
            make_trajectory("B", 1.0),
            make_trajectory("B", 0.0),
        ]
        advantages = estimator.compute(trajs)

        # Within group A, high reward traj gets positive advantage
        assert advantages[0][0] > 0
        assert advantages[1][0] < 0

        # Within group B, high reward traj gets positive advantage
        assert advantages[2][0] > 0
        assert advantages[3][0] < 0

        # The normalized advantage magnitudes within each group should be equal
        # since both groups have 2 trajs with rewards differing by equal z-score
        assert abs(advantages[0][0]) == pytest.approx(abs(advantages[1][0]))
        assert abs(advantages[2][0]) == pytest.approx(abs(advantages[3][0]))

    def test_all_steps_in_trajectory_same_advantage(self):
        """All steps within a trajectory must share the same episode-level advantage."""
        estimator = GRPOEstimator()
        trajs = [make_trajectory("g1", 3.0, n_steps=4), make_trajectory("g1", 1.0, n_steps=3)]
        advantages = estimator.compute(trajs)
        for adv_list in advantages:
            assert len(set(adv_list)) == 1, "All steps should share the same advantage"

    def test_output_shape_matches_trajectories(self):
        """Output shape should match number of trajectories and steps."""
        estimator = GRPOEstimator()
        trajs = [
            make_trajectory("g1", 1.0, n_steps=3),
            make_trajectory("g1", 2.0, n_steps=5),
        ]
        advantages = estimator.compute(trajs)
        assert len(advantages) == 2
        assert len(advantages[0]) == 3
        assert len(advantages[1]) == 5
