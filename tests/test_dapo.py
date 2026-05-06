import pytest
from mlx_agent_rl.algorithms.dapo import DAPOEstimator
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


class TestDAPOEstimator:
    def test_default_clip_params(self):
        """Default epsilon_low and epsilon_high should be set correctly."""
        estimator = DAPOEstimator()
        assert estimator.epsilon_low == pytest.approx(0.2)
        assert estimator.epsilon_high == pytest.approx(0.28)

    def test_custom_clip_params_stored(self):
        """Custom clip params should be stored on the estimator."""
        estimator = DAPOEstimator(epsilon_low=0.1, epsilon_high=0.5)
        assert estimator.epsilon_low == pytest.approx(0.1)
        assert estimator.epsilon_high == pytest.approx(0.5)

    def test_advantages_match_grpo(self):
        """DAPO advantage computation should be identical to GRPO."""
        epsilon = 1e-4
        trajs = [
            make_trajectory("g1", 3.0),
            make_trajectory("g1", 1.0),
            make_trajectory("g2", 5.0),
            make_trajectory("g2", 2.0),
        ]
        dapo = DAPOEstimator(epsilon=epsilon)
        grpo = GRPOEstimator(epsilon=epsilon)

        dapo_advs = dapo.compute(trajs)
        grpo_advs = grpo.compute(trajs)

        for traj_idx in range(len(trajs)):
            for step_idx in range(trajs[traj_idx].total_steps):
                assert dapo_advs[traj_idx][step_idx] == pytest.approx(
                    grpo_advs[traj_idx][step_idx]
                ), f"Mismatch at traj={traj_idx}, step={step_idx}"

    def test_inherits_from_grpo(self):
        """DAPOEstimator should be a subclass of GRPOEstimator."""
        assert issubclass(DAPOEstimator, GRPOEstimator)

    def test_epsilon_param_forwarded(self):
        """Epsilon param should be forwarded to the parent GRPO estimator."""
        estimator = DAPOEstimator(epsilon=1e-6)
        assert estimator.epsilon == pytest.approx(1e-6)
