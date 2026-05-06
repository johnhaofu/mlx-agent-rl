import pytest
from mlx_agent_rl.algorithms.gigpo import GiGPOEstimator
from mlx_agent_rl.algorithms.grpo import GRPOEstimator
from mlx_agent_rl.data.trajectory import Step, Trajectory


def make_step(reward: float = 0.0, done: bool = False, anchor_obs: str = "s0") -> Step:
    return Step(
        prompt_tokens=[1],
        action_tokens=[2],
        log_probs=[-0.5],
        reward=reward,
        done=done,
        anchor_obs=anchor_obs,
    )


def make_trajectory(uid: str, episode_reward: float, steps: list[Step]) -> Trajectory:
    return Trajectory(steps=steps, episode_reward=episode_reward, uid=uid)


class TestGiGPOEstimator:
    def test_shared_anchor_state_different_outcomes(self):
        """Two trajectories sharing anchor at step 0 but diverging should get different step advs."""
        # Both start at "state_A"; traj1 gets high reward, traj2 gets low reward
        traj1 = make_trajectory(
            uid="g1",
            episode_reward=2.0,
            steps=[
                make_step(reward=1.0, anchor_obs="state_A"),
                make_step(reward=1.0, done=True, anchor_obs="state_B"),
            ],
        )
        traj2 = make_trajectory(
            uid="g1",
            episode_reward=0.0,
            steps=[
                make_step(reward=0.0, anchor_obs="state_A"),
                make_step(reward=0.0, done=True, anchor_obs="state_C"),
            ],
        )

        estimator = GiGPOEstimator(step_advantage_w=1.0, gamma=0.99)
        advantages = estimator.compute([traj1, traj2])

        # Traj1 step 0 should have higher advantage than traj2 step 0
        # (it has better discounted return from "state_A")
        assert advantages[0][0] > advantages[1][0], (
            f"Traj1 step 0 adv {advantages[0][0]} should exceed traj2 step 0 adv {advantages[1][0]}"
        )

    def test_no_shared_states_falls_back_to_episode(self):
        """When no anchor states are shared, step advantages are 0 and result equals GRPO."""
        traj1 = make_trajectory(
            uid="g1",
            episode_reward=2.0,
            steps=[
                make_step(reward=1.0, anchor_obs="state_A"),
                make_step(reward=1.0, done=True, anchor_obs="state_B"),
            ],
        )
        traj2 = make_trajectory(
            uid="g1",
            episode_reward=0.0,
            steps=[
                make_step(reward=0.0, anchor_obs="state_C"),  # unique anchor
                make_step(reward=0.0, done=True, anchor_obs="state_D"),
            ],
        )

        epsilon = 1e-4
        gigpo = GiGPOEstimator(epsilon=epsilon, step_advantage_w=1.0, gamma=0.99)
        grpo = GRPOEstimator(epsilon=epsilon)

        gigpo_advs = gigpo.compute([traj1, traj2])
        grpo_advs = grpo.compute([traj1, traj2])

        # When no states are shared, step_adv = 0 for all steps, so final == episode_adv
        for traj_idx in range(2):
            for step_idx in range(2):
                assert gigpo_advs[traj_idx][step_idx] == pytest.approx(
                    grpo_advs[traj_idx][step_idx]
                ), f"Mismatch at traj={traj_idx}, step={step_idx}"

    def test_step_advantage_weight_zero(self):
        """With step_advantage_w=0, GiGPO should match pure GRPO."""
        traj1 = make_trajectory(
            uid="g1",
            episode_reward=3.0,
            steps=[make_step(reward=1.5, anchor_obs="shared"), make_step(reward=1.5, done=True, anchor_obs="shared")],
        )
        traj2 = make_trajectory(
            uid="g1",
            episode_reward=1.0,
            steps=[make_step(reward=0.5, anchor_obs="shared"), make_step(reward=0.5, done=True, anchor_obs="shared")],
        )

        epsilon = 1e-4
        gigpo = GiGPOEstimator(epsilon=epsilon, step_advantage_w=0.0, gamma=0.99)
        grpo = GRPOEstimator(epsilon=epsilon)

        gigpo_advs = gigpo.compute([traj1, traj2])
        grpo_advs = grpo.compute([traj1, traj2])

        for traj_idx in range(2):
            for step_idx in range(2):
                assert gigpo_advs[traj_idx][step_idx] == pytest.approx(
                    grpo_advs[traj_idx][step_idx]
                )

    def test_inherits_from_grpo(self):
        """GiGPOEstimator should inherit from GRPOEstimator."""
        assert issubclass(GiGPOEstimator, GRPOEstimator)

    def test_constructor_params_stored(self):
        """Constructor params should be stored correctly."""
        estimator = GiGPOEstimator(epsilon=1e-3, step_advantage_w=0.5, gamma=0.95)
        assert estimator.epsilon == pytest.approx(1e-3)
        assert estimator.step_advantage_w == pytest.approx(0.5)
        assert estimator.gamma == pytest.approx(0.95)

    def test_discounted_return_uses_gamma(self):
        """Verify that discounted returns correctly apply gamma."""
        # Single trajectory, no group peers, so step advantages = 0
        # We test via two trajectories sharing anchor to see gamma effect
        traj1 = make_trajectory(
            uid="g1",
            episode_reward=1.0,
            steps=[
                make_step(reward=0.0, anchor_obs="shared"),
                make_step(reward=1.0, done=True, anchor_obs="end_A"),
            ],
        )
        traj2 = make_trajectory(
            uid="g1",
            episode_reward=0.99,
            steps=[
                make_step(reward=0.99, anchor_obs="shared"),
                make_step(reward=0.0, done=True, anchor_obs="end_B"),
            ],
        )
        # At step 0 (shared anchor), traj1 discounted return = 0 + 0.99*1 = 0.99
        # traj2 discounted return = 0.99 + 0.99*0 = 0.99
        # Returns are equal → step advantages at step 0 should both be ~0
        estimator = GiGPOEstimator(step_advantage_w=1.0, gamma=0.99)
        advantages = estimator.compute([traj1, traj2])
        # Episode advantages cancel out the step component when returns are equal
        # We just check the step advantage part doesn't explode
        # Both should have small step_adv since returns at "shared" are approx equal
        grpo = GRPOEstimator()
        grpo_advs = grpo.compute([traj1, traj2])
        # The difference between gigpo and grpo should be small when returns are equal
        diff0 = abs(advantages[0][0] - grpo_advs[0][0])
        diff1 = abs(advantages[1][0] - grpo_advs[1][0])
        assert diff0 < 0.1, f"Step advantage contribution too large: {diff0}"
        assert diff1 < 0.1, f"Step advantage contribution too large: {diff1}"
