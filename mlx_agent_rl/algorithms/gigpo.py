import math
from collections import defaultdict

from mlx_agent_rl.algorithms.grpo import GRPOEstimator
from mlx_agent_rl.data.trajectory import Trajectory


class GiGPOEstimator(GRPOEstimator):
    """GiGPO (Group-indexed GRPO) estimator with two-level advantage.

    Combines episode-level advantage (from GRPO) with a step-level advantage
    computed by grouping steps that share the same anchor state across
    trajectories in the same uid group.

    Final advantage = episode_adv + step_advantage_w * step_adv
    """

    def __init__(
        self,
        epsilon: float = 1e-4,
        step_advantage_w: float = 1.0,
        gamma: float = 0.99,
    ) -> None:
        super().__init__(epsilon=epsilon)
        self.step_advantage_w = step_advantage_w
        self.gamma = gamma

    def _discounted_returns(self, trajectory: Trajectory) -> list[float]:
        """Compute discounted future return for each step."""
        returns = [0.0] * trajectory.total_steps
        running = 0.0
        for i in reversed(range(trajectory.total_steps)):
            running = trajectory.steps[i].reward + self.gamma * running
            returns[i] = running
        return returns

    def compute(self, trajectories: list[Trajectory]) -> list[list[float]]:
        # Episode-level advantages via GRPO
        episode_advantages = super().compute(trajectories)

        # Compute discounted returns for every step in every trajectory
        all_returns: list[list[float]] = [
            self._discounted_returns(traj) for traj in trajectories
        ]

        # Group (step_idx, return_value) by (uid, anchor_obs)
        # Key: (uid, anchor_obs) -> list of (traj_idx, step_idx, return_value)
        anchor_groups: dict[tuple[str, str], list[tuple[int, int, float]]] = defaultdict(list)
        for traj_idx, traj in enumerate(trajectories):
            for step_idx, step in enumerate(traj.steps):
                key = (traj.uid, step.anchor_obs)
                anchor_groups[key].append((traj_idx, step_idx, all_returns[traj_idx][step_idx]))

        # Normalize returns within each anchor group; groups of size 1 get 0
        step_advantages: list[list[float]] = [
            [0.0] * traj.total_steps for traj in trajectories
        ]
        for key, entries in anchor_groups.items():
            if len(entries) < 2:
                # No peer steps to compare against; step advantage stays 0
                continue
            ret_values = [e[2] for e in entries]
            mean = sum(ret_values) / len(ret_values)
            variance = sum((r - mean) ** 2 for r in ret_values) / len(ret_values)
            std = math.sqrt(variance)
            for traj_idx, step_idx, ret in entries:
                step_advantages[traj_idx][step_idx] = (ret - mean) / (std + self.epsilon)

        # Combine episode and step advantages
        final_advantages: list[list[float]] = []
        for traj_idx, traj in enumerate(trajectories):
            combined = [
                episode_advantages[traj_idx][s] + self.step_advantage_w * step_advantages[traj_idx][s]
                for s in range(traj.total_steps)
            ]
            final_advantages.append(combined)

        return final_advantages
