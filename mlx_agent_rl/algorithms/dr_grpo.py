from collections import defaultdict

from mlx_agent_rl.algorithms.base import AdvantageEstimator
from mlx_agent_rl.data.trajectory import Trajectory


class DrGRPOEstimator(AdvantageEstimator):
    """Decoupled GRPO (Dr. GRPO) advantage estimator.

    Uses mean-only normalization (no std division) for improved training
    stability. advantage = reward - mean_reward within the group.
    """

    def compute(self, trajectories: list[Trajectory]) -> list[list[float]]:
        # Group trajectories by uid
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, traj in enumerate(trajectories):
            groups[traj.uid].append(idx)

        # Compute per-group mean of episode rewards
        group_mean: dict[str, float] = {}
        for uid, indices in groups.items():
            rewards = [trajectories[i].episode_reward for i in indices]
            group_mean[uid] = sum(rewards) / len(rewards)

        # Assign advantages
        advantages: list[list[float]] = []
        for traj in trajectories:
            mean = group_mean[traj.uid]
            episode_adv = traj.episode_reward - mean
            advantages.append([episode_adv] * traj.total_steps)

        return advantages
