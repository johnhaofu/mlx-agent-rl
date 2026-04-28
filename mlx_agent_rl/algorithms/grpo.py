from collections import defaultdict
import math

from mlx_agent_rl.algorithms.base import AdvantageEstimator
from mlx_agent_rl.data.trajectory import Trajectory


class GRPOEstimator(AdvantageEstimator):
    """Group Relative Policy Optimization advantage estimator.

    Within each group (identified by uid), normalizes episode rewards using
    mean and standard deviation. All steps within a trajectory share the same
    episode-level advantage.
    """

    def __init__(self, epsilon: float = 1e-4) -> None:
        self.epsilon = epsilon

    def compute(self, trajectories: list[Trajectory]) -> list[list[float]]:
        # Group trajectories by uid
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, traj in enumerate(trajectories):
            groups[traj.uid].append(idx)

        # Compute per-group mean and std of episode rewards
        group_stats: dict[str, tuple[float, float]] = {}
        for uid, indices in groups.items():
            rewards = [trajectories[i].episode_reward for i in indices]
            mean = sum(rewards) / len(rewards)
            variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
            std = math.sqrt(variance)
            group_stats[uid] = (mean, std)

        # Assign advantages
        advantages: list[list[float]] = []
        for traj in trajectories:
            mean, std = group_stats[traj.uid]
            episode_adv = (traj.episode_reward - mean) / (std + self.epsilon)
            advantages.append([episode_adv] * traj.total_steps)

        return advantages
