from abc import ABC, abstractmethod

from mlx_agent_rl.data.trajectory import Trajectory


class AdvantageEstimator(ABC):
    @abstractmethod
    def compute(self, trajectories: list[Trajectory]) -> list[list[float]]:
        """Returns advantages[traj_idx][step_idx]."""
