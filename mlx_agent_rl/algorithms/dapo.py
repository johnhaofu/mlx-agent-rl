from mlx_agent_rl.algorithms.grpo import GRPOEstimator
from mlx_agent_rl.data.trajectory import Trajectory


class DAPOEstimator(GRPOEstimator):
    """DAPO (Dual-clipping Advantage Policy Optimization) estimator.

    Advantage computation is identical to GRPO. Stores epsilon_low and
    epsilon_high clip parameters that are applied in the loss computation
    (not during advantage estimation).
    """

    def __init__(
        self,
        epsilon: float = 1e-4,
        epsilon_low: float = 0.2,
        epsilon_high: float = 0.28,
    ) -> None:
        super().__init__(epsilon=epsilon)
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
