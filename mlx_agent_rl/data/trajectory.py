from dataclasses import dataclass


@dataclass
class Step:
    prompt_tokens: list[int]
    action_tokens: list[int]
    log_probs: list[float]
    reward: float
    done: bool
    anchor_obs: str


@dataclass
class Trajectory:
    steps: list[Step]
    episode_reward: float
    uid: str

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def succeeded(self) -> bool:
        return self.steps[-1].done if self.steps else False
