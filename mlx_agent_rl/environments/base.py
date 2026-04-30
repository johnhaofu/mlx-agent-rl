from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Observation:
    text: str
    done: bool = False
    anchor: str = ""


class BaseEnvironment(ABC):
    @abstractmethod
    def reset(self, prompt: str, **kwargs) -> Observation:
        pass

    @abstractmethod
    def step(self, action: str) -> tuple[Observation, float, bool]:
        pass

    @abstractmethod
    def extract_action(self, model_output: str) -> str | None:
        pass

    @property
    def invalid_action_message(self) -> str:
        """Observation text shown to the model after an unparseable response."""
        return (
            "Error: could not parse a valid action from your last response. "
            "Re-emit your action in the required format."
        )

    @property
    def last_step_info(self) -> dict:
        """Side-channel metrics from the most recent step (for logging only)."""
        return {}

    @property
    def stop_strings(self) -> list[str]:
        """Optional list of substrings that should halt generation early.

        Envs with structured output formats (e.g. ``</action>``,
        ``</tool_call>``) should override to cap wall-clock per turn.
        """
        return []
