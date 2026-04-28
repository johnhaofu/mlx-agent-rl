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
