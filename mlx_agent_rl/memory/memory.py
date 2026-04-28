class SlidingMemory:
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self._history: list[tuple[str, str]] = []

    def reset(self):
        self._history = []

    def update(self, observation: str, action: str):
        self._history.append((observation, action))

    def get_context(self) -> str:
        if not self._history:
            return ""
        window = self._history[-self.window_size:]
        parts = []
        for i, (obs, act) in enumerate(window, 1):
            parts.append(f"[Step {i}] Observation: {obs}\nAction: {act}")
        return "\n".join(parts)

    @property
    def length(self) -> int:
        return len(self._history)
