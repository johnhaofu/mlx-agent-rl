"""NumberLine environment — text port of verl-agent's gym-cards NumberLine.

Task: an integer ``current`` position is given on a number line [0, max_position].
The agent picks ``+`` or ``-`` each turn to step ``current`` toward ``goal``.
Reward (per verl-agent's NumberLineEnv):
    +1 when current == goal (done)
    -1 when distance to goal increased or did not decrease
     0 when distance to goal decreased

The episode is truncated after ``2 * max_position`` steps. Observation is plain
text "Target number: G. Current number: C", same wording as verl-agent's
gym-cards text wrapper, which makes prompt comparisons across frameworks easy.

For dataset interop with the existing trainer, ``reset`` accepts a prompt of
the form ``"start=S|goal=G"`` (encoded by load_numberline) so episodes are
deterministic per dataset entry. If the prompt cannot be parsed, fall back to
random initialization.
"""

from __future__ import annotations

import json
import random
import re

from mlx_agent_rl.environments.base import BaseEnvironment, Observation


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_PROMPT_RE = re.compile(r"start=(\d+)\|goal=(\d+)")
_PLUS_RE = re.compile(r'(?<![\w<])\+(?![\w/>])')
_MINUS_RE = re.compile(r'(?<![\w<\-])-(?![\w/>])')


_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": (
                "Move the current position by one step on the number line. "
                "Pass direction='+' to increment by 1 or '-' to decrement by 1. "
                "Each call advances the game by one step."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["+", "-"],
                        "description": "'+' to add 1, '-' to subtract 1.",
                    }
                },
                "required": ["direction"],
            },
        },
    },
]


class NumberLineEnvironment(BaseEnvironment):
    """Text-only NumberLine env."""

    def __init__(self, max_position: int = 5) -> None:
        self.max_position = max_position
        self._goal: int = 0
        self._position: int = 0
        self._steps: int = 0

    # ------------------------------------------------------------------
    # Public env interface
    # ------------------------------------------------------------------

    def reset(
        self,
        prompt: str,
        answer: float | int | str | None = None,
        **kwargs,
    ) -> Observation:
        """Initialize an episode. If the prompt encodes ``start=S|goal=G`` we
        use those values verbatim so a fixed dataset gives reproducible
        episodes; otherwise fall back to a fresh random pair."""
        match = _PROMPT_RE.search(prompt or "")
        if match:
            start = int(match.group(1))
            goal = int(match.group(2))
        else:
            start = random.randint(0, self.max_position)
            goal = random.randint(0, self.max_position)
            if goal == start:
                goal = (goal + 1) % (self.max_position + 1)

        self._goal = goal
        self._position = start
        self._steps = 0
        return Observation(
            text=self._format_obs(),
            done=False,
            anchor=self._format_anchor(),
        )

    def step(self, action: str) -> tuple[Observation, float, bool]:
        """Apply a parsed action ('+' or '-'). Reward matches verl-agent's
        NumberLineEnv: +1 on hit, -1 on no-progress, 0 on progress."""
        action = action.strip()
        prev_dist = abs(self._position - self._goal)

        if action == "+":
            if self._position < self.max_position:
                self._position += 1
        elif action == "-":
            if self._position > 0:
                self._position -= 1
        else:
            obs = Observation(
                text=f"Error: unknown direction {action!r}. Use '+' or '-'.",
                done=False,
                anchor=self._format_anchor(),
            )
            return obs, 0.0, False

        self._steps += 1
        new_dist = abs(self._position - self._goal)

        if new_dist == 0:
            reward, done = 1.0, True
        elif new_dist >= prev_dist:
            reward, done = -1.0, False
        else:
            reward, done = 0.0, False

        if not done and self._steps >= 2 * self.max_position:
            done = True  # truncation

        return (
            Observation(
                text=self._format_obs(),
                done=done,
                anchor=self._format_anchor(),
            ),
            reward,
            done,
        )

    def extract_action(self, model_output: str) -> str | None:
        """Parse a <tool_call>{"name":"move","arguments":{"direction":"+"}}</tool_call>.
        Falls back to scanning the raw output for a bare '+' or '-' so a model
        that almost-obeys (e.g. Qwen3-0.6B early in training) still gets credit.
        """
        match = _TOOL_CALL_RE.search(model_output)
        if match:
            try:
                call = json.loads(match.group(1))
                args = call.get("arguments", {}) or {}
                if isinstance(args, str):
                    args = json.loads(args)
                if call.get("name") == "move":
                    direction = str(args.get("direction", "")).strip()
                    if direction in {"+", "-"}:
                        return direction
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass

        # Permissive fallback: a stray '+' or '-' in the response.
        plus = bool(_PLUS_RE.search(model_output))
        minus = bool(_MINUS_RE.search(model_output))
        if plus and not minus:
            return "+"
        if minus and not plus:
            return "-"
        return None

    def get_tools_schema(self) -> list[dict]:
        return _TOOLS_SCHEMA

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _format_obs(self) -> str:
        return f"Target number: {self._goal}. Current number: {self._position}"

    def _format_anchor(self) -> str:
        """Anchor is just the (position, goal) state — used by GiGPO to group
        peer steps across trajectories that arrived at the same state. Without
        a meaningful anchor, every turn-2+ step would land in a single empty
        group and step-level advantages would be uninformative."""
        return f"pos={self._position}|goal={self._goal}"
