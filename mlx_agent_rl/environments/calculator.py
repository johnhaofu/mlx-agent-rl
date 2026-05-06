import json
import re

from mlx_agent_rl.environments.base import BaseEnvironment, Observation

_SAFE_EXPR_RE = re.compile(r"^[\d\s\+\-\*\/\.\(\)\%\*\*]+$")
_CALCULATE_RE = re.compile(r"calculate\((.+?)\)")
_ANSWER_RE = re.compile(r"answer\(([^\)]+)\)")
_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate an arithmetic expression and return the numeric result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A math expression using digits and + - * / ( ) %, e.g. '60 / 2' or '12*7'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer",
            "description": (
                "Submit the final numeric answer. You MUST call this tool to "
                "finish the task — the conversation does not end until you do. "
                "Plain-text answers, \\boxed{...}, or 'The answer is N' are NOT "
                "accepted. Always wrap your final number in a call to this tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "The final numeric answer."}
                },
                "required": ["value"],
            },
        },
    },
]


class CalculatorEnvironment(BaseEnvironment):
    def __init__(self):
        self._question: str = ""
        self._expected: float | None = None

    def reset(self, prompt: str, answer: float | int | str | None = None, **kwargs) -> Observation:
        self._question = prompt
        if answer is not None:
            self._expected = float(answer)
        else:
            self._expected = None
        return Observation(text=prompt, done=False, anchor=prompt)

    def get_tools_schema(self) -> list[dict]:
        """Return the JSON-schema tool definitions for chat-template tools= argument."""
        return _TOOLS_SCHEMA

    def extract_action(self, model_output: str) -> str | None:
        # Prefer Qwen3-native <tool_call>{json}</tool_call> format
        tc = _TOOL_CALL_RE.search(model_output)
        if tc:
            try:
                call = json.loads(tc.group(1))
                name = call.get("name", "")
                args = call.get("arguments", {}) or {}
                if isinstance(args, str):
                    args = json.loads(args)
            except (json.JSONDecodeError, AttributeError, TypeError):
                return None
            if name == "calculate":
                expr = args.get("expression", "")
                return f"calculate({expr})"
            if name == "answer":
                val = args.get("value", "")
                return f"answer({val})"
            return None

        # Fallback: legacy <action>...</action> format
        match = _ACTION_RE.search(model_output)
        if not match:
            return None
        content = match.group(1).strip()
        if _CALCULATE_RE.fullmatch(content) or _ANSWER_RE.fullmatch(content):
            return content
        return None

    def step(self, action: str) -> tuple[Observation, float, bool]:
        calc_match = _CALCULATE_RE.fullmatch(action.strip())
        if calc_match:
            expr = calc_match.group(1)
            result = self._safe_eval(expr)
            if result is None:
                obs = Observation(text="Error: invalid expression", done=False)
                return obs, 0.0, False
            obs = Observation(text=str(result), done=False)
            return obs, 0.0, False

        answer_match = _ANSWER_RE.fullmatch(action.strip())
        if answer_match:
            try:
                given = float(answer_match.group(1))
            except ValueError:
                obs = Observation(text="Error: invalid answer format", done=True)
                return obs, 0.0, True

            if self._expected is not None and abs(given - self._expected) < 1e-9:
                obs = Observation(text="Correct!", done=True)
                return obs, 1.0, True
            else:
                obs = Observation(text="Wrong answer.", done=True)
                return obs, 0.0, True

        obs = Observation(text="Error: unrecognized action", done=False)
        return obs, 0.0, False

    def _safe_eval(self, expr: str) -> float | None:
        if not _SAFE_EXPR_RE.match(expr):
            return None
        try:
            result = eval(expr, {"__builtins__": {}}, {})  # noqa: S307
            return float(result)
        except Exception:
            return None
