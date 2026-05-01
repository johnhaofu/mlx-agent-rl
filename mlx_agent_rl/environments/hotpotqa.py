"""HotpotQA environment — HTTP client backed by the BM25 sidecar.

Mirrors the WebShop env pattern. The actual data and BM25 index live on a
remote machine; we just call ``/reset`` and ``/step`` endpoints. Reward is
the SQuAD-style token F1 of the model's final ``answer[…]`` against the
HotpotQA reference answer (continuous 0-1 — denser signal than the binary
won/lost shaping that hurt WebShop at small batch sizes).

Action format follows the same ``<action>...</action>`` envelope as
WebShop with verl-style tags. Two verbs:

  * ``search[query]`` — BM25 search over the 10 candidate paragraphs for
    the current question. Returns top-3 passages.
  * ``answer[text]``  — terminal; reward = F1(text, reference).

The dataset for HotpotQA training is just goal indices into the
distractor split — the sidecar picks the question by ``seed_idx`` so val
remains reproducible. ``reset(prompt=..., answer=goal_idx)`` interprets
``answer`` as the seed index.
"""

from __future__ import annotations

import json
import re
from typing import Optional

import urllib.error
import urllib.request

from mlx_agent_rl.environments.base import BaseEnvironment, Observation


_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_RAW_ACTION_RE = re.compile(
    r"(search|answer)\[([^\[\]]+)\](?!.*(?:search|answer)\[)",
    re.DOTALL,
)
# Accept slop forms the model produces in practice and normalise to the
# canonical ``verb[arg]`` syntax. Without this guard, ``<action>answer>X</action>``
# is captured by ``_ACTION_RE`` but rejected by the sidecar, and the
# untrained model gets stuck repeating the same malformed output every
# turn — the dominant failure mode at baseline (78% timeout w/o ever
# answering).
_SLOP_VERB_RE = re.compile(
    r"^(search|answer)\s*[>:\-=]\s*(.+?)\s*$", re.DOTALL
)


def _normalise_action(action: str) -> str | None:
    """Coerce common malformed forms (``answer>X``, ``search: q``, etc.) into
    the strict ``verb[arg]`` syntax the sidecar expects. Returns the
    normalised string, or the original if it already looks valid, or
    None if nothing salvageable remains."""
    a = action.strip()
    if not a:
        return None
    # Already canonical
    if re.match(r"^(search|answer)\[.+\]\s*$", a, re.DOTALL):
        return a
    m = _SLOP_VERB_RE.match(a)
    if m:
        verb, arg = m.group(1), m.group(2).strip()
        if arg:
            return f"{verb}[{arg}]"
    return a  # let the sidecar reject it explicitly
_DEFAULT_TIMEOUT = 30.0


_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search the candidate passages with a BM25 query. Returns the "
                "top-3 passages by relevance score."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "free-text query, ideally specific keywords",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer",
            "description": (
                "Submit your final answer. Must be a short factual span "
                "(one phrase, no explanation). Ends the episode."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "the answer text, kept brief",
                    }
                },
                "required": ["text"],
            },
        },
    },
]


class HotpotQAEnvironment(BaseEnvironment):
    """Talks to a remote HotpotQA sidecar over HTTP.

    Parameters
    ----------
    base_url:
        Where the sidecar lives, e.g. ``http://192.168.0.117:3002``.
    split:
        Which dataset split the sidecar should sample from (``train`` or
        ``validation``).
    require_think:
        When True a model output without a ``<think>...</think>`` block is
        rejected by ``extract_action``. Default False — base models often
        emit raw ``search[…]`` / ``answer[…]`` without thinking and we
        want to accept those during early training.
    timeout:
        Per-request HTTP timeout in seconds.
    """

    TASK_TYPE = "hotpotqa"

    def __init__(
        self,
        base_url: str = "http://192.168.0.117:3002",
        split: str = "train",
        require_think: bool = False,
        timeout: float = _DEFAULT_TIMEOUT,
        use_tools_schema: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.split = split
        self.require_think = require_think
        self.timeout = timeout
        self.use_tools_schema = use_tools_schema
        self._session_id: str | None = None
        self._instruction: str = ""
        self._titles: list[str] = []
        self._last_obs: str = ""
        self._last_f1: float = 0.0
        self._last_em: float = 0.0
        self._last_won: bool = False

    # ------------------------------------------------------------------
    # Public env interface
    # ------------------------------------------------------------------

    def reset(
        self,
        prompt: str,
        answer: Optional[int] = None,
        **kwargs,
    ) -> Observation:
        """Open a session for one question.

        ``answer`` (a goal_idx) pins which question the sidecar serves so
        validation is reproducible. ``prompt`` is ignored — the question
        text comes back from the sidecar's reset response."""
        body: dict = {"split": self.split}
        if isinstance(answer, (int, float)):
            body["seed_idx"] = int(answer)
        if self._session_id is not None:
            body["session_id"] = self._session_id

        data = self._post("/reset", body)
        self._session_id = data["session_id"]
        self._instruction = data.get("instruction") or ""
        self._titles = list((data.get("available_actions") or {}).get("titles") or [])
        obs_text = data.get("obs") or ""
        self._last_obs = obs_text
        self._last_f1 = 0.0
        self._last_em = 0.0
        self._last_won = False
        return Observation(
            text=obs_text,
            done=False,
            anchor=self._anchor(),
        )

    def step(self, action: str) -> tuple[Observation, float, bool]:
        """Forward a parsed action ('search[…]' / 'answer[…]') to the sidecar."""
        if self._session_id is None:
            raise RuntimeError("HotpotQAEnvironment.step called before reset")
        data = self._post(
            "/step",
            {"session_id": self._session_id, "action": action},
        )
        if "error" in data and data.get("done") is False and data.get("obs", "") == "":
            self._last_f1 = 0.0
            self._last_em = 0.0
            self._last_won = False
            obs = Observation(text=self._last_obs, done=False, anchor=self._anchor())
            return obs, 0.0, False

        reward = float(data.get("reward", 0.0))
        done = bool(data.get("done", False))
        won = bool(data.get("won", False))
        self._last_f1 = float(data.get("f1", reward if done else 0.0))
        self._last_em = float(data.get("em", 0.0))
        self._last_won = won

        obs_text = data.get("obs") or ""
        self._last_obs = obs_text
        return (
            Observation(
                text=obs_text,
                done=done,
                anchor=self._anchor(),
            ),
            reward,
            done,
        )

    def extract_action(self, model_output: str) -> str | None:
        """Return a HotpotQA-flavored action string or None.

        Accepts (in order):
          1. ``<action>search[...]</action>`` / ``<action>answer[...]</action>``
          2. raw ``search[...]`` / ``answer[...]`` (when no envelope)
        """
        if self.require_think and not _THINK_RE.search(model_output):
            return None
        if re.search(r"[\u4e00-\u9fff]", model_output):
            return None
        match = _ACTION_RE.search(model_output)
        if match:
            action = match.group(1).strip()
            return _normalise_action(action) if action else None
        raw = _RAW_ACTION_RE.search(model_output)
        if raw:
            verb = raw.group(1)
            arg = raw.group(2).strip()
            return f"{verb}[{arg}]" if arg else None
        return None

    def get_tools_schema(self) -> list[dict] | None:
        return _TOOLS_SCHEMA if self.use_tools_schema else None

    @property
    def invalid_action_message(self) -> str:
        if self.use_tools_schema:
            return (
                "Error: no <tool_call> detected. Respond with a "
                "<tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call>."
            )
        return (
            "Error: could not parse <action>. Respond with "
            "<think>your reasoning</think>"
            "<action>search[query] or answer[text]</action>."
        )

    @property
    def last_step_info(self) -> dict:
        return {
            "f1": self._last_f1,
            "em": self._last_em,
            "won": self._last_won,
        }

    @property
    def stop_strings(self) -> list[str]:
        return ["</action>", "</tool_call>"]

    @property
    def instruction(self) -> str:
        return self._instruction

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _anchor(self) -> str:
        """Anchor for GiGPO step grouping. Trajectories at the same
        retrieval-state observation are interchangeable for credit
        assignment purposes."""
        return self._last_obs

    def _post(self, path: str, body: dict) -> dict:
        url = self.base_url + path
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"HotpotQA sidecar request failed: {url} — {exc}"
            ) from exc
        return json.loads(raw)
