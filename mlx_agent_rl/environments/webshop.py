"""WebShop environment — HTTP client backed by the text-mode sidecar.

Two extraction modes are supported, picked automatically from the model's
output:
- ``<tool_call>{"name":"search","arguments":{"query":...}}</tool_call>``
  matches Qwen3's native tool-calling format. The action is reassembled into
  the WebShop string the sidecar expects (``search[query]`` / ``click[target]``).
- ``<think>...</think><action>...</action>`` mirrors verl-agent's
  webshop_projection (kept for backwards compat / for non-tool-aware base
  models).


The actual WebShop simulator (Flask + 1k product DB + faiss-cpu + spaCy) is too
heavy to host on a Mac, so we deploy verl-agent's text-mode env wrapper as a
sidecar on a separate Linux box and call it over LAN. See
``examples/webshop/webshop_text_api.py`` (deployed to the server) for the
sidecar implementation.

Action format follows verl-agent (NOT our Qwen3 tool-call style): the model
emits

    <think>...</think><action>search[red jacket]</action>

because WebShop's action space is open-ended (any free-text search query, any
ASIN click). Forcing it into a JSON tool-call schema would either restrict the
search query or invite parsing ambiguity. Sticking with the textual
``<action>...</action>`` envelope keeps us aligned with verl's prompt and
lets the existing GiGPO/KL/PPO machinery work unchanged.

The dataset for WebShop training is just goal indices (the sidecar picks the
goal text itself). ``reset(prompt=..., answer=...)`` interprets ``prompt`` as
the literal task description (instruction text returned by /reset) and
``answer`` as an optional ``goal_idx`` — when supplied, the env asks the
sidecar for that specific deterministic goal so val sets stay reproducible.
"""

from __future__ import annotations

import json
import re
from typing import Optional

import urllib.error
import urllib.request

from mlx_agent_rl.environments.base import BaseEnvironment, Observation


_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)
# verl-agent's projection also requires <think> tags; we mirror that in
# extract_action so trajectories without reasoning get the invalid-action
# penalty just like in their setup.
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# Raw "search[…]" / "click[…]" fallback — matches the LAST occurrence in
# the output so any preceding admissible-actions echo doesn't get picked
# up. Greedy on the bracket body for queries with embedded special chars.
_RAW_ACTION_RE = re.compile(
    r"(search|click)\[([^\[\]]+)\](?!.*(?:search|click)\[)",
    re.DOTALL,
)
_DEFAULT_TIMEOUT = 30.0


_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search the catalog by free-text query. Only available on the "
                "home page (when has_search_bar=True)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "natural-language search query, keep it short",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": (
                "Click a clickable on the current page. ``target`` may be: an "
                "ASIN like B09KP78G37, an option value like 'red' / 'large', "
                "or a button label like 'Buy Now' / 'Next >' / '< Prev' / "
                "'Back to Search'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "asin, option, or button label",
                    }
                },
                "required": ["target"],
            },
        },
    },
]


class WebShopEnvironment(BaseEnvironment):
    """Talks to a remote WebAgentTextEnv via the HTTP sidecar.

    Parameters
    ----------
    base_url:
        Where the sidecar lives, e.g. ``http://192.168.0.117:3001``.
    require_think:
        When True (verl default) a model output without a <think>...</think>
        block is rejected by ``extract_action``, mirroring
        ``webshop_projection``.
    timeout:
        Per-request HTTP timeout in seconds.
    """

    TASK_TYPE = "webshop"

    def __init__(
        self,
        base_url: str = "http://192.168.0.117:3001",
        require_think: bool = False,
        timeout: float = _DEFAULT_TIMEOUT,
        embed_available_actions: bool = True,
        use_tools_schema: bool = True,
        dense_reward: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.require_think = require_think
        self.timeout = timeout
        # When True (default), obs.text gains a trailing "Admissible actions: [...]"
        # block listing the current page's clickables. Empirically (8-goal A/B
        # probe) this lifted Qwen3-4B baseline from 12.5% → 25% by giving the
        # model an explicit click target list, matching verl-agent's prompt
        # template.
        self.embed_available_actions = embed_available_actions
        # When False, the chat template won't be given a tools schema, so the
        # model is expected to emit verl-style <think>...</think><action>...</action>.
        self.use_tools_schema = use_tools_schema
        # When True, the terminal-step reward is the sidecar's continuous
        # attribute-match score scaled to 0-10 (instead of binary 0/10
        # depending on full-match). This trades verl-paper fidelity for a
        # denser learning signal, which matters at small batch sizes where
        # binary-only rewards leave most groups with zero advantage.
        self.dense_reward = dense_reward
        self._session_id: str | None = None
        self._instruction: str = ""
        self._available_actions: dict = {}
        self._last_obs: str = ""
        self._last_raw_reward: float = 0.0
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
        """Start (or restart) a session.

        ``answer`` may be used to pin a deterministic goal index — the sidecar
        forwards it as ``seed_idx`` to ``WebAgentTextEnv.reset``. If absent,
        the sidecar picks the next default goal."""
        body: dict = {}
        if isinstance(answer, (int, float)):
            body["seed_idx"] = int(answer)
        if self._session_id is None:
            # let the server allocate one
            pass
        else:
            body["session_id"] = self._session_id

        data = self._post("/reset", body)
        self._session_id = data["session_id"]
        self._instruction = data.get("instruction") or ""
        self._available_actions = data.get("available_actions") or {}
        obs_text = self._format_obs(data.get("obs") or "")
        self._last_obs = obs_text
        self._last_raw_reward = 0.0
        self._last_won = False
        return Observation(
            text=obs_text,
            done=False,
            anchor=self._anchor(),
        )

    def step(self, action: str) -> tuple[Observation, float, bool]:
        """Forward a parsed action ('search[…]' or 'click[…]') to the sidecar."""
        if self._session_id is None:
            raise RuntimeError("WebShopEnvironment.step called before reset")
        data = self._post(
            "/step",
            {"session_id": self._session_id, "action": action},
        )
        if "error" in data and data.get("done") is False and data.get("obs") == "":
            # Sidecar returned a graceful 200 with error info — treat as
            # invalid action: zero reward, same observation, not done.
            self._last_raw_reward = 0.0
            self._last_won = False
            obs = Observation(text=self._last_obs, done=False, anchor=self._anchor())
            return obs, 0.0, False

        reward = float(data.get("reward", 0.0))
        done = bool(data.get("done", False))
        won = bool(data.get("won", False))
        if self.dense_reward:
            # Continuous 0-10 reward only on terminal step (matches verl's
            # convention of zero intermediate reward but uses task_score
            # instead of binary won-score). Non-terminal steps still 0.
            shaped_reward = reward * 10.0 if done else 0.0
        else:
            # verl-agent reward shaping: 0 / 10 sparse rule-based.
            shaped_reward = 10.0 if (done and won) else 0.0
        # Stash the continuous attribute-match score for info-only logging
        # (matches verl-agent's `episode/webshop_task_score` side metric).
        self._last_raw_reward = reward
        self._last_won = won

        self._available_actions = data.get("available_actions") or {}
        obs_text = self._format_obs(data.get("obs") or "")
        self._last_obs = obs_text
        return (
            Observation(
                text=obs_text,
                done=done,
                anchor=self._anchor(),
            ),
            shaped_reward,
            done,
        )

    def extract_action(self, model_output: str) -> str | None:
        """Return a WebShop-flavored action string or ``None``.

        Tries Qwen3-style ``<tool_call>{"name":..., "arguments":...}</tool_call>``
        first; reassembles the JSON into the ``search[query]`` /
        ``click[target]`` strings the sidecar expects.

        Falls back to verl-agent's ``<think>...</think><action>...</action>``
        envelope for non-tool-aware base models. The CJK guard in the verl
        projection is preserved for the ``<action>`` path; it is *not*
        applied to the JSON path because tool_call arguments often legitimately
        include the (English) instruction text verbatim."""
        # 1) Qwen3 native tool_call
        tc = _TOOL_CALL_RE.search(model_output)
        if tc:
            try:
                call = json.loads(tc.group(1))
            except json.JSONDecodeError:
                return None
            name = (call.get("name") or "").strip()
            args = call.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if name == "search":
                q = (args.get("query") or "").strip()
                return f"search[{q}]" if q else None
            if name == "click":
                t = (args.get("target") or "").strip()
                return f"click[{t}]" if t else None
            return None

        # 2) verl-agent <think>+<action> envelope
        if self.require_think and not _THINK_RE.search(model_output):
            return None
        if re.search(r"[\u4e00-\u9fff]", model_output):
            return None
        match = _ACTION_RE.search(model_output)
        if match:
            action = match.group(1).strip()
            return action or None
        # 3) raw "search[…]" / "click[…]" fallback — base models often emit
        # the action string directly when not yet trained on the envelope.
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
            "<think>your reasoning</think><action>search[...] or click[...]</action>."
        )

    @property
    def last_step_info(self) -> dict:
        return {"task_score": self._last_raw_reward, "won": self._last_won}

    @property
    def stop_strings(self) -> list[str]:
        # `<action>...</action>` is always usable; `</tool_call>` only
        # matters when use_tools_schema=True but listing both is harmless.
        return ["</action>", "</tool_call>"]

    def get_available_actions(self) -> dict:
        """Latest sidecar 'available_actions' dict.

        Trainer / prompt code can include this in the system prompt so the
        model sees the explicit admissible-action list — same as
        ``WEBSHOP_TEMPLATE``."""
        return dict(self._available_actions)

    @property
    def instruction(self) -> str:
        return self._instruction

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _format_obs(self, raw_obs: str) -> str:
        """Append admissible-actions list when ``embed_available_actions``."""
        if not self.embed_available_actions:
            return raw_obs
        clickables = (self._available_actions or {}).get("clickables") or []
        has_search = (self._available_actions or {}).get("has_search_bar")
        actions = []
        if has_search:
            actions.append("search[query]")
        # Limit to 12 to keep prompt tight; clickables on result pages can
        # number ~15 (10 results + nav buttons).
        for c in clickables[:12]:
            actions.append(f"click[{c}]")
        if not actions:
            return raw_obs
        return f"{raw_obs}\n\nAdmissible actions: [{', '.join(actions)}]"

    def _anchor(self) -> str:
        """Anchor for GiGPO step grouping. Trajectories at the same WebShop
        page state should match. We use the last observation text — pages
        with identical visible content are interchangeable for credit
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
                f"WebShop sidecar request failed: {url} — {exc}"
            ) from exc
        return json.loads(raw)
