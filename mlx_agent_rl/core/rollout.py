"""Rollout collector — runs policy in an environment and builds Trajectory objects."""

from __future__ import annotations

import re
import uuid

from mlx_agent_rl.data.trajectory import Step, Trajectory
from mlx_agent_rl.environments.base import BaseEnvironment
from mlx_agent_rl.memory.memory import SlidingMemory


_TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*\{.*?\}\s*</tool_call>", re.DOTALL)


def _strip_to_tool_call(model_output: str) -> str:
    """Reduce assistant output stored in chat history to just the <tool_call>
    block. Qwen3's best-practice docs say multi-turn history should contain
    only the final output, not thinking/reasoning prose. When no tool_call
    block is present (invalid action) we keep the original output so the
    model still sees its own malformed attempt as feedback context.
    """
    m = _TOOL_CALL_BLOCK.search(model_output)
    return m.group(0) if m else model_output


class RolloutCollector:
    """Collects trajectories by running a policy inside an environment.

    Parameters
    ----------
    policy:
        A Policy instance (or any object exposing ``generate_with_log_probs``,
        ``compute_log_probs``, and ``tokenizer``).
    env:
        A :class:`BaseEnvironment` instance.
    memory:
        A :class:`SlidingMemory` instance.
    max_steps:
        Maximum number of environment steps per episode.
    max_tokens:
        Maximum tokens to generate per step.
    invalid_action_penalty:
        Reward applied when the model outputs an unrecognised action.
    system_prompt:
        Optional system-level instruction prepended to every prompt.
    backend:
        Optional backend instance for batched generation (e.g.
        :class:`~mlx_agent_rl.core.mlc_backend.MLCBackend` or
        :class:`~mlx_agent_rl.core.llamacpp_backend.LlamaCppBackend`).
        When provided, all active trajectories at each rollout step are batched
        together and sent to the backend's continuous-batching engine for faster
        generation.  Log-probs are computed in a second pass after all rollouts
        complete.  When ``None`` (the default) the original sequential MLX-LM
        path is used unchanged.
    """

    def __init__(
        self,
        policy,
        env: BaseEnvironment,
        memory: SlidingMemory,
        max_steps: int = 5,
        max_tokens: int = 256,
        invalid_action_penalty: float = -0.1,
        system_prompt: str = "",
        backend=None,
    ) -> None:
        self.policy = policy
        self.env = env
        self.memory = memory
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.invalid_action_penalty = invalid_action_penalty
        self.system_prompt = system_prompt
        self.backend = backend
        # Native chat-template tool descriptors (Qwen3 etc.). When the env
        # exposes get_tools_schema we pass it through to apply_chat_template,
        # which prepends a <tools> system block and biases the model toward
        # <tool_call> output.
        self._tools = env.get_tools_schema() if hasattr(env, "get_tools_schema") else None

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def collect(
        self, prompts: list[dict], group_size: int = 4
    ) -> list[Trajectory]:
        """Collect rollouts for a list of prompt dicts.

        Each *prompt dict* should have at least a ``"prompt"`` key and optionally
        an ``"answer"`` key used to score the final answer.

        For each prompt ``group_size`` independent rollouts are collected, all
        sharing the same ``uid`` so advantage estimators can group them.

        When a backend is configured, all active trajectories at each step
        are batched together for parallel generation, then log-probs are filled
        in via a second MLX forward pass after the rollout loop completes.

        Returns
        -------
        list[Trajectory]
            All collected trajectories (``len(prompts) * group_size`` entries).
        """
        if self.backend is not None:
            return self._collect_batched(prompts, group_size)
        return self._collect_sequential(prompts, group_size)

    # ------------------------------------------------------------------
    # Sequential path (original, MLX-LM)
    # ------------------------------------------------------------------

    def _collect_sequential(
        self, prompts: list[dict], group_size: int
    ) -> list[Trajectory]:
        """Original sequential collection using policy.generate_with_log_probs."""
        all_trajectories: list[Trajectory] = []

        for prompt_dict in prompts:
            uid = str(uuid.uuid4())
            question = prompt_dict.get("prompt", "")
            answer = prompt_dict.get("answer", None)

            for _ in range(group_size):
                traj = self._run_episode(question, answer, uid)
                all_trajectories.append(traj)

        return all_trajectories

    # ------------------------------------------------------------------
    # Batched path (MLC / llama.cpp / any backend with generate_batch_sync)
    # ------------------------------------------------------------------

    def _collect_batched(
        self, prompts: list[dict], group_size: int
    ) -> list[Trajectory]:
        """Collect rollouts using a batched generation backend.

        All active trajectories across all prompts/group-members are batched
        together at each step, then log-probs are computed in a second pass.
        """
        # ----------------------------------------------------------------
        # Build initial state for every trajectory slot
        # ----------------------------------------------------------------
        # Each slot tracks: question, answer, uid, memory, current obs,
        # accumulated steps, total_reward, and whether the episode is done.

        slots: list[dict] = []
        for prompt_dict in prompts:
            uid = str(uuid.uuid4())
            question = prompt_dict.get("prompt", "")
            answer = prompt_dict.get("answer", None)
            for _ in range(group_size):
                mem = SlidingMemory(window_size=self.memory.window_size)
                obs = self.env.reset(question, answer=answer)
                slots.append(
                    {
                        "uid": uid,
                        "question": question,
                        "answer": answer,
                        "memory": mem,
                        "obs": obs,
                        "steps": [],
                        "total_reward": 0.0,
                        "done": False,
                    }
                )

        # ----------------------------------------------------------------
        # Step-level batched generation loop
        # ----------------------------------------------------------------
        for _step_idx in range(self.max_steps):
            active_indices = [i for i, s in enumerate(slots) if not s["done"]]
            if not active_indices:
                break

            # Build prompts for all active slots
            batch_prompts: list[str] = []
            batch_prompt_tokens: list[list[int]] = []
            for idx in active_indices:
                slot = slots[idx]
                prompt_text = self._build_prompt_for(
                    slot["obs"].text, slot["memory"], slot["question"]
                )
                batch_prompts.append(prompt_text)
                batch_prompt_tokens.append(
                    list(self.policy.tokenizer.encode(prompt_text))
                )

            # Batch generate via backend (continuous batching, no log_probs yet)
            batch_outputs: list[str] = self.backend.generate_batch_sync(
                batch_prompts, max_tokens=self.max_tokens
            )

            # Process each result and advance environment
            for local_i, idx in enumerate(active_indices):
                slot = slots[idx]
                model_output = batch_outputs[local_i]
                prompt_tokens = batch_prompt_tokens[local_i]

                action = self.env.extract_action(model_output)
                action_tokens: list[int] = list(
                    self.policy.tokenizer.encode(model_output)
                )

                if action is None:
                    reward = self.invalid_action_penalty
                    done = False
                    new_obs_text = (
                        "Error: no <tool_call> detected. Respond with a "
                        "<tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call>."
                    )
                else:
                    next_obs, reward, done = self.env.step(action)
                    new_obs_text = next_obs.text
                    slot["obs"] = next_obs

                slot["total_reward"] += reward

                # Store step with empty log_probs — filled in after the loop
                step = Step(
                    prompt_tokens=prompt_tokens,
                    action_tokens=action_tokens,
                    log_probs=[],  # placeholder; filled in second pass below
                    reward=reward,
                    done=done,
                    anchor_obs=slot["obs"].anchor
                    if hasattr(slot["obs"], "anchor")
                    else slot["question"],
                )
                slot["steps"].append(step)

                if done:
                    slot["done"] = True
                else:
                    slot["memory"].update(
                        new_obs_text, _strip_to_tool_call(model_output)
                    )

        # ----------------------------------------------------------------
        # Second pass: compute log_probs for every step via MLX policy
        # ----------------------------------------------------------------
        for slot in slots:
            for step in slot["steps"]:
                if len(step.action_tokens) > 0:
                    step.log_probs = self.policy.compute_log_probs(
                        step.prompt_tokens, step.action_tokens
                    )

        # ----------------------------------------------------------------
        # Assemble Trajectory objects
        # ----------------------------------------------------------------
        trajectories: list[Trajectory] = []
        for slot in slots:
            trajectories.append(
                Trajectory(
                    steps=slot["steps"],
                    episode_reward=slot["total_reward"],
                    uid=slot["uid"],
                )
            )

        return trajectories

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, obs_text: str, question: str | None = None) -> str:
        return self._build_prompt_for(obs_text, self.memory, question)

    def _build_prompt_for(
        self,
        obs_text: str,
        memory: SlidingMemory,
        question: str | None = None,
    ) -> str:
        """Build prompt via chat template (with tools) when available.

        Multi-turn structure (Qwen3 / OpenAI tool-calling style):
            system: <system_prompt> + auto-generated <tools> block
            user:   <question>
            assistant: <prev model output, contains <tool_call>...</tool_call>>
            tool:   <env response 1>
            assistant: <prev model output 2>
            tool:   <env response 2>
            ...
        Memory entries store ``(env_response_text, full_model_output_text)``
        so each replay matches what the model actually emitted.
        Falls back to plain-text concatenation if no chat template is available.
        """
        tokenizer = self.policy.tokenizer
        if hasattr(tokenizer, "apply_chat_template") and question is not None:
            messages: list[dict] = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": question})
            history = memory._history[-memory.window_size:] if memory._history else []
            for resp_obs, model_output in history:
                messages.append({"role": "assistant", "content": model_output})
                messages.append({"role": "tool", "content": resp_obs})
            kwargs = {"tokenize": False, "add_generation_prompt": True}
            if self._tools is not None:
                kwargs["tools"] = self._tools
            # enable_thinking is Qwen3-specific; some tokenizers reject it.
            try:
                return tokenizer.apply_chat_template(
                    messages, enable_thinking=False, **kwargs
                )
            except TypeError:
                return tokenizer.apply_chat_template(messages, **kwargs)

        # Fallback: plain text
        parts: list[str] = []
        if self.system_prompt:
            parts.append(self.system_prompt)
        ctx = memory.get_context()
        if ctx:
            parts.append(ctx)
        parts.append(f"Observation: {obs_text}")
        return "\n".join(parts)

    def _run_episode(
        self, question: str, answer, uid: str
    ) -> Trajectory:
        """Run a single episode and return a :class:`Trajectory`."""
        self.memory.reset()
        obs = self.env.reset(question, answer=answer)

        steps: list[Step] = []
        total_reward = 0.0

        for _ in range(self.max_steps):
            prompt_text = self._build_prompt(obs.text, question=question)

            # Tokenize prompt for storage
            prompt_tokens: list[int] = list(
                self.policy.tokenizer.encode(prompt_text)
            )

            # Generate action
            model_output, log_probs = self.policy.generate_with_log_probs(
                prompt_text, max_tokens=self.max_tokens
            )

            action = self.env.extract_action(model_output)

            # Tokenize the generated text for action_tokens
            action_tokens: list[int] = list(
                self.policy.tokenizer.encode(model_output)
            )

            if action is None:
                # Invalid action: apply penalty, do not step environment
                reward = self.invalid_action_penalty
                done = False
                obs_text_new = (
                    "Error: no <tool_call> detected. Respond with a "
                    "<tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call>."
                )
            else:
                next_obs, reward, done = self.env.step(action)
                obs_text_new = next_obs.text
                obs = next_obs

            total_reward += reward

            step = Step(
                prompt_tokens=prompt_tokens,
                action_tokens=action_tokens,
                log_probs=log_probs,
                reward=reward,
                done=done,
                anchor_obs=obs.anchor if hasattr(obs, "anchor") else question,
            )
            steps.append(step)

            if done:
                break

            self.memory.update(obs_text_new, _strip_to_tool_call(model_output))

        return Trajectory(steps=steps, episode_reward=total_reward, uid=uid)
