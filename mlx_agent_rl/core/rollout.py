"""Rollout collector — runs policy in an environment and builds Trajectory objects."""

from __future__ import annotations

import uuid

from mlx_agent_rl.data.trajectory import Step, Trajectory
from mlx_agent_rl.environments.base import BaseEnvironment
from mlx_agent_rl.memory.memory import SlidingMemory


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
                prompt_text = self._build_prompt_for(slot["obs"].text, slot["memory"])
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
                    new_obs_text = slot["obs"].text
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
                    taken_action = action if action is not None else model_output
                    slot["memory"].update(new_obs_text, taken_action)

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

    def _build_prompt(self, obs_text: str) -> str:
        """Combine system prompt, memory context, and current observation."""
        return self._build_prompt_for(obs_text, self.memory)

    def _build_prompt_for(self, obs_text: str, memory: SlidingMemory) -> str:
        """Build prompt using a specific memory instance."""
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
            prompt_text = self._build_prompt(obs.text)

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
                obs_text_new = obs.text
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

            # Update memory with the action that was taken
            taken_action = action if action is not None else model_output
            self.memory.update(obs_text_new, taken_action)

        return Trajectory(steps=steps, episode_reward=total_reward, uid=uid)
