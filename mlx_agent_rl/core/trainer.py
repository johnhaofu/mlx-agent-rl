"""Trainer module — orchestrates policy training with RL algorithms."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx_agent_rl.algorithms.base import AdvantageEstimator
from mlx_agent_rl.algorithms.dapo import DAPOEstimator
from mlx_agent_rl.algorithms.dr_grpo import DrGRPOEstimator
from mlx_agent_rl.algorithms.gigpo import GiGPOEstimator
from mlx_agent_rl.algorithms.grpo import GRPOEstimator
from mlx_agent_rl.core.rollout import RolloutCollector
from mlx_agent_rl.data.trajectory import Trajectory
from mlx_agent_rl.environments.base import BaseEnvironment
from mlx_agent_rl.memory.memory import SlidingMemory


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    path: str = ""
    quantize: int | None = None
    lora_rank: int = 8
    lora_layers: int = 8


@dataclass
class RolloutConfig:
    group_size: int = 4
    max_steps: int = 5
    max_tokens: int = 256
    system_prompt: str = ""


@dataclass
class TrainingConfig:
    algorithm: str = "grpo"
    lr: float = 1e-5
    epochs: int = 10
    batch_size: int = 4
    epsilon: float = 0.2
    epsilon_high: float = 0.28
    clip_grad: float = 1.0
    micro_batch_size: int = 4  # samples per backward pass; controls peak memory
    val_interval: int = 5  # run val eval every N train batches (0 disables)
    val_temperature: float = 0.4  # low-variance sampling for val (not greedy)
    val_top_p: float = 1.0


@dataclass
class EnvironmentConfig:
    type: str = "calculator"
    invalid_action_penalty: float = -0.1


@dataclass
class MemoryConfig:
    window_size: int = 3


@dataclass
class MLCConfig:
    enabled: bool = False
    model_id: str = "HF://mlc-ai/Qwen2.5-0.5B-Instruct-q4f16_1-MLC"


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "mlx-agent-rl"
    run_name: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class LlamaCppConfig:
    enabled: bool = False
    gguf_path: str = ""
    port: int = 8090
    n_parallel: int = 8
    n_ctx: int = 16384


@dataclass
class TrainerConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    mlc: MLCConfig = field(default_factory=MLCConfig)
    llamacpp: LlamaCppConfig = field(default_factory=LlamaCppConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def from_yaml(path: str | Path) -> "TrainerConfig":
        """Load config from a YAML file."""
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("PyYAML is required to load configs from YAML files.") from exc

        with open(path) as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)

        cfg = TrainerConfig()

        if "model" in raw:
            m = raw["model"]
            lora = m.get("lora", {})
            cfg.model = ModelConfig(
                path=m.get("path", ""),
                quantize=m.get("quantize", None),
                lora_rank=lora.get("rank", 8),
                lora_layers=lora.get("layers", 8),
            )

        if "rollout" in raw:
            r = raw["rollout"]
            cfg.rollout = RolloutConfig(
                group_size=r.get("group_size", 4),
                max_steps=r.get("max_steps", 5),
                max_tokens=r.get("max_tokens", 256),
                system_prompt=r.get("system_prompt", ""),
            )

        if "training" in raw:
            t = raw["training"]
            cfg.training = TrainingConfig(
                algorithm=t.get("algorithm", "grpo"),
                lr=t.get("lr", 1e-5),
                epochs=t.get("epochs", 10),
                batch_size=t.get("batch_size", 4),
                epsilon=t.get("epsilon", 0.2),
                epsilon_high=t.get("epsilon_high", 0.28),
                clip_grad=t.get("clip_grad", 1.0),
                micro_batch_size=t.get("micro_batch_size", 4),
                val_interval=t.get("val_interval", 5),
                val_temperature=t.get("val_temperature", 0.4),
                val_top_p=t.get("val_top_p", 1.0),
            )

        if "environment" in raw:
            e = raw["environment"]
            cfg.environment = EnvironmentConfig(
                type=e.get("type", "calculator"),
                invalid_action_penalty=e.get("invalid_action_penalty", -0.1),
            )

        if "memory" in raw:
            mem = raw["memory"]
            cfg.memory = MemoryConfig(
                window_size=mem.get("window_size", 3),
            )

        if "mlc" in raw:
            mlc = raw["mlc"]
            cfg.mlc = MLCConfig(
                enabled=mlc.get("enabled", False),
                model_id=mlc.get(
                    "model_id",
                    "HF://mlc-ai/Qwen2.5-0.5B-Instruct-q4f16_1-MLC",
                ),
            )

        if "llamacpp" in raw:
            lc = raw["llamacpp"]
            cfg.llamacpp = LlamaCppConfig(
                enabled=lc.get("enabled", False),
                gguf_path=lc.get("gguf_path", ""),
                port=lc.get("port", 8090),
                n_parallel=lc.get("n_parallel", 8),
                n_ctx=lc.get("n_ctx", 16384),
            )

        if "wandb" in raw:
            wb = raw["wandb"]
            cfg.wandb = WandbConfig(
                enabled=wb.get("enabled", False),
                project=wb.get("project", "mlx-agent-rl"),
                run_name=wb.get("run_name", None),
                tags=wb.get("tags", []),
            )

        return cfg


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Orchestrates RL training of a language model policy.

    Parameters
    ----------
    config:
        A fully populated :class:`TrainerConfig`.
    dataset:
        List of dicts with ``"prompt"`` (and optional ``"answer"``) keys.
    """

    def __init__(
        self,
        config: TrainerConfig,
        dataset: list[dict],
        val_dataset: list[dict] | None = None,
    ) -> None:
        self.config = config
        self.dataset = dataset
        self.val_dataset = val_dataset

        # Build components
        from mlx_agent_rl.core.policy import Policy

        self.policy = Policy(
            model_path=config.model.path,
            quantize=config.model.quantize,
            lora_rank=config.model.lora_rank,
            lora_layers=config.model.lora_layers,
        )

        self.env = self._create_environment(config.environment)
        self.memory = SlidingMemory(window_size=config.memory.window_size)
        self.algorithm = self._create_algorithm(
            config.training.algorithm,
            epsilon=config.training.epsilon,
            epsilon_high=config.training.epsilon_high,
        )
        # Optionally create a batched generation backend.
        # llama.cpp takes priority over MLC when both are enabled.
        rollout_backend = None
        if config.llamacpp.enabled:
            from mlx_agent_rl.core.llamacpp_backend import LlamaCppBackend

            rollout_backend = LlamaCppBackend(
                gguf_path=config.llamacpp.gguf_path,
                port=config.llamacpp.port,
                n_parallel=config.llamacpp.n_parallel,
                n_ctx=config.llamacpp.n_ctx,
            )
        elif config.mlc.enabled:
            from mlx_agent_rl.core.mlc_backend import MLCBackend

            rollout_backend = MLCBackend(config.mlc.model_id)

        self.collector = RolloutCollector(
            policy=self.policy,
            env=self.env,
            memory=self.memory,
            max_steps=config.rollout.max_steps,
            max_tokens=config.rollout.max_tokens,
            invalid_action_penalty=config.environment.invalid_action_penalty,
            system_prompt=config.rollout.system_prompt,
            backend=rollout_backend,
        )

        # Optimizer — only update LoRA parameters
        self.optimizer = optim.Adam(learning_rate=config.training.lr)

        # Optional W&B logging. Initialized lazily so that pure-test imports
        # don't pull wandb or trigger network calls.
        self._wandb = None
        if config.wandb.enabled:
            try:
                import wandb  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "wandb.enabled=true but the wandb package is not installed."
                ) from exc
            wandb.init(
                project=config.wandb.project,
                name=config.wandb.run_name,
                tags=config.wandb.tags,
                config={
                    "model": config.model.__dict__,
                    "rollout": config.rollout.__dict__,
                    "training": config.training.__dict__,
                    "environment": config.environment.__dict__,
                    "memory": config.memory.__dict__,
                },
            )
            self._wandb = wandb

    # ------------------------------------------------------------------
    # Static factories
    # ------------------------------------------------------------------

    @staticmethod
    def _create_algorithm(name: str, **kwargs) -> AdvantageEstimator:
        """Instantiate an advantage estimator by name.

        Note: ``epsilon`` in the YAML config is the **PPO-clip** epsilon used in
        the policy update. The advantage-normalization epsilon (denominator
        stabilizer in (r - mean) / (std + eps)) is a separate, much smaller
        constant — exposing the same name for both would conflate two unrelated
        quantities, so we hard-code the normalization eps here.
        """
        name = name.lower()
        epsilon_clip = kwargs.get("epsilon", 0.2)
        epsilon_high = kwargs.get("epsilon_high", 0.28)
        norm_eps = 1e-4  # for std denominator stability, NOT PPO clip

        if name == "grpo":
            return GRPOEstimator(epsilon=norm_eps)
        elif name == "dr_grpo":
            return DrGRPOEstimator()
        elif name == "dapo":
            return DAPOEstimator(
                epsilon=norm_eps,
                epsilon_low=epsilon_clip,
                epsilon_high=epsilon_high,
            )
        elif name == "gigpo":
            return GiGPOEstimator(epsilon=norm_eps)
        else:
            raise ValueError(f"Unknown algorithm: {name!r}. Choose from grpo, dr_grpo, dapo, gigpo.")

    @staticmethod
    def _create_environment(config: EnvironmentConfig) -> BaseEnvironment:
        """Instantiate an environment by type name."""
        env_type = config.type.lower()
        if env_type == "calculator":
            from mlx_agent_rl.environments.calculator import CalculatorEnvironment

            return CalculatorEnvironment()
        else:
            raise ValueError(f"Unknown environment type: {env_type!r}. Choose from: calculator.")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Main training loop."""
        from collections import deque

        cfg = self.config
        self.policy.train()
        rolling_window = 10
        rolling = deque(maxlen=rolling_window)  # tuples (reward, answered, correct)

        for epoch in range(cfg.training.epochs):
            shuffled = list(self.dataset)
            random.shuffle(shuffled)

            # Process in batches
            batch_size = cfg.training.batch_size
            total_loss = 0.0
            num_updates = 0

            num_batches = (len(shuffled) + batch_size - 1) // batch_size
            for batch_idx, batch_start in enumerate(range(0, len(shuffled), batch_size)):
                batch = shuffled[batch_start : batch_start + batch_size]

                # Collect rollouts
                trajectories = self.collector.collect(
                    batch, group_size=cfg.rollout.group_size
                )

                n = len(trajectories)
                avg_reward = sum(t.episode_reward for t in trajectories) / n
                # answered = called answer(...) within max_steps (right or wrong)
                answered_rate = sum(1 for t in trajectories if t.succeeded) / n
                # correct = at least one step had reward >= 1.0 (calculator env's
                # answer-correct signal); distinct from answered.
                correct_rate = (
                    sum(
                        1 for t in trajectories
                        if any(s.reward >= 1.0 for s in t.steps)
                    ) / n
                )

                rolling.append((avg_reward, answered_rate, correct_rate))
                rw_r = sum(x[0] for x in rolling) / len(rolling)
                rw_a = sum(x[1] for x in rolling) / len(rolling)
                rw_c = sum(x[2] for x in rolling) / len(rolling)

                # Compute advantages
                advantages = self.algorithm.compute(trajectories)

                # Policy update
                loss = self._update_policy(trajectories, advantages)
                total_loss += loss
                num_updates += 1

                global_step = epoch * num_batches + batch_idx + 1
                print(
                    f"  [{batch_idx+1}/{num_batches}] "
                    f"r={avg_reward:+.2f} a={answered_rate:.0%} c={correct_rate:.0%} "
                    f"| ma{len(rolling)}: r={rw_r:+.2f} a={rw_a:.0%} c={rw_c:.0%} "
                    f"loss={loss:.4f}",
                    flush=True,
                )

                if self._wandb is not None:
                    self._wandb.log(
                        {
                            "train/reward": avg_reward,
                            "train/answered": answered_rate,
                            "train/correct": correct_rate,
                            "train/loss": loss,
                            "train/ma_reward": rw_r,
                            "train/ma_answered": rw_a,
                            "train/ma_correct": rw_c,
                            "train/epoch": epoch + 1,
                        },
                        step=global_step,
                    )

                if (
                    self.val_dataset is not None
                    and cfg.training.val_interval > 0
                    and global_step % cfg.training.val_interval == 0
                ):
                    self._evaluate(global_step)

            avg_loss = total_loss / max(num_updates, 1)
            print(f"Epoch {epoch + 1}/{cfg.training.epochs}  avg_loss={avg_loss:.4f}", flush=True)

    # ------------------------------------------------------------------
    # Validation eval (low-variance sampling, group_size=1, no policy update)
    # ------------------------------------------------------------------

    def _evaluate(self, global_step: int) -> None:
        """Run a held-out evaluation pass and print val metrics.

        Uses low-variance sampling (val_temperature, default 0.4) and
        group_size=1 so each val problem yields exactly one trajectory.
        Following verl-agent's convention, val is sampling-based but with
        lower temperature than training, giving a smoother trend curve
        than the per-batch RL metrics without committing to fully greedy
        decoding (which often degrades into repetition for small models).
        """
        cfg = self.config.training
        if not self.val_dataset:
            return

        old_temp = self.policy.temperature
        old_top_p = self.policy.top_p
        self.policy.temperature = cfg.val_temperature
        self.policy.top_p = cfg.val_top_p
        self.policy.eval()
        try:
            trajectories = self.collector.collect(self.val_dataset, group_size=1)
        finally:
            self.policy.temperature = old_temp
            self.policy.top_p = old_top_p
            self.policy.train()

        n = len(trajectories)
        if n == 0:
            return
        avg_reward = sum(t.episode_reward for t in trajectories) / n
        answered_rate = sum(1 for t in trajectories if t.succeeded) / n
        correct_rate = (
            sum(
                1 for t in trajectories
                if any(s.reward >= 1.0 for s in t.steps)
            ) / n
        )
        avg_steps = sum(t.total_steps for t in trajectories) / n
        print(
            f"  [val @ step {global_step}] r={avg_reward:+.2f} "
            f"a={answered_rate:.0%} c={correct_rate:.0%} "
            f"avg_steps={avg_steps:.1f}  (n={n})",
            flush=True,
        )

        if self._wandb is not None:
            self._wandb.log(
                {
                    "val/reward": avg_reward,
                    "val/answered": answered_rate,
                    "val/correct": correct_rate,
                    "val/avg_steps": avg_steps,
                    "val/n": n,
                },
                step=global_step,
            )

    # ------------------------------------------------------------------
    # Policy update (PPO-clip)
    # ------------------------------------------------------------------

    def _update_policy(
        self,
        trajectories: list[Trajectory],
        advantages: list[list[float]],
    ) -> float:
        """Compute PPO-clip loss and perform one gradient step.

        For DAPO the clipping uses ``epsilon`` and ``epsilon_high``; for all
        other algorithms ``epsilon`` is used for both sides.

        Returns the scalar loss value.
        """
        eps_low = self.config.training.epsilon
        eps_high = (
            self.config.training.epsilon_high
            if isinstance(self.algorithm, DAPOEstimator)
            else self.config.training.epsilon
        )

        # Collect all (prompt_tokens, action_tokens, old_log_probs, advantage) pairs
        samples = []
        for traj_idx, traj in enumerate(trajectories):
            for step_idx, step in enumerate(traj.steps):
                if len(step.action_tokens) == 0:
                    continue
                samples.append((
                    step.prompt_tokens,
                    step.action_tokens,
                    step.log_probs,
                    advantages[traj_idx][step_idx],
                ))

        if not samples:
            return 0.0

        from mlx.utils import tree_map

        n_samples = len(samples)
        micro_bs = max(1, self.config.training.micro_batch_size)

        def chunk_loss_sum(model, chunk):
            """Sum (not mean) of per-sample PPO-clip losses for a chunk."""
            loss = mx.array(0.0)
            for prompt_toks, action_toks, old_lps_list, adv in chunk:
                new_lps = _compute_log_probs_mx(model, prompt_toks, action_toks)
                old_lps = mx.array(old_lps_list)
                min_len = min(new_lps.shape[0], old_lps.shape[0])
                if min_len == 0:
                    continue
                new_lps = new_lps[:min_len]
                old_lps = mx.stop_gradient(old_lps[:min_len])
                ratio = mx.exp(new_lps - old_lps)
                adv_arr = mx.array(adv)
                clipped = mx.clip(ratio, 1.0 - eps_low, 1.0 + eps_high)
                loss = loss + (-mx.minimum(ratio * adv_arr, clipped * adv_arr)).mean()
            return loss

        self.policy.train()

        # Accumulate gradients across micro-batches. Each micro-batch is its own
        # autograd graph that gets evaluated and freed before the next one starts,
        # bounding peak activation memory to ~micro_bs sequences worth of layers.
        accumulated_grads = None
        total_loss_sum = 0.0
        for start in range(0, n_samples, micro_bs):
            chunk = samples[start : start + micro_bs]

            def fn(model, chunk=chunk):
                return chunk_loss_sum(model, chunk)

            chunk_loss, chunk_grads = nn.value_and_grad(self.policy.model, fn)(
                self.policy.model
            )
            mx.eval(chunk_loss, chunk_grads)
            total_loss_sum += float(chunk_loss)

            if accumulated_grads is None:
                accumulated_grads = chunk_grads
            else:
                accumulated_grads = tree_map(
                    lambda a, g: a + g, accumulated_grads, chunk_grads
                )
            mx.eval(accumulated_grads)

        # Match the original mean-over-samples reduction.
        final_grads = tree_map(lambda g: g / n_samples, accumulated_grads)
        self.optimizer.update(self.policy.model, final_grads)
        mx.eval(self.policy.model.parameters(), self.optimizer.state)
        self.policy.eval()

        return total_loss_sum / n_samples


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_log_probs_mx(
    model: nn.Module,
    prompt_tokens: list[int],
    action_tokens: list[int],
) -> mx.array:
    """Differentiable forward pass returning log-probs for action_tokens.

    Returns mx.array of shape (len(action_tokens),).
    """
    full_tokens = prompt_tokens + action_tokens
    input_ids = mx.array(full_tokens[:-1])  # (T-1,)

    logits = model(input_ids[None])  # (1, T-1, vocab)
    logits = logits[0]  # (T-1, vocab)

    log_probs_all = nn.log_softmax(logits, axis=-1)  # (T-1, vocab)

    action_start = len(prompt_tokens) - 1
    n_action = len(action_tokens)
    indices = mx.array(action_tokens)  # (n_action,)

    # Gather log-probs for each action token position
    result = log_probs_all[action_start : action_start + n_action, :]
    # Advanced indexing: pick the log-prob at each action token id
    gathered = result[mx.arange(n_action), indices]
    return gathered
