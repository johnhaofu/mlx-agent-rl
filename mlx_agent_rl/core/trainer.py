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
class LlamaCppConfig:
    enabled: bool = False
    gguf_path: str = ""
    port: int = 8090
    n_parallel: int = 16


@dataclass
class TrainerConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    mlc: MLCConfig = field(default_factory=MLCConfig)
    llamacpp: LlamaCppConfig = field(default_factory=LlamaCppConfig)

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
                n_parallel=lc.get("n_parallel", 16),
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

    def __init__(self, config: TrainerConfig, dataset: list[dict]) -> None:
        self.config = config
        self.dataset = dataset

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

    # ------------------------------------------------------------------
    # Static factories
    # ------------------------------------------------------------------

    @staticmethod
    def _create_algorithm(name: str, **kwargs) -> AdvantageEstimator:
        """Instantiate an advantage estimator by name."""
        name = name.lower()
        epsilon = kwargs.get("epsilon", 0.2)
        epsilon_high = kwargs.get("epsilon_high", 0.28)

        if name == "grpo":
            return GRPOEstimator(epsilon=epsilon)
        elif name == "dr_grpo":
            return DrGRPOEstimator()
        elif name == "dapo":
            return DAPOEstimator(
                epsilon=epsilon,
                epsilon_low=epsilon,
                epsilon_high=epsilon_high,
            )
        elif name == "gigpo":
            return GiGPOEstimator(epsilon=epsilon)
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
        cfg = self.config
        self.policy.train()

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

                avg_reward = sum(t.episode_reward for t in trajectories) / len(trajectories)
                success_rate = sum(1 for t in trajectories if t.succeeded) / len(trajectories)

                # Compute advantages
                advantages = self.algorithm.compute(trajectories)

                # Policy update
                loss = self._update_policy(trajectories, advantages)
                total_loss += loss
                num_updates += 1

                print(
                    f"  [{batch_idx+1}/{num_batches}] "
                    f"reward={avg_reward:.2f} success={success_rate:.0%} loss={loss:.4f}",
                    flush=True,
                )

            avg_loss = total_loss / max(num_updates, 1)
            print(f"Epoch {epoch + 1}/{cfg.training.epochs}  avg_loss={avg_loss:.4f}", flush=True)

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

        def loss_fn(model):
            total_loss = mx.array(0.0)
            for prompt_toks, action_toks, old_lps_list, adv in samples:
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
                total_loss = total_loss + (-mx.minimum(ratio * adv_arr, clipped * adv_arr)).mean()
            return total_loss / len(samples)

        self.policy.train()
        loss_val, grads = nn.value_and_grad(self.policy.model, loss_fn)(self.policy.model)
        self.optimizer.update(self.policy.model, grads)
        mx.eval(self.policy.model.parameters(), self.optimizer.state)
        self.policy.eval()

        return float(loss_val)


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
