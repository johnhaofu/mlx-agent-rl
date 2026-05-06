"""Trainer module — orchestrates policy training with RL algorithms."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm.models.cache import (
    can_trim_prompt_cache,
    make_prompt_cache,
    trim_prompt_cache,
)

from mlx_agent_rl.algorithms.base import AdvantageEstimator
from mlx_agent_rl.algorithms.dapo import DAPOEstimator
from mlx_agent_rl.algorithms.dr_grpo import DrGRPOEstimator
from mlx_agent_rl.algorithms.gigpo import GiGPOEstimator
from mlx_agent_rl.algorithms.grpo import GRPOEstimator
from mlx_agent_rl.core.rollout import RolloutCollector
from mlx_agent_rl.data.trajectory import Trajectory
from mlx_agent_rl.environments.base import BaseEnvironment
from mlx_agent_rl.memory.memory import SlidingMemory


class AdaptiveKLController:
    """verl-agent style proportional KL coefficient controller.

    Adjusts ``kl_coef`` to drive the observed KL toward ``target_kl`` over
    ``horizon`` steps. ``proportional_error`` is clipped to ±0.2 to prevent
    runaway swings on a single noisy KL estimate.
    """

    def __init__(self, init_kl_coef: float, target_kl: float, horizon: int) -> None:
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int) -> None:
        proportional_error = max(-0.2, min(0.2, current_kl / self.target - 1))
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Constant kl_coef; the trivial controller."""

    def __init__(self, init_kl_coef: float) -> None:
        self.value = init_kl_coef

    def update(self, current_kl: float, n_steps: int) -> None:
        return None


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
    enable_thinking: bool = False  # Qwen3 reasoning <think>...</think> mode
    per_traj_temperatures: list[float] | None = None  # cycle temps across group members for diversity


@dataclass
class TrainingConfig:
    algorithm: str = "grpo"
    lr: float = 1e-5
    epochs: int = 10
    batch_size: int = 4
    epsilon: float = 0.2
    epsilon_high: float = 0.28
    clip_grad: float = 1.0  # max global L2 norm of LoRA gradients per update
    micro_batch_size: int = 4  # samples per backward pass; controls peak memory
    val_interval: int = 5  # run val eval every N train batches (0 disables)
    val_temperature: float = 0.4  # low-variance sampling for val (not greedy)
    val_top_p: float = 1.0
    val_before_train: bool = True  # run a baseline val pass at step 0
    ppo_epochs: int = 4  # number of update passes over each rollout batch
    kl_coef: float = 0.05  # initial KL(current || reference) penalty weight (0 disables)
    clip_ratio_c: float = 3.0  # dual-clip lower bound for negative-advantage steps
    entropy_coef: float = 0.0  # weight on entropy bonus (0 disables)
    kl_ctrl_type: str = "fixed"  # 'fixed' or 'adaptive'
    kl_target: float = 0.01  # target KL when kl_ctrl_type='adaptive'
    kl_horizon: int = 10000  # adaptation horizon for the adaptive controller
    gigpo_mode: str = "mean_std_norm"  # 'mean_std_norm' (default) or 'mean_norm'
    dynamic_sampling: bool = False  # DAPO-style: drop zero-reward-variance groups before update


@dataclass
class EnvironmentConfig:
    type: str = "calculator"
    invalid_action_penalty: float = -0.1
    base_url: str | None = None  # sidecar URL for HTTP envs (webshop/hotpotqa)
    use_tools_schema: bool = True  # pass tools to chat template (Qwen3 native)
    dense_reward: bool = False  # WebShop: use continuous task_score instead of binary won
    split: str = "train"  # HotpotQA / SQL agent: which split to sample from
    data_dir: str | None = None  # SQL agent: path to data/spider/spider_data
    schema_max_chars: int = 4000  # SQL agent: cap on initial-obs schema block
    rows_per_query: int = 10  # SQL agent: rows shown after sql[…]
    partial_credit: float = 0.1  # SQL agent: reward for valid SQL with wrong result; 0 disables (v4 ablation)
    format_reward: float = 0.0  # SQL agent: small bonus per well-formed action; 0 disables (default)


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
                enable_thinking=r.get("enable_thinking", False),
                per_traj_temperatures=r.get("per_traj_temperatures", None),
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
                val_before_train=t.get("val_before_train", True),
                ppo_epochs=t.get("ppo_epochs", 4),
                kl_coef=t.get("kl_coef", 0.05),
                clip_ratio_c=t.get("clip_ratio_c", 3.0),
                entropy_coef=t.get("entropy_coef", 0.0),
                kl_ctrl_type=t.get("kl_ctrl_type", "fixed"),
                kl_target=t.get("kl_target", 0.01),
                kl_horizon=t.get("kl_horizon", 10000),
                gigpo_mode=t.get("gigpo_mode", "mean_std_norm"),
                dynamic_sampling=t.get("dynamic_sampling", False),
            )

        if "environment" in raw:
            e = raw["environment"]
            cfg.environment = EnvironmentConfig(
                type=e.get("type", "calculator"),
                invalid_action_penalty=e.get("invalid_action_penalty", -0.1),
                base_url=e.get("base_url", None),
                use_tools_schema=e.get("use_tools_schema", True),
                dense_reward=e.get("dense_reward", False),
                split=e.get("split", "train"),
                data_dir=e.get("data_dir", None),
                schema_max_chars=e.get("schema_max_chars", 4000),
                rows_per_query=e.get("rows_per_query", 10),
                partial_credit=e.get("partial_credit", 0.1),
                format_reward=e.get("format_reward", 0.0),
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
            gigpo_mode=config.training.gigpo_mode,
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
            enable_thinking=config.rollout.enable_thinking,
        )

        # Optimizer — only update LoRA parameters
        self.optimizer = optim.Adam(learning_rate=config.training.lr)

        # KL coefficient controller (constant or proportional/adaptive).
        if config.training.kl_ctrl_type == "adaptive":
            self._kl_ctrl: AdaptiveKLController | FixedKLController | None = (
                AdaptiveKLController(
                    init_kl_coef=config.training.kl_coef,
                    target_kl=config.training.kl_target,
                    horizon=config.training.kl_horizon,
                )
            )
        elif config.training.kl_ctrl_type == "fixed":
            self._kl_ctrl = FixedKLController(init_kl_coef=config.training.kl_coef)
        else:
            raise ValueError(
                f"Unknown kl_ctrl_type: {config.training.kl_ctrl_type!r}. "
                f"Choose 'fixed' or 'adaptive'."
            )

        # Optional W&B logging. Initialized lazily so that pure-test imports
        # Output directory for LoRA checkpoints. ``run_name`` defaults to
        # the wandb run name; falls back to a generic ``run`` so even
        # offline runs leave behind a recoverable artifact.
        from pathlib import Path as _Path
        self._out_dir = _Path("outputs") / (
            config.wandb.run_name or "run"
        )
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._best_val_score: float | None = None

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
        gigpo_mode = kwargs.get("gigpo_mode", "mean_std_norm")
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
            return GiGPOEstimator(epsilon=norm_eps, mode=gigpo_mode)
        else:
            raise ValueError(f"Unknown algorithm: {name!r}. Choose from grpo, dr_grpo, dapo, gigpo.")

    @staticmethod
    def _create_environment(config: EnvironmentConfig) -> BaseEnvironment:
        """Instantiate an environment by type name."""
        env_type = config.type.lower()
        if env_type == "calculator":
            from mlx_agent_rl.environments.calculator import CalculatorEnvironment

            return CalculatorEnvironment()
        elif env_type == "numberline":
            from mlx_agent_rl.environments.numberline import NumberLineEnvironment

            return NumberLineEnvironment()
        elif env_type == "webshop":
            from mlx_agent_rl.environments.webshop import WebShopEnvironment

            base_url = getattr(config, "base_url", None) or "http://192.168.0.117:3001"
            return WebShopEnvironment(
                base_url=base_url,
                use_tools_schema=getattr(config, "use_tools_schema", True),
                dense_reward=getattr(config, "dense_reward", False),
            )
        elif env_type == "hotpotqa":
            from mlx_agent_rl.environments.hotpotqa import HotpotQAEnvironment

            base_url = getattr(config, "base_url", None) or "http://192.168.0.117:3002"
            return HotpotQAEnvironment(
                base_url=base_url,
                split=getattr(config, "split", "train"),
                use_tools_schema=getattr(config, "use_tools_schema", False),
            )
        elif env_type == "sql_agent":
            from mlx_agent_rl.environments.sql_agent import SQLAgentEnvironment

            return SQLAgentEnvironment(
                data_dir=getattr(config, "data_dir", None),
                split=getattr(config, "split", "train"),
                schema_max_chars=getattr(config, "schema_max_chars", 4000),
                rows_per_query=getattr(config, "rows_per_query", 10),
                use_tools_schema=getattr(config, "use_tools_schema", False),
                partial_credit=getattr(config, "partial_credit", 0.1),
                format_reward=getattr(config, "format_reward", 0.0),
            )
        else:
            raise ValueError(
                f"Unknown environment type: {env_type!r}. "
                f"Choose from: calculator, numberline, webshop, hotpotqa, sql_agent."
            )

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

        if (
            cfg.training.val_before_train
            and self.val_dataset is not None
            and cfg.training.val_interval > 0
        ):
            self._evaluate(global_step=0)

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
                task_score = _mean_last_step_metric(trajectories, "task_score")
                won_rate = _mean_last_step_metric(trajectories, "won")

                rolling.append((avg_reward, answered_rate, correct_rate))
                rw_r = sum(x[0] for x in rolling) / len(rolling)
                rw_a = sum(x[1] for x in rolling) / len(rolling)
                rw_c = sum(x[2] for x in rolling) / len(rolling)

                # Compute advantages
                advantages = self.algorithm.compute(trajectories)

                # DAPO-style dynamic sampling (Yu et al. 2025): drop groups
                # whose trajectories ALL got the maximum or ALL got the
                # minimum reward — these groups have no learning signal
                # since (r_i - mean) = 0 for every trajectory. The original
                # DAPO paper specifies "remove all samples with reward = 0
                # or reward = 1 for all samples in the batch (no learning
                # signal)" — for binary reward this matches our prior
                # implementation; for partial-credit setups it's strictly
                # less aggressive (preserves groups that mix 0 and partial).
                used_trajectories = trajectories
                used_advantages = advantages
                n_dropped_groups = 0
                if cfg.training.dynamic_sampling:
                    from collections import defaultdict as _dd
                    groups: dict[str, list[int]] = _dd(list)
                    for i, t in enumerate(trajectories):
                        groups[t.uid].append(i)
                    # Reward extremes across the whole batch (DAPO uses 0/1
                    # explicitly; we use empirical max/min so the rule
                    # generalizes when partial_credit > 0 or rewards are
                    # shifted by format_reward).
                    all_rewards = [t.episode_reward for t in trajectories]
                    r_max, r_min = max(all_rewards), min(all_rewards)
                    keep_idx: list[int] = []
                    for idxs in groups.values():
                        rewards = [trajectories[i].episode_reward for i in idxs]
                        all_max = all(r >= r_max - 1e-6 for r in rewards)
                        all_min = all(r <= r_min + 1e-6 for r in rewards)
                        if all_max or all_min:
                            n_dropped_groups += 1
                        else:
                            keep_idx.extend(idxs)
                    if keep_idx and n_dropped_groups > 0:
                        used_trajectories = [trajectories[i] for i in keep_idx]
                        used_advantages = [advantages[i] for i in keep_idx]

                # Policy update
                loss = self._update_policy(used_trajectories, used_advantages)
                total_loss += loss
                num_updates += 1

                global_step = epoch * num_batches + batch_idx + 1
                drop_str = f" drop={n_dropped_groups}" if n_dropped_groups > 0 else ""
                print(
                    f"  [{batch_idx+1}/{num_batches}] "
                    f"r={avg_reward:+.2f} a={answered_rate:.0%} c={correct_rate:.0%} "
                    f"| ma{len(rolling)}: r={rw_r:+.2f} a={rw_a:.0%} c={rw_c:.0%} "
                    f"loss={loss:.4f}{drop_str}",
                    flush=True,
                )

                if self._wandb is not None:
                    payload = {
                        "train/reward": avg_reward,
                        "train/answered": answered_rate,
                        "train/correct": correct_rate,
                        "train/loss": loss,
                        "train/ma_reward": rw_r,
                        "train/ma_answered": rw_a,
                        "train/ma_correct": rw_c,
                        "train/epoch": epoch + 1,
                        "train/task_score": task_score,
                        "train/won": won_rate,
                    }
                    if self._kl_ctrl is not None:
                        payload["train/kl_coef"] = self._kl_ctrl.value
                    self._wandb.log(payload, step=global_step)

                if (
                    self.val_dataset is not None
                    and cfg.training.val_interval > 0
                    and global_step % cfg.training.val_interval == 0
                ):
                    self._evaluate(global_step)

            avg_loss = total_loss / max(num_updates, 1)
            print(f"Epoch {epoch + 1}/{cfg.training.epochs}  avg_loss={avg_loss:.4f}", flush=True)
            epoch_dir = self._out_dir / f"epoch_{epoch + 1:02d}"
            self.policy.save_adapters(epoch_dir)
            print(f"  [save] epoch {epoch + 1} adapters → {epoch_dir}", flush=True)

        final_dir = self._out_dir / "final"
        self.policy.save_adapters(final_dir)
        print(f"  [save] final adapters → {final_dir}", flush=True)

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
        task_score = _mean_last_step_metric(trajectories, "task_score")
        won_rate = _mean_last_step_metric(trajectories, "won")
        print(
            f"  [val @ step {global_step}] r={avg_reward:+.2f} "
            f"a={answered_rate:.0%} c={correct_rate:.0%} "
            f"score={task_score:.2f} won={won_rate:.0%} "
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
                    "val/task_score": task_score,
                    "val/won": won_rate,
                    "val/n": n,
                },
                step=global_step,
            )

        # Save adapters on val improvement so we always have a recoverable
        # checkpoint matching the best known val score. ``won_rate`` tracks
        # the env's primary success metric (EX for Spider, F1≥thresh for
        # HotpotQA).
        score = won_rate if won_rate > 0 else correct_rate
        if self._best_val_score is None or score > self._best_val_score:
            self._best_val_score = score
            best_dir = self._out_dir / "best"
            self.policy.save_adapters(best_dir)
            print(f"  [save] best val score={score:.3f} → {best_dir}", flush=True)
        latest_dir = self._out_dir / f"step_{global_step:06d}"
        self.policy.save_adapters(latest_dir)

    # ------------------------------------------------------------------
    # Policy update (PPO-clip)
    # ------------------------------------------------------------------

    def _update_policy(
        self,
        trajectories: list[Trajectory],
        advantages: list[list[float]],
    ) -> float:
        """Compute PPO-clip loss and perform one gradient step.

        Asymmetric clipping (clip-higher) is active for ALL algorithms when
        ``epsilon_high > epsilon`` — historically only DAPO used it, but
        looser upward clip helps any policy gradient method capture
        positive-advantage updates that get cropped at the symmetric
        boundary. Set ``epsilon_high == epsilon`` in YAML to fall back
        to standard symmetric PPO.

        Returns the scalar loss value.
        """
        eps_low = self.config.training.epsilon
        eps_high = self.config.training.epsilon_high

        # Collect all (prompt_tokens, action_tokens, old_log_probs, advantage) pairs.
        # We also remember which traj each sample belongs to so the ref_lps
        # loop below can reuse a per-trajectory KV cache (samples within a
        # trajectory share a monotonically extending prompt prefix).
        samples = []
        sample_traj_idx: list[int] = []
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
                sample_traj_idx.append(traj_idx)

        if not samples:
            return 0.0

        from mlx.utils import tree_map
        from mlx.optimizers import clip_grad_norm

        cfg = self.config.training
        n_samples = len(samples)
        micro_bs = max(1, cfg.micro_batch_size)
        kl_coef = self._kl_ctrl.value if self._kl_ctrl is not None else cfg.kl_coef
        clip_ratio_c = cfg.clip_ratio_c
        clip_grad = cfg.clip_grad
        entropy_coef = cfg.entropy_coef

        # ----- Compute frozen-reference log probs once for this batch ----- #
        # The reference is the base model (LoRA scale=0). KL keeps the policy
        # within a global trust region of the initial model — without this
        # we observed empirical collapse around batch 18 of the Calculator
        # and NumberLine runs.
        #
        # ref_lps is the second-largest forward cost of the update (about
        # 6 min/batch on Qwen3-4B with 16 trajs × ~7 steps × ~2k-token
        # prompts). Within a trajectory each step's prompt extends the
        # previous, so we can keep a per-trajectory KV cache and feed only
        # the delta tokens — matching the rollout-side prompt cache trick.
        # On the bench (8-turn WebShop) this cuts ref_lps token volume to
        # ~24% of the no-cache version.
        ref_lps_per_sample: list[mx.array] = []
        self.policy.eval()
        with self.policy.reference():
            # NOTE: an episode-aware cached version of this loop was
            # prototyped (see scripts/verify_ref_lps_cache.py) but the
            # cached forward produced different log-probs from a fresh
            # forward at intermediate positions on Qwen3-4B (max |Δ| ≈
            # 0.4 logit), suggesting a subtle interaction with
            # mx.fast.scaled_dot_product_attention's "causal" mask when
            # query length < key length. Reverted to per-sample fresh
            # forward; revisit when MLX behaviour is verified.
            for prompt_toks, action_toks, _old_lps_list, _adv in samples:
                rl = _compute_log_probs_mx(self.policy.model, prompt_toks, action_toks)
                mx.eval(rl)
                ref_lps_per_sample.append(mx.stop_gradient(rl))

        def chunk_loss_sum(model, chunk, ref_chunk):
            """Sum of per-sample losses: PPO-dual-clip + KL(p||ref) + entropy.

            mlx-lm-lora-style ratio convention:
              ratio = exp(log_p_current - log_p_ref)   # drift from base, NOT from rollout
            This keeps PPO clip *active* even with ppo_epochs=1, where the
            verl/standard convention (ratio = current/old_at_rollout) would
            collapse to 1 and turn the clip into dead code on the first inner
            update. Using ref_lps as the denominator makes the trust region
            a global bound on "how far the LoRA has moved the policy from
            the base model" rather than a per-batch update bound.

            KL uses the k3 unbiased estimator (Schulman) against the same
            frozen base; advantage is broadcast across action tokens; entropy
            bonus is the single-sample -log_p estimate.
            """
            loss = mx.array(0.0)
            for (prompt_toks, action_toks, _old_lps_list, adv), ref_lps in zip(
                chunk, ref_chunk
            ):
                new_lps = _compute_log_probs_mx(model, prompt_toks, action_toks)
                min_len = min(new_lps.shape[0], ref_lps.shape[0])
                if min_len == 0:
                    continue
                new_lps = new_lps[:min_len]
                ref_l = ref_lps[:min_len]
                # ratio = π_current / π_ref ; PPO clip bounds drift from base
                ratio = mx.exp(new_lps - ref_l)
                adv_arr = mx.array(adv) * mx.ones_like(new_lps)

                pg1 = -ratio * adv_arr
                pg2 = -mx.clip(ratio, 1.0 - eps_low, 1.0 + eps_high) * adv_arr
                pg_max = mx.maximum(pg1, pg2)
                if adv < 0:
                    upper = -clip_ratio_c * adv_arr
                    pg_loss = mx.minimum(pg_max, upper)
                else:
                    pg_loss = pg_max

                sample_loss = pg_loss
                if kl_coef > 0:
                    delta = ref_l - new_lps  # log(p_ref / p_new)
                    kl = mx.clip(mx.exp(delta) - delta - 1.0, -10.0, 10.0)
                    sample_loss = sample_loss + kl_coef * kl
                if entropy_coef > 0:
                    sample_loss = sample_loss - entropy_coef * (-new_lps)

                loss = loss + sample_loss.mean()
            return loss

        self.policy.train()
        total_loss = 0.0
        total_kl = 0.0
        n_updates = 0

        # Multi-epoch PPO update: reuse the same rollout (and the frozen
        # ref_lps cached above) for several gradient passes.
        for _ppo_epoch in range(cfg.ppo_epochs):
            accumulated_grads = None
            epoch_loss_sum = 0.0
            for start in range(0, n_samples, micro_bs):
                chunk = samples[start : start + micro_bs]
                ref_chunk = ref_lps_per_sample[start : start + micro_bs]

                def fn(model, chunk=chunk, ref_chunk=ref_chunk):
                    return chunk_loss_sum(model, chunk, ref_chunk)

                chunk_loss, chunk_grads = nn.value_and_grad(self.policy.model, fn)(
                    self.policy.model
                )
                mx.eval(chunk_loss, chunk_grads)
                epoch_loss_sum += float(chunk_loss)

                if accumulated_grads is None:
                    accumulated_grads = chunk_grads
                else:
                    accumulated_grads = tree_map(
                        lambda a, g: a + g, accumulated_grads, chunk_grads
                    )
                mx.eval(accumulated_grads)

            final_grads = tree_map(lambda g: g / n_samples, accumulated_grads)
            if clip_grad and clip_grad > 0:
                final_grads, _gnorm = clip_grad_norm(final_grads, clip_grad)
            self.optimizer.update(self.policy.model, final_grads)
            mx.eval(self.policy.model.parameters(), self.optimizer.state)

            total_loss += epoch_loss_sum / n_samples
            n_updates += 1

        # Estimate observed KL post-update so the *adaptive* controller can
        # tune kl_coef. The fixed controller ignores the value, so the loop
        # is pure waste in that mode — and at ~6 min/batch on Qwen3-4B with
        # WebShop-length prompts it is the second-largest item in the per-
        # batch budget after PPO backprop. Skip it for fixed mode; revisit
        # if we ever want to log observed-KL as a side metric.
        if isinstance(self._kl_ctrl, AdaptiveKLController):
            kls = []
            for (prompt_toks, action_toks, _ol, _adv), ref_lps in zip(
                samples, ref_lps_per_sample
            ):
                new_lps = _compute_log_probs_mx(
                    self.policy.model, prompt_toks, action_toks
                )
                n = min(new_lps.shape[0], ref_lps.shape[0])
                if n == 0:
                    continue
                delta = (ref_lps[:n] - new_lps[:n])
                kl = (mx.exp(delta) - delta - 1.0).mean()
                mx.eval(kl)
                kls.append(float(kl))
            if kls:
                total_kl = sum(kls) / len(kls)
                self._kl_ctrl.update(total_kl, n_steps=1)

        self.policy.eval()
        return total_loss / max(n_updates, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mean_last_step_metric(trajectories: list[Trajectory], key: str) -> float:
    """Mean of ``step.info[key]`` taken from each trajectory's last step.

    Trajectories whose last step lacks the key are skipped. Returns 0.0 when
    no trajectory contributes a value (e.g. envs that don't populate info).
    """
    vals: list[float] = []
    for t in trajectories:
        if not t.steps:
            continue
        v = t.steps[-1].info.get(key)
        if v is None:
            continue
        vals.append(float(v))
    return sum(vals) / len(vals) if vals else 0.0


def _compute_log_probs_mx_cached(
    model: nn.Module,
    prompt_tokens: list[int],
    action_tokens: list[int],
    prompt_cache,
    cache_offset: int,
) -> mx.array:
    """Like ``_compute_log_probs_mx`` but reuses an existing KV cache.

    The caller must guarantee that ``prompt_cache`` already contains exactly
    ``cache_offset`` tokens, and those tokens are a prefix of
    ``prompt_tokens``. We feed the remaining ``prompt_tokens[cache_offset:]
    + action_tokens[:-1]`` through the model with the cache attached, then
    gather log-probs for ``action_tokens`` from the resulting logits.

    After this call the cache holds ``prompt_tokens + action_tokens[:-1]``
    (i.e. ``cache_offset + len(delta_prompt) + len(action) - 1`` tokens).

    Returns mx.array of shape (len(action_tokens),).
    """
    delta_prompt = prompt_tokens[cache_offset:]
    n_action = len(action_tokens)
    if not delta_prompt and n_action == 1:
        # Cache covers full prompt and action is a single token. We can't
        # call model with 0 inputs, so re-feed the last prompt token (after
        # trimming cache by 1) to get a logit position for action[0].
        trim_prompt_cache(prompt_cache, 1)
        delta_prompt = prompt_tokens[-1:]

    full_input = list(delta_prompt) + list(action_tokens[:-1])
    if not full_input:
        # Single-action with empty delta-prompt and no action[:-1]: shouldn't
        # happen after the trim above, but guard anyway.
        raise RuntimeError(
            "_compute_log_probs_mx_cached: cannot forward zero tokens"
        )
    input_ids = mx.array(full_input)
    logits = model(input_ids[None], cache=prompt_cache)  # (1, T_in, V)
    log_probs_all = nn.log_softmax(logits[0], axis=-1)  # (T_in, V)

    # Position of logit predicting action[k] = (len(delta_prompt) - 1) + k
    start = len(delta_prompt) - 1
    result = log_probs_all[start : start + n_action, :]
    indices = mx.array(action_tokens)
    gathered = result[mx.arange(n_action), indices]
    return gathered


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
