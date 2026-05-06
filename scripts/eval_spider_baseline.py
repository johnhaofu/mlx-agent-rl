"""Baseline-only evaluation of Qwen3-4B on Spider dev.

No training. Uses the configured val_temperature for low-variance
sampling and reports:
    - exec_match rate (Spider's EX, the primary metric)
    - answered rate (episodes that called answer[…])
    - mean steps used per episode
    - per-hardness breakdown when ``hardness`` is available

Usage:
    uv run python scripts/eval_spider_baseline.py [config] [n]
"""

from __future__ import annotations

import sys
import time

from mlx_agent_rl.core.policy import Policy
from mlx_agent_rl.core.rollout import RolloutCollector
from mlx_agent_rl.core.trainer import TrainerConfig
from mlx_agent_rl.environments.sql_agent import SQLAgentEnvironment
from mlx_agent_rl.memory.memory import SlidingMemory


def main() -> None:
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/spider.yaml"
    n_val = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    cfg = TrainerConfig.from_yaml(cfg_path)

    print(f"[setup] loading {cfg.model.path}", flush=True)
    policy = Policy(
        model_path=cfg.model.path,
        quantize=cfg.model.quantize,
        lora_rank=cfg.model.lora_rank,
        lora_layers=cfg.model.lora_layers,
    )
    policy.eval()
    policy.temperature = cfg.training.val_temperature
    policy.top_p = cfg.training.val_top_p

    env = SQLAgentEnvironment(
        data_dir=cfg.environment.data_dir,
        split="validation",  # always evaluate on dev
        schema_max_chars=cfg.environment.schema_max_chars,
        rows_per_query=cfg.environment.rows_per_query,
    )
    memory = SlidingMemory(window_size=cfg.memory.window_size)
    collector = RolloutCollector(
        policy=policy,
        env=env,
        memory=memory,
        max_steps=cfg.rollout.max_steps,
        max_tokens=cfg.rollout.max_tokens,
        invalid_action_penalty=cfg.environment.invalid_action_penalty,
        system_prompt=cfg.rollout.system_prompt,
        enable_thinking=cfg.rollout.enable_thinking,
    )

    val_dataset = [{"prompt": "", "answer": i} for i in range(n_val)]
    print(f"[eval] running {n_val} val episodes (split=validation, "
          f"temp={policy.temperature}) …", flush=True)

    t0 = time.perf_counter()
    trajectories = collector.collect(val_dataset, group_size=1)
    elapsed = time.perf_counter() - t0

    n = len(trajectories)
    if n == 0:
        print("[eval] no trajectories", flush=True)
        return

    matches = 0
    answered = 0
    steps_total = 0
    for t in trajectories:
        info = t.steps[-1].info if t.steps else {}
        if info.get("won"):
            matches += 1
        if t.succeeded:
            answered += 1
        steps_total += t.total_steps

    print()
    print(f"[result]  n={n}   wall={elapsed:.1f}s   {elapsed/n:.1f}s/episode")
    print(f"          EX (exec match) = {matches/n:.3f}  ({matches}/{n})")
    print(f"          answered        = {answered/n:.3f}  ({answered}/{n})")
    print(f"          avg steps       = {steps_total/n:.2f}")


if __name__ == "__main__":
    main()
