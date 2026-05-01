"""Baseline-only evaluation of Qwen3-4B on HotpotQA distractor.

No training. We just run N episodes against the validation split using
the same RolloutCollector path the trainer uses, then report:
    - mean F1 (proxy for HotpotQA's official metric)
    - mean EM (exact match)
    - won rate (EM == 1.0)
    - mean steps used per episode
    - answered rate (episodes that called answer[…] before max_steps)

Usage:
    uv run python scripts/eval_hotpotqa_baseline.py [config] [n_val]
"""

from __future__ import annotations

import sys
import time

from mlx_agent_rl.core.policy import Policy
from mlx_agent_rl.core.rollout import RolloutCollector
from mlx_agent_rl.core.trainer import TrainerConfig
from mlx_agent_rl.environments.hotpotqa import HotpotQAEnvironment
from mlx_agent_rl.memory.memory import SlidingMemory


def main() -> None:
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/hotpotqa.yaml"
    n_val = int(sys.argv[2]) if len(sys.argv) > 2 else 128
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

    env = HotpotQAEnvironment(
        base_url=cfg.environment.base_url or "http://192.168.0.117:3002",
        split="validation",
        use_tools_schema=cfg.environment.use_tools_schema,
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
        print("[eval] no trajectories — sidecar issue?", flush=True)
        return

    f1_total = 0.0
    em_total = 0.0
    won_total = 0
    answered = 0
    step_total = 0
    for t in trajectories:
        last_info = t.steps[-1].info if t.steps else {}
        f1_total += float(last_info.get("f1", 0.0))
        em_total += float(last_info.get("em", 0.0))
        if last_info.get("won"):
            won_total += 1
        if t.succeeded:  # last step done == True (called answer or env terminated)
            answered += 1
        step_total += t.total_steps

    mean_f1 = f1_total / n
    mean_em = em_total / n
    won_rate = won_total / n
    answered_rate = answered / n
    mean_steps = step_total / n

    print()
    print(f"[result]  n={n}   wall={elapsed:.1f}s   {elapsed/n:.1f}s/episode")
    print(f"          mean F1   = {mean_f1:.3f}")
    print(f"          mean EM   = {mean_em:.3f}")
    print(f"          won rate  = {won_rate:.1%}  ({won_total}/{n})")
    print(f"          answered  = {answered_rate:.1%}  ({answered}/{n})")
    print(f"          avg steps = {mean_steps:.2f}")


if __name__ == "__main__":
    main()
