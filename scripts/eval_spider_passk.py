"""Pass^k evaluation on Spider — measure consistency, not luck.

For each question, sample K trajectories (with temperature > 0). Then:
  - pass@1   : at least 1 of K succeeded
  - pass@k   : at least 1 of K succeeded (= pass@1 with K samples)
  - pass^k   : ALL K of K succeeded — agent is *consistent*

The pass^k metric (introduced by τ-bench, Yao et al. 2024) is the better
metric for agent deployment readiness: a 50% pass@1 model that's coin-flip
random is much worse than a 45% pass@1 model that's deterministic.

Usage:
    uv run python scripts/eval_spider_passk.py [config] [n] [split] [k] [adapter]

    n:         number of dev/test questions (default 128)
    split:     train | validation | test  (default test)
    k:         samples per question (default 5)
    adapter:   optional path to LoRA adapter checkpoint dir
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from mlx_agent_rl.core.policy import Policy
from mlx_agent_rl.core.rollout import RolloutCollector
from mlx_agent_rl.core.trainer import TrainerConfig
from mlx_agent_rl.environments.sql_agent import SQLAgentEnvironment
from mlx_agent_rl.memory.memory import SlidingMemory


def main() -> None:
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/spider.yaml"
    n_val = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    split = sys.argv[3] if len(sys.argv) > 3 else "test"
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    adapter_dir = sys.argv[5] if len(sys.argv) > 5 else None
    cfg = TrainerConfig.from_yaml(cfg_path)

    print(f"[setup] split={split}  n={n_val}  k={k}  adapter={adapter_dir}",
          flush=True)
    print(f"[setup] loading {cfg.model.path}", flush=True)
    policy = Policy(
        model_path=cfg.model.path,
        quantize=cfg.model.quantize,
        lora_rank=cfg.model.lora_rank,
        lora_layers=cfg.model.lora_layers,
    )
    if adapter_dir:
        print(f"[setup] loading adapter from {adapter_dir}", flush=True)
        policy.load_adapters(adapter_dir)
    policy.eval()
    # Sample with temperature > 0 — pass^k only makes sense under stochastic
    # decoding. We use the val_temperature (0.4) as a moderate-noise default.
    policy.temperature = cfg.training.val_temperature
    policy.top_p = cfg.training.val_top_p

    env = SQLAgentEnvironment(
        data_dir=cfg.environment.data_dir,
        split=split,
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
    print(f"[run] {n_val} questions × k={k} samples = {n_val*k} rollouts "
          f"(temp={policy.temperature}) …", flush=True)

    t0 = time.perf_counter()
    # group_size=k: for each prompt, collector returns k trajectories sharing uid.
    trajectories = collector.collect(val_dataset, group_size=k)
    elapsed = time.perf_counter() - t0

    # Group successes per question
    by_question: dict[int, list[bool]] = {}
    for t, prompt_dict in zip(trajectories, _expand(val_dataset, k)):
        qid = prompt_dict["answer"]
        info = t.steps[-1].info if t.steps else {}
        won = bool(info.get("won"))
        by_question.setdefault(qid, []).append(won)

    n = len(by_question)
    pass_at_1 = sum(any(s) for s in by_question.values()) / n
    pass_caret_k = sum(all(s) for s in by_question.values()) / n
    # also report fraction-of-k distribution
    consistency = [sum(s) / k for s in by_question.values()]
    avg_consistency = sum(consistency) / n

    print()
    print(f"[result] n={n}   k={k}   wall={elapsed:.1f}s   "
          f"{elapsed/(n*k):.1f}s/rollout")
    print(f"         pass@1   = {pass_at_1:.3f}  "
          f"({sum(any(s) for s in by_question.values())}/{n})")
    print(f"         pass^{k}    = {pass_caret_k:.3f}  "
          f"({sum(all(s) for s in by_question.values())}/{n})")
    print(f"         avg_consistency (mean k-success rate per question) "
          f"= {avg_consistency:.3f}")

    # Stratify pass^k by question consistency level
    buckets = {"all_k": 0, "k-1": 0, "half_or_more": 0, "any": 0, "none": 0}
    for s in by_question.values():
        succ = sum(s)
        if succ == k:
            buckets["all_k"] += 1
        elif succ == k - 1:
            buckets["k-1"] += 1
        elif succ >= k // 2 + 1:
            buckets["half_or_more"] += 1
        elif succ >= 1:
            buckets["any"] += 1
        else:
            buckets["none"] += 1
    print(f"\n[consistency distribution]")
    for label, count in buckets.items():
        print(f"  {label:<14} {count:>4}/{n}  ({count/n*100:5.1f}%)")


def _expand(prompts: list[dict], k: int) -> list[dict]:
    """Mirror the rollout collector: each prompt yields k trajectories in order."""
    out = []
    for p in prompts:
        out.extend([p] * k)
    return out


if __name__ == "__main__":
    main()
