"""Build a difficulty-filtered training subset for RLVR.

Per Lambert (2026) §7.2.3: "If the starting model can solve a problem
either 100% of the time or 0% of the time, there will be no gradient.
Many models have used difficulty filtering before starting a large-scale
RL to restrict the training problems to those that the starting model
solves only 20-80% of the time."

This script samples k trajectories from the BASE policy (no adapter) for
each candidate question, computes per-question success rate, and writes
the indices that fall into the [20%, 80%] band — those are the
"learnable" questions where RLVR will see real gradient signal.

Usage:
    uv run python scripts/build_difficulty_filter.py [n_candidates] [k] [out_file]

    n_candidates : how many of train_spider.json[0:n] to scan (default 1000)
    k            : samples per question (default 4)
    out_file     : output JSON (default data/spider/filter_20_80.json)
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
    n_candidates = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    out_file = (
        sys.argv[3] if len(sys.argv) > 3
        else "data/spider/filter_20_80.json"
    )

    cfg = TrainerConfig.from_yaml("configs/spider.yaml")
    print(f"[setup] n={n_candidates}  k={k}  out={out_file}", flush=True)
    print(f"[setup] loading {cfg.model.path} (base policy, no adapter)",
          flush=True)
    policy = Policy(
        model_path=cfg.model.path,
        quantize=cfg.model.quantize,
        lora_rank=cfg.model.lora_rank,
        lora_layers=cfg.model.lora_layers,
    )
    policy.eval()
    # Sample at training temp (default 1.0) so the filter reflects what
    # gradient signal the policy actually sees during rollout.
    policy.temperature = 1.0
    policy.top_p = 1.0

    env = SQLAgentEnvironment(
        data_dir=cfg.environment.data_dir,
        split="train",
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

    # Chunked collection for progress
    CHUNK = 8
    candidates = [{"prompt": "", "answer": i} for i in range(n_candidates)]
    success_rates: dict[int, float] = {}

    t0 = time.perf_counter()
    n_done = 0
    n_kept = 0
    n_easy = 0    # solved 100%
    n_hard = 0    # solved 0%

    for chunk_start in range(0, n_candidates, CHUNK):
        chunk = candidates[chunk_start : chunk_start + CHUNK]
        chunk_trajs = collector.collect(chunk, group_size=k)

        # Group trajs by question idx (chunked answers)
        # collector returns chunk[0]_traj1..chunk[0]_trajk, chunk[1]_traj1.., etc.
        for q_offset in range(len(chunk)):
            qid = chunk[q_offset]["answer"]
            traj_slice = chunk_trajs[q_offset * k : (q_offset + 1) * k]
            wins = sum(
                1 for t in traj_slice
                if t.steps and t.steps[-1].info.get("won")
            )
            rate = wins / k
            success_rates[qid] = rate
            n_done += 1
            if rate >= 0.99:
                n_easy += 1
            elif rate <= 0.01:
                n_hard += 1
            elif 0.20 <= rate <= 0.80:
                n_kept += 1

        elapsed = time.perf_counter() - t0
        remaining = n_candidates - n_done
        rate_per_q = n_done / max(elapsed, 1e-6)
        eta = remaining / max(rate_per_q, 1e-6)
        print(
            f"  [{n_done}/{n_candidates}]  kept={n_kept}  "
            f"easy(100%)={n_easy}  hard(0%)={n_hard}  "
            f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
            flush=True,
        )

    # Save filter
    kept_indices = sorted(
        qid for qid, rate in success_rates.items()
        if 0.20 <= rate <= 0.80
    )
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "k": k,
        "n_candidates": n_candidates,
        "n_kept": len(kept_indices),
        "n_easy": n_easy,
        "n_hard": n_hard,
        "indices": kept_indices,
        "success_rates": {str(k): v for k, v in sorted(success_rates.items())},
    }
    out_path.write_text(json.dumps(payload, indent=2))

    print()
    print(f"[done] scanned {n_candidates} questions in "
          f"{time.perf_counter() - t0:.0f}s")
    print(f"       easy (100%): {n_easy}/{n_candidates}  "
          f"({n_easy/n_candidates*100:.1f}%)")
    print(f"       hard ( 0%):  {n_hard}/{n_candidates}  "
          f"({n_hard/n_candidates*100:.1f}%)")
    print(f"       kept (20-80%): {len(kept_indices)}/{n_candidates}  "
          f"({len(kept_indices)/n_candidates*100:.1f}%)")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
