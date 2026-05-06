"""Find Spider episodes the agent failed to commit (no answer[]).

Runs N greedy episodes on the chosen split, identifies trajectories whose
last step has done=False (i.e. the agent never reached an ``answer[…]``),
and dumps the full per-turn rollout for each: question, gold, and what
the model actually emitted at every step.

Usage:
    uv run python scripts/debug_spider_unanswered.py [n=128] [split=test]
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

from mlx_agent_rl.core.policy import Policy
from mlx_agent_rl.core.rollout import RolloutCollector
from mlx_agent_rl.core.trainer import TrainerConfig
from mlx_agent_rl.environments.sql_agent import SQLAgentEnvironment
from mlx_agent_rl.memory.memory import SlidingMemory


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    split = sys.argv[2] if len(sys.argv) > 2 else "test"
    cfg = TrainerConfig.from_yaml("configs/spider.yaml")

    split_file = {"train": "train_spider.json",
                  "validation": "dev.json", "dev": "dev.json",
                  "test": "test.json"}[split]
    raw = json.load(open(Path(cfg.environment.data_dir) / split_file))

    print(f"[setup] loading {cfg.model.path}", flush=True)
    policy = Policy(
        model_path=cfg.model.path,
        quantize=cfg.model.quantize,
        lora_rank=cfg.model.lora_rank,
        lora_layers=cfg.model.lora_layers,
    )
    policy.eval()
    policy.temperature = 0.0
    policy.top_p = 1.0
    tok = policy.tokenizer

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

    val_dataset = [{"prompt": "", "answer": i} for i in range(n)]
    print(f"[run] {n} episodes on split={split} (greedy) …", flush=True)
    trajectories = collector.collect(val_dataset, group_size=1)

    unanswered = [(i, t) for i, t in enumerate(trajectories) if not t.succeeded]
    print(f"\n[summary] {len(unanswered)}/{n} unanswered "
          f"({len(unanswered)/n*100:.1f}%)\n", flush=True)

    cause_counts: dict[str, int] = {}
    for dev_idx, traj in unanswered:
        item = raw[dev_idx]
        question = item["question"]
        gold = item["query"]
        db_id = item["db_id"]

        # Classify the failure cause from the last step.
        last = traj.steps[-1]
        if not last.action_tokens:
            cause = "empty_output"
        else:
            last_text = tok.decode(last.action_tokens)
            if env.extract_action(last_text) is None:
                cause = "unparseable_action"
            else:
                cause = "max_steps_no_answer"
        cause_counts[cause] = cause_counts.get(cause, 0) + 1

        print(f"{'='*80}")
        print(f"[idx={dev_idx}] db={db_id}  cause={cause}  steps={len(traj.steps)}")
        print(f"[Q]    {question}")
        print(f"[gold] {textwrap.shorten(gold, width=200)}")

        for s_i, step in enumerate(traj.steps):
            txt = tok.decode(step.action_tokens) if step.action_tokens else "(empty)"
            parsed = env.extract_action(txt) if step.action_tokens else None
            print(f"\n  step {s_i}  reward={step.reward:.2f}  done={step.done}")
            print(f"  out:    {textwrap.shorten(txt, width=240)!r}")
            print(f"  parsed: {parsed!r}")
            if step.info:
                err = step.info.get("err") or step.info.get("error")
                if err:
                    print(f"  err:    {err}")
        print()

    print(f"{'='*80}\n[cause histogram]")
    for k, v in sorted(cause_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<22} {v}")


if __name__ == "__main__":
    main()
