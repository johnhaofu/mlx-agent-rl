"""Spider baseline with per-hardness breakdown.

Reports EX/answered/avg-steps bucketed by Spider's official hardness label
(easy/medium/hard/extra), computed inline from the parsed AST in the split
JSON — no nltk dependency.

Greedy decoding (temp=0) for deterministic, reproducible baseline numbers
that match Spider leaderboard reporting conventions.

Usage:
    uv run python scripts/eval_spider_by_hardness.py [config] [n] [split]

    split: one of train | validation | test  (default: test)
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


HARDNESS_ORDER = ["easy", "medium", "hard", "extra"]
_WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like',
              'is', 'exists')
_AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')


def _has_agg(unit) -> bool:
    return unit[0] != _AGG_OPS.index('none')


def _count_agg(units) -> int:
    return sum(1 for u in units if _has_agg(u))


def _get_nested_sql(sql):
    nested = []
    for cond in (sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]):
        if isinstance(cond[3], dict):
            nested.append(cond[3])
        if isinstance(cond[4], dict):
            nested.append(cond[4])
    for k in ('intersect', 'except', 'union'):
        if sql.get(k) is not None:
            nested.append(sql[k])
    return nested


def _count_component1(sql) -> int:
    c = 0
    if len(sql['where']) > 0: c += 1
    if len(sql['groupBy']) > 0: c += 1
    if len(sql['orderBy']) > 0: c += 1
    if sql['limit'] is not None: c += 1
    if len(sql['from']['table_units']) > 0:
        c += len(sql['from']['table_units']) - 1
    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    c += sum(1 for t in ao if t == 'or')
    cu = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    c += sum(1 for u in cu if u[1] == _WHERE_OPS.index('like'))
    return c


def _count_component2(sql) -> int:
    return len(_get_nested_sql(sql))


def _count_others(sql) -> int:
    c = 0
    agg = _count_agg(sql['select'][1])
    agg += _count_agg(sql['where'][::2])
    agg += _count_agg(sql['groupBy'])
    if len(sql['orderBy']) > 0:
        agg += _count_agg(
            [u[1] for u in sql['orderBy'][1] if u[1]] +
            [u[2] for u in sql['orderBy'][1] if u[2]])
    agg += _count_agg(sql['having'])
    if agg > 1: c += 1
    if len(sql['select'][1]) > 1: c += 1
    if len(sql['where']) > 1: c += 1
    if len(sql['groupBy']) > 1: c += 1
    return c


def _eval_hardness(sql) -> str:
    c1 = _count_component1(sql)
    c2 = _count_component2(sql)
    co = _count_others(sql)
    if c1 <= 1 and co == 0 and c2 == 0:
        return "easy"
    if (co <= 2 and c1 <= 1 and c2 == 0) or (c1 <= 2 and co < 2 and c2 == 0):
        return "medium"
    if ((co > 2 and c1 <= 2 and c2 == 0) or
            (2 < c1 <= 3 and co <= 2 and c2 == 0) or
            (c1 <= 1 and co == 0 and c2 <= 1)):
        return "hard"
    return "extra"


def _load_hardness(dev_path: Path, n: int) -> list[str]:
    data = json.load(open(dev_path))
    return [_eval_hardness(item["sql"]) for item in data[:n]]


def main() -> None:
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/spider.yaml"
    n_val = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    split = sys.argv[3] if len(sys.argv) > 3 else "test"
    adapter_dir = sys.argv[4] if len(sys.argv) > 4 else None
    cfg = TrainerConfig.from_yaml(cfg_path)

    split_file = {
        "train": "train_spider.json",
        "validation": "dev.json",
        "dev": "dev.json",
        "test": "test.json",
    }[split]
    json_path = Path(cfg.environment.data_dir) / split_file
    hardness = _load_hardness(json_path, n_val)
    bucket_n: dict[str, int] = {h: 0 for h in HARDNESS_ORDER}
    for h in hardness:
        bucket_n[h] = bucket_n.get(h, 0) + 1
    print(f"[split] {split} ({split_file})")
    print(f"[bucket sizes among first {n_val}]")
    for h in HARDNESS_ORDER:
        print(f"  {h:<6} {bucket_n[h]}")

    print(f"\n[setup] loading {cfg.model.path}", flush=True)
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
    # Greedy decoding for baseline: deterministic + reproducible. Spider
    # leaderboard reports single-shot greedy, so this matches convention.
    policy.temperature = 0.0
    policy.top_p = 1.0

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
    print(f"\n[eval] running {n_val} episodes on split={split} "
          f"(temp={policy.temperature}, greedy) …", flush=True)

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
    bucket_w: dict[str, int] = {h: 0 for h in HARDNESS_ORDER}
    bucket_steps: dict[str, int] = {h: 0 for h in HARDNESS_ORDER}
    for i, t in enumerate(trajectories):
        info = t.steps[-1].info if t.steps else {}
        h = hardness[i] if i < len(hardness) else "easy"
        won = bool(info.get("won"))
        if won:
            matches += 1
            bucket_w[h] = bucket_w.get(h, 0) + 1
        if t.succeeded:
            answered += 1
        steps_total += t.total_steps
        bucket_steps[h] = bucket_steps.get(h, 0) + t.total_steps

    print()
    print(f"[result]  n={n}   wall={elapsed:.1f}s   {elapsed/n:.1f}s/episode")
    print(f"          EX (overall)    = {matches/n:.3f}  ({matches}/{n})")
    print(f"          answered        = {answered/n:.3f}  ({answered}/{n})")
    print(f"          avg steps       = {steps_total/n:.2f}")
    print()
    print(f"[per-hardness EX]")
    for h in HARDNESS_ORDER:
        bn = bucket_n[h]
        if bn == 0:
            continue
        bw = bucket_w[h]
        bs = bucket_steps[h] / bn
        print(f"  {h:<6} {bw:>3}/{bn:<3}  ({bw/bn*100:5.1f}%)   "
              f"avg_steps={bs:.2f}")


if __name__ == "__main__":
    main()
