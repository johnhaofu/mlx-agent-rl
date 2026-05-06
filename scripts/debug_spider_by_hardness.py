"""Per-hardness rollout traces on Spider dev (4 each: easy/medium/hard/extra).

Computes hardness with Spider's official evaluator (data/spider/eval_scripts),
which reads the pre-parsed ``sql`` AST already in dev.json — no re-parsing.

Usage:
    uv run python scripts/debug_spider_by_hardness.py [k_per_bucket=4]
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

from mlx_agent_rl.core.policy import Policy
from mlx_agent_rl.core.trainer import TrainerConfig
from mlx_agent_rl.environments.sql_agent import SQLAgentEnvironment


HARDNESS_ORDER = ["easy", "medium", "hard", "extra"]


# Inlined from data/spider/eval_scripts/evaluation.py to avoid the nltk
# dependency that process_sql.py drags in. We only need eval_hardness,
# which operates on dev.json's already-parsed ``sql`` AST.
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


def _build_chat_prompt(tokenizer, system_prompt, question, obs, history,
                       enable_thinking, tools_schema):
    messages = [{"role": "system", "content": system_prompt}]
    if not history and obs and obs != question:
        user_content = f"{question}\n\n{obs}"
    else:
        user_content = question
    messages.append({"role": "user", "content": user_content})
    for resp_obs, model_output in history:
        messages.append({"role": "assistant", "content": model_output})
        messages.append({"role": "tool", "content": resp_obs})
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if tools_schema is not None:
        kwargs["tools"] = tools_schema
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def _bucket_indices(dev_path: Path, k: int) -> dict[str, list[int]]:
    data = json.load(open(dev_path))
    buckets: dict[str, list[int]] = {h: [] for h in HARDNESS_ORDER}
    for i, item in enumerate(data):
        try:
            h = _eval_hardness(item["sql"])
        except Exception:
            continue
        if h in buckets and len(buckets[h]) < k:
            buckets[h].append(i)
        if all(len(v) >= k for v in buckets.values()):
            break
    return buckets


def main() -> None:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    cfg = TrainerConfig.from_yaml("configs/spider.yaml")

    dev_path = Path(cfg.environment.data_dir) / "dev.json"
    buckets = _bucket_indices(dev_path, k)
    for h in HARDNESS_ORDER:
        print(f"[bucket] {h:<6} -> {buckets[h]}", flush=True)

    print(f"\n[setup] loading {cfg.model.path}", flush=True)
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
        split="validation",
        schema_max_chars=cfg.environment.schema_max_chars,
        rows_per_query=cfg.environment.rows_per_query,
    )

    summary: dict[str, list[int]] = {h: [] for h in HARDNESS_ORDER}

    for hardness in HARDNESS_ORDER:
        for ep_idx, dev_idx in enumerate(buckets[hardness]):
            print(f"\n{'='*80}\n[{hardness.upper()} {ep_idx}/{k}] dev_idx={dev_idx}",
                  flush=True)
            obs = env.reset(prompt="", answer=dev_idx)
            question = env._current["question"]
            gold = env.gold_sql
            db_id = env.db_id
            print(f"[db_id] {db_id}", flush=True)
            print(f"[Q]     {question}", flush=True)
            print(f"[gold]  {gold}", flush=True)

            history: list[tuple[str, str]] = []
            won = 0
            for step in range(cfg.rollout.max_steps):
                prompt_text = _build_chat_prompt(
                    policy.tokenizer, cfg.rollout.system_prompt, question,
                    obs.text, history,
                    cfg.rollout.enable_thinking,
                    env.get_tools_schema() if hasattr(env, "get_tools_schema") else None,
                )
                out, _, _ = policy.generate_with_log_probs(
                    prompt_text,
                    max_tokens=cfg.rollout.max_tokens,
                    stop_strings=env.stop_strings or None,
                )
                action = env.extract_action(out)
                print(f"\n  --- step {step} ---", flush=True)
                print(f"  [raw_out] {textwrap.shorten(out, width=300)!r}", flush=True)
                print(f"  [parsed]  {action!r}", flush=True)
                if action is None:
                    print(f"  [abort]   could not parse action", flush=True)
                    break
                next_obs, reward, done = env.step(action)
                info = env.last_step_info
                print(f"  [reward]  {reward:.2f}  done={done}  info={info}", flush=True)
                print(f"  [obs]     {textwrap.shorten(next_obs.text, width=280)}", flush=True)
                if done:
                    won = 1 if info.get("won") else 0
                    break
                history.append((next_obs.text, out))
                obs = next_obs
            summary[hardness].append(won)

    print(f"\n{'='*80}\n[summary] EX by hardness:")
    for h in HARDNESS_ORDER:
        wins = sum(summary[h])
        n = len(summary[h])
        print(f"  {h:<6} {wins}/{n}  ({wins/n*100:.0f}%)")
    total_w = sum(sum(v) for v in summary.values())
    total_n = sum(len(v) for v in summary.values())
    print(f"  {'all':<6} {total_w}/{total_n}  ({total_w/total_n*100:.0f}%)")


if __name__ == "__main__":
    main()
