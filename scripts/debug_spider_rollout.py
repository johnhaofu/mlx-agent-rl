"""Print raw rollout traces on Spider dev — see what the model actually
emits per turn (full prompt tail / model output / parsed action / obs).

Useful for diagnosing format issues and seeing the SQL the agent writes.

Usage:
    uv run python scripts/debug_spider_rollout.py [n=4]
"""

from __future__ import annotations

import sys
import textwrap

from mlx_agent_rl.core.policy import Policy
from mlx_agent_rl.core.trainer import TrainerConfig
from mlx_agent_rl.environments.sql_agent import SQLAgentEnvironment
from mlx_agent_rl.memory.memory import SlidingMemory


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


def main() -> None:
    n_eps = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    cfg = TrainerConfig.from_yaml("configs/spider.yaml")
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
        split="validation",
        schema_max_chars=cfg.environment.schema_max_chars,
        rows_per_query=cfg.environment.rows_per_query,
    )

    for ep in range(n_eps):
        print(f"\n{'='*80}\nEPISODE {ep} (db_id, gold below)", flush=True)
        memory = SlidingMemory(window_size=cfg.memory.window_size)
        obs = env.reset(prompt="", answer=ep)
        question = env._current["question"]
        gold = env.gold_sql
        db_id = env.db_id
        print(f"[db_id] {db_id}", flush=True)
        print(f"[Q]     {question}", flush=True)
        print(f"[gold]  {gold}", flush=True)
        print(f"[obs0]  {textwrap.shorten(obs.text, width=600)}", flush=True)

        history: list[tuple[str, str]] = []
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
            print(f"  [raw_out] {textwrap.shorten(out, width=400)!r}", flush=True)
            print(f"  [parsed]  {action!r}", flush=True)
            if action is None:
                print(f"  [abort]   could not parse action", flush=True)
                break
            next_obs, reward, done = env.step(action)
            info = env.last_step_info
            print(f"  [reward]  {reward:.2f}  done={done}  info={info}", flush=True)
            print(f"  [obs]     {textwrap.shorten(next_obs.text, width=400)}", flush=True)
            if done:
                break
            history.append((next_obs.text, out))
            obs = next_obs


if __name__ == "__main__":
    main()
