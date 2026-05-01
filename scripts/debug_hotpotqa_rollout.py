"""Inspect what the model actually emits during a HotpotQA rollout.

Runs N episodes step-by-step and prints each turn's prompt-tail, raw model
output, parsed action, and reward — so we can see whether the model
fails to emit answer[…] format, gets stuck searching, or hits some other
extract_action bug.

Usage:
    uv run python scripts/debug_hotpotqa_rollout.py [n=4]
"""

from __future__ import annotations

import sys
import textwrap

from mlx_agent_rl.core.policy import Policy
from mlx_agent_rl.core.trainer import TrainerConfig
from mlx_agent_rl.environments.hotpotqa import HotpotQAEnvironment
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
    cfg = TrainerConfig.from_yaml("configs/hotpotqa.yaml")
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

    for ep in range(n_eps):
        print(f"\n========== EPISODE {ep} (seed_idx={ep}) ==========", flush=True)
        memory = SlidingMemory(window_size=cfg.memory.window_size)
        obs = env.reset(prompt="", answer=ep)
        question = env.instruction
        print(f"[Q] {question}", flush=True)
        print(f"[obs0] {textwrap.shorten(obs.text, width=400)}", flush=True)

        history: list[tuple[str, str]] = []
        for step in range(cfg.rollout.max_steps):
            prompt_text = _build_chat_prompt(
                policy.tokenizer, cfg.rollout.system_prompt, question,
                obs.text, history,
                cfg.rollout.enable_thinking,
                env.get_tools_schema(),
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
            print(f"  [reward]  {reward:.3f}  done={done}  info={env.last_step_info}", flush=True)
            print(f"  [obs]     {textwrap.shorten(next_obs.text, width=300)}", flush=True)
            if done:
                break
            history.append((next_obs.text, out))
            obs = next_obs


if __name__ == "__main__":
    main()
