"""Single-episode smoke test for the verl-style <action> webshop format.

Runs one rollout against the sidecar with the current trainer config and
prints prompt-snippet, raw model output, and parsed action per step. Aborts
after 5 steps regardless of done.

Usage:
    uv run python scripts/smoke_webshop_action.py
"""

from __future__ import annotations

import sys
import textwrap

from mlx_agent_rl.core.policy import Policy
from mlx_agent_rl.core.rollout import RolloutCollector
from mlx_agent_rl.core.trainer import TrainerConfig
from mlx_agent_rl.environments.webshop import WebShopEnvironment
from mlx_agent_rl.memory.memory import SlidingMemory


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/webshop.yaml"
    cfg = TrainerConfig.from_yaml(config_path)

    print("[setup] loading policy …", flush=True)
    policy = Policy(
        model_path=cfg.model.path,
        quantize=cfg.model.quantize,
        lora_rank=cfg.model.lora_rank,
        lora_layers=cfg.model.lora_layers,
    )

    env = WebShopEnvironment(
        base_url=cfg.environment.base_url or "http://192.168.0.117:3001",
        use_tools_schema=cfg.environment.use_tools_schema,
    )
    print(f"[setup] env tools_schema={env.get_tools_schema() is not None} "
          f"thinking={cfg.rollout.enable_thinking}", flush=True)

    memory = SlidingMemory(window_size=cfg.memory.window_size)
    collector = RolloutCollector(
        policy=policy,
        env=env,
        memory=memory,
        max_steps=5,
        max_tokens=cfg.rollout.max_tokens,
        invalid_action_penalty=cfg.environment.invalid_action_penalty,
        system_prompt=cfg.rollout.system_prompt,
        enable_thinking=cfg.rollout.enable_thinking,
    )

    # Patch _run_episode telemetry: monkey-patch generate_with_log_probs to
    # log each turn. Simpler: directly drive a single episode using the
    # collector's internal helper.
    policy.eval()
    policy.temperature = 0.4
    policy.top_p = 1.0

    # Use val goal idx 0 for determinism.
    obs = env.reset(prompt="", answer=0)
    print(f"\n[task] {env.instruction!r}", flush=True)
    print(f"[obs0] {textwrap.shorten(obs.text, width=400)}", flush=True)

    for step in range(5):
        prompt_text = collector._build_prompt(obs.text, question=env.instruction)
        out, _lps, _toks = policy.generate_with_log_probs(
            prompt_text, max_tokens=cfg.rollout.max_tokens
        )
        action = env.extract_action(out)
        print(f"\n--- step {step} ---", flush=True)
        print(f"[raw_out] {textwrap.shorten(out, width=600)}", flush=True)
        print(f"[parsed_action] {action!r}", flush=True)
        if action is None:
            print("[abort] could not parse action", flush=True)
            break
        next_obs, reward, done = env.step(action)
        info = env.last_step_info
        print(f"[reward] shaped={reward} task_score={info.get('task_score'):.3f} "
              f"won={info.get('won')} done={done}", flush=True)
        if done:
            print("[done]", flush=True)
            break
        memory.update(next_obs.text, out)
        obs = next_obs


if __name__ == "__main__":
    main()
