"""Time RolloutCollector._run_episode against the live sidecar, with and
without cache. This isolates the *integration* cost rather than the raw
policy generation cost (which the standalone bench already measured at
3.16x).

Usage:
    uv run python scripts/bench_collector_episode.py
"""

from __future__ import annotations

import time

from mlx_agent_rl.core.policy import Policy
from mlx_agent_rl.core.rollout import RolloutCollector
from mlx_agent_rl.core.trainer import TrainerConfig
from mlx_agent_rl.environments.webshop import WebShopEnvironment
from mlx_agent_rl.memory.memory import SlidingMemory


def main() -> None:
    cfg = TrainerConfig.from_yaml("configs/webshop_smoke.yaml")
    print(f"[setup] loading {cfg.model.path}", flush=True)
    policy = Policy(
        model_path=cfg.model.path,
        quantize=cfg.model.quantize,
        lora_rank=cfg.model.lora_rank,
        lora_layers=cfg.model.lora_layers,
    )
    policy.eval()
    policy.temperature = 0.4
    policy.top_p = 1.0

    env = WebShopEnvironment(
        base_url=cfg.environment.base_url or "http://192.168.0.117:3001",
        use_tools_schema=cfg.environment.use_tools_schema,
    )

    # Warmup: a tiny prompt to compile MLX kernels
    _ = policy.generate_with_log_probs("hi", max_tokens=4)

    for use_cache in [False, True]:
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
            use_prompt_cache=use_cache,
        )

        label = "cached" if use_cache else "no-cache"
        print(f"\n[{label}] running 1 episode against sidecar (goal_idx=0) …",
              flush=True)
        t0 = time.perf_counter()
        traj = collector._run_episode(question="", answer=0, uid="bench")
        elapsed = time.perf_counter() - t0
        print(f"[{label}] total = {elapsed:.2f}s   "
              f"steps = {traj.total_steps}  "
              f"per-turn = {elapsed / max(traj.total_steps, 1):.2f}s  "
              f"reward = {traj.episode_reward}", flush=True)


if __name__ == "__main__":
    main()
