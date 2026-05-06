"""End-to-end episode rollout bench: cache on vs cache off.

We simulate an 8-turn WebShop episode by injecting fake observations
(the sidecar isn't required for the speed comparison). Each path runs
the same prompt sequence; only the cache strategy differs.

Usage:
    uv run python scripts/bench_episode_rollout.py
"""

from __future__ import annotations

import time
from typing import List

from mlx_lm.models.cache import (
    can_trim_prompt_cache,
    make_prompt_cache,
    trim_prompt_cache,
)

from mlx_agent_rl.core.policy import Policy
from mlx_agent_rl.core.trainer import TrainerConfig

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import bench_prompt_cache as bp  # noqa: E402


def run_episode_no_cache(policy, n_turns: int, max_tokens: int) -> tuple[float, list[str]]:
    """Fresh prefill each turn — what we did before this PR."""
    outputs: List[str] = []
    t0 = time.perf_counter()
    for t in range(1, n_turns + 1):
        toks = bp.build_turn_prompt(policy.tokenizer, t)
        prompt_text = policy.tokenizer.decode(toks)
        out, _, _ = policy.generate_with_log_probs(
            prompt_text, max_tokens=max_tokens, stop_strings=["</action>"]
        )
        outputs.append(out)
    return time.perf_counter() - t0, outputs


def run_episode_cached(policy, n_turns: int, max_tokens: int) -> tuple[float, list[str]]:
    """Per-episode shared cache + LCP delta-feed (this PR)."""
    cache = make_prompt_cache(policy.model)
    assert can_trim_prompt_cache(cache)
    cache_tokens: List[int] = []
    outputs: List[str] = []
    t0 = time.perf_counter()
    for t in range(1, n_turns + 1):
        toks = bp.build_turn_prompt(policy.tokenizer, t)
        prompt_text = policy.tokenizer.decode(toks)
        prompt_tokens = list(policy.tokenizer.encode(prompt_text))
        lcp = 0
        m = min(len(cache_tokens), len(prompt_tokens))
        while lcp < m and cache_tokens[lcp] == prompt_tokens[lcp]:
            lcp += 1
        excess = len(cache_tokens) - lcp
        if excess > 0:
            trim_prompt_cache(cache, excess)
            cache_tokens = cache_tokens[:lcp]
        delta = prompt_tokens[lcp:]
        if delta:
            out, _, action_tokens = policy.generate_with_log_probs(
                prompt_text, max_tokens=max_tokens, stop_strings=["</action>"],
                prompt_cache=cache, delta_tokens=delta,
            )
        else:
            out, _, action_tokens = policy.generate_with_log_probs(
                prompt_text, max_tokens=max_tokens, stop_strings=["</action>"],
                prompt_cache=cache,
            )
        cache_tokens = list(prompt_tokens) + list(action_tokens)
        outputs.append(out)
    return time.perf_counter() - t0, outputs


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

    n_turns = 8
    max_tokens = 96

    # Warmup
    print("[warmup]", flush=True)
    _ = run_episode_no_cache(policy, n_turns=2, max_tokens=16)

    # No-cache run
    print("\n[no-cache] running 8-turn episode …", flush=True)
    t_nc, out_nc = run_episode_no_cache(policy, n_turns, max_tokens)
    print(f"[no-cache] total = {t_nc:.2f}s   per-turn avg = {t_nc / n_turns:.2f}s", flush=True)
    for i, o in enumerate(out_nc):
        print(f"  turn {i+1}: {o[:60]!r}", flush=True)

    # Cached run
    print("\n[cached] running 8-turn episode …", flush=True)
    t_c, out_c = run_episode_cached(policy, n_turns, max_tokens)
    print(f"[cached] total = {t_c:.2f}s   per-turn avg = {t_c / n_turns:.2f}s", flush=True)
    for i, o in enumerate(out_c):
        print(f"  turn {i+1}: {o[:60]!r}", flush=True)

    print(f"\n[speedup] {t_nc / t_c:.2f}x ({t_nc:.1f}s → {t_c:.1f}s)")


if __name__ == "__main__":
    main()
