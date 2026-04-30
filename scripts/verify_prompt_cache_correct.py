"""Correctness check: cached rollout produces the same outputs as fresh prefill.

Greedy sampling makes the comparison exact. We synthesize an 8-turn
conversation (no real WebShop sidecar required) and run policy generation
in two modes:

  A. fresh KV cache built per turn (current default)
  B. one shared KV cache reused across turns with LCP-based delta-feed

Outputs (decoded text) at every turn must match. If they don't, the cache
state diverges from the fresh forward and we have a bug.

Usage:
    uv run python scripts/verify_prompt_cache_correct.py
"""

from __future__ import annotations

from typing import List
import sys
from pathlib import Path

from mlx_lm.models.cache import (
    can_trim_prompt_cache,
    make_prompt_cache,
    trim_prompt_cache,
)

from mlx_agent_rl.core.policy import Policy
from mlx_agent_rl.core.trainer import TrainerConfig

# Reuse the synthetic 8-turn convo from the bench script.
sys.path.insert(0, str(Path(__file__).parent))
import bench_prompt_cache as bp  # noqa: E402


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
    # Greedy: no temperature noise so the two paths must produce identical
    # tokens (same model, same logits, same argmax).
    policy.temperature = 0.0
    policy.top_p = 1.0

    n_turns = 6
    max_tokens = 24  # short — only need the action envelope
    prompts: List[str] = []
    for t in range(1, n_turns + 1):
        toks = bp.build_turn_prompt(policy.tokenizer, t)
        prompts.append(policy.tokenizer.decode(toks))

    # --- Path A: fresh prefill per turn ---
    print("\n[A] fresh prefill per turn …", flush=True)
    fresh_outputs: List[str] = []
    for p in prompts:
        out, _, _ = policy.generate_with_log_probs(
            p, max_tokens=max_tokens, stop_strings=["</action>"]
        )
        fresh_outputs.append(out)
        print(f"  out = {out!r}", flush=True)

    # --- Path B: shared cache + LCP delta-feed ---
    print("\n[B] cached rollout (LCP delta-feed) …", flush=True)
    cache = make_prompt_cache(policy.model)
    assert can_trim_prompt_cache(cache), "cache must be trimmable"
    cache_tokens: List[int] = []
    cached_outputs: List[str] = []
    for p in prompts:
        prompt_tokens = list(policy.tokenizer.encode(p))
        # LCP between cache_tokens and current prompt
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
                p, max_tokens=max_tokens, stop_strings=["</action>"],
                prompt_cache=cache, delta_tokens=delta,
            )
            cache_tokens = list(prompt_tokens) + list(action_tokens)
        else:
            out, _, action_tokens = policy.generate_with_log_probs(
                p, max_tokens=max_tokens, stop_strings=["</action>"],
                prompt_cache=cache,
            )
            cache_tokens = list(prompt_tokens) + list(action_tokens)
        cached_outputs.append(out)
        print(f"  out = {out!r}", flush=True)

    # --- Compare ---
    print("\n[compare]")
    n_match = 0
    for i, (a, b) in enumerate(zip(fresh_outputs, cached_outputs)):
        ok = a == b
        n_match += int(ok)
        print(f"  turn {i+1}: {'MATCH' if ok else 'MISMATCH'}", flush=True)
        if not ok:
            print(f"    fresh:  {a!r}")
            print(f"    cached: {b!r}")
    print(f"\n[result] {n_match}/{n_turns} turns match",
          flush=True)
    if n_match != n_turns:
        raise SystemExit("prompt cache produces different outputs than fresh prefill")


if __name__ == "__main__":
    main()
