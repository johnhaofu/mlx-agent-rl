"""Correctness check: ref_lps cached path matches the non-cached one.

We build a tiny synthetic trajectory (3 steps, monotonically extending
prompts) and run the ref-log-probs computation two ways:

  A. The original ``_compute_log_probs_mx`` per sample (full forward each).
  B. The new per-trajectory cached path with ``_compute_log_probs_mx_cached``.

Each sample's output must match (allclose at fp16 precision). If they
don't, the LCP/trim/feed math has a bug.

Usage:
    uv run python scripts/verify_ref_lps_cache.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import mlx.core as mx
from mlx_lm.models.cache import (
    can_trim_prompt_cache,
    make_prompt_cache,
    trim_prompt_cache,
)

from mlx_agent_rl.core.policy import Policy
from mlx_agent_rl.core.trainer import (
    _compute_log_probs_mx,
    _compute_log_probs_mx_cached,
    TrainerConfig,
)

sys.path.insert(0, str(Path(__file__).parent))
import bench_prompt_cache as bp  # noqa: E402


def _build_synthetic_trajectory(tokenizer, n_steps: int) -> List[tuple]:
    """Return a list of (prompt_tokens, action_tokens) where the prompt at
    step k is a strict extension of step k-1's prompt (mirrors what real
    rollouts produce within an episode)."""
    samples: List[tuple] = []
    for k in range(1, n_steps + 1):
        prompt_toks = bp.build_turn_prompt(tokenizer, k)
        # Synthetic short action: tokens for "<action>click[xyz]</action>"
        action_text = f"<action>click[step{k}]</action>"
        action_toks = list(tokenizer.encode(action_text))
        samples.append((prompt_toks, action_toks))
    return samples


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

    n_steps = 4
    samples = _build_synthetic_trajectory(policy.tokenizer, n_steps)
    print(f"[setup] {n_steps} synthetic samples", flush=True)
    for i, (p, a) in enumerate(samples):
        print(f"  step {i+1}: prompt={len(p)} toks, action={len(a)} toks", flush=True)

    # --- Path A: per-sample fresh forward ---
    print("\n[A] per-sample fresh forward …", flush=True)
    a_results: List[mx.array] = []
    with policy.reference():
        for prompt_toks, action_toks in samples:
            rl = _compute_log_probs_mx(policy.model, prompt_toks, action_toks)
            mx.eval(rl)
            a_results.append(mx.stop_gradient(rl))
    for i, r in enumerate(a_results):
        print(f"  step {i+1}: log_probs[:3] = {r[:3].tolist()}", flush=True)

    # --- Path B: per-trajectory cached forward ---
    print("\n[B] per-trajectory cached forward …", flush=True)
    b_results: List[mx.array] = []
    cache = make_prompt_cache(policy.model)
    assert can_trim_prompt_cache(cache)
    cache_tokens: List[int] = []
    with policy.reference():
        for prompt_toks, action_toks in samples:
            lcp = 0
            m = min(len(cache_tokens), len(prompt_toks))
            while lcp < m and cache_tokens[lcp] == prompt_toks[lcp]:
                lcp += 1
            excess = len(cache_tokens) - lcp
            if excess > 0:
                trim_prompt_cache(cache, excess)
                cache_tokens = cache_tokens[:lcp]
            rl = _compute_log_probs_mx_cached(
                policy.model, prompt_toks, action_toks,
                prompt_cache=cache, cache_offset=lcp,
            )
            mx.eval(rl)
            b_results.append(mx.stop_gradient(rl))
            cache_tokens = list(prompt_toks) + list(action_toks[:-1])
    for i, r in enumerate(b_results):
        print(f"  step {i+1}: log_probs[:3] = {r[:3].tolist()}", flush=True)

    # --- Compare ---
    print("\n[compare]")
    n_match = 0
    for i, (a, b) in enumerate(zip(a_results, b_results)):
        diff = mx.abs(a - b).max().item()
        ok = diff < 1e-3  # bf16 / fp16 tolerance
        n_match += int(ok)
        print(f"  step {i+1}: max|Δ| = {diff:.6f}  {'MATCH' if ok else 'MISMATCH'}",
              flush=True)
    print(f"\n[result] {n_match}/{n_steps} samples match", flush=True)
    if n_match != n_steps:
        raise SystemExit("ref_lps cache produces different log_probs than fresh forward")


if __name__ == "__main__":
    main()
