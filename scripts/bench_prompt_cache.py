"""Benchmark prompt-cache reuse across turns of one WebShop episode.

We don't run the actual env — we synthesize an 8-turn conversation and
compare two approaches for processing each turn's prompt:

  A. Fresh prefill every turn (current behavior)
  B. Reuse the cache from the prior turn; only prefill the delta tokens

The delta is computed by finding the longest-common-prefix of consecutive
turns' tokenized prompts, then feeding only the new tail to the model
with the prior cache attached.

If the chat template produces a strict extension across turns (it should,
since Qwen3's template just appends new <im_start>...<im_end> blocks),
LCP-based reuse is exact and lossless.

Run:
    uv run python scripts/bench_prompt_cache.py
"""

from __future__ import annotations

import time
from typing import List

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from mlx_agent_rl.core.policy import Policy
from mlx_agent_rl.core.trainer import TrainerConfig


SYSTEM_PROMPT = (
    "You are an autonomous shopping agent in the WebShop e-commerce environment. "
    "Find and buy the product that best matches the instruction.\n\n"
    "Each turn you receive the current page text plus a list of admissible actions. "
    "Pick exactly one and reply with ONLY this format:\n\n"
    "<action>search[free-text query]</action>      (when search[query] is admissible)\n"
    "<action>click[clickable label or asin]</action>  (any other admissible click)\n\n"
    "No prose, no commentary, no JSON. Just one <action>...</action> tag.\n"
    "ASCII only inside <action>."
)

INSTRUCTION = (
    "Find me double sided, machine washable decorative pillows with size 28x28 "
    "under 40 dollars"
)

RESULTS_OBS = (
    "Instruction: [SEP] {inst} [SEP] "
    "Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] "
    "B07XTK3P89 [SEP] Premium Decorative Pillows 28x28 Machine Washable Double Sided Print [SEP] $34.99 [SEP] "
    "B08QVDNJC7 [SEP] Throw Pillows Square Cotton Decorative Cushion Cover Set [SEP] $29.95 [SEP] "
    "B09KP78G37 [SEP] Modern Geometric Throw Pillow Cover Linen Blend [SEP] $42.50 [SEP] "
    "B0928YQXR4 [SEP] Velvet Decorative Pillow Cover with Tassels Premium Quality [SEP] $38.00 [SEP] "
    "B07GJ9NM4B [SEP] Polyester Throw Pillow 26x26 Reversible Print Hypoallergenic [SEP] $24.99 [SEP] "
    "B08YHC4N9V [SEP] Boho Style Decorative Pillow Cover Linen Cotton Blend [SEP] $32.75 [SEP] "
    "B07VPKL3HR [SEP] Minimalist Decorative Pillow Cover Set of 2 Soft Velvet [SEP] $45.99 [SEP] "
    "B0BR7M82WX [SEP] Square Decorative Pillow Cover with Hidden Zipper Easy Wash [SEP] $19.99 [SEP] "
    "B09F8N3KQM [SEP] Premium Decorative Pillows Indoor Outdoor Use Fade Resistant [SEP] $54.00 [SEP] "
    "B08LRN5VBT [SEP] Throw Pillow Cushion Set 4-Pack with Insert Soft Polyester [SEP] $39.95"
)
ADMISSIBLE = (
    "Admissible actions: [click[back to search], click[next >], "
    "click[b07xtk3p89], click[b08qvdnjc7], click[b09kp78g37], click[b0928yqxr4], "
    "click[b07gj9nm4b], click[b08yhc4n9v], click[b07vpkl3hr], click[b0br7m82wx], "
    "click[b09f8n3kqm], click[b08lrn5vbt]]"
)

# Synthetic 8-turn trajectory: alternating search / click actions.
ACTIONS = [
    "<action>search[double sided decorative pillows 28x28 under 40 dollars]</action>",
    "<action>click[b07xtk3p89]</action>",
    "<action>click[back to search]</action>",
    "<action>click[b08qvdnjc7]</action>",
    "<action>click[< prev]</action>",
    "<action>click[b09kp78g37]</action>",
    "<action>click[back to search]</action>",
    "<action>click[b07gj9nm4b]</action>",
]


def build_turn_prompt(tokenizer, turn: int) -> List[int]:
    """Render the chat history for ``turn`` (1-indexed). Returns token ids."""
    first_obs = (
        f"WebShop [SEP] Instruction: [SEP] {INSTRUCTION} [SEP] Search\n"
        "Admissible actions: [search[query], click[search]]"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{INSTRUCTION}\n\n{first_obs}"},
    ]
    for i in range(turn - 1):
        messages.append({"role": "assistant", "content": ACTIONS[i]})
        messages.append({
            "role": "tool",
            "content": RESULTS_OBS.format(inst=INSTRUCTION) + "\n\n" + ADMISSIBLE,
        })
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    return list(tokenizer.encode(text))


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

    n_turns = 8
    turns_tokens: List[List[int]] = [
        build_turn_prompt(policy.tokenizer, t + 1) for t in range(n_turns)
    ]
    print("[setup] tokens per turn:", [len(t) for t in turns_tokens], flush=True)

    # Verify chat template produces strict-extension across turns: each turn's
    # tokenized prompt should start with the prior turn's tokenized prompt up
    # to (and including) the prior assistant's <|im_start|>assistant\n marker.
    # (We sanity-check that the LCP is large enough for cache reuse to matter.)
    lcps: List[int] = []
    for i in range(1, n_turns):
        a, b = turns_tokens[i - 1], turns_tokens[i]
        j = 0
        while j < min(len(a), len(b)) and a[j] == b[j]:
            j += 1
        lcps.append(j)
    print("[setup] LCP(turn N-1, turn N):", lcps, flush=True)
    print("[setup] delta tokens per turn:", [
        len(turns_tokens[i]) - lcps[i - 1] for i in range(1, n_turns)
    ], flush=True)

    # Warmup
    print("\n[warmup]", flush=True)
    cache = make_prompt_cache(policy.model)
    _ = policy.model(mx.array(turns_tokens[0])[None], cache=cache)
    mx.eval(_)

    # ----- Approach A: fresh prefill every turn -----
    print("\n[A] fresh prefill every turn …", flush=True)
    t0 = time.perf_counter()
    total_a_tokens = 0
    for toks in turns_tokens:
        cache = make_prompt_cache(policy.model)
        logits = policy.model(mx.array(toks)[None], cache=cache)
        mx.eval(logits)
        total_a_tokens += len(toks)
    a_time = time.perf_counter() - t0
    print(f"[A] total = {a_time:.2f}s   prefill_tokens = {total_a_tokens}", flush=True)

    # ----- Approach B: incremental prefill via shared cache -----
    print("\n[B] incremental prefill via cache …", flush=True)
    t0 = time.perf_counter()
    cache = make_prompt_cache(policy.model)
    fed = 0
    total_b_tokens = 0
    for i, toks in enumerate(turns_tokens):
        # Feed only the delta past what cache already covers.
        delta = toks[fed:]
        if len(delta) == 0:
            # Same prompt as prior turn — nothing to feed (shouldn't happen).
            continue
        logits = policy.model(mx.array(delta)[None], cache=cache)
        mx.eval(logits)
        fed += len(delta)
        total_b_tokens += len(delta)
    b_time = time.perf_counter() - t0
    print(f"[B] total = {b_time:.2f}s   prefill_tokens = {total_b_tokens}", flush=True)

    print(f"\n[speedup] {a_time / b_time:.2f}x")
    print(f"[token-savings] {total_a_tokens} → {total_b_tokens}  "
          f"({100 * (1 - total_b_tokens / total_a_tokens):.1f}% fewer prefill tokens)")


if __name__ == "__main__":
    main()
