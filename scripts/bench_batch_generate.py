"""Benchmark mlx-lm's batch_generate vs sequential per-prompt generation.

Goal: measure the realistic wall-clock speedup we'd get by switching the
WebShop rollout path to mlx-lm's continuous-batching API.

We use representative WebShop-style prompts at the lengths the agent
actually sees during a rollout, with the live system prompt + admissible
actions block. Stop tokens encode ``</action>`` so each sample halts when
the action is emitted.

Run:
    uv run python scripts/bench_batch_generate.py
"""

from __future__ import annotations

import time
from typing import List

from mlx_lm.generate import BatchGenerator, batch_generate
from mlx_lm.sample_utils import make_sampler

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

INSTRUCTIONS = [
    "Find me double sided, machine washable decorative pillows with size 28x28 under 40 dollars",
    "I need women's pumps with closed toe, rubber sole, color nude, size 10, under 70 dollars",
    "Show me a blue cotton t-shirt for men, size large, under 25 dollars",
    "Find vegan leather wallet for women, with multiple card slots, under 35 dollars",
    "I want a stainless steel water bottle, 32oz, with straw lid, under 30 dollars",
    "Get me a wooden picture frame, 8x10 inches, rustic style, under 20 dollars",
    "Find ergonomic office chair with lumbar support, mesh back, under 200 dollars",
    "Show me bluetooth headphones with noise cancellation, over-ear, under 100 dollars",
    "I want a yoga mat, 6mm thick, non-slip, blue color, under 30 dollars",
    "Find me a cast iron skillet, 12 inch, pre-seasoned, under 50 dollars",
    "Show me an electric toothbrush with multiple brush heads, under 80 dollars",
    "Get me a backpack for hiking, 40 liter capacity, water resistant, under 100 dollars",
    "Find vegan leather laptop sleeve for 15-inch laptop, gray color, under 35 dollars",
    "Show me running shoes for men, size 10, with arch support, under 90 dollars",
    "I need a bamboo cutting board, large size, juice groove, under 25 dollars",
    "Find a desk lamp with adjustable arm, USB charging port, under 45 dollars",
]

# A representative search-results page with 10 products + nav buttons.
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

ADMISSIBLE_ACTIONS = (
    "Admissible actions: [click[back to search], click[next >], "
    "click[b07xtk3p89], click[b08qvdnjc7], click[b09kp78g37], click[b0928yqxr4], "
    "click[b07gj9nm4b], click[b08yhc4n9v], click[b07vpkl3hr], click[b0br7m82wx], "
    "click[b09f8n3kqm], click[b08lrn5vbt]]"
)


def build_prompts(tokenizer, n: int, *, turn: int = 3) -> List[str]:
    """Realistic rollout prompts at ``turn`` depth (more turns -> longer ctx).

    We simulate ``turn`` previous (assistant action, tool obs) pairs so the
    chat history resembles what the model sees at mid-episode. The default
    turn=3 puts us at ~1200 tokens, similar to a typical WebShop step.
    """
    prompts: List[str] = []
    for inst in INSTRUCTIONS[:n]:
        first_obs = (
            f"WebShop [SEP] Instruction: [SEP] {inst} [SEP] Search\n"
            "Admissible actions: [search[query], click[search]]"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{inst}\n\n{first_obs}"},
        ]
        # Fake history: action 1 = search, then action 2 = click ASIN, then click back.
        # Each tool message is a search-results page (~1000 tokens worth).
        history = [
            ("<action>search[" + inst.replace(",", "") + "]</action>",
             RESULTS_OBS.format(inst=inst) + "\n\n" + ADMISSIBLE_ACTIONS),
            ("<action>click[b07xtk3p89]</action>",
             RESULTS_OBS.format(inst=inst) + "\n\n" + ADMISSIBLE_ACTIONS),
            ("<action>click[back to search]</action>",
             RESULTS_OBS.format(inst=inst) + "\n\n" + ADMISSIBLE_ACTIONS),
        ][:turn]
        for assistant_out, tool_obs in history:
            messages.append({"role": "assistant", "content": assistant_out})
            messages.append({"role": "tool", "content": tool_obs})
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(text)
    return prompts


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

    import sys as _sys
    n_prompts = int(_sys.argv[1]) if len(_sys.argv) > 1 else 16
    turn = int(_sys.argv[2]) if len(_sys.argv) > 2 else 3
    max_tokens = 96
    prompts = build_prompts(policy.tokenizer, n_prompts, turn=turn)
    print(f"[setup] {n_prompts} prompts, max_tokens={max_tokens}, history_turn={turn}", flush=True)
    print(f"[setup] avg prompt tokens = "
          f"{sum(len(policy.tokenizer.encode(p)) for p in prompts) // n_prompts}",
          flush=True)

    sampler = make_sampler(temp=policy.temperature, top_p=policy.top_p)
    stop_tokens = []
    for s in ["</action>", "</tool_call>"]:
        toks = policy.tokenizer.encode(s, add_special_tokens=False)
        stop_tokens.append(list(toks))
    print(f"[setup] stop_tokens (encoded): {stop_tokens}", flush=True)

    # Warmup — prime MLX kernel cache so first measurements aren't polluted
    print("\n[warmup] one prompt to compile kernels …", flush=True)
    _ = policy.generate_with_log_probs(
        prompts[0], max_tokens=16, stop_strings=["</action>"],
    )

    # ----- Path A: sequential -----
    print("\n[seq] running sequentially …", flush=True)
    t0 = time.perf_counter()
    seq_outputs: List[str] = []
    for p in prompts:
        out, _lps, _toks = policy.generate_with_log_probs(
            p, max_tokens=max_tokens,
            stop_strings=["</action>", "</tool_call>"],
        )
        seq_outputs.append(out)
    seq_time = time.perf_counter() - t0
    print(f"[seq] total = {seq_time:.2f}s   "
          f"avg per prompt = {seq_time / n_prompts:.2f}s", flush=True)

    # ----- Path B: BatchGenerator with custom stop_tokens -----
    # Note: batch_generate() forces stop_tokens from EOS only and won't let
    # us add </action>. So we drive BatchGenerator directly to pass our
    # action-envelope stop sequences alongside EOS.
    print("\n[batch] running BatchGenerator with custom stop_tokens …", flush=True)
    tokenized = [list(policy.tokenizer.encode(p)) for p in prompts]
    eos_seqs = [[t] for t in policy.tokenizer.eos_token_ids]
    all_stop = eos_seqs + stop_tokens
    t0 = time.perf_counter()
    gen = BatchGenerator(
        policy.model,
        stop_tokens=all_stop,
        sampler=sampler,
        prefill_batch_size=min(n_prompts, 8),
        completion_batch_size=n_prompts,
    )
    uids = gen.insert(tokenized, [max_tokens] * len(tokenized))
    results: dict = {uid: [] for uid in uids}
    with gen.stats() as stats:
        while responses := gen.next_generated():
            for r in responses:
                if r.finish_reason != "stop":
                    results[r.uid].append(r.token)
    gen.close()
    batch_outputs = [policy.tokenizer.decode(results[uid]) for uid in uids]
    batch_time = time.perf_counter() - t0
    print(f"[batch] total = {batch_time:.2f}s   "
          f"avg per prompt = {batch_time / n_prompts:.2f}s", flush=True)
    print(f"[batch] stats: prompt_tps={stats.prompt_tps:.1f}  "
          f"gen_tps={stats.generation_tps:.1f}  "
          f"peak_mem={stats.peak_memory:.2f}GB", flush=True)

    # ----- Compare outputs -----
    print(f"\n[speedup] {seq_time / batch_time:.2f}x")
    print("\n[outputs] (first 80 chars per prompt)")
    for i, (s, b) in enumerate(zip(seq_outputs, batch_outputs)):
        print(f"  #{i} seq   = {s[:80]!r}")
        print(f"  #{i} batch = {b[:80]!r}")


if __name__ == "__main__":
    main()
