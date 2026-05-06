# Training a Spider SQL agent with GiGPO on Apple Silicon

**Author:** Junhao Fu
**Date:** 2026-05-06
**Stack:** Qwen3-4B (4-bit MLX) · GiGPO · single M-series Mac

Over four days I ran 9 RL training experiments on Yale Spider, with a strict
goal: see how far a 4B-parameter open-weight model could go on text-to-SQL
when trained on a single Mac, and figure out which knobs actually matter.

The headline number is **+7.6pp** on Spider dev (44.0% → 51.6% execution
accuracy, n=1034). The more interesting result: the recipe that won is the
opposite of what the conventional wisdom suggested.

## TL;DR

- Partial-credit reward (0.1 for valid-but-wrong SQL) **hurts** vs strict
  binary 1.0/0.0. -2.4pp on test, -9pp on medium queries. Agents settle for
  syntactically valid mediocrity.
- KL anchor at 0.001 doesn't lift overall EX, it **redistributes**: protects
  easy/extra at the cost of medium/hard.
- Scaling training data from n=300 to n=400-600 **regressed** at fixed
  compute budget. More data without more updates dilutes the signal.
- 87% of GiGPO groups had **zero reward variance** (8 trajectories per
  prompt collapsing to identical outcomes). The hyperparameter knobs we
  tried address symptoms, not the root cause.
- A **dead config** in the trainer was silently coercing `epsilon_high`
  back to `epsilon` for every algorithm except DAPO. Six asymmetric-clip
  experiments that weren't actually asymmetric.

## 1. Setup

The task is multi-turn text-to-SQL on Yale's
[Spider 1.0](https://yale-lily.github.io/spider). Each episode the agent
sees a SQLite schema and a natural-language question; it can issue
read-only `sql[…]` queries (returning up to 10 rows) and finally commits
with `answer[…]`. Reward is **execution accuracy** — does the predicted
query produce the same result-set as the gold.

Base model: `Qwen3-4B-MLX-4bit` running locally on M-series silicon.
LoRA rank 8, last 4 transformer layers (~10M trainable params, 0.25% of
the 4B base) for v2-v6 and v8-v9b; rank 16, layers 8 (~40M, 1%) for v7.

Algorithm: GiGPO with mean-norm advantages, lr 5e-5, no KL anchor.
Each experiment trains 2 epochs on 300 questions (76 batches × group_size 8
= 4800 trajectories), runs ~10 hours wall-time, saves the best/ checkpoint
based on held-out val won_rate, evaluated greedy on dev (n=1034) and a test
sample (n=128).

## 2. Results across all runs

### Test n=128 greedy (small sample, run-to-run ranking)

| Run     | Config delta                         | Test EX   | Δ vs baseline |
|---------|--------------------------------------|-----------|---------------|
| baseline| no training                          | 50.0%     | —             |
| v2      | partial credit 0.1                   | 53.1%     | +3.1          |
| v3      | v2 + n_train=600 (stopped)           | 52.3%     | +2.3          |
| **v4 ★**| **partial=0 (binary)**               | **55.5%** | **+5.5**      |
| v5      | v4 + kl=0.001                        | 53.1%     | +3.1          |
| v6      | v4 + n_train=400                     | 51.6%     | +1.6          |
| v7      | v4 + lora 2× (rank 16, layers 8)     | 57.8%     | +7.8          |
| v8      | v4 + asym clip 0.4 + dyn-sample      | (stopped) | val flat      |
| v9b     | v4 + entropy_coef 0.001              | 52.3%     | +2.3          |

The headline test-n=128 winner is v7 (57.8%). But test n=128 has ±2-3pp
run-to-run variance, which we discovered after running both v4 and v7 on
the full dev set (n=1034).

### Dev n=1034 greedy (the reliable estimate)

| Bucket          | Baseline | v4 ★      | v7        |
|-----------------|----------|-----------|-----------|
| Overall         | 44.0%    | **51.6%** | 50.5%     |
| easy (248)      | 60.1%    | 70.2%     | 69.0%     |
| medium (446)    | 50.0%    | **61.7%** | 56.1%     |
| hard (174)      | 31.0%    | 28.7%     | **33.9%** |
| extra (166)     | 17.5%    | 21.1%     | **25.3%** |
| answered        | 90.3%    | **98.6%** | 92.9%     |
| avg_steps       | 2.54     | **1.97**  | 2.47      |

On the larger sample, v4 wins overall (+7.6pp vs baseline) while v7
trades medium for hard/extra. Test n=128 was misleading.

**Takeaway:** always confirm RL deltas on n≥1000 before claiming a winner.
Two-three pp is well inside the noise floor of an n=128 estimate.

## 3. Three findings that surprised me

### 3.1 Partial credit reward hurts

I started with what looked like the cleaner reward design: 1.0 for
execution match, 0.1 for "valid SQL but different result", 0.0 for syntax
error. The intuition: GiGPO's group-relative advantage needs variance, so
3-tier reward gives more signal than binary.

The result was the opposite. v4 (binary 1.0/0.0) beats v2 (with 0.1
partial credit) by **+2.4pp on test**, with the biggest gain on medium
queries (+9.3pp vs v2's +4.7pp).

**Why?** The 0.1 reward distinguishes "valid SQL" from "garbage" but isn't
strong enough to push toward "actually correct". The agent learns to
*settle* for queries that parse but miss the right column — they earn a
guaranteed 0.1 instead of risking the 0 from a syntax error.

The behavioral evidence is striking. avg_steps drops from 2.49 (v2) to
**2.08 (v4)** — a 16% reduction. answered rate (% of episodes that ever
called `answer[…]`) jumps from 90.6% to **96.1%**. With binary reward, the
agent commits faster and harder.

### 3.2 KL anchor redistributes, doesn't lift

v5 = v4 + kl_coef=0.001. I expected the anchor to stabilize v4's volatile
val curve and push the best/ checkpoint higher.

It did stabilize. But it also *capped* the upside: v5 ended at the same
overall test EX as v2 (53.1%). The per-bucket breakdown shows what
actually happened.

| Bucket  | v4 (no KL)    | v5 (kl=0.001) | Δ      |
|---------|---------------|---------------|--------|
| easy    | 66.7%         | 70.0% ★       | +3.3   |
| medium  | 74.4% ★       | 65.1%         | -9.3   |
| hard    | 37.5% ★       | 34.4%         | -3.1   |
| extra   | 30.4%         | 34.8% ★       | +4.4   |

**Why?** KL anchor controls how far the policy can drift from the base
model. Drift = ability to learn new behaviors. v5 gives up the medium/hard
gains v4 found via aggressive drift, in exchange for preserving the base
model's strengths on easy/extra. Net change: zero overall.

The deeper mechanism is *RL's Razor* (Shenfeld et al. 2026): on-policy RL
already implicitly minimizes reverse-KL to the base policy, biasing toward
high-reward solutions that stay close in KL. Adding an *explicit* KL term
on top stacks two regularizers. Empirically the authors show forgetting
correlates R²=0.96 with KL drift, so the implicit version is plenty.
Stacking explicit KL just clamps the drift further, which protects
already-strong buckets (easy/extra) but suppresses useful exploration on
the buckets where RL is finding new behavior (medium/hard).

For tasks where you want raw EX, skip the KL anchor — RL's Razor handles
the regularization for free. For tasks where you specifically need to
protect base behavior on a subset, KL is a knob.

### 3.3 Scaling up training data doesn't help (at fixed compute)

v3 (n_train 300 → 600) and v6 (300 → 400) both *underperformed* v4.
Same hyperparameters, just more questions. Final test EX was 52.3% and
51.6%, vs v4's 55.5%.

**What's going on?** Holding total update steps roughly constant, doubling
data means each unique question gets fewer gradient updates. With sparse
binary reward, fewer updates per question = fewer chances to lock in a
useful policy. The result is a thinner policy spread over a wider
distribution. Going wider needs to scale both data and updates together.

In hindsight this matches Lambert (2026) Ch 7's framing of RLVR: "While
standard instruction tuning is done with 1 or 2 epochs of loss updates
over the data, RLVR gets its name by doing **hundreds or thousands of
epochs over the same few data points** to give the model time to learn
new behaviors." We ran 2 epochs over n=300 — already at the floor of the
RLVR regime, and our v3/v6 attempts went the wrong direction by adding
data instead of adding epochs. The right scaling axis is "epochs per
problem", not "problem count".

## 4. The bug we found

While digging into v8 (asymmetric clip + dynamic sampling), I found this
guard in the trainer:

```python
# Original (buggy) code in core/trainer.py:_update_policy
eps_low = self.config.training.epsilon
eps_high = (
    self.config.training.epsilon_high
    if isinstance(self.algorithm, DAPOEstimator)
    else self.config.training.epsilon  # silently ignores epsilon_high!
)
# ...
pg2 = -mx.clip(ratio, 1.0 - eps_low, 1.0 + eps_high) * adv_arr
```

`epsilon_high` in the YAML config was **only** applied for DAPO. For
GiGPO, GRPO, Dr.GRPO and everything else, the clip was symmetric — even
when the user wrote `epsilon_high: 0.28` in the config, it was silently
coerced back to `epsilon`.

This means our v2-v7 results were all using symmetric 0.2/0.2 clip,
despite the YAML files claiming 0.28. Six runs of "asymmetric clip"
experiments that weren't actually asymmetric.

The fix is one line:

```python
# Fixed
eps_low = self.config.training.epsilon
eps_high = self.config.training.epsilon_high  # respected by all algos
```

If you're running a similar GRPO / GiGPO / verl-agent stack, **check
whether your trainer actually uses your `epsilon_high`**. The DAPO paper
recommends `epsilon_high > epsilon` ("clip-higher") for stronger upward
updates on positive advantages, and several open-source frameworks have
similar dead-config bugs.

**Caveat on v8's failure (added after rereading Lambert 2026 §6.3.3).**
Even with the fix in place, v8 (asymmetric clip 0.2/0.4 + dynamic
sampling) failed to outperform v4. The simple "clip was disabled"
explanation isn't sufficient. Two structural reasons:

1. We use `ppo_epochs=1` and our PPO ratio is `π_θ / π_ref` (against the
   *base* model, not the rollout-time policy). With one gradient step
   per batch, the rollout policy and the policy at update time are
   effectively the same, so the per-token ratio is dominated by the LoRA
   delta from base. Early in training that delta is tiny — `ratio ≈ 1`
   — and clipping is a near no-op regardless of `epsilon_high`. Lambert
   notes this directly: "PPO and GRPO are often run with only one
   gradient step per batch, which means that the PPO-native
   regularization is never applied".

2. Our `dynamic_sampling` filter drops zero-variance *groups* before the
   loss, but DAPO's original dynamic sampling drops *prompts* whose
   *whole batch* is uniform. Dropping at the group granularity is more
   aggressive and removes ~80% of trajectories under our binary reward
   regime, leaving very little signal per gradient step.

The right tool for the "rare-but-important token gets zero gradient"
problem isn't asymmetric PPO clipping — it's CISPO (Lambert §6.1.8),
which clips importance *weights* (with a stop-gradient) instead of the
*objective*, so every token still receives a gradient proportional to
its advantage. We haven't tried CISPO yet; it's near the top of the
follow-up list below.

## 5. The real bottleneck: group-variance collapse

With binary reward and group_size=8, GiGPO computes a relative advantage
by comparing each trajectory's reward against the group mean. If all 8
trajectories in a group get the same reward, the advantage is zero and the
gradient contribution is zero.

I instrumented the trainer to count zero-variance groups per batch
(`drop=N`) and the result was sobering:

```
[1/38] r=+0.50 c=50% loss=0.0083 drop=7
[2/38] r=+0.50 c=50% loss=0.0000 drop=8
[3/38] r=+0.46 c=46% loss=0.0000 drop=8
...
```

For most batches, **87% of groups had zero variance**. Out of 8 prompts ×
8 trajectories = 64 trajectories per batch, only 8 contributed meaningful
gradient.

The math doesn't predict this. With per-trajectory success rate p ≈ 0.42,
the probability that all 8 trajectories in a group land on the same reward
is:

```
P(all 8 same) = p^8 + (1-p)^8 = 0.42^8 + 0.58^8 ≈ 1.4%
```

Expected drop rate ~1.4%, observed ~87.5%. That's a **60× discrepancy**.
The trajectories within a group are nowhere near i.i.d. — they're highly
correlated because the policy has very concentrated modes: at temp=1.0,
sampling 8 trajectories on the same prompt gives 8 nearly-identical SQL
queries.

### Things we tried for the variance bottleneck

**v8: Dynamic sampling.** Drop zero-variance groups before computing the
loss (DAPO-style). Effect on test: training collapsed at the same rate as
v4, val never broke 0.42. Filtering doesn't fix the problem; it just
throws away wasted compute.

**v9b: Entropy bonus.** entropy_coef=0.001 to directly reward high-entropy
output distributions. Effect on val: much *smoother* curve. Where v4
oscillated 38-45%, v9b held 44% across four consecutive validation points
(steps 40, 50, 60, 70). But the peak was lower (0.441 vs v4's 0.451), and
on test n=128 v9b lands at 52.3% — same as v3, 3.2pp below v4. Entropy
bonus trades peak performance for stability across the late training tail.
Useful if you can't run multi-checkpoint selection; not a free lunch.

**Open: per-trajectory temperature schedules.** Vary temperature across
the 8 trajectories in a group so some explore and some exploit. PPO's
importance-ratio interaction with low-natural-probability sampled tokens
needs care.

## 6. Practical recipe (v4)

If you want to reproduce: **v4 is the recommended config for overall EX**.
v7 is recommended only if your downstream is hard/extra-skewed.

```yaml
model:
  path: Qwen3-4B-MLX-4bit
  lora:
    rank: 8
    layers: 4          # last 4 transformer blocks

training:
  algorithm: gigpo
  lr: 0.00005
  kl_coef: 0.0           # no anchor
  epsilon: 0.2
  epsilon_high: 0.2      # symmetric clip
  epochs: 2
  batch_size: 8          # 8 prompts × group=8 = 64 trajs/batch
  ppo_epochs: 1
  entropy_coef: 0.0      # set to 0.001 if you want stability over peak

rollout:
  group_size: 8
  max_steps: 6
  max_tokens: 384

environment:
  type: sql_agent
  partial_credit: 0.0    # BINARY 1.0/0.0 reward — important
  invalid_action_penalty: -0.1

data:
  n_train: 300           # first 300 of train_spider.json
  val_set: indices [6000:6128] from train (held-out tail)
```

## 7. Reproducibility

- **Code:** [github.com/johnhaofu/mlx-agent-rl](https://github.com/johnhaofu/mlx-agent-rl)
- **Adapter checkpoint:**
  - v4 best/ (recommended): https://huggingface.co/x32/spider-rl-qwen3-4b
  - v7 / v9b: available locally in `outputs/`, not yet uploaded
- **One-command reproduce on M-series Mac:**

```bash
git clone https://github.com/johnhaofu/mlx-agent-rl
cd mlx-agent-rl

# download Spider data into data/spider/spider_data/

uv run python examples/train_spider.py configs/spider_v4.yaml 300 128
# ~10 hours on M3 Max, ~14 hours on M2 Pro

uv run python scripts/eval_spider_by_hardness.py configs/spider.yaml 1034 dev \
  outputs/spider_qwen3_4b_gigpo_v4_no_partial/best
```

## 8. What's next

The group-variance bottleneck is the most interesting open problem from
this work. Three directions I haven't tried:

1. **Per-trajectory temperature schedule** — sample the 8 trajectories in
   a group at temperatures 0.5, 0.7, ..., 1.8 to force diversity. PPO
   importance-ratio interaction needs care.
2. **Dynamic resampling** — when a group's reward variance is below
   threshold, sample more trajectories until variance appears. Costs
   compute, may pay off for binary-reward tasks.
3. **State-aware KL** — let the KL coefficient depend on per-state
   advantage magnitude, so the policy can drift in productive regions
   and stay anchored in saturated ones.

If you've tried any of these on similar setups, I'd love to hear how they
went.

## Compute & cost summary

| Run     | n_train | Batches done | Wall time |
|---------|---------|--------------|-----------|
| v2      | 300     | 76 (full)    | ~10.5 h   |
| v3      | 600     | 60 of 150    | ~9.5 h    |
| v4 ★    | 300     | 76 (full)    | ~10.7 h   |
| v5      | 300     | ~58 of 76    | ~8 h      |
| v6      | 400     | 80 of 100    | ~12 h     |
| v7      | 300     | 76 (full)    | ~12 h     |
| v8      | 300     | ~33 of 76    | ~5 h      |
| v9b     | 300     | 76 (full)    | ~10.5 h   |
| **Total training** |  |          | **~78 h** |

Plus ~3 hours of test/dev evals across 7 final checkpoints.

## Acknowledgments

Thanks to the [verl-agent](https://github.com/langfengq/verl-agent) team
for the GiGPO reference implementation, and to Apple for MLX. This work
was done on a single M-series Mac across four days.
