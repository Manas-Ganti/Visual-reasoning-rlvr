# GRPO run 1 — what moved, what could not, and why

**Date:** 2026-08-26 · **Job:** 7263147 · **Policy:** Qwen2.5-VL-32B + SFT LoRA ·
**Substrate:** `synth1024`, `OVERVIEW_LONG_EDGE=56`, G=8, 4 inspects

Terminated by the 24-hour walltime at **step 696 of 1258 (55%)**, 23:57:37
elapsed, ~120 s/step. Not a crash. `checkpoint-600` is a usable adapter and is
what the eval below measures.

## The result in one line

GRPO taught the policy to **tell the classes apart**, and could not change **how
often it commits to the rarer one** — and the second is structural, not a tuning
failure.

## What moved

| | start of run | end of run |
|---|---|---|
| mean reward | −0.032 | **+0.183** |
| correct rate | 48.3% | **57.9%** |
| AI recall | 19.5% | **29.5%** |
| false "AI" on real photos | 22.9% | **13.7%** |
| discrimination gap | **−3.4%** | **+15.8%** |

The starting gap is the interesting number: it is *negative*. Before GRPO the
policy was marginally **more** likely to call a real photograph AI than an actual
AI image. Whatever accuracy it had came from answering REAL often on a set that is
half real — not from discrimination, of which it had none.

600 steps produced a genuine 15.8-point gap. The gradient stayed healthy
throughout: advantage sd ~0.935, only **2.7%** of groups unanimous.

Per-class rates are inferred from the marginal AI-share and the reward sign,
assuming a balanced train stream (it is: 629/629) and reward > 0 ⇒ correct
verdict. The eval measures them directly.

## What did not move at all

```
AI share     21.2% -> 21.6%    (balanced data; 50% would be unbiased)
confidence   0.950 -> 0.955    (~50% of episodes state exactly 1.0)
P(fake)      0.209 -> 0.222    (~22% pinned at 0.0 or 1.0)
inspects     4.09 per episode  (the budget is exhausted every single time)
```

Four behaviours, flat across 5,568 rollouts.

### Why the prior cannot move under GRPO

This is the finding worth keeping. GRPO's advantage is computed **within a group
of rollouts on one image**. On an AI image the advantage pushes toward answering
AI; on a real image it pushes toward REAL. Across balanced data those pressures
**cancel**.

So the method sharpens *which* images earn the rare AI call, and has no mechanism
to change *how often* one is made. Observed exactly: discrimination +19 points,
marginal AI-share ±0.4 points.

A policy that answers AI on 21% of images caps its own AI recall at **42%**, even
with perfect targeting. It reached 29.5% — about two thirds of the room it allows
itself. **The binding constraint is the prior, not the discriminator.**

## Three defects this run exposed

**1. One image per optimizer step.** `grpo.py:219` derives
`grad_accum = max(G // (per_device_bs × world_size), 1)` = `max(8 // 8, 1)` = **1**,
so the global rollout batch is exactly one group. That is the highest-variance
configuration available, and it is why an epoch is 1258 steps and ~42 hours. Set
`GRAD_ACCUM=8` for 8 images per step — same total generation, one eighth the
optimizer updates, far lower gradient variance.

**2. The ceiling gate measured a task the agent never faces.** Gate 1 scored
**0.930 AUC on the full image**. The agent sees 4 cells of 16 — **25% of the
image** at high resolution. The gate certified a substrate the policy cannot
actually reach, and when the agent lacks evidence it falls back on its prior,
which is REAL. **A budget-limited ceiling probe** — reveal 4 random cells, ask
once — would measure the ceiling that exists inside the environment. That number,
not 0.930, is what SFT and GRPO are chasing.

**3. Confidence never learned to hedge.** Mean stated confidence 0.95, with half
of all episodes at exactly 1.0, unchanged. `confident_wrong` (−0.50 × confidence)
is meant to make loud wrong answers expensive, but against a ±1.00 outcome term it
never became the dominant consideration.

## Rebuild plan, in order of expected value

1. **Attack the prior directly.** GRPO will not do it. Options, cheapest first:
   raise `--max-inspects` (25% of the image may simply be too little evidence);
   make `confident_wrong` bite hard enough that hedging beats a confident REAL;
   add explicit class-balance pressure to the reward.
2. **Fix the batch shape** — `GRAD_ACCUM=8`.
3. **Budget-limited ceiling gate** before any further training.
4. **Prompt:** `RECONCILIATION` never states that CONFIRMED means the predicted
   AI artifact was *present*, so `P(fake)` must **rise**. 91% of reconciliations
   were CONFIRMED while beliefs fell — `belief_coherence` scored ~0 on every
   REAL-leaning rollout, which turned that term into a proxy for verdict direction
   rather than a check on reasoning. Needs an SFT redo (23 minutes), not a GRPO
   restart.
5. **Walltime:** a full epoch needs ~42 h. Either `--max-steps 600` for a clean
   finish inside 24 h, or the 48 h base QOS, or halve G.

## Eval

Held-out test split, `checkpoint-600` vs. base, plus the unseen-generator set.

```
synth1024      (seen generator, SDXL)   PENDING  job 7283097
synth1024flux  (unseen generator, FLUX) PENDING  job 7283103
```

**Read the class breakdown, not the accuracy.** A policy answering REAL 79% of the
time scores ~50% on a balanced split and looks like a coin flip; the per-class
recall is what shows the bias. That is the failure that killed the faces
substrate, and `eval/harness.py` prints it for this reason.
