# GRPO run 3 — the environment fixes held; accuracy did not move

*Job 7588418, 140 steps, 8×A100 single node, budget 6, 900-token turns, from
`sft-qwen2.5-vl-32b-c3`. Eval at budgets 2/4/6 against the base model.*

Cycle 3 changed three things in the environment and one in distillation: the
evidence-asymmetry paragraph, explicit CONFIRMED/REFUTED/UNCLEAR definitions,
`w_belief_coherence` → 0.0, and a floor on the teacher's REAL conclusions plus a
filter (`--min-real-belief`) for the traces that ignored it.

## Every intervention worked, measured in the policy's own rollouts

| | run 2 | run 3 |
|---|---|---|
| answer rate | 0.758 (gate) | **0.997** |
| CONFIRMED → belief rises | 24.8% | **86.8%** |
| median belief after 5 inspections | 0.01 | **0.20** |
| %AI, start → end of run | 21% → 22% | **46.8% → 40.7%** |

The last row is the one three cycles had failed at. The balanced prior **survived
GRPO**, drifting 6 points where cycle 1 sat at 21% start to finish. The answer-rate
fix needed `MAX_NEW_TOKENS=900`: the longer prompt pushed turns past the 640-token
cap, and a turn cut off before its ACTION line is unparseable.

## And it made no difference

| held-out, clean, budget 2 | base | SFT (c3) | GRPO (c3) |
|---|---|---|---|
| accuracy | 0.404 | 0.417 | **0.385** |
| McNemar vs base | — | p=1.0 | **p=0.80, −0.019** |
| predicts AI | 45.8% | 46.3% | 38.7% |
| recall(AI) | 0.441 | 0.493 | 0.375 |

Training-set accuracy was flat across the run too: 0.551 ± 0.020 whole-run, first
70 steps vs last 70 giving z=1.05, slope +0.016 over 140 steps.

**`kl` = 0.0006 in every quartile.** The policy barely left where SFT put it.
`grad_norm` ~0.13, `entropy` flat at 0.82, `frac_reward_zero_std` 0.000 — the
gradient was healthy and available at every step and almost nothing happened with
it. Cycle 1's only real gain (discrimination −3.4% → +15.8%) took **600** steps.
This run stopped at 140.

## The finding that matters

**The class collapse was a symptom, not the bottleneck.** We removed it cleanly —
13.4% → 46.3% predicted-AI, confirmed in both the traces and the policy — and
accuracy did not follow. Anyone continuing this work should stop treating the
marginal class rate as the target.

Note also that the *base* model moved 16.5% → 50.4% under the same prompt change,
with no training at all. The prompt did the work; SFT added nothing on top of it.

## One encouraging sign, within noise

The inverted budget curve reversed. The SFT checkpoint got worse with more reveals
(0.421 → 0.357 averaged over degradations); the GRPO policy is flat to slightly
positive (0.376 → 0.380 → 0.389). Choosing cells is precisely what GRPO optimises,
so this is where an effect would first appear. At n=156 per cell it is not
significant, and it should be treated as a lead rather than a result.

## What is left, ranked

1. **600 steps.** The only run that moved anything took 600; this took 140 at a KL
   of 0.0006. Compute, not insight.
2. **Batch the group rollouts.** `make_rollout_func` generates the 8 group members
   sequentially — ~50s each, 401s/step — though they share one image. A 4–6×
   speedup, and the reason every run so far saw only ~140 images and could not
   measure its own effect. This is the prerequisite for (1).
3. **Drop the teacher's label hint.** The teacher is shown the answer and is right
   98.4% of the time, so it writes confident reasoning it never earned. Removing it
   costs keep rate and buys conclusions a student can actually reach.
4. **The budget curve.** More reveals improve the model's internal ranking
   (0.586 → 0.820 AUC) and degraded its written verdict before GRPO. Whatever
   happens between seeing and writing is the largest unexplained effect in the
   project.

## Status

Three cycles, three hypotheses, no held-out accuracy gain. The environment is
validated with measured headroom (0.930 ceiling / 0.591 floor / 0.763 at budget 6),
the reward is hack-resistant after two documented failures, and the ~35-point gap
between what the model can rank and what it answers is characterised but not
closed.
