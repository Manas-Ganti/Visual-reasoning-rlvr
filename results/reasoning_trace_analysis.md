# Why the policy says REAL to everything

*GRPO run 2 (job 7520138), 701 rollouts recovered from the SLURM log with
`tools/traces_from_log.py`. Substrate `synth1024`, budget 6, policy
`sft-qwen2.5-vl-32b-a100off`.*

Cycle 2 changed three things: a Brier term on the stated `P(fake)`, SFT class
weighting at 65% AI, and an inspect budget of 6. Held-out accuracy came back
**0.455 against a 0.449 base** (McNemar p=1.0), with the policy predicting AI on
**13.4%** of answered episodes — *less* than the untouched base's 16.5%.

The aggregate numbers say the intervention did nothing. The trajectories say
why, and it is not what we assumed.

---

## The reasoning is sound. The inference from it is not.

The trajectories are not degenerate. They contain real observations, targeted
hypotheses, and updates in the right direction:

> **HYPOTHESIS:** If the image is AI-generated, the text on the garment will
> appear distorted, incomplete, or nonsensical.
> **RECONCILIATION:** The text in cell 6 is legible and consistent. It reads
> "SALVA"… the font is uniform and natural-looking.
> **BELIEF_UPDATE:** P(fake)=0.2 because readable and logically placed text
> reduces suspicion but does not entirely rule out AI generation.

That is good investigative work. The failure is what happens when several of
those accumulate.

### Absence of evidence is being read as evidence of absence

Median `P(fake)` across 685 belief paths, starting from a 0.5 prior:

| after N inspections | median P(fake) | fraction below 0.1 |
|---|---|---|
| 1 | 0.200 | 0% |
| 2 | 0.100 | 8% |
| 3 | 0.100 | 48% |
| 4 | 0.050 | 59% |
| 5 | **0.010** | 61% |

65% of all belief changes are downward. After inspecting 5 of 16 cells the
policy asserts 100-to-1 odds of REAL — on the strength of not having found
anything in under a third of the image.

This is a reasoning error, not a perception failure. Finding an artifact proves
AI; finding none in a few tiles proves very little, because the artifact may sit
in any of the ten cells never opened. The policy weighs both the same, so every
uneventful look ratchets it further toward REAL. **That is the whole 87%-REAL
bias**, and it explains why neither GRPO nor class-weighted SFT could move the
prior: they were correcting an output whose cause was an inference rule.

It does this even when the inspection was simply *mis-aimed*. One trajectory
hypothesises about a face five times, inspects five cells that do not contain
the face, and records each miss as evidence for REAL:

> **REFUTED** - Cell 11 continues to deviate from the original hypothesis by
> providing no view of the face whatsoever.

Not finding what you were looking for in the wrong place is evidence of nothing.

### The verdict is faithful to the belief

| final P(fake) | episodes |
|---|---|
| 0.0 – 0.1 | **436** |
| 0.1 – 0.9 | 45 |
| 0.9 – 1.0 | **178** |

The verdict follows the policy's own `P(fake) > 0.5` rule **99.4%** of the time.
So this is not a broken decision rule sitting on top of good beliefs — it is a
faithful decision on a belief that collapsed. It also rules out the cheap fix:
the distribution is bimodal, and no threshold separates within a cluster piled
at 0.0.

---

## We were training against the correct behaviour

`RECONCILIATION` was specified as *"CONFIRMED or REFUTED — did the reveal match
your last hypothesis, and how"*. The policy read that as **"was my description
accurate"**. The reward read it as **"was the fake-hypothesis borne out"**, and
`belief_coherence` (weight 0.30) paid out accordingly.

| CONFIRMED, then the belief… | count | share |
|---|---|---|
| rose — what the reward assumes | 561 | 24.8% |
| **fell — the policy verified something clean** | **1,367** | **60.5%** |
| held flat | 333 | 14.7% |

CONFIRMED outnumbered REFUTED 87:13. So the dominant pattern in the data is the
policy writing "CONFIRMED" to mean *I checked this*, correctly lowering its
estimate — and scoring 0.0 on a term that paid 0.30 to siblings whose tag
happened to agree.

The term never subtracted; it withheld. Under GRPO's group-relative advantage
that is the same thing: a 0.30 gap against the other seven members of the group
is a gradient pushing away from the behaviour. Ten hours of GRPO pushed the
policy away from updating its beliefs sensibly. Measured coherence was 29%.

---

## Changes made

1. **`env/prompts.py` — the evidence asymmetry is now stated.** Finding an
   artifact is strong evidence of AI; a clean cell is weak evidence of REAL and
   should move `P(fake)` by less than 0.1; a cell that does not contain the
   predicted feature is evidence of *nothing*. The paragraph names the real
   budget ("at most 6 of 16 cells") because the argument depends on how much
   stays unseen.
2. **`env/prompts.py` — the three tags are defined against the fake-hypothesis**,
   with an example of each and an explicit direction: *CONFIRMED means you FOUND
   something and P(fake) RISES. If you looked and everything seemed fine, that
   is REFUTED.* `UNCLEAR` now has a stated use — the mis-aimed inspection.
3. **`env/reward.py` — `w_belief_coherence` 0.30 → 0.0.** The scorer is kept and
   still tested; only the weight is zero. Re-enable it once a run shows the
   policy using the tags as now defined — coherence well above 29% — and not
   before, or it will mistrain again.

`w_prediction_tracking` was already 0.0 from cycle 2 for a related reason: it
paid for the CONFIRMED *fraction*, which the policy maxed at 87%.

## What this does not fix

The prompt cannot repair the SFT traces. Distillation used a **label-revealed
teacher**: the teacher was told the answer, then wrote a trajectory justifying
it. The student inherits the format and the certainty without the
discrimination, which is the most likely source of the saturation at 0.98
confidence. Cycle 3 needs the traces regenerated under the new prompt — and
should consider whether the teacher hint can be weakened, since a trace whose
evidence→verdict link was never real is teaching a pattern the student cannot
reproduce.

## Open, and worth measuring before anything expensive

- **`MAX_INSPECTS` is silently ignored by the `eval` branch** of
  `arc_infer.slurm`, which uses `--budgets`. The 0.455 was measured at budgets 2
  and 4; the policy was trained at 6. There is no eval at the trained budget.
- **Answer rate is 99.8% in training but 81.4% at eval.** The final-turn warning
  that fixed the former does not appear to carry to the shorter budgets.
- **The run could not measure itself.** With `G=8` on 8 ranks and `grad_accum=1`,
  each step is *one image* — 100 steps saw 100 images. Whole-run training
  accuracy was 0.539 ± 0.035. An apparent peak at steps 31–50 (0.725 ± 0.072,
  z=2.89 against the remainder) does not survive the fact that the window was
  chosen after seeing the data; the pre-committed halves give z=1.69. `kl` sat
  at ~0.0008 throughout with no excursion, so there is no evidence of a real
  policy change in either direction.
- **Rollouts are sequential** (`for prompt in prompts` in `make_rollout_func`),
  ~50 s per episode, 401 s per step. The 8 group members share an image and
  could be batched through `generate()` together. That is the lever that would
  make a future run large enough to measure its own effect.
