# A substrate that clears both gates — built, because none was found

**Date:** 2026-08-23 · **Model:** Qwen2.5-VL-32B-Instruct ·
**Dataset:** `synth1024` — DIV2K photographs vs. SDXL, 1570 images, 1024×1024 ·
**`OVERVIEW_LONG_EDGE=56`**

Three public substrates were measured and none passed. The faces set had no
evidence to reveal (300px images → 75px cells). GenImage/Wukong's gates turned out
to be measuring image size ([`geometry_confound.md`](geometry_confound.md)), and
once that was removed its floor stayed at 0.673 — two thirds of the answer free in
the overview. A survey of paired alternatives found the structural reason: these
datasets are downsampled to 256–512px because they are built to train CNN
classifiers, which do not need more. This environment needs the opposite.

So the substrate was built to specification instead.

## Result

```
ceiling  AUC 0.930   95% CI [0.89, 0.97]   calibrated accuracy 0.878     gate >=0.85  PASS
floor    AUC 0.591   95% CI [0.50, 0.68]   at OVERVIEW_LONG_EDGE=56      gate ~0.50   PASS
gap      0.339  — 79% of the discriminative signal requires investigation
```

n = 156 test images, 78 per class.

## How it was built

| requirement | how it is satisfied |
|---|---|
| native ≥1024 | DIV2K photographs (~2040×1356), center-cropped to 1024 |
| pairing | Qwen2.5-VL captions each **cropped** real; SDXL generates from that caption |
| no geometry shortcut | both classes exactly 1024×1024 PNG — aspect, area, long edge all 0.500 |
| local artifacts | SDXL is globally photorealistic and locally broken (hands, text, fine structure) |

`data/build_paired_synthetic.py`, four restartable stages: crop → caption →
generate → assemble. 785 pairs from 800 DIV2K images (15 fell under 1024px).

Two details are load-bearing and were both wrong on the first attempt:

**Caption ordering.** The first pass produced dense, accurate captions of 90–110
words — and SDXL's CLIP encoders keep ~77 tokens and discard the rest silently.
The model's natural ordering opens with the subject and closes with *"marine
debris ... coral fragments"*, *"small imperfections like shadows and
reflections"*, so truncation ate precisely the incidental detail the pairing
exists to preserve. A tidy render against a cluttered photograph is how
"tidy = fake" becomes readable from the blurred overview. Fixed by asking for 50
words as a fragment, clutter first, subject and lighting last.

**Guidance scale.** CFG 5.5 rather than the usual 7.5+, and no negative prompt.
High guidance and "blurry, low quality" negatives both push toward saturated,
over-stylised output — a *global* difference that survives any downsample and
hands the floor a free answer.

## Correcting the overview heuristic

`scripts/arc_env.sh` advised aiming at roughly **native/7**. On this substrate
that is `OVERVIEW_LONG_EDGE=140`, and it fails:

| overview | zoom | floor AUC | 95% CI | signal free in the overview |
|---|---|---|---|---|
| 140 | 7× | 0.740 | [0.66, 0.82] | 56% |
| 80 | 13× | 0.637 | [0.55, 0.72] | 32% |
| 64 | 16× | 0.601 | [0.51, 0.69] | 24% |
| **56** | **18×** | **0.591** | **[0.50, 0.68]** | **21%**  ← chosen |
| 48 | 21× | 0.556 | [0.47, 0.65] | 13% |

native/7 was calibrated when overview resolution and cell size were coupled: at
512px natives a harsher overview bought nothing, because the 4×4 cells were
already too small for `INSPECT` to reveal anything. At 1024 they are decoupled —
the cell is 256px whatever the overview is — so the blur can be pushed far harder
at no cost. **The right rule is not a fixed ratio but a sweep: take the mildest
blur whose floor CI contains chance.**

The ceiling is measured at native resolution and is therefore independent of this
setting; only the floor needs re-running when it changes.

**48, 56 and 64 are statistically indistinguishable.** At n=156 the standard error
on each is ~0.045, against differences of 0.035–0.045; the intervals overlap
almost entirely. The sweep resolves the *shape* of the curve — 140 and 80 clearly
fail — but not the bottom of it, and further probes would resample noise rather
than settle anything.

**56 was chosen on a criterion the gate cannot measure.** At 48 the overview is
hard for a human to plan from, and an overview nobody can direct an investigation
from breaks the environment in a different way: cell selection degenerates to
random, and the predict-then-verify mechanism has nothing to condition a
hypothesis on. The floor gate is blind to that failure — it only asks whether the
answer is free, not whether the search can be aimed. 56 is also 2×28, an exact
multiple of Qwen's patch size.

What that buys, stated plainly: an overview-only policy reaches ~0.60 accuracy
against a 0.50 baseline, while a full-resolution investigator reaches 0.878. The
reward gradient points at investigating by a wide margin, which is the property
the environment needs.

## Why the accuracy framing would have rejected this substrate

The raw ceiling reading is **accuracy 0.878 only after calibration**; the model's
argmax answer is far more conservative. Every gate in this project now runs with
`--auc`, because argmax accuracy on a forced one-word answer measures the model's
prior as much as its eyesight and cannot separate "no signal" from "signal the
policy will not commit to". That distinction is what rescued Wukong from a 0.600
reading, and it is what makes 0.930 legible here.

## What this does not yet establish

* **One generator.** Every fake is SDXL, so a trained detector may learn SDXL
  artifacts rather than AI artifacts. The fix is a held-out set from a second
  generator (FLUX) used in eval only — `eval/harness.py` already breaks results
  down per generator, and it upgrades the claim from "detects SDXL" to "the
  investigation transfers".
* **Caption-mediated composition.** Fakes depict what a caption described, so they
  may be systematically simpler than the photographs that seeded them. The floor
  gate is the instrument that would catch it, and at 48 it reads chance — but the
  dense-caption prompt is what keeps it that way, and it should not be loosened.
* **Gate 2 is unrun.** Group variance is measured against the SFT checkpoint, not
  the base model. A passing Gate 1 says the substrate is learnable; it says
  nothing about whether GRPO will have a gradient.
