# The gates were measuring image size, not image content

**Date:** 2026-08-22 · **Model:** Qwen2.5-VL-32B-Instruct ·
**Dataset:** `genwukong` (GenImage Wukong, 800 images, streamed via `build_manifest_hf.py`)

Both pre-training gates returned numbers that looked interpretable and were not.
Every generated image in the substrate was exactly 512×512 while the real half
was varied and slightly smaller, and **image area alone predicts the label at
0.850 AUC** — no model, no pixels, just the file. That is larger than the
measured floor and statistically indistinguishable from the measured ceiling.

The substrate was rebuilt with both classes center-cropped to a common size. The
signal that survived is real but the gap is narrow, and the residual floor points
at a different problem. Details below.

## What was measured, and what it was actually measuring

| | before crop | geometry alone | after crop |
|---|---|---|---|
| ceiling AUC (full res) | 0.874 | **0.850** | 0.809 |
| floor AUC (56px overview) | 0.745 | **0.850** | 0.673 |

The ceiling and the confound agree to within noise. Nothing in the original
numbers is evidence that the model ever looked at a picture.

## How it hid

Three things had to line up, and each is individually reasonable.

**1. Accuracy could not see the model's actual ability.** The first ceiling read
`accuracy=0.600` on a 50/50 split — apparently a hard fail. But the confusion
matrix showed the model answering `REAL` on 70 of 80 images while being right on
9 of the 10 it called `AI`. It was ranking correctly and refusing to commit.
Argmax accuracy on a forced one-word answer measures the model's *prior* as much
as its eyesight, and it cannot distinguish "no signal" from "signal the policy
will not act on" — the two outcomes that imply opposite decisions.

Scoring the first-token distribution instead (`--auc`) put the same run at
**0.874**. Same model, same images, same prompt.

**2. The floor gate was equally blind, in the same direction.** A model that
answers `REAL` to nearly everything scores ≈0.50 on the floor *whether or not the
overview leaks the answer*. Measured by accuracy the floor would have read
"≈chance — perfect", from a measurement structurally incapable of detecting the
failure it exists to catch. By AUC it read **0.745**: two thirds of the
discriminative signal was already available through the blur.

**3. Size survives everything the environment does to an image.**
`env/grid.make_overview` scales width and height by a single factor and restores
uniformly, so aspect ratio and relative size pass through the downsample
untouched — unlike the texture artifacts the task is *about*, which the blur is
designed to destroy. Worse, Qwen bins images to a patch grid, so a 512×512 input
and a 490×470 input arrive as **different numbers of image tokens**. The shortcut
reaches the policy before any content is perceived.

An encoding difference would at least die when the pixels are resampled. A
geometry difference never does.

## The fix

`data/recrop_manifest.py` center-crops an existing substrate to a fixed size and
writes a new namespace. Cropping is allowed where resizing is not: it *selects*
pixels and invents nothing, so the artifacts the task depends on survive, whereas
resampling would manufacture frequency content the sources never had — from two
different starting sizes, which is a fresh confound in place of the old one.

`genwukong` → `genwukong392` (392 = 14×28, an exact multiple of Qwen's patch
size) kept 796 of 800 images with test at exactly 40/40:

```
                 before          after
  aspect (w/h)   0.532           0.500
  long edge      0.840           0.500
  area (px)      0.850           0.500
  bytes/px       0.604           0.622
```

`bytes/px` is deliberately *not* driven to 0.5. The model never sees file size;
that number is an indicator that AI images are smoother and so more compressible,
which is a genuine visual property a detector should use. Only the channels that
mechanically reach the policy needed removing.

## What the clean numbers say

```
ceiling AUC 0.809   95% CI ≈ [0.71, 0.91]   calibrated accuracy 0.812
floor   AUC 0.673   95% CI ≈ [0.56, 0.79]
```

Real signal: the ceiling survived removing a 0.850 confound and stays well clear
of chance. Against the ≥0.85 gate it is short but inside the error bar at n=80.

The floor is the remaining problem. At 0.673, **56% of the discriminative signal
is still free in the overview**, and the ceiling-floor gap (0.136) is not cleanly
significant at this sample size.

Whatever leaks now is not geometry and not texture — 56px destroys texture. It is
global: colour statistics, saturation, smoothness, composition. That is the
shortcut `build_manifest_hf.py` names in its own header — *"Pairing an arbitrary
generator against an arbitrary photo set adds a content-style shortcut
('arty-looking = fake')."* GenImage's Wukong split is diffusion renderings
prompted from ImageNet class names against real ImageNet photographs; those two
populations differ in style, and style survives any downsample. No crop fixes it.

**Next:** a genuinely paired source (bitmind's `<real>___<generator>`, where fakes
are generated from the real set's own captions) attacks the cause; the 72B may
widen the gap by lifting the ceiling without lifting the floor.

## Tooling this produced

* `tools/manifest_stats.py` — scores aspect, long edge, area and bytes/px as
  standalone label predictors. Run it before trusting any gate number.
* `tools/ceiling_probe.py --auc` — first-token ranking AUC plus a sweep over the
  observed scores, replacing an argmax accuracy that cannot separate the two
  failure modes.
* `data/recrop_manifest.py`, `build_manifest_hf.py --center-crop` — equalise
  geometry without re-downloading or resampling.

## The transferable lesson

Both gates were honest measurements of the wrong quantity, and both agreed with
each other, which is what made it convincing. The check that broke the tie cost
ten seconds and needed no GPU: **ask whether the label is readable from the file
before asking whether the model can read it from the picture.**

Sequence to run on any new substrate, in this order:

1. `tools/manifest_stats.py` — every predictor ~0.5, or fix the data first.
2. `ceiling_probe --auc --condition full` — want ≥0.85.
3. `ceiling_probe --auc --condition overview` — want ≈0.5.

A ceiling that passes with an unchecked manifest is not evidence of anything.
