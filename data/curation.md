# Data curation

## Substrate: GenImage

[GenImage](https://github.com/GenImage-Dataset/GenImage) (Zhu et al., NeurIPS 2023
Datasets & Benchmarks) — real ImageNet photographs paired against generations from
**eight** models: ADM, BigGAN, GLIDE, Midjourney, Stable Diffusion 1.4, Stable
Diffusion 1.5, VQDM, Wukong.

Chosen against four requirements, each derived from a *measured* failure of the
previous substrate (see [`../results/faces_negative_result.md`](../results/faces_negative_result.md)):

| Requirement | Why | Enforced by |
|---|---|---|
| ≥512px short edge | a 4×4 cell must carry real detail, or `INSPECT` just upscales a blur | `--min-edge` |
| ceiling ≥0.85 | the model must be *able* to know the answer | `tools/ceiling_probe.py --condition full` |
| floor ≈ chance | otherwise investigation buys nothing | `tools/ceiling_probe.py --condition overview` |
| localized, semantic artifacts | *which* cell you inspect has to matter | substrate choice |

The artifact type is the substantive change. GAN face artifacts are
high-frequency (pore texture, iris edges, hair strands) and do not survive
downsampling. Diffusion artifacts here are **semantic and spatially
concentrated** — malformed hands, garbled text on signs, impossible geometry,
shadows that disagree with their light source. VLMs demonstrably detect these,
they survive the overview downsample as *hints* rather than as answers, and they
live in specific grid cells. That is what makes `INSPECT <n>` a real decision.

### Why image-level labels are enough

The verifiable reward (`env/reward.py`) is anchored on the agent's *own
documented trajectory* — whether P(fake) moves coherently with its recorded
reconciliations, and whether the final verdict follows from the accumulated
evidence — plus ground-truth verdict correctness and calibration. None of those
terms need region labels. Region-level ground truth appears **only** in the
eval-only evidence slice below, never in the training reward.

### The known confound, and how we detect it

GenImage's real and generated images have different compression and resize
histories, so a detector can score well on low-level statistics rather than on
anything semantic. The **floor probe is the test for this**: if accuracy from the
blurred overview is well above chance, the model is reading a global statistic
that survives downsampling, investigation is decorative, and the substrate is not
usable as-is. A high ceiling is only good news when paired with a low floor.

## Manifest (`build_manifest_genimage.py` → `data/genimage/manifest.jsonl`)

```bash
export VRR_DATASET=genimage
python data/build_manifest_genimage.py --src /path/to/GenImage --per-generator 200
python data/build_manifest_genimage.py --src /path/to/GenImage \
    --generators sdv1.4,midjourney --per-generator 400 --copy
```

Each row: `{"id", "file_name" (repo-relative), "label" (0 real / 1 AI), "split", "generator"}`.
Images are symlinked (or `--copy`ed) under `data/genimage/images/<generator>/{ai,real}/`.

**Splits**: stratified by `(generator, label)`, default 80/10/10, seed 0 — so every
generator and class appears in test and the per-generator breakdown has rows to
report. Test is held out from all training (SFT + GRPO) and used only by
`eval/harness.py`.

> ⚠️ **Check the per-class pool counts the builder prints before committing to a
> manifest.** GenImage's `nature` images are ImageNet, typically around 500×375,
> while SD/Midjourney generations are natively 512 or 1024. At `--min-edge 512`
> you can reject most of the real class while keeping nearly all of the fake one —
> which makes *resolution itself* a shortcut feature and quietly invalidates the
> whole run. If the real pools come back small, lower `--min-edge` (≈384) rather
> than accepting the imbalance.

## Difficulty axis 1 — image degradation (`degradation.py`)

Applied to the *same* images to erode the cues that betray a generation:

| level | transform | intent |
|---|---|---|
| `clean` | none | baseline |
| `jpeg` | re-encode at quality 30 | block/ringing artifacts mask subtle tells |
| `blur_downscale` | ½-res round-trip + Gaussian blur | destroy fine detail |

Deterministic given `(image, level)` and applied consistently to the overview and
every inspect crop. Stage-3 verification **holds out an unseen level** to prove
generalization to difficulty never trained on.

## Difficulty axis 2 — budget tightness

Fewer allowed inspects (`InvestigationEnv(max_inspects=k)`) = harder. The
calibration deliverable reports pass-rate × degradation × budget, so you can state
"to target model X at ~50% pass, use level Y at budget k".

Axes 1 and 2 are not only reporting dimensions — they are the knobs that move the
policy into the band where GRPO has a gradient. See
[`../results/reward_failure_history.md`](../results/reward_failure_history.md) and
`tools/group_variance_probe.py`.

## Axis 3 (reporting only) — generator

`generator` rides along in every row, so `eval/harness.py` breaks pass rate down
per generator. This distinguishes "detects BigGAN, blind to Midjourney" from
"uniformly weak" — two situations the aggregate accuracy cannot tell apart. It is
categorical, not ordered, so it is a breakdown rather than a difficulty ladder.

## Evidence slice (`build_evidence_slice.py`, eval-only) — honest scoping

To ask *"does the RL agent inspect true-artifact cells more than the SFT
baseline?"* we need a small set of fakes with known artifact locations. We build
~60–80 **confidently-labeled fake** images:

1. Compute a per-cell **occlusion saliency** map over each fake: gray-box each
   cell in turn, ask the base VLM "AI or REAL?", and score the cell by how much
   hiding it *lowers* P(AI).
2. Map the top-k cells to the 4×4 grid as *proposed* artifact cells.
3. **Human-verify** at zoom: keep only cells where a real, nameable artifact is
   visible; set `artifact_cells: []` for genuinely indistinguishable fakes.

Written to `data/<dataset>/evidence_slice.jsonl`. Used **evaluation-only**, never
in any reward. Indistinguishable images are bucketed out of the inspect-hit metric
but still counted in verdict accuracy. Being explicit about that scoping is itself
part of the honest-reporting story.

## Retired substrate: Fake-Vs-Real-Faces (Hard)

`data/build_manifest.py` (namespace `faces`) still builds
[`hamzaboulahia/hardfakevsrealfaces`](https://www.kaggle.com/datasets/hamzaboulahia/hardfakevsrealfaces)
— 1,288 300×300 images, 700 StyleGAN2 fakes vs 589 Unsplash reals. It is kept
only so the negative result reproduces; every artifact it produces is namespaced
under `faces/` and cannot collide with GenImage. Do not start new work on it.
