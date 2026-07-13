# Data curation

## Substrate: Fake-Vs-Real-Faces (Hard)

[`hamzaboulahia/hardfakevsrealfaces`](https://www.kaggle.com/datasets/hamzaboulahia/hardfakevsrealfaces)
— 1,288 images at 300×300:

- **700 fake** — StyleGAN2 samples from *thispersondoesnotexist.com*.
- **589 real** — diverse faces from the Unsplash API, cropped with OpenCV.

The set is deliberately *hard*: modern GAN faces at photo resolution, chosen so
the task is non-trivial for humans and models alike. Labels are **image-level
only** — there are no tiers, no generator diversity (single generator), and no
per-region artifact annotations.

### Why image-level labels are enough here

The verifiable reward (`env/reward.py`) is anchored on the agent's *own
documented trajectory* — whether P(fake) moves coherently with its recorded
reconciliations, and whether the final verdict follows from the accumulated
evidence — plus ground-truth verdict correctness and calibration. None of those
terms need region labels. Region-level ground truth appears **only** in the
eval-only evidence slice below, never in the training reward. This is the design
decision that let us drop the generator-tier / artifact-region labeling the
original plan assumed.

## Manifest (`build_manifest.py` → `manifest.jsonl`)

```bash
python data/build_manifest.py                 # kagglehub download (needs Kaggle creds)
python data/build_manifest.py --src /path/to/unzipped/dataset
```

Each row: `{"id", "file_name" (repo-relative), "label" (0 real / 1 AI), "split"}`.
Images are copied+normalized to PNG under `data/images/{fake,real}/` so the
manifest is portable and byte-reproducible (rerun on the A100 with the same
`--seed` to regenerate identical splits). The builder tolerates both the
dataset's `data.csv` label file and a `fake/real` subfolder layout.

**Splits**: stratified by label, default 80/10/10 train/val/test, seed 0. Test is
held out from all training (SFT + GRPO) and used only by `eval/harness.py`.

## Difficulty axis 1 — image degradation (`degradation.py`)

Single-generator data gives no generator-tier ladder, so we manufacture
difficulty by degrading the *same* images, eroding the high-frequency cues (hair
strands, iris edges, skin micro-texture) that betray a GAN face:

| level | transform | intent |
|---|---|---|
| `clean` | none | baseline |
| `jpeg` | re-encode at quality 30 | block/ringing artifacts mask subtle tells |
| `blur_downscale` | ½-res round-trip + Gaussian blur | destroy fine detail |

Transforms are deterministic given `(image, level)` and applied consistently to
the overview and every inspect crop. The harness reports pass rate per level;
Stage-3 verification **holds out an unseen level** to prove generalization to
difficulty never trained on.

## Difficulty axis 2 — budget tightness

Fewer allowed inspects (`InvestigationEnv(max_inspects=k)`) = harder. Tunable per
target model; the calibration deliverable reports pass-rate × degradation ×
budget so you can state "to target model X at ~50% pass, use level Y at budget k".

## Evidence slice (`build_evidence_slice.py`, eval-only) — honest scoping

To ask *"does the RL agent inspect true-artifact cells more than the SFT
baseline?"* we need a small set of fakes with known artifact locations. We build
~60–80 **confidently-labeled fake** images:

1. Run a face-forgery detector / the base VLM and compute a **GradCAM** saliency
   map over each fake.
2. Map saliency peaks to 4×4 grid cells (`env/grid.point_to_cell`) as *proposed*
   artifact cells.
3. **Human-verify** at zoom: keep only cells where a real, nameable artifact is
   visible; discard images where the fake is genuinely indistinguishable.

This slice is used **evaluation-only**, never in any reward. Indistinguishable
images are bucketed out of the inspect-hit metric but still counted in verdict
accuracy. Being explicit about that scoping is itself part of the honest-reporting
story (see `results/reward_failure_history.md` and the blog).
