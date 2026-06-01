# VLM RL Forgery Detection

Train a Vision-Language Model (VLM) to tell **real photographs** apart from
**AI-generated images** — not with a single one-shot classification, but through
**active perception**: the model investigates each image over several turns,
zooming into regions and requesting forensic metadata, documents its reasoning,
and only then commits to a verdict. The policy is fine-tuned with
**GRPO** (Group Relative Policy Optimization) so that *correct, efficient,
well-reasoned* investigations are reinforced.

> 📐 **Full design doc:** [`ARCHITECTURE.md`](ARCHITECTURE.md) — the end-to-end
> system (data pipeline, Gymnasium environment, agent protocol, GRPO loop,
> reward structure, and known limitations).

---

## Why active perception?

A downsampled full image often hides the tell-tale signs of synthesis. The
artifacts that betray a generated image — warped textures, fused or extra
fingers, garbled text, asymmetric eyes/teeth, melted backgrounds, missing
camera EXIF — live in *local detail* or in *metadata*. So instead of forcing a
verdict from one glance, this environment gives the model **tools** and a
**turn budget**, and rewards it for gathering exactly the evidence it needs
before deciding. The model learns *how to look*, not just *what to label*.

---

## How it works

Each episode presents one image. The image is divided into a 3×3 grid:

```
1 2 3
4 5 6
7 8 9
```

Every turn the model emits a reasoning line and exactly one action:

```
THOUGHT: <reasoning about what it sees and what to do next>
ACTION:  <ZOOM n | METADATA | ANSWER AI | ANSWER REAL>
```

| Action | Effect |
|--------|--------|
| `ZOOM <1-9>` | inspect a grid cell at higher (upscaled) resolution |
| `METADATA` | reveal the image's forensic metadata report (EXIF, software signature, color space) |
| `ANSWER <AI\|REAL>` | commit a verdict and end the episode |

The sequence of `THOUGHT`s across the turns forms a **chain of reasoning** that
culminates in the final `ANSWER`. The parser (`src/utils.py`) is deliberately
tolerant — case-insensitive, accepts `ZOOM(5)`, honors only the last `ACTION`
line, and normalizes verdict synonyms — so a slightly malformed completion is
still usable instead of wasting the rollout.

---

## Reward design (high level)

Reward is computed inside `ForgeryDetectionEnv.step` and parameterized by a
`RewardConfig` dataclass, so the scheme can be tuned without touching the
rollout or training code. The signals:

- **Terminal correctness** — `+1.0` correct verdict, `−1.0` wrong.
- **Investigation cost** — small per-action step cost (efficiency).
- **Action validity / redundancy** — penalties for malformed actions and
  re-zooming an already-viewed cell.
- **Budget exhaustion** — penalty for running out of turns without answering.
- **Metadata-shortcut penalty** — the simulated metadata is label-correlated, so
  a *correct verdict reached after using `METADATA`* is docked, pushing the
  policy to discriminate from pixels alone while still keeping such answers
  net-positive.
- **Reasoning-presence reward (live)** — a small per-turn bonus for a
  substantive, non-boilerplate `THOUGHT`, so the reasoning channel doesn't
  collapse under outcome-only credit assignment.
- **Judged reasoning quality (stubbed, off by default)** — an optional,
  correctness-gated bonus from a separate VLM judge that scores whether the
  final reasoning is grounded and coherent; it can only break ties between
  correct answers, never reward an eloquent wrong one.

See [`ARCHITECTURE.md` §5](ARCHITECTURE.md) for the full table, default weights,
and the rationale behind each lever.

---

## Repository layout

```text
vlm-rl-forgery-detection/
├── data/                     # storage layer (gitignored)
│   ├── processed_images/     # standardized lossless RGB PNGs
│   ├── raw_data/             # source videos / images (Sora frames, real photos)
│   └── metadata.jsonl        # dataset catalog (one JSON record per image)
├── src/
│   ├── dataset_builder.py    # ingestion, standardization, metadata pairing
│   ├── sort_data.py          # local corpus build + metadata injection
│   ├── extract_frames.sh     # ffmpeg frame extraction from source video
│   ├── environment.py        # Gymnasium env: episode state machine + reward
│   ├── prompt_templates.py   # system grounding, tool specs, output format
│   ├── utils.py              # 3×3 crop math + tolerant action/thought parsers
│   └── trace_logger.py       # per-episode reasoning-trace logging
├── train_grpo.py             # multi-turn GRPO rollout + LoRA fine-tuning (TRL)
├── evaluate.py               # base VLM vs RL policy benchmark (+ McNemar test)
├── visualize.py              # Gradio reasoning-trace replay viewer
├── ARCHITECTURE.md           # full design document
└── requirements.txt          # pinned, mutually-compatible dependency set
```

---

## Setup

Requires **Python 3.12** (pins verified on macOS / Apple Silicon). `ffmpeg` is
needed only for frame extraction from source video (`brew install ffmpeg`).

```bash
pip install -r requirements.txt
```

A CUDA GPU is effectively required to actually *train*; data prep, the
environment smoke test, and a training dry run all run CPU-only. The default
policy model is the small `HuggingFaceTB/SmolVLM-500M-Instruct` (trainable on
modest hardware); swap in `Qwen/Qwen2.5-VL-3B-Instruct` or larger on a CUDA box
via `--model`. Fine-tuning uses **LoRA**.

---

## Quickstart

```bash
# 1. Build the standardized image corpus + metadata manifest
python src/dataset_builder.py

# 2. Smoke-test the environment with a scripted policy (prints transitions)
python src/environment.py

# 3. Run the GRPO fine-tuning loop
python train_grpo.py

# 4. Benchmark base VLM vs the RL-fine-tuned policy
python evaluate.py

# 5. (optional) Replay episode reasoning traces in a browser
python visualize.py
```

---

## Training & evaluation

- **Algorithm:** GRPO via TRL's `GRPOTrainer`. A custom `rollout_func` makes each
  GRPO "completion" a full multi-turn episode; the dataset row is just a seed
  carrying the manifest index, and the per-episode scalar return is fed straight
  through as the reward. GRPO normalizes returns *within each group of rollouts
  of the same image*, so reward **spread across rollouts** matters more than
  absolute magnitude.
- **Policy:** SmolVLM-500M (default) + LoRA adapter; precision follows the device
  (bf16 on CUDA, fp16 on MPS, fp32 on CPU).
- **Evaluation:** `evaluate.py` runs the untrained base model and the fine-tuned
  policy through the *same* environment and compares accuracy, with a McNemar
  test for paired significance (degrades gracefully if SciPy is absent).

---

## Known limitations

- **Reward design is not finalized** — `RewardConfig` is a tunable hook. The
  reasoning-presence reward is live but untested in training and its weight needs
  tuning against the correctness scale; the judged-reasoning reward is stubbed
  behind a flag and needs a real VLM judge wired in.
- **Synthetic metadata leakage** — manifest metadata can be label-correlated, so
  the `METADATA` tool is a potential shortcut; mitigated by both a data-side
  informativeness knob and the reward-side metadata-shortcut penalty.
- **Multi-turn token masking** — the masking convention for interleaved
  tool-result tokens must be validated against the installed TRL version.

See [`ARCHITECTURE.md` §9](ARCHITECTURE.md) for details.
