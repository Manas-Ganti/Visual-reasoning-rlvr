# Architecture

VLM RL Forgery Detection trains a Vision-Language Model (VLM) to distinguish
**real photographs** from **AI-generated images** using *active perception*:
instead of classifying a single image in one shot, the model investigates over
several turns — zooming into regions, requesting forensic metadata — and then
commits to a verdict with written reasoning. The policy is fine-tuned with
**GRPO** (Group Relative Policy Optimization) so that correct, efficient
investigations are reinforced.

---

## 1. System overview

```mermaid
flowchart TD
    subgraph DATA["Data layer (offline, one-time)"]
        SORA["raw_data/OpenAI_Sora/*.mp4"] -->|extract_frames.sh<br/>ffmpeg 1 fps| FAKE["raw_data/fake_images"]
        REAL["raw_data/real_images"] --> SORT
        FAKE --> SORT["sort_data.py<br/>standardize + inject metadata"]
        HUB["HF Hub: artifacts dataset"] -.->|dataset_builder.py<br/>(streaming alt.)| SORT
        SORT --> IMGS["processed_images/*.png"]
        SORT --> MANIFEST["metadata.jsonl<br/>(catalog)"]
    end

    subgraph ENV["Environment (src/)"]
        MANIFEST --> GENV["ForgeryDetectionEnv<br/>(Gymnasium)"]
        IMGS --> GENV
        PROMPTS["prompt_templates.py"] --> GENV
        UTILS["utils.py<br/>crop math + parsers"] --> GENV
    end

    subgraph RL["RL loop"]
        GENV <-->|"messages+images / action text"| ROLLOUT["rollout_func<br/>(multi-turn episode)"]
        ROLLOUT --> TRAINER["TRL GRPOTrainer"]
        TRAINER -->|LoRA update| POLICY["SmolVLM-500M<br/>+ LoRA adapter"]
        POLICY --> ROLLOUT
    end

    POLICY --> EVAL["evaluate.py<br/>base vs policy + McNemar"]
    GENV --> EVAL
```

The pipeline has three stages: an **offline data layer** that produces a
standardized image corpus + manifest, an **environment** that turns each image
into an interactive episode, and an **RL loop** that rolls out episodes and
updates the policy. Evaluation reuses the same environment to benchmark the
trained policy against the untrained base model.

---

## 2. Repository layout

```text
vlm-rl-forgery-detection/
├── data/
│   ├── processed_images/    # standardized lossless RGB PNGs
│   ├── raw_data/            # source videos/images (Sora frames, real photos)
│   └── metadata.jsonl       # dataset catalog (one JSON record per image)
├── src/
│   ├── __init__.py          # exports ForgeryDetectionEnv, RewardConfig
│   ├── dataset_builder.py   # streaming ingestion from HF Hub (alternative source)
│   ├── sort_data.py         # local ingestion: standardize pixels + inject metadata
│   ├── extract_frames.sh    # ffmpeg: video -> frames (AI "images" from Sora clips)
│   ├── environment.py       # Gymnasium env: episode state machine + reward
│   ├── prompt_templates.py  # system prompt, tool descriptions, result templates
│   └── utils.py             # 3x3 crop geometry + action/verdict parsers
├── train_grpo.py            # GRPO fine-tuning via TRL (rollout_func drives the env)
└── evaluate.py              # base-vs-policy benchmark with statistical comparison
```

---

## 3. Data layer

### 3.1 Sources & ingestion

- **`extract_frames.sh`** — runs `ffmpeg -vf fps=1` over OpenAI Sora `.mp4`
  clips, emitting one PNG frame per second into `data/raw_data/fake_images/`.
  These frames serve as AI-generated examples.
- **`sort_data.py`** — the primary local pipeline. For each raw real/fake image
  it standardizes the pixels and writes a manifest record:
  - converts to strict **RGB** (drops alpha/grayscale edge cases),
  - resizes preserving aspect ratio so the long edge ≤ **1024 px**,
  - saves as **lossless PNG** (`compress_level=0`),
  - balances classes up to a target count,
  - injects **simulated forensic metadata** (see §3.3).
- **`dataset_builder.py`** — an alternative streaming source that pulls a
  balanced real/fake split directly from the Hugging Face Hub
  (`polytechnique-montreal/artifacts`) with near-zero local disk use. Useful for
  quick experiments without a local corpus.

### 3.2 Manifest schema (`data/metadata.jsonl`)

One JSON object per line:

```json
{
  "id": "sample_0000",
  "file_name": "/abs/path/to/data/processed_images/real_0000.png",
  "label": 0,
  "label_text": "Real",
  "metadata": {
    "software_sig": "Camera Internal Firmware",
    "exif_profile": {"Make": "Canon", "Model": "EOS R5", "FocalLength": "50mm"},
    "color_space": "Display P3",
    "compression_ratio": "Standard Variable Bitrate Camera Matrix"
  }
}
```

`label`: `0` = Real, `1` = AI. The environment loads images by absolute
`file_name` and exposes `metadata` only when the agent requests it.

### 3.3 Synthetic metadata (tunable predictiveness)

Metadata is **fabricated**, not extracted from files, so how strongly it
correlates with the label is a free parameter. `sort_data.py` exposes a
`metadata_informativeness` knob (0..1) via `simulate_metadata()`:

| Value | Metadata-only accuracy | Meaning |
|-------|------------------------|---------|
| `0.0` | ~50% | pure noise — independent of the label, forces image-only reasoning |
| `0.5` (default) | ~75% | weak, realistic forensic clue |
| `1.0` | 100% | perfectly diagnostic (the original leaky behavior) |

**Mechanism.** There is a single informative axis — *does the file carry camera
provenance or not*. Real photos usually do, AI images usually don't; with
probability `1 - p_consistent` that is flipped (a stripped real photo, or an AI
image with spoofed/inherited EXIF), where `p_consistent = 0.5 + 0.5 ·
informativeness`. All label correlation flows through this single rate, so at
`0.0` the two classes' metadata distributions are identical. Crucially, both
classes draw every field (`software_sig`, `exif_profile`, `color_space`,
`compression_ratio`) from the **same vocabularies** — there is no structural
tell (e.g. a named generator string) that lets a policy bypass the knob.

This is the **data-side** lever against the metadata shortcut; it pairs with the
**reward-side** `metadata_correct_penalty` (§5). Changing the knob requires
rebuilding the manifest (`python src/sort_data.py`).

---

## 4. Environment (`src/environment.py`)

`ForgeryDetectionEnv` subclasses `gymnasium.Env` and implements a multi-turn,
multimodal episode.

### 4.1 Episode lifecycle

```
reset(index?) ─► full image + system prompt + initial user turn
   │
   ▼  loop (≤ max_steps turns)
step(action_text):
   ├─ append model's turn to the conversation
   ├─ parse action  (utils.parse_action)
   ├─ ZOOM n     → crop+upscale cell n, append image+result to conversation
   ├─ METADATA   → reveal manifest metadata for this image
   ├─ ANSWER v   → score verdict vs ground truth, TERMINATE
   └─ INVALID    → corrective feedback, turn consumed
   │
   ▼
terminated (answered) | truncated (turn budget exhausted)
```

### 4.2 Observation & action contract

This is a conversational, multimodal environment, so the rich payload does not
fit Gymnasium's `spaces` primitives. Each **observation** is a dict:

```python
{
  "messages":  [ {role, content:[{type:"image"} | {type:"text", text}]} ... ],
  "images":    [PIL.Image, ...],   # in placeholder order
  "step":      int,
  "max_steps": int,
}
```

`messages` + `images` are exactly what a HuggingFace VLM processor consumes. The
declared `observation_space` (a `Dict` of the scalar bookkeeping fields) is
informational; the dict above is the real contract.

The **action** is the model's full text completion for the turn
(`action_space = spaces.Text`). The agent must emit a `THOUGHT` line followed by
exactly one `ACTION` line — see §6.

### 4.3 Configuration

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `max_steps` | 6 | investigation turns before a forced decision |
| `grid` | 3 | 3×3 inspection grid |
| `upscale_to` | 512 | long-edge size a cropped cell is upscaled to ("zoom") |
| `shuffle` | True | random vs sequential sample order |
| `reward_config` | `RewardConfig()` | reward shaping hook (see §5) |

---

## 5. Reward structure (placeholder — not yet finalized)

> The reward design is **intentionally generic for now.** What follows describes
> the *mechanism* and the signals available; the specific shaping and weights
> are open and expected to change.

Reward is computed entirely inside `ForgeryDetectionEnv.step` and is
parameterized by a `RewardConfig` dataclass, so the scheme can be tuned without
touching the rollout or training code. The environment combines these signals:

- **Terminal correctness** — whether the final `ANSWER` matched ground truth.
- **Investigation cost** — a per-action step cost (efficiency signal).
- **Action validity** — penalty for a malformed / unparseable action.
- **Redundancy** — penalty for re-inspecting an already-zoomed cell.
- **Budget exhaustion** — penalty for ending without a verdict.
- **Metadata-shortcut penalty** — a penalty applied to a *correct* verdict that
  was reached *after using `METADATA`* (see below).
- **Reasoning-presence reward (Option A)** — a small per-turn bonus for a
  substantive, non-boilerplate `THOUGHT` (see below).
- **Judged reasoning quality (Option C, stubbed)** — an optional, correctness-
  gated bonus from a separate reasoning judge (off by default; see below).

### Current `RewardConfig` defaults (provisional)

| Field | Value | Fires when |
|-------|-------|-----------|
| `correct` | +1.0 | final `ANSWER` is right |
| `incorrect` | −1.0 | final `ANSWER` is wrong |
| `step_cost` | −0.05 | every non-terminal action |
| `invalid_penalty` | −0.10 | malformed action (+ step_cost) |
| `repeated_zoom_penalty` | −0.10 | re-zoom a viewed cell (+ step_cost) |
| `no_answer_penalty` | −1.0 | turn budget exhausted, no verdict |
| `format_bonus` | 0.0 | optional, per well-formed action (off) |
| `metadata_correct_penalty` | −0.5 | correct verdict **and** metadata was used |
| `reasoning_bonus` | +0.05 | well-formed turn carries a substantive `THOUGHT` |
| `reasoning_min_words` | 4 | min words for a `THOUGHT` to count (not a reward) |
| `use_reasoning_judge` | `False` | master switch for the Option-C judge (off) |
| `reasoning_judge_weight` | 0.2 | scales the judge's [0,1] score when enabled |

### Metadata-shortcut penalty (anti-leakage)

Because the simulated metadata is strongly label-correlated (§3.3), a policy
could win by reading `METADATA` and ignoring the pixels. `metadata_correct_penalty`
counteracts this: it only docks a verdict that is **both correct and
metadata-assisted**, producing the ordering

```
image-only correct (+0.95)  >  metadata-assisted correct (+0.40)  >  any wrong (-1.05)
```

So pixel-only reasoning out-scores the shortcut, yet a metadata-assisted correct
answer stays net-positive (the model isn't pushed to answer *wrong*). Wrong
answers are not additionally penalized for using metadata. Set the field to
`0.0` to disable, or larger (≈ −0.9) to make the shortcut roughly neutral.

### Rewarding the reasoning channel

Under outcome-only reward, the `THOUGHT` tokens still receive credit — GRPO
back-propagates the episode return over *every* generated token, so reasoning
that reliably leads to correct verdicts is reinforced indirectly. The gaps that
direct shaping addresses are (a) no floor keeping a `THOUGHT` present at all, and
(b) no pressure for the reasoning to be *faithful* (the model can confabulate and
still collect full reward on a lucky-but-right verdict). Two layered mechanisms:

**Option A — reasoning-presence floor (live).** Each well-formed (`!= INVALID`)
turn earns `reasoning_bonus` when its `THOUGHT` clears `utils.is_substantive_thought`:
at least `reasoning_min_words` words and not a verbatim repeat of the previous
turn's thought (a cheap anti-boilerplate guard). This rewards *having* a reason,
not its correctness, and keeps the channel from collapsing. Caveat: at +0.05 over
a 6-turn budget it can contribute up to ~0.3 — non-trivial against `correct = 1.0`
— so if it begins to dominate or encourages stalling to farm the bonus, lower it
toward +0.02.

**Option C — judged reasoning quality (stubbed, off by default).** When
`use_reasoning_judge` is set and a judge callable is supplied to
`ForgeryDetectionEnv(reasoning_judge=…)`, the final-answer `THOUGHT` is scored in
`[0, 1]` (grounded in the image / coherent / entails the verdict) and added as
`reasoning_judge_weight × score`. It fires **only on a correct verdict**, so it can
break ties between right answers but can never reward an eloquent *wrong* one. The
score is clamped to `[0, 1]` so a miscalibrated judge can't blow up the advantage
scale. Without a judge callable, `_judge_reasoning` is a no-op returning `0.0`, so
flipping the flag alone changes nothing. The judge runs inside the rollout hot
loop (one call per terminated episode), so a real implementation should be
batched/cached at the call site.

GRPO consumes a **single scalar return per episode** (the sum of per-step
rewards), then normalizes it within each group of sampled rollouts to form
advantages — so absolute reward magnitudes matter less than their *spread across
rollouts of the same image*. These weights remain provisional; when tuning,
update `RewardConfig` and this section together.

---

## 6. Agent interaction protocol

Defined in `prompt_templates.py`, parsed in `utils.py`. Every turn the model
must output:

```
THOUGHT: <one or two sentences of reasoning>
ACTION: <ZOOM n | METADATA | ANSWER AI | ANSWER REAL>
```

| Action | Effect |
|--------|--------|
| `ZOOM <1-9>` | inspect a grid cell (row-major) at higher resolution |
| `METADATA` | reveal the image's forensic metadata report |
| `ANSWER <AI\|REAL>` | commit a verdict and end the episode |

The parser (`utils.parse_action`) is deliberately tolerant: case-insensitive,
accepts `ZOOM(5)`, honors only the **last** `ACTION` line (so a model can narrate
a plan first), and normalizes verdict synonyms (`fake`/`synthetic` → AI,
`authentic`/`photo` → REAL). Unparseable turns become `INVALID` and the
environment returns corrective feedback instead of crashing the rollout. The
`THOUGHT` line is mandatory by prompt instruction and is extracted every turn
(`utils.extract_thought`); the reasoning-presence reward (§5) backs that
instruction with a reward signal so it isn't prompt-only.

`utils.py` also owns the crop geometry: `grid_cell_bbox` computes pixel boxes
with rounding so edge cells reach the true image border, and `crop_grid_cell`
upscales the crop (LANCZOS) to surface fine artifacts.

---

## 7. RL training & evaluation

### 7.1 Training (`train_grpo.py`)

- **Algorithm:** GRPO via TRL `GRPOTrainer`.
- **Policy model:** `HuggingFaceTB/SmolVLM-500M-Instruct` (default — small enough
  to train on Apple Silicon / modest disk); swap for `Qwen/Qwen2.5-VL-3B-Instruct`
  or larger on a CUDA box via `--model`. Fine-tuned with **LoRA** (`r=16`,
  `alpha=32`, targeting `q_proj`/`v_proj`). Precision follows the device: bf16 on
  CUDA, fp16 on MPS, fp32 on CPU (see `--device` / `--no-bf16`).
- **Custom multi-turn rollout:** a `rollout_func` makes each GRPO "completion" a
  full episode. The TRL dataset row is just a **seed** carrying the manifest
  index (`index=<i>`); the rollout parses it, resets the env to that sample, and
  runs the real image-grounded conversation. This decouples TRL's
  one-row-per-prompt model from multi-turn, multi-image rollouts.
- **Group sampling:** `num_generations` (GRPO group size G) makes TRL repeat each
  seed G times; rollouts sample with temperature so each repeat is a distinct
  trajectory, giving GRPO the within-group reward spread it needs.
- **Token bookkeeping:** the policy's generated tokens (concatenated across
  turns) form the completion; env tool-result tokens are part of the running
  context but excluded from the gradient. The exact masking convention is
  TRL-version-dependent and flagged in-code as a validation point (§8).
- **Dry run:** `python train_grpo.py --dry-run` exercises env + dataset wiring
  with no model and no `trl` installed.

### 7.2 Evaluation (`evaluate.py`)

Self-contained (no `trl`). Runs **greedy** multi-turn episodes over the held-out
manifest tail for the base model and, if `--adapter` is given, the trained
policy — over the **same** items. Reports:

- accuracy, precision, recall, F1 (AI = positive class),
- mean investigation steps and metadata-usage rate,
- a **McNemar test** on paired correct/incorrect outcomes for significance
  (uses `scipy` if available; otherwise reports the discordant counts).

The train/eval split is shared via `HOLDOUT_FRACTION` (default 0.1, the manifest
tail), so evaluation never touches a trained-on sample.

---

## 8. Frameworks & dependencies

| Framework | Role | Notes |
|-----------|------|-------|
| **Gymnasium** | environment API (`reset`/`step`, spaces) | installed |
| **Pillow (PIL)** | image I/O, cropping, resizing | installed |
| **NumPy** | RNG / sample selection | installed |
| **PyTorch** | model execution, generation | installed |
| **Transformers** | VLM + processor (`AutoModelForImageTextToText`, `AutoProcessor`) | installed |
| **Datasets** | seed dataset; HF Hub streaming | installed |
| **TRL** | GRPO trainer + custom `rollout_func` | **required for training, not yet installed** |
| **PEFT** | LoRA adapters | **required for training, not yet installed** |
| **Accelerate** | distributed / device orchestration | **required for training, not yet installed** |
| **SciPy** | McNemar p-value in evaluation | optional |
| **ffmpeg** | frame extraction (`extract_frames.sh`) | external CLI |

Install the training stack with:

```bash
pip install "trl>=0.29" peft accelerate
```

A CUDA GPU is effectively required to train; data prep, the env smoke test, and
the training dry run all run CPU-only.

---

## 9. Known limitations / open work

1. **Reward design is not finalized** — `RewardConfig` is a placeholder hook
   (§5). The reasoning-presence reward (Option A) is live but untested in
   training; its weight needs tuning against the correctness scale. The judged-
   reasoning reward (Option C) is stubbed behind `use_reasoning_judge` and needs
   a real VLM judge (batched in the rollout loop) before it does anything.
2. **Synthetic metadata leakage** — manifest metadata can be label-correlated, so
   the `METADATA` tool is a potential shortcut. Now addressed by two levers: the
   data-side `metadata_informativeness` knob (§3.3) and the reward-side
   `metadata_correct_penalty` (§5). Open work: pick final settings for both, and
   note the default-built `data/metadata.jsonl` only reflects the knob after a
   rebuild (`python src/sort_data.py`).
3. **Multi-turn token masking** — the gradient/masking convention for
   interleaved tool-result tokens must be validated against the installed TRL
   version (cross-reference TRL's openenv / browsergym GRPO examples).

---

## 10. Common commands

```bash
# Data prep
python src/sort_data.py                 # standardize + build manifest (local)
bash  src/extract_frames.sh             # Sora videos -> frames

# Environment
python src/environment.py               # scripted-policy smoke test

# Training
python train_grpo.py --dry-run          # verify wiring (no GPU / no trl)
python train_grpo.py                     # GRPO fine-tune (needs trl + GPU)

# Evaluation
python evaluate.py --limit 50                          # base model only
python evaluate.py --adapter checkpoints/grpo-...      # base vs RL policy
```
