# Negative result — the StyleGAN2 face substrate is not learnable by this model

**Date:** 2026-08-16 · **Model:** Qwen2.5-VL-32B-Instruct ·
**Dataset:** `hamzaboulahia/hardfakevsrealfaces` (1,288 images, 300×300)

The full SFT→GRPO pipeline was built and run against this substrate. It does not
work, and the reason is not the environment, the reward, or the training — it is
that **the base model cannot distinguish these two classes at all.** GRPO was
stopped before the 24-hour run on this evidence.

## The measurement that settled it

A "ceiling probe" (`tools/ceiling_probe.py`) strips away everything the
environment withholds: native resolution, one question, one word, no budget, no
format requirement. Best possible conditions.

```
[full] 100 test images — {'REAL': 43, 'AI': 57}; majority baseline 0.570

n=100  accuracy=0.390  answer_rate=1.000
  truth AI   {'AI': 0,  'REAL': 57, None: 0}
  truth REAL {'AI': 4,  'REAL': 39, None: 0}
  predicted AI on 4/100 answered (4.0%)
```

**Recall on the AI class is 0.00.** Of 57 StyleGAN2 faces, it identified zero.
All four of its "AI" answers were false positives on real photographs. It
answered every prompt (`answer_rate=1.0`), so this is not a formatting or
truncation artifact — the model's "AI" signal is noise.

Accuracy 0.390 is below both random guessing (0.50) and the majority-class
baseline (0.570).

## Why the environment could not have rescued it

Two structural problems, both invisible until measured:

**1. There was no evidence to reveal.** Images are 300×300, so a 4×4 cell is
**75×75 real pixels**. `INSPECT` upscaled that crop to 336px — an enlarged blur,
not new information. The core mechanic was returning nothing, by construction.

**2. StyleGAN2 artifacts are high-frequency.** Iris structure, pore texture, hair
strands, tooth blending. Downsampling 1024×1024 generations to 300×300 destroys
exactly the detail that makes them detectable. The dataset is named *hard* for
this reason.

## What the pipeline reported before the probe

Every stage "worked" and every number was misleading, which is the lesson worth
keeping:

| Stage | Result | What it actually meant |
|---|---|---|
| Distill | 263/1033 traces (25.5% keep) | teacher *with the label revealed* still failed 42% of the time |
| SFT | loss 1.69→1.42, 18 steps | learned the format; learned nothing about detection |
| Eval (policy) | 0.280 accuracy | below the 0.547 baseline |
| Eval (base) | 0.320 accuracy | statistically identical — McNemar p=0.68 |
| Calibration | mean conf 0.97, acc 0.25 | overconfidence inherited from the label-revealed teacher |
| Confusion matrix | 6 AI predictions in 151 answered | **total class collapse — the actual finding** |

The accuracy numbers alone read as "training didn't help." The confusion matrix
read as "the model has no signal." Only the second is actionable, and it was
never printed by the harness — it had to be recovered from the trace log.

## Three bugs found on the way, all real, none the cause

Worth recording because each independently produced a plausible-looking wrong
number (details in [`../SLURMandjobscheduler.md`](../SLURMandjobscheduler.md)):

1. **Eval truncated completions at 320 tokens** — 52 of 100 episodes never
   reached their `ACTION:` line and scored as failures. Same bug held distillation
   at 13% keep rate before `--max-new-tokens 640` doubled it.
2. **vLLM silently drops vision-tower LoRA** — `visual.* will be ignored`, so the
   adapter was half-loaded at eval.
3. **peft keys are transformers-version-specific** — an adapter trained under
   transformers ≥4.56 (`model.language_model.layers`) loads *zero* weights under
   4.51 (`model.layers`), with a warning rather than an error. Two eval runs
   measured a pristine base model while reporting on the "policy."

## What carries forward

The environment, the verifiable reward, the trajectory parser, the distillation
→ SFT → GRPO → eval pipeline, and the SLURM tooling are all substrate-agnostic
and all work. What failed is the choice of images.

Requirements for a replacement, derived from the two structural problems above:

- **≥1024px native** so a 4×4 cell carries ≥256 real pixels
- **Ceiling ≥0.85** at full resolution — the model must be able to know
- **Floor ≈chance** from the overview — otherwise investigation buys nothing
- **Localized, semantic artifacts** so *which cell* is inspected matters

Successor: **GenImage** (`data/build_manifest_genimage.py`). Diffusion and GAN
artifacts there are semantic and spatially concentrated — malformed hands,
garbled text, impossible geometry — the kind VLMs demonstrably detect and that
survive downsampling.

## The process lesson

**Measure the ceiling before building the pipeline.** This probe is 15 minutes on
2 GPUs. Running it first would have redirected the substrate choice before any
distillation, SFT, or eval work — and it is now the first step in the
dataset-selection workflow, with `--condition overview` added to measure the
floor as well.
