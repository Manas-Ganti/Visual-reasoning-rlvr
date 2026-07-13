# visual-reasoning-rlvr

**A verifiable RL environment and full post-training pipeline for investigative
visual reasoning.**

An agentic RL environment where a VLM investigates an image under a
resolution/action budget — forming testable hypotheses, inspecting regions to
confirm or refute them, and committing a verdict — trained so that *reasoning is
the only efficient path to reward*. The task: decide whether a face is a real
photograph or AI-generated (StyleGAN2). The point: a reward that is **mechanically
verifiable end to end — no LLM judge anywhere.**

> The interesting artifact here is the **reward**, not the detector. See
> [`results/reward_failure_history.md`](results/reward_failure_history.md) for the
> four iterations it took to get a signal that is both *faithful* and *learnable*.

---

## The core idea

The environment makes the correct answer **unreachable without reasoning**. The
agent starts with only a low-resolution overview (partial observability), must
spend a limited budget to sharpen regions, and must commit a **falsifiable
hypothesis before each reveal**. A correct final verdict is therefore evidence
that genuine investigation occurred. Reasoning is forced *structurally* — never
scored for eloquence.

### Action space (two actions, locked)

| Action | Effect |
|---|---|
| `INSPECT <n>` | reveal grid cell *n* (1–16, a 4×4 grid) at high resolution — the only information-acquisition action; costs budget |
| `VERDICT <AI\|REAL> confidence=<c>` | commit and end the episode |

### Predict-then-verify (the locked invariant)

Every turn is one structured block. The **hypothesis is committed before the
reveal**; the reveal is reconciled against it on the next turn. Predict → observe,
never observe → narrate.

```
RECONCILIATION: CONFIRMED/REFUTED — did the last reveal match my hypothesis?   (post)
BELIEF_UPDATE:  P(fake)=0.75 because …                                          (post)
OBSERVATION:    what I perceive at current resolution                           (pre)
REASONING:      why this region matters / my uncertainty                        (pre)
HYPOTHESIS:     if AI, the left iris in cell 6 will be malformed                (pre)
ACTION:         INSPECT 6   |   VERDICT AI confidence=0.85
```

## The verifiable reward

Every term is a mechanical function of the trajectory + ground-truth label
(`env/reward.py`, unit-tested in `tests/`):

```
R = +1.00·verdict_correct        final label vs ground truth
    +0.30·belief_coherence       P(fake) moved as the agent's own reconciliations imply
    +0.30·verdict_consistency    the final call follows the accumulated evidence
    +0.10·prediction_tracking    (soft) fraction of hypotheses confirmed
    −0.05·per_inspect            budget pressure → hypothesis-driven, not exhaustive
    −0.50·confident_wrong·conf    calibration pressure → hedge when indistinguishable
    −1.00·no_answer
```

`verdict_correct` dominates so process credit can never rescue a confidently-wrong
episode. The reward reads only numeric beliefs and reconciliation *direction* —
never prose — so boilerplate reasoning earns nothing. Full rationale + the
reward-hacking surface: [`results/reward_failure_history.md`](results/reward_failure_history.md).

## Difficulty (two honest axes)

Single-generator data (StyleGAN2) gives no generator tiers, so difficulty is
manufactured two ways: **image degradation** (`clean → jpeg → blur_downscale`) and
**budget tightness** (fewer allowed inspects). The eval harness reports pass rate
across both — the calibration deliverable: *"to target model X at ~50% pass, use
degradation Y at budget k."*

## Pipeline

| Stage | What | Script |
|---|---|---|
| 0 | Baseline eval (zero-shot pass rate per degradation) | `eval/harness.py` |
| 1 | **SFT** — teach the pre/post format from distilled traces | `training/sft.py` |
| 2 | **GRPO** — online RL vs the verifiable reward, KL-anchored to SFT | `training/grpo.py` |
| 3 | **Verification** — prove the gains are real (below) | `eval/harness.py` |

Base model **Qwen2-VL-7B-Instruct** (fallback 2B). Inference via **vLLM**
(eval + trace distillation); training rollouts via HF `generate` (needs logprobs).
All stages logged to **W&B**. DPO was cut — data-starved at this scale;
SFT→GRPO suffices.

### Headline result (populate after runs)

Pass rate per stage × degradation, `test` split:

| Stage | clean | jpeg | blur_downscale |
|---|---|---|---|
| Baseline | – | – | – |
| SFT | – | – | – |
| GRPO | – | – | – |

### Stage 3 — proving the learning is real

Held-out degradation · trajectory audit (`demo/app.py`) · grounding-term ablation
· adversarial-trajectory probe (unit-tested) · calibration curve · evidence slice
(GradCAM-proposed, human-verified fake cells; eval-only) — do RL rollouts inspect
true-artifact cells more than SFT? Indistinguishable images are scoped out of that
metric but still counted in accuracy.

---

## Quickstart

```bash
pip install -r requirements.txt            # CORE profile is CPU-only

# --- runs anywhere (no GPU) ---
pytest tests/                              # verifiable-reward + trajectory + env tests
python -m env.environment                  # scripted-policy smoke test (+ reward breakdown)

# --- data (needs Kaggle creds) ---
python data/build_manifest.py              # → data/manifest.jsonl (+ data/images/)

# --- A100 ---
python data/build_sft_traces.py --limit 800   # distill Stage-1 traces from base Qwen2-VL
python training/sft.py                          # Stage 1
python training/grpo.py --sft-checkpoint checkpoints/sft-qwen2vl   # Stage 2
python eval/harness.py --adapter checkpoints/grpo-qwen2vl \
    --budgets 2,4 --degradations clean,jpeg,blur_downscale --compare-base

# --- demo ---
python demo/app.py --log logs/grpo_episodes.jsonl
```

## Repository

```
env/         environment.py · reward.py · trajectory.py · grid.py · prompts.py · trace_logger.py
data/        build_manifest.py · degradation.py · build_sft_traces.py · build_evidence_slice.py · curation.md
training/    common.py · sft.py · grpo.py
eval/        harness.py               (pass-rate × degradation × budget, calibration, evidence slice)
tests/       test_reward.py · test_trajectory.py · test_environment.py   (CI target)
demo/        app.py                   (Gradio step-by-step trajectory viewer)
results/     reward_failure_history.md · curves/tables
Dockerfile · .github/workflows/ci.yml
```

Substrate: [Fake-Vs-Real-Faces (Hard)](https://www.kaggle.com/datasets/hamzaboulahia/hardfakevsrealfaces)
— 1,288 300×300 images (700 StyleGAN2 fakes, 589 real), image-level labels only.
See [`data/curation.md`](data/curation.md).
