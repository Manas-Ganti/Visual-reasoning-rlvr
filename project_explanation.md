# Project Explanation — what each stage does and why

A walkthrough of the post-training pipeline: what each stage is *for*, the
mechanism it relies on, and what actually happened on the first full run
(Qwen2.5-VL-32B on VT ARC A100-80s, August 2026).

For the environment and reward design see [`README.md`](README.md); for the
cluster/scheduler mechanics see
[`SLURMandjobscheduler.md`](SLURMandjobscheduler.md).

---

## The task, in one paragraph

A VLM must decide whether a face image is a real photograph or StyleGAN2-
generated — but it never gets a clear look. It starts on a deliberately
unresolvable low-resolution **overview**, and the only information-acquisition
action is `INSPECT <n>`, which sharpens one cell of a 4×4 grid at the cost of a
limited budget. Before each reveal it must commit a **falsifiable hypothesis**;
after the reveal it must **reconcile** what it saw against that hypothesis and
restate `P(fake)`. Then `VERDICT <AI|REAL> confidence=<c>` ends the episode.

The reward is computed mechanically from structured fields — the numeric
`P(fake)`, the reconciliation direction, the verdict, the confidence — with **no
LLM judge anywhere**. Prose is never scored. That is what makes the signal
verifiable, and it is why reasoning is the only efficient path to reward.

> **Status (2026-08-16).** The original StyleGAN2 face substrate was retired
> after a ceiling probe showed the base model scores 0.390 against a 0.570
> majority baseline and identifies **0 of 57 fakes** at full resolution — see
> [`results/faces_negative_result.md`](results/faces_negative_result.md). The
> pipeline below is unchanged and substrate-agnostic; it is being re-run on
> **GenImage**. Stage results quoted here are from the faces run unless noted.

## Artifact layout (dataset-namespaced)

Every artifact is namespaced by dataset, set once via `VRR_DATASET` (default
`genimage`), so substrates cannot overwrite each other and a stale file cannot
silently feed the next stage:

```
data/<ds>/{manifest,sft_traces}.jsonl · data/<ds>/images/
checkpoints/<ds>/{sft,grpo}-<model-tag>/
logs/<ds>/{eval,grpo,distill}_episodes.jsonl · logs/<ds>/runs.tsv
results/<ds>/
logs/slurm/<stage>-<jobid>.{out,err}      # unique by job id; runs.tsv is the index
```

`logs/<ds>/runs.tsv` records every stage launch — timestamp, dataset, stage, job
id, node, and key arguments — which is what makes a directory full of
`infer-71xxxxx.out` navigable later.

## Pipeline at a glance

```
Stage -1   build_manifest.py     data/manifest.jsonl + data/images/    1,290 images
              │                                                        1033 / 128 / 128
Stage  0   build_sft_traces.py   data/sft_traces.jsonl                 263 traces (25.5%)
              │                  rejection-sampled demonstrations
Stage  1   training/sft.py       checkpoints/sft-qwen2.5-vl-32b        566 MB LoRA
              │                  format compliance, learned unaided
   ── GATE ── eval/harness.py    answer rate · pass rate · calibration
              │                  can GRPO get a gradient from this?
Stage  2   training/grpo.py      checkpoints/grpo-qwen2.5-vl-32b       the actual RL
              │                  strategy, scored on outcomes
Stage  3   eval/harness.py       degradation × budget grid, McNemar,
                                 calibration, evidence slice
```

Each stage consumes exactly one artifact from the stage above, and **nothing
downstream re-validates it**. A thin trace file or a half-loaded adapter
propagates silently. That is the single most important structural fact about
this pipeline, and the reason the gate exists.

---

## Stage -1 — the dataset

`data/build_manifest.py` ingests `hamzaboulahia/hardfakevsrealfaces` (1,288
300×300 JPEGs: StyleGAN2 fakes from thispersondoesnotexist, real photos from
Unsplash), copies images into `data/images/`, and writes a minimal manifest:

```json
{"id": "face_00000", "file_name": "data/images/real/real_00000.png", "label": 0, "split": "val"}
```

`label` is 0=REAL / 1=AI. Splits are 1033 train / 128 val / 128 test, roughly
54% AI throughout — so **the majority-class baseline is 0.547**, which is the
number any accuracy claim has to beat.

Image-level labels only. There are no per-region artifact annotations anywhere
in the training path; region labels exist solely in the eval-only evidence
slice, and the reward never sees them.

---

## Stage 0 — distillation (rejection-sampled trace generation)

### The problem it solves

GRPO learns from *variance within a group*. For each image it samples `G=8`
episodes, scores them, and pushes the policy toward the above-average ones.

Hand the raw base model this environment cold and most completions won't parse:
it has never seen a blurry overview, a 4×4 cell numbering, or an output format
demanding `RECONCILIATION` / `BELIEF_UPDATE` / `HYPOTHESIS` / `ACTION` blocks in
a fixed order. Malformed episodes terminate as invalid and all score the same.

**When every rollout in a group scores identically, the advantage is zero, so
the gradient is zero.** GRPO doesn't learn slowly in that regime — it learns
nothing, and burns a day of 8 GPUs doing it. Something has to clear the
formatting bar first.

### The mechanism: privileged information

You cannot simply *ask* the base model for good investigations — that is the
capability being built. But you can remove the hard part: the answer.

`data/build_sft_traces.py:44` appends an extra user message after reset:

> INSTRUCTOR NOTE (not part of the record): the ground-truth label for this
> image is **AI**. Produce a *genuine* investigation that a careful analyst
> would run to reach AI: inspect the cells most likely to be decisive, commit a
> testable HYPOTHESIS before each inspect, and RECONCILE honestly after each
> reveal…

The model is no longer detecting anything. It is **narrating a plausible path to
a known destination** — a far easier generative task that a 32B handles well.

The hint is injected via `on_reset` and lives only in the conversation.
`harvest()` (line 65) extracts *only* the assistant turns, so the note never
reaches `sft_traces.jsonl`. What SFT trains on is indistinguishable from an
episode played blind.

This is why the stage is **context distillation** rather than model compression:
teacher and student are the same 32B, and what gets distilled is an advantage
that existed only in the prompt. Nothing is trained here — Stage 0 is pure data
generation.

### The rejection filter

Generation is cheap; the filter is what makes the data trustworthy
(`build_sft_traces.py:144`):

```python
if r["correct"] and r["well_formed"] and r["actions"]
```

- **`correct`** — final verdict matches ground truth. Even *with* the label
  revealed, only 600/1033 (58%) complied.
- **`well_formed`** — every turn parsed to a valid action. 264/1033 (25.6%).
- **`actions`** — non-empty.

`well_formed` is conjunctive across turns. At ~5 turns per episode, a ~76%
per-turn parse rate compounds to ~25% at episode level: one bad turn out of five
discards the whole thing.

The technique has a name — **rejection sampling fine-tuning**, and with
reasoning traces specifically, **STaR** (Self-Taught Reasoner). The rejection
step is the quality control: nothing enters the curriculum unless it reached the
right answer through a fully parseable investigation.

### What actually ran

Batched 32-wide through vLLM (`run_episodes_batched`), `max_turns =
max_inspects + 3`, temperature 0.4, sampling on, TP=2 on two A100-80s.

| `max_new_tokens` | well_formed | kept | keep rate |
|---|---|---|---|
| 320 (default) | — | 134 | 13.0% |
| 640 | 264/1033 | 263 | **25.5%** |

Doubling the token budget doubled the yield, which localizes the failure exactly:
completions were being **truncated before their `ACTION:` line**. With 264
well-formed and 263 kept, the `correct` filter costs essentially nothing on top —
**format was the entire bottleneck**. (The same truncation later corrupted the
eval; see the gate.)

### What the 263 survivors look like

- 4.32 turns, 3.32 inspects per episode
- Cell targeting: 11, 6, 14, 10 dominate — the center-face region (eyes, nose,
  mouth) where StyleGAN2 artifacts live. Not random looking-around.
- `P(fake)` stated numerically in 979 turns, mean 0.48 — beliefs are real
  numbers that move, not a collapsed prior.
- Reconciliation: 661 confirmed / 234 refuted / 242 unclear.

Two defects inherited directly from a teacher that knew the answer:

- **Confidence saturation** — 245/263 end at ≥0.95, 127 at exactly 1.0. An
  analyst who already knows the answer never hedges.
- **Budget exhaustion** — 145/263 use all four inspects; only 17 stop at one,
  despite the system prompt explicitly rewarding a correct verdict with *fewer*
  inspects.

Both are penalized by the reward, so SFT initializes GRPO at a policy that is
confidently wrong and maximally wasteful. Tolerable — GRPO exists to unlearn
exactly this — but it predicts the shape of the early reward curves: format
terms should start high, calibration and efficiency terms low and climbing.

### Why the keep rate matters more than it looks

263 examples is the **entire curriculum** for Stage 1, and GRPO can only amplify
what SFT established. It is the cheapest stage to improve (≈1 hour on 2 GPUs)
and the most expensive to get wrong (24h+ on 8 GPUs downstream). If revisited,
the highest-leverage change is not more images but `--max-new-tokens 1024`:
`correct=58%` is the ceiling, and format is what costs the gap.

---

## Stage 1 — SFT

### What it is for

Teach the model to produce the format **unaided**. The traces were generated
with the answer visible; SFT trains on them with the hint stripped, so the model
learns the investigative shape as an unconditional habit rather than a response
to being told.

SFT can only imitate. It will reproduce the good (hypothesis-before-reveal,
center-face targeting, moving beliefs) *and* the bad (0.97 confidence, budget
exhaustion) with equal fidelity. It cannot exceed its demonstrations — that is
Stage 2's job.

### Configuration

LoRA r=32, `alpha=2r`, target `all-linear`, 2 epochs, per-device batch 1 ×
grad-accum 8, lr 1e-5 cosine, bf16, gradient checkpointing, DeepSpeed **ZeRO-2**,
4× A100-80.

One ordering constraint is load-bearing (`training/sft.py:167`): **`SFTConfig`
must be constructed before the model.** Building `TrainingArguments` with a
DeepSpeed config installs the global `HfDeepSpeedConfig`, which is what lets
ZeRO-3 shard parameters *during* `from_pretrained` instead of materializing the
full model on every rank. Irrelevant at 32B/ZeRO-2; an instant OOM at 72B.

### What actually ran

263 traces ÷ (1 × 8 × 4 GPUs) × 2 epochs = **18 optimizer steps**, ~45 s/step,
13.5 minutes of training.

```
step  5/18   loss 1.691   mean_token_accuracy 0.681   entropy 0.775
step 10/18   loss 1.550   mean_token_accuracy 0.694   entropy 0.802
step 15/18   loss 1.417   mean_token_accuracy 0.707   entropy 0.825
final        train_loss 1.536                          entropy 0.852
```

Output: `checkpoints/sft-qwen2.5-vl-32b`, a 566 MB adapter.

Loss falls, token accuracy rises modestly, entropy rises slightly — consistent
with light format adaptation rather than memorization, which is what 18 steps on
263 examples should produce.

### Two things worth knowing

**It OOM'd on the first attempt at the save step, not during training.** All 18
steps completed, then `SFTTrainer.save_model()` gathers a full state dict on CPU;
peak RSS was 169 GB against a 200 GB cgroup limit. Host RAM, not VRAM — the GPUs
were never the constraint. Rerun with `--mem=600G` succeeded.

**The `all-linear` target reaches the vision tower.** `training/common.py:633`
claims the Qwen2.5-VL vision blocks use `fc1/fc2` names and therefore stay
frozen. That was true of Qwen2-VL; in Qwen2.5-VL the vision MLP uses
`gate_proj/up_proj/down_proj`, which **do** match the target list. The adapter
therefore includes vision-tower weights — which matters at eval, because vLLM
silently drops them (below). The docstring is stale and should be corrected.

---

## The gate — evaluating SFT before spending a day on GRPO

### Why it exists

GRPO is 24h+ on 8 GPUs with **no checkpoint-resume**, and it can only amplify
what SFT established. Two questions must be answered first:

1. **Does the policy produce well-formed episodes?** If half the rollouts fail
   to emit a parseable verdict, most groups score uniformly zero, the advantage
   collapses, and the run learns nothing. **This is the real gate.**
2. **Is it at least at chance (0.547 majority class)?** Below chance means
   something systematic is wrong, and RL would optimize on top of a broken
   signal.

Bad calibration is **not** a gate — it is precisely what GRPO is meant to fix.

### What the harness measures

1. **Pass rate × degradation × budget** — accuracy across clean / jpeg /
   blur_downscale and budgets 2 / 4. The *shape* is the deliverable: accuracy
   should fall as images degrade and budget shrinks. Flat means the model isn't
   using its inspects.
2. **Calibration curve** — bucket by stated confidence, measure accuracy per
   bucket. Does 0.9 mean right 90% of the time?
3. **Base vs policy** — the same items with the adapter enabled and disabled,
   plus **McNemar** on paired disagreements. Pairing matters: 4 base-only-correct
   vs 3 policy-only-correct is noise, not a regression.
4. **Evidence slice** (optional) — on human-verified fakes with known artifact
   cells, does the policy inspect true-artifact cells more than the base? The
   strongest available evidence that reasoning is real rather than lucky.

Also reported: precision / recall / F1, average inspects, and **answer rate** —
the fraction of episodes producing a parseable verdict. That last metric is the
gate, and watching it would have caught the bug below immediately.

### First run: two measurement bugs and one real finding

```
Policy pass rate (clean, budget 4): 0.120        Base: 0.130
McNemar: base-only=4  policy-only=3  p=1.0000    delta -0.010
Calibration: conf [0.90,1.01)  n=48  mean_conf=0.97  acc=0.250
```

**Bug 1 — eval truncated every completion at 320 tokens.** `eval/harness.py`
never passed `max_new_tokens`, so `run_episode` / `run_episodes_batched` fell
back to the backend default of 320 — the same truncation that held distillation
at 13%. `n=48` in the calibration table is the tell: **52 of 100 episodes never
produced a verdict** and were scored as failures. This applies equally to base
and policy, which is why both sat near 0.12. Fixed by threading a
`--max-new-tokens` flag (default 640) through `rollout_set`.

**Bug 2 — vLLM discards the vision half of the adapter.** vLLM only applies
LoRA to the language model of a multimodal model, emitting ~200 warnings
(`visual.blocks.N.mlp.gate_proj will be ignored`). Since Stage 1 trained vision
LoRA (above), the eval measured a **partially loaded adapter**, making
base-vs-policy not a fair comparison. Workarounds: evaluate with `--backend hf`
(applies the full adapter), or retrain with `--lora-target attn` so training and
inference agree.

**Real finding — overconfidence, exactly as predicted.** Mean stated confidence
0.97, accuracy 0.25 in that bucket. This is the distilled-trace artifact
(245/263 traces ended ≥0.95) surviving into the policy. It is the calibration
term GRPO must unlearn, now measured rather than hypothesized. Worth recording
in `results/reward_failure_history.md`.

**Still unexplained:** 25% accuracy among *answered* episodes is below chance on
a 57/43 slice. Truncation explains the missing 52, not why the answered ones
underperform random. If it persists after the fix, suspect something systematic
(verdict polarity, prompt) rather than task difficulty.

---

## Stage 2 — GRPO

### What it is for

The first and only stage that can **exceed** its demonstrations. SFT is scored
on resembling a transcript; GRPO is scored on outcomes, via the verifiable
reward. It is where investigation stops being an imitated format and becomes a
learned strategy — and where the inherited defects (0.97 confidence, always
spending the full budget) get penalized rather than reproduced.

### The mechanism

For each prompt, sample `G=8` full episodes. Score each with `env/reward.py`.
The group's mean is the baseline; each episode's advantage is its deviation from
it. No value network — the group *is* the baseline, which is what makes GRPO
practical for multi-turn agentic rollouts.

Implemented through TRL's `GRPOTrainer` with a custom `rollout_func`
(`training/grpo.py:302-309`), so one "completion" is an entire multi-turn
episode rather than a single response. Rollouts use HF `generate` (not vLLM)
because GRPO needs per-token logprobs.

### Two shape decisions that are not arbitrary

**ZeRO-2, not ZeRO-3.** Every rollout turn calls `generate()`. Under ZeRO-3 that
re-gathers sharded parameters *on every call*, and an episode is
`max_inspects + 1` sequential generates. ZeRO-2 replicates the full policy per
GPU and avoids the gather. Move to ZeRO-3 only if the policy genuinely does not
fit, and expect it to be slower.

**Wall clock is dominated by rollouts, not the optimizer step.** Decode is
HBM-bound: ZeRO-2 keeps the full 66 GB bf16 32B policy on each GPU, so every
token is a full weight sweep (~18 s/turn on A100-80, ~8 s on H200). One epoch
over the ~1k train split is one optimizer step per unique prompt. Estimate from
generate latency, never from training throughput. Calibrate with `--max-steps 5`
before committing an allocation.

### What to watch

- **Reward variance within groups.** If all 8 rollouts score identically, the
  advantage is zero and nothing is learning — the same failure the gate checks
  for, visible in the first few steps.
- **Term-by-term reward,** not just the total. Format terms should start high
  (SFT taught those); calibration and efficiency should start low and climb.
  If calibration never moves, the SFT prior is too strong and the fix is
  upstream — instruct the Stage-0 teacher to state realistic confidence.
- **Average inspects.** Should fall below SFT's 3.32 if the efficiency term is
  doing its job.
- **KL to the reference** (`beta=0.04`). Runaway KL means the policy is drifting
  off-distribution and format compliance will collapse.

### Operational note

No checkpoint-resume is wired up, so **never run this on a preemptable queue**.
`save_steps=100` with a few hundred total steps means intermediate checkpoints
are sparse — worth lowering if the queue is unstable.

---

## Stage 3 — final evaluation

The same harness as the gate, run properly and in full: the complete
degradation × budget grid, calibration, base-vs-policy McNemar on paired items,
and the evidence slice.

The headline deliverable is **not** raw accuracy. Plenty of ordinary classifiers
detect StyleGAN2 faces without any of this machinery. The claims that matter:

1. **Does accuracy degrade gracefully with resolution and budget?** That is what
   demonstrates the task is genuinely investigation-limited rather than solvable
   from the overview.
2. **Is the trained policy better than the base under a paired test?** McNemar
   on the same items, not two independent accuracy numbers.
3. **Did calibration improve?** The SFT policy claims 0.97 and is right 25% of
   the time. If GRPO moves that toward the diagonal, the verifiable reward did
   something an imitation objective could not.
4. **Does the policy inspect true-artifact cells more than the base?** The
   evidence slice is the closest thing to direct evidence that the investigation
   is real.

Trajectory logs (`--trace-log`) are the audit surface: every claim above is
traceable to individual episodes, and `demo/app.py` (or `tools/make_demo_gif.py`)
renders them turn by turn.

---

## Open items

- [ ] Rerun the gate with `--max-new-tokens 640` and `--backend hf`; confirm the
      answer rate recovers toward ~100 and accuracy clears 0.547.
- [ ] Explain or rule out the below-chance accuracy among answered episodes.
- [ ] Fix the stale vision-tower docstring at `training/common.py:633`, and
      decide between `--lora-target attn` and HF-backend eval.
- [ ] `env/trajectory.py:49` anchors on the *first* occurrence of a label token,
      so prose containing the word "observation"/"hypothesis" before the real
      labelled field captures from the wrong position. Mostly cosmetic, but
      `RECONCILIATION` feeds `classify_reconciliation`, which the reward reads —
      a mis-anchored capture can flip the direction being scored. Needs a test
      case in `tests/test_trajectory.py`.
- [ ] Record the confidence-saturation finding in
      `results/reward_failure_history.md`.
