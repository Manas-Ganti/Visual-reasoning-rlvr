# What we know

*Running synthesis, current as of 2026-09-18 (cycle 3 complete: see `grpo_run3.md`). Written to
stop us re-running things that are already settled. Every number here was
measured; where something is untested it says so.*

Individual docs go deeper: [`geometry_confound.md`](geometry_confound.md),
[`substrate_synth1024.md`](substrate_synth1024.md),
[`grpo_run1.md`](grpo_run1.md),
[`reasoning_trace_analysis.md`](reasoning_trace_analysis.md),
[`reward_failure_history.md`](reward_failure_history.md),
[`faces_negative_result.md`](faces_negative_result.md).

---

## 1. The one sentence version

The model **can** tell these images apart — 0.763 accuracy is reachable from its
own scores at budget 6 — and **delivers 0.455**, because it answers REAL to 87%
of everything. Three attempts to move that prior have failed. The cause is now
localised to a reasoning error we can point at in the trajectories, and cycle 3
is the first attempt that targets it.

## 2. What is settled — do not re-litigate

### The substrate is sound
| gate | value | verdict |
|---|---|---|
| geometry predicts the label | 0.500 | clean (was **0.850** on GenImage) |
| ceiling, full image | 0.930 AUC | PASS (≥0.85) |
| floor, 56px overview | 0.591 AUC | PASS (≈0.5) |
| **available at budget 6, random cells** | **0.820 AUC / 0.763 acc** | the target |
| group variance on the SFT policy | 0.938 usable | PASS (≥0.40) |

`synth1024`: 1,570 paired 1024px images, DIV2K photographs vs SDXL renderings of
their own captions, geometry matched by construction. Three substrates were
rejected before it (faces, GenImage/Wukong, a survey of paired HF datasets).

**The gap between 0.763 available and 0.455 delivered is the whole problem.** It
is not a perception ceiling and not a substrate defect.

### GRPO cannot move a class prior
Run 1: 600 steps moved the marginal AI-rate **21.2% → 21.6%**. Structural, not a
tuning failure — the advantage is computed within a group of rollouts on ONE
image, so it pushes toward AI on AI images and REAL on real ones, and across
balanced data those pressures cancel. **Whatever prior SFT hands over is the
prior GRPO still has at the end.** Confirmed twice.

Corollary, and the reason the Brier term failed: **GRPO can only reinforce what
it samples.** A policy emitting 0.98 confidence on every rollout never produces a
hedge for the reward to pay for, however well the reward is shaped.

### GRPO does improve discrimination
Run 1 moved the separation gap **−3.4% → +15.8%** over 600 steps — from
anti-correlated to genuinely discriminating — while held-out accuracy stayed at
chance. The method works; it was pointed at the wrong bottleneck.

### The model's reasoning is good and its inference is not
From 701 recovered rollouts (`tools/traces_from_log.py`):
- Median P(fake) fell **0.50 → 0.20 → 0.10 → 0.05 → 0.01** across five
  inspections of a 16-cell grid. 65% of all belief changes were downward.
- It asserts 100-to-1 odds having seen **under a third of the image**.
- A mis-aimed inspection (cell doesn't contain the predicted feature) was counted
  as evidence *for* REAL.
- The verdict follows its own P(fake)>0.5 rule **99.4%** of the time.

So: not a broken decision rule on good beliefs — a faithful decision on a belief
that collapsed. **Finding an artifact proves AI; finding none in 6 of 16 cells
proves very little.** The policy weighted both the same.

## 3. What we tried that did not work

| attempt | result | what it taught |
|---|---|---|
| GRPO, 600 steps (cycle 1) | 0.545 vs 0.513 base, p=0.23 | discrimination improved, prior did not move |
| Brier term on P(fake), w=0.40 | confidence 0.92 → **0.98** | a proper scoring rule cannot teach hedging to a policy that never hedges |
| SFT class weighting at 65% AI | eval AI-rate **13.4%** vs 16.5% base | weighting the *training set* does not move the *policy's* prior |
| `prediction_tracking`, w=0.10 | 91% CONFIRMED | paid for a verbal tic; now 0.0 |
| `belief_coherence`, w=0.30 | 60.5% of CONFIRMED preceded a *falling* belief | the term and the policy disagreed on what the tag meant; it withheld 0.30 from 1,367 sound updates. Now 0.0 |
| budget 4 → 6 | 0.820 AUC at 6 vs 0.786 at 4 | genuinely better, but not the bottleneck |

## 4. What worked

| change | before | after |
|---|---|---|
| token cap 320 → 640 | 19.4% distillation keep | **77.2%** |
| `strip_emphasis` in the parser | `**ACTION:**` parsed INVALID | parses |
| final-turn warning | answer rate 0.688 | **99.8%** in training |
| evidence-asymmetry paragraph | traces ratcheted to 0.01 | hold 0.35–0.50 *(traces only — untested in a policy)* |
| tag definitions | CONFIRMED→rising 24.8% | **89.8%** *(traces only)* |
| teacher-hint floor on REAL | 88% at the rail | 60% *(partial)* |
| `--min-real-belief` filter | — | drops the remaining 255 *(in flight)* |

## 5. Rejected designs — and why, so they stay rejected

- **Analysis actions (FFT, lighting, geometry).** Return a global near-sufficient
  statistic, so the optimal policy is "call it once, verdict" and the
  investigation becomes decoration.
- **REAL-side tells in the prompt** (sensor grain, chromatic aberration). Same
  collapse by a different route: AI artifacts are *spatially localised*, which is
  what makes choosing a cell a real decision — grain is *uniform*, so one look
  anywhere settles it. It would also destroy the asymmetry the environment
  exists to study: AI is provable, REAL is only ever not-yet-disproved.
- **Forcing exhaustive inspection.** Contradicts partial observability, and the
  policy already spends 5.05 of its 6 inspections. Coverage was never the
  problem.
- **Full fine-tuning.** Out of scope; LoRA only.

## 6. Infrastructure — solved, but costly to rediscover

`--mem=0` blocks backfill for small jobs but is *correct* for full-node ones (a
300G cap caused a host OOM that killed run 2 at step 100 of 140, mid-save, with
no checkpoint written). ZeRO-3 does not shard on this stack; parameter-only
offload runs 32B SFT on 2×A100 in ~4h. Full offload forces `DeepSpeedCPUAdam`,
which needs an `nvcc` the compute nodes lack. GPU partitions are heterogeneous —
a 2-node request mixed a DGX and a standard node whose InfiniBand could not talk.
Three conda envs, and crossing them fails. **`VAR=x sbatch` never errors on a
name the launcher does not read** — `MAX_INSPECTS` was silently dropped twice,
and the `eval` branch still ignores it, so no policy has ever been evaluated at
the budget it trained on. Details in `docs/arc_runbook.md` and
`docs/arc_quickref.md`.

## 7. Open questions

- ~~**Does calibrated SFT move the prior?**~~ **ANSWERED, cycle 3: yes, and it did
  not matter.** Predicted-AI went 13.4% → 46.3% and held through GRPO. Held-out
  accuracy did not move (0.417 SFT, 0.385 GRPO, against 0.404 base). The prior was
  a symptom. Note the *base* model moved 16.5% → 50.4% on the prompt change alone —
  SFT added nothing on top.
- **Does the teacher hint poison the traces?** The teacher is shown the label and
  is right 98.4% of the time, so it has no reason to hedge. Dropping the hint
  entirely — accepting a much lower keep rate for genuinely earned conclusions —
  is the untried lever.
- **GRPO runs are underpowered.** With `G=8` on 8 ranks and `grad_accum=1`, each
  step is *one image*; 100 steps saw 100 images. Rollouts are sequential
  (`for prompt in prompts`), ~50s each, 401s/step. Batching the 8 group members
  through `generate()` together is a 4–6× speedup and the single highest-leverage
  engineering change available.
- **Eval at budget 6.** Never measured. `BUDGETS=2,4,6` from now on.

## 8. The rule that has paid off every time

Measure the cheap thing first. `manifest_stats` caught a confound that had
invalidated every prior number. The budget probe proved the signal survived SFT
before we spent a cycle on it. `trace_stats` rejected a trace set in five minutes
that would have cost 20 GPU-hours to reject at the eval.

Two cycles were lost by running the full pipeline and reading the result at the
end. Every stage now has a gate in front of it, and each gate is minutes against
the hours it protects.
