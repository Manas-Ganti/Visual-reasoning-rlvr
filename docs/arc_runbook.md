# ARC runbook — operational gotchas

Everything here cost at least one wasted GPU allocation. Read it before debugging
a job that "just won't start", dies on an import, completes suspiciously fast, or
returns a gate number that looks interpretable.

**Start with *Where the project is* below** — it carries the current substrate,
the settings every stage must share, and what is in flight.

Design docs live elsewhere — [`README.md`](../README.md),
[`.claude/CLAUDE.md`](../.claude/CLAUDE.md). This file is only about the
cluster — with one exception: the gate methodology below, because the way
the gates were being measured wasted more GPU than every scheduling problem
here combined. See [`results/geometry_confound.md`](../results/geometry_confound.md).

---

## Where the project is (2026-08-23)

**Substrate: `synth1024`. Both Gate-1 halves pass. `OVERVIEW_LONG_EDGE=56`.**

```
ceiling  AUC 0.930  [0.89, 0.97]   gate >=0.85   PASS
floor    AUC 0.591  [0.50, 0.68]   gate ~0.50    PASS   (at OVERVIEW_LONG_EDGE=56)
geometry aspect / long edge / area all 0.500 by construction
```

Full account: [`results/substrate_synth1024.md`](../results/substrate_synth1024.md).
Why the earlier substrates failed: [`results/geometry_confound.md`](../results/geometry_confound.md).

Three substrates were tried before this one:

| substrate | outcome |
|---|---|
| faces (300px) | retired — no evidence to reveal, 75px cells |
| `genwukong` (512px) | gates were measuring image size (area AUC 0.850) |
| `genwukong392` | confound removed; ceiling 0.809, floor 0.673 — 56% of the answer free |
| **`synth1024`** | **built to spec — both gates pass** |

In flight as of the last session: distill `7242560` → SFT `7242576`
(`afterok` chained). Not yet run: Gate 2 (group variance), GRPO, eval.

**The settings every stage must share** — a chain that trains at one overview
resolution and evaluates at another produces numbers that are quietly meaningless:

```
VRR_DATASET=synth1024
OVERVIEW_LONG_EDGE=56
HF_HOME=/home/manasganti/hf_cache
CONDA_ENV=/home/manasganti/miniconda3/envs/vrr        # vrr-gen for diffusers stages
```

---

## The known-good submit line

```bash
cd ~/ondemand/data/VLM-RL-aicontent-detection
git pull --ff-only origin rebuild/visual-reasoning-rlvr

HF_HOME=/home/manasganti/hf_cache CONDA_ENV=/home/manasganti/miniconda3/envs/vrr \
OVERVIEW_LONG_EDGE=56 VRR_DATASET=synth1024 JOB=ceiling AUC=1 TP=1 \
sbatch --account=ece-6474-spring2026 --partition=h200_normal_q --qos=tc_h200_normal_short \
       --gres=gpu:h200:1 --cpus-per-task=8 --mem=96G --time=00:30:00 \
       --mail-user=manasganti@vt.edu scripts/arc_infer.slurm
```

`JOB=floor` for the other half of Gate 1. Every flag on that line is load-bearing;
the sections below say why. **This line is confirmed working** — it ran the first
successful ceiling probe on `tc-xe003`.

If your terminal mangles multi-line pastes (see *Terminal paste corruption*
below), use the one-line form, which has nothing for bracketed paste to break:

```bash
HF_HOME=/home/manasganti/hf_cache CONDA_ENV=/home/manasganti/miniconda3/envs/vrr OVERVIEW_LONG_EDGE=56 VRR_DATASET=synth1024 JOB=ceiling AUC=1 TP=1 sbatch --account=ece-6474-spring2026 --partition=h200_normal_q --qos=tc_h200_normal_short --gres=gpu:h200:1 --cpus-per-task=8 --mem=96G --time=00:30:00 --mail-user=manasganti@vt.edu scripts/arc_infer.slurm
```

---

## 1. Two checkouts, one branch

There are two copies of this repo and they are **not** the same machine:

| where | path | role |
|---|---|---|
| Mac | `/Users/manasganti/portfolio-projects/VLM-RL-aicontent-detection` | editing |
| ARC | `/home/manasganti/ondemand/data/VLM-RL-aicontent-detection` | running |

Both track `origin/rebuild/visual-reasoning-rlvr`. **An edit on one is invisible
to the other until it is committed, pushed, and pulled.** This is not a
theoretical concern — it is exactly why the first `--mail-type` change appeared
to do nothing: the directive existed on the Mac while the job ran from the ARC
copy.

The remote has moved: `Manas-Ganti/RL-Based-AI-content-detector` →
`Manas-Ganti/Visual-reasoning-rlvr`. The old URL still redirects, so pushes
succeed with a warning. Worth a `git remote set-url` eventually.

**Rule: after any change to `scripts/`, push from the Mac and pull on ARC before
submitting.** A job runs the file that is on ARC, not the one you just edited.

---

## 2. Outlook / SLURM mail

`--mail-user` says *where*. `--mail-type` says *when*. **Passing only
`--mail-user` sends nothing** — that was the bug.

`--mail-type=BEGIN,END,FAIL,TIME_LIMIT_80` now lives in all three launchers
(commit `fb11a31`). `TIME_LIMIT_80` warns at 24 minutes of a 30-minute wall,
which is enough notice to tell whether a probe will finish.

The address is deliberately **not** hardcoded — these files are tracked in git
and this is a public portfolio repo. Two ways to supply it:

* `scontrol show config | grep -i mail` → if `MailDomain` is set, SLURM resolves
  `$USER` automatically and nothing more is needed.
* otherwise pass `--mail-user=manasganti@vt.edu` on every submit.

Outlook junk-filters SLURM mail aggressively (bare cluster hostname, no SPF).
**Check the Junk folder on the first BEGIN notification** and add the sender to
Safe Senders.

On `*_preemptable_q`, add `REQUEUE` to the mail-type list.

---

## 3. The conda environment

### Two envs, and they must stay separate

| env | path | holds | used by |
|---|---|---|---|
| `vrr` | `/home/manasganti/miniconda3/envs/vrr` | vLLM 0.8.5, torch 2.6.0+cu124, transformers 4.51.3 | gates, distill, eval |
| `vrr-train` | `/home/manasganti/.conda/envs/vrr-train` | trl (no vLLM) | SFT, GRPO |

`vllm 0.8.5` pins transformers near 4.51; `trl 1.9.2` wants `>=4.56.2`. Those
cannot coexist. Do not try to unify them. This mirrors the repo's own split —
training rollouts use HF `generate` for logprobs, eval and distillation use vLLM.

`pip check` reports `trl 1.9.2 requires transformers>=4.56.2` inside `vrr`. It is
a declarative constraint, not a runtime break, and the gate path never imports
`trl` (`tools/ceiling_probe.py:29` pulls only `training.common` and
`training.vllm_backend`). Ignore it in `vrr`; it will matter in `vrr-train`.

### Fixes already applied to `vrr`

* `huggingface-hub` was `1.28.0`, which transformers rejects (`>=0.30,<1.0`).
  Pinned back to `0.36.2`. **Constrain the leaf package, don't bump transformers**
  — that would fight vLLM's pin.

### Activation: the silent failure (commits `cef7d8f`, `c7b1665`)

> **In hindsight this was not the bug** — the real cause was the zero-byte
> interpreter in the next section. Both commits are still worth keeping (they
> turn a silent fall-through into a loud abort), but do not let this section
> send you down the same path: run `python -V` *first*.

`arc_env.sh` runs `module reset` + `module load Miniforge3`, which swaps in the
**cluster's** conda. That conda cannot see a personal `~/miniconda3` root, so:

* a bare `CONDA_ENV=vrr` that activates fine on the login node resolves to
  nothing inside the job → `EnvironmentNameNotFound`, and with `set -e` the job
  dies having done nothing;
* worse, `source activate` can report **success** in a non-interactive batch
  shell without switching interpreters. The `||` fallback never fires, the job
  silently runs in `base`, and you find out from a bare `ModuleNotFoundError:
  No module named 'PIL'` — after the GPU allocation is spent.

`arc_env.sh` now puts `$CONDA_ENV/bin` on `PATH` directly when `CONDA_ENV` is an
absolute path, then **asserts** `sys.executable` is inside the env and that `PIL`
imports, aborting with a readable message otherwise.

**Always pass an absolute path**, and confirm this line near the top of the log:

```
[arc_env] python=/home/manasganti/miniconda3/envs/vrr/bin/python
```

`c7b1665` went further: `arc_env.sh` exports `PY="$CONDA_ENV/bin/python"` and all
four invocations in `arc_infer.slurm` call `"$PY"` rather than a bare `python`.
Nothing resolves through `PATH` any more.

### A zero-byte interpreter (the expensive one)

Symptom: **every** `python` invocation prints nothing and exits 0. A GPU job
"succeeds" in 2 seconds with `State=COMPLETED ExitCode=0:0`, an empty `.err`,
and a `.out` holding only the shell `echo`s. `type -a python` looks correct.
`[ -x ... ]` passes. Nothing errors anywhere.

Cause: `envs/vrr/bin/python3.11` was a **0-byte file** (`file` reports `empty`);
`python`, `python3`, `python3.1` all symlink to it. `execve` on an empty file
returns `ENOEXEC`, so bash runs it as a shell script — an empty script, which
does nothing and exits 0, whatever arguments you pass.

Diagnose in one line — a real CPython can never fail this:

```bash
/home/manasganti/miniconda3/envs/vrr/bin/python -V; echo "exit=$?"
```

No output means the binary is broken, **not** that stdout is being swallowed.
Then check the extent:

```bash
find /home/manasganti/miniconda3/envs/vrr/bin -type f -size 0
find /home/manasganti/miniconda3/envs/vrr/lib -name "*.so*" -size 0 | head
```

**What actually fixed it.** Only that one file was damaged — no zero-byte `.so`s,
and `vrr-train` was untouched. Conda's package cache still held an intact copy
(they were *not* hardlinked: cache 25,548,416 bytes, env 0), so restoring the
byte-identical build was a straight copy, with no dependency solve and no risk to
the torch/vLLM pairing:

```bash
ls -la /home/manasganti/miniconda3/pkgs/python-3.11*/bin/python3.11   # confirm non-zero
cp -p /home/manasganti/miniconda3/pkgs/python-3.11.15-h17756b0_1/bin/python3.11 \
      /home/manasganti/miniconda3/envs/vrr/bin/python3.11
/home/manasganti/miniconda3/envs/vrr/bin/python -V                    # Python 3.11.15
```

Then re-verify the stack, because a broken interpreter can mask other damage:

```bash
/home/manasganti/miniconda3/envs/vrr/bin/python -c "import torch, vllm, PIL, transformers; print(torch.__version__, torch.version.cuda, vllm.__version__)"
# 2.6.0+cu124 12.4 0.8.5
```

If the cache copy is *also* 0 bytes they share an inode; fall back to
`conda install -p <env> --force-reinstall "python=3.11.15"` and re-check the
imports afterwards, since a forced reinstall can reshuffle dependencies.

**Cause: still unknown.** It was **not** quota (334 of 640 GB used). Permissions
and the symlinks were intact; only the file contents vanished. The mtime was
`Aug 22 18:15`, the same minute job 7239565 ran — but nothing in `arc_env.sh` or
`arc_infer.slurm` writes to that path. If a binary is ever silently truncated
again, that is an ARC `/home` support ticket, not something to debug locally.

**Rule: if a job completes in seconds with an empty `.err`, check that the
interpreter is a real binary before debugging anything else.** Hours went into
the launcher scripts, PATH ordering, heredocs and `sitecustomize` before anyone
ran `python -V`.

### Terminal paste corruption

The VS Code remote terminal leaks bracketed-paste markers, which silently
corrupts commands: a stray `~` appended to a filename (`manifest.jsonl~`), a
literal `[200~` prefix, assignments that never take effect, and heredocs that
swallow the next command. This cost several rounds of chasing failures that were
paste damage rather than real.

Two specific traps seen here:

* a mangled `V=...` assignment meant `$V` was empty, so `env FOO=1 $V -c ...` ran
  `env` with no command — which prints the whole environment and exits 0, and
  looks nothing like the failure you were testing for;
* pasting previous *output* back into the prompt produces a cascade of
  `command not found` and `syntax error` lines that mask the real result.

Fix it once per shell:

```bash
bind 'set enable-bracketed-paste off'
```

Otherwise: one command per line, prefer single-line commands over backslash
continuations, and write anything long to a file (`cat > /tmp/sub.sh <<'EOF'`)
and run that instead.

### HF cache

`HF_HOME=/home/manasganti/hf_cache` (Qwen2.5-VL-32B present: 68.3 GB, 32 files).
`arc_env.sh` still *defaults* to `/projects/$USER/hf_cache`, so **`HF_HOME` must
be passed or exported** — left deliberately explicit rather than silently
redirecting a 68 GB cache. `~/.bashrc` exports both `HF_HOME` and `CONDA_ENV`.

`huggingface-cli` / `hf` only exist inside the env; they are not on the base PATH.

---

## 4. Running the gates (read before trusting a number)

Measure the substrate before the model, and rank before argmax. Both gates on
`genwukong` returned interpretable-looking numbers that were measuring image
size; the full account is in
[`results/geometry_confound.md`](../results/geometry_confound.md).

**Order matters.** Step 1 costs ten seconds and no GPU:

```bash
# 1. is the label readable from the FILE?  every predictor should be ~0.5
python tools/manifest_stats.py --dataset <ds>

# 2. ceiling — can the model do this at all?   want AUC >=0.85
JOB=ceiling AUC=1 ... sbatch scripts/arc_infer.slurm

# 3. floor — does the overview actually hide it?   want AUC ~0.5
JOB=floor   AUC=1 ... sbatch scripts/arc_infer.slurm
```

**Always pass `AUC=1`.** Argmax accuracy measures the model's prior as much as
its eyesight: a policy that ranks every AI image above every real one still
scores at the majority baseline if it never crosses its own threshold for saying
the rarer word. On this substrate that gap was 0.600 accuracy versus 0.874 AUC —
the difference between "abandon the dataset" and "the dataset is fine".

It matters even more on the floor, and in the more dangerous direction: a model
answering `REAL` to everything scores ~0.50 there *whether or not the overview
leaks the answer*, so an accuracy floor reads "perfect" from a measurement that
cannot detect the failure it exists to catch.

**`bytes/px` is not a confound.** The model sees pixels, never file size. A
raised `bytes/px` means AI images are smoother and more compressible, which is a
real visual property a detector should use. Geometry is the one that must be
driven to 0.5 — aspect and size survive `make_overview` untouched, and Qwen bins
images to a patch grid, so different sizes arrive as different token counts.

**Choosing `OVERVIEW_LONG_EDGE`:** sweep the floor, don't use a ratio. The
ceiling is measured at native resolution so it is unaffected — only the floor
needs re-running per value. Take the mildest blur whose floor CI contains 0.5.
On `synth1024`: 140 → 0.740, 80 → 0.637, 48 → 0.556 (passes). Full result in
[`results/substrate_synth1024.md`](../results/substrate_synth1024.md).

**Fixing geometry:** `data/recrop_manifest.py --src <ds> --dst <ds>N --size N`
crops images already on disk (no re-download, same selection), or
`build_manifest_hf.py --center-crop N` at build time. Pick N as a multiple of 28
and check what it drops — `392` kept 796/800 on genwukong where `448` dropped 49,
all from the real class.

## 5. Why the scheduling flags look like that

Both jobs pended a full day before any of this was understood.

* **`#SBATCH --mem=0` is still in the launcher headers** and means *all memory on
  the node* (~2 TB). It is correct for the 8-GPU full-node `distill`/`eval` jobs,
  and fatal for a small gate probe: it can only be satisfied on a node where
  nobody else holds a byte, so the job can never backfill. **Always pass an
  explicit `--mem` on gate submissions.**
* **`--qos=tc_h200_normal_short`** is the single biggest lever: priority went
  1330 → 2312 and the pend reason went `Priority` → `Resources` (first in line).
  The name is misleading — it caps at a **full day** and has the *highest*
  priority on the cluster:

  | QOS | priority | MaxWall |
  |---|---|---|
  | `tc_h200_normal_short` | **2000** | 1-00:00:00 |
  | `tc_h200_normal_base` | 1000 | 7-00:00:00 |
  | `tc_h200_normal_long` | 500 | 14-00:00:00 |

  So every stage except GRPO (48h) belongs in `short`: gates, captioning,
  generation, distillation, SFT (12h), eval. Only GRPO needs `base`. Confirm the
  A100 equivalents with `sacctmgr show qos format=name%28,priority,maxwall`
  before queueing there.
* **Do not fall back to `*_preemptable_q`.** `h200_normal_q` is `PriorityTier=16`
  with `PreemptMode=OFF`; `h200_preemptable_q` is tier 8 and evictable. With
  `PreemptType=preempt/partition_prio`, normal preempts preemptable — moving down
  makes you the preemptee.
* **`PARTITION` and `ARC_ACCOUNT` are dead knobs.** Both appear only in comments
  (`arc_env.sh:12`, `:18`); nothing consumes them, and `#SBATCH` directives cannot
  read environment variables anyway. The hardcoded
  `#SBATCH --partition=a100_normal_q` wins unless `--partition=` is on the CLI.
* Partitions have `MaxTime=UNLIMITED`, so jobs ahead of you may have no time
  limit and the backfill scheduler cannot plan around them. Being small and short
  is the only reliable way through.
* `sinfo` state `mix` means *partially* allocated — it does not imply free GPUs.
  Check `AllocTRES` for `gres/gpu=` per node.

### Cluster shape

| partition | nodes | GPUs |
|---|---|---|
| `a100_normal_q` / `a100_preemptable_q` | 14 — `tc-dgx[001-010]`, `tc-gpu[001-004]` | `gpu:a100:8` |
| `h200_normal_q` / `h200_preemptable_q` | 6 — `tc-xe[001-006]` | `gpu:h200:8` |

H200 = 141 GB, so Qwen2.5-VL-32B (~66 GB) fits on **one** GPU at the stock
`max_model_len=16384` / `gpu_memory_utilization=0.90`. Hence `--gres=gpu:h200:1`
and `TP=1`. **`TP` must equal the GPU count** — `arc_infer.slurm` defaults it to
`$GPUS_PER_NODE`, so it must be set explicitly on a partial-node request.

---

## 6. Building a substrate (when none off the shelf fits)

No public AI-detection dataset surveyed had what this environment needs — they
are downsampled to 256–512px for CNN classifiers, and a 4×4 cell of a 256px image
is 64 pixels, so `INSPECT` upscales a blur. `data/build_paired_synthetic.py`
builds one instead: real photographs cropped to a fixed size, captioned by the
VLM, and a synthetic half generated from those captions.

```bash
# 1. reals — login node, no GPU. DIV2K (~2040x1356) at /home/manasganti/realsrc/
python data/build_paired_synthetic.py --stage crop --dataset synth1024 \
    --real-src /home/manasganti/realsrc/DIV2K_train_HR --per-class 800

# 2. captions — vLLM env
STAGE=caption DATASET=synth1024 CONDA_ENV=.../vrr     sbatch ... scripts/arc_synth.slurm
# 3. generate — diffusers env
STAGE=generate DATASET=synth1024 CONDA_ENV=.../vrr-gen sbatch ... scripts/arc_synth.slurm

# 4. assemble + verify — login node
python data/build_paired_synthetic.py --stage assemble --dataset synth1024
python tools/manifest_stats.py --dataset synth1024      # geometry must be 0.500
```

Every stage is restartable — each skips what is already on disk — so a preempted
job resumes by resubmitting the identical line. To re-caption, move
`captions.jsonl` aside first or the stage will consider the work done.

Three settings are load-bearing, and two were wrong on the first attempt:

* **Caption length and ordering.** SDXL's CLIP encoders keep ~77 tokens and
  discard the rest silently. Qwen overshoots any word limit, and its natural
  ordering puts the subject first and the incidental detail last — so truncation
  eats exactly the clutter the pairing exists to preserve. Ask for ~50 words as a
  fragment, **clutter first, subject and lighting last**.
* **Guidance scale 5.5, no negative prompt.** High CFG and "blurry, low quality"
  negatives both push toward saturated, over-stylised output — a *global*
  difference that survives any downsample and hands the floor a free answer.
* **Crop before captioning.** The captioner must describe the cropped view the
  model will actually be shown; otherwise every fake depicts a wider scene than
  its paired real. The stage order already enforces this.

An existing substrate can be re-cropped without rebuilding:
`python data/recrop_manifest.py --src <ds> --dst <ds>N --size N`.

## 7. The training chain

`./scripts/train_all.sh` submits distill → SFT → GRPO → eval as one `afterok`
chain. It refuses to run without `GATES_OK=1`, and `DRY_RUN=1` prints every
sbatch line without submitting.

```bash
VRR_DATASET=synth1024 OVERVIEW_LONG_EDGE=56 GATES_OK=1 DRY_RUN=1 \
  SBATCH_ACCOUNT=ece-6474-spring2026 MAIL_USER=manasganti@vt.edu \
  CONDA_ENV=/home/manasganti/miniconda3/envs/vrr HF_HOME=/home/manasganti/hf_cache \
  ./scripts/train_all.sh
```

**Gate 2 is deliberately not in the chain.** Group variance is measured against
the SFT checkpoint, and an `afterok` dependency cannot express "stop if this
number is too low" — which is the whole point when the next stage costs 48 hours:

```bash
JOB=groupvar ADAPTER=checkpoints/synth1024/sft-qwen2.5-vl-32b ... scripts/arc_infer.slurm
# want usable_groups >= 0.40
```

Two numbers to read rather than judge:

* **Distillation keep rate** (printed at the end of the distill log).
  `build_sft_traces.py` writes a trace only if every turn parses AND the verdict
  matches ground truth, so malformed traces cannot reach the file — the risk is
  too few, not bad. ≥40% is healthy; under ~15% means SFT will underfit, and the
  fix is to distil with `--model 72b` (the docstring recommends a bigger teacher
  than the student for exactly this).
* **`usable_groups`** from Gate 2. Under 0.40, GRPO has no gradient to work with.

Queue shapes: inference and generation are one GPU (`--gres=gpu:h200:1`); SFT and
GRPO are data-parallel across a full node (`gpu:h200:8`) and queue much slower —
there are only six H200 nodes. GRPO's 48h exceeds the short QOS cap, so it must
use `tc_h200_normal_base`.

## Preflight (all cheap, all on the login node)

```bash
ls -l data/genwukong/manifest.jsonl                    # probe exits instantly without it
python -c "
import json, collections
rows=[json.loads(l) for l in open('data/genwukong/manifest.jsonl') if l.strip()]
print(len(rows),'rows')
for k,v in sorted(collections.Counter((r['split'],r['label']) for r in rows).items()): print(' ',k,v)"

V=/home/manasganti/miniconda3/envs/vrr                  # gate import chain, no GPU needed
$V/bin/python -c "
from PIL import Image
import sys; sys.path.insert(0,'.')
from env import grid, prompts
from training import common, vllm_backend
print('OK ->', sys.executable)"

quota                                                   # ~105 GB lives in \$HOME (2 envs + cache)
```

`ceiling_probe` defaults to `--split test`, so the test split needs a healthy
number of **both** labels — an imbalanced test set can read ≥0.85 purely from
class collapse, which is the exact failure mode that killed the faces substrate.

---

## Watching a running job

```bash
squeue -j <id> -o "%.10i %.9T %.11M %.11L %.22R %N"
srun --jobid=<id> --overlap --pty nvidia-smi      # weights load ~66 GB, then util spikes
sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,ReqTRES%40,Start,End,ExitCode
seff <id>
```

**The gate probes write nothing but stdout** — no W&B (despite `arc_env.sh`
configuring it, `ceiling_probe` never calls `wandb.init`), no results file.
Progress and results live only in `logs/slurm/infer-<jobid>.out`.
`training/common.py:114` defines `results_dir()` but the probes don't use it, so
gate numbers are lost if that log is lost. **Open item: persist them.**

---

## Open items

- [ ] **Gate results are stdout-only.** `training/common.py:114` defines
      `results_dir()` but neither probe uses it, so every number in
      `results/*.md` was recovered from SLURM logs by hand. Persist to
      `results/$VRR_DATASET/gate_*.json` (+ optional `wandb.init`).
- [ ] **Held-out generator.** Every fake in `synth1024` is SDXL, so a trained
      detector may learn SDXL artifacts rather than AI artifacts. Generate a
      second set from FLUX with the same captions under a different
      `--generator` tag, eval only — `eval/harness.py` already breaks results
      down per generator, and it turns "detects SDXL" into "the investigation
      transfers to a generator it never saw".
- [ ] **`.claude/CLAUDE.md` is gitignored**, so the corrected gate commands
      (manifest_stats first, `--auc` always) exist only on the Mac. The ARC copy
      still documents the accuracy-based gate that caused the confound.
- [ ] `arc_env.sh` could probe for a usable `HF_HOME` the way
      `scripts/fetch_genimage.sh:72` probes for data storage — it still defaults
      to `/projects/$USER/hf_cache`, which is wrong here, so `HF_HOME` must be
      passed or exported on every submit.
- [ ] `git remote set-url origin git@github.com:Manas-Ganti/Visual-reasoning-rlvr.git`
      (the old URL redirects with a warning on every push).
- [ ] Confirm `vrr-train` has a trl-compatible transformers before Stage 1 —
      `trl 1.9.2` wants `transformers>=4.56.2` while `vrr`'s vLLM pins 4.51.3.
      That is why the two envs exist; do not try to unify them.
- [x] `synth1024` gates — ceiling 0.930, floor 0.591 at
      `OVERVIEW_LONG_EDGE=56`. See `results/substrate_synth1024.md`.
- [x] `train_all.sh` namespacing and `SBATCH_PARTITION` — fixed, and it now
      refuses to submit without `GATES_OK=1`.
