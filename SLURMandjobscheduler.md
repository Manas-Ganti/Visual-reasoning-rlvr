# SLURM & Job Scheduler — Field Notes

Everything that broke while getting this pipeline onto VT ARC (TinkerCliffs,
A100-80GB), why it broke, and what the fix was. Written for whoever (human or
agent) runs the next distributed RL job here.

**Read this before submitting anything.** Every failure below cost a queue slot,
and on a fully-backfilled partition a slot can be hours or days.

Cluster context for the concrete examples: VT ARC TinkerCliffs, Slurm, Lmod
modules, `a100_normal_q` (9 usable DGX nodes, 8×A100-80GB each) and
`h200_normal_q` (6 nodes). Both were saturated throughout.

---

## 0. The one-paragraph version

A distributed RL pipeline has four independent things that must *all* be right
before a single GPU-second does useful work: **the Python environment**, **the
job script's ability to find and activate that environment**, **the collective
comms layer (NCCL)**, and **a resource request the scheduler can actually
satisfy**. Each is a separate failure domain. Each fails *fast* and *silently*
in its own way. Validate them in that order, cheaply, before queuing anything
long — because `afterok` chains propagate a stage-0 failure into days of dead
pending time.

---

## 1. Environment: the dependency graph is the hard part

### 1.1 Python version pins everything downstream

**Symptom:** `ERROR: Could not find a version that satisfies the requirement gradio>=5.0`,
followed by a wall of "Requires-Python >=3.10".

**Cause:** the conda env was Python 3.9.

**Fix:** Python **3.11**. Don't patch around the individual package — modern
`vllm`, `trl`, and `gradio` all require ≥3.10, so 3.9 just moves the failure
later.

```bash
conda create -y -n <name> python=3.11
```

### 1.2 vLLM will build from source and fail if you don't pin it

**Symptom:**

```
Downloading vllm-0.20.2.tar.gz (33.5 MB)
  Getting requirements to build wheel ... error
  AssertionError: CUDA_HOME is not set
```

**Cause:** an unpinned `vllm>=0.8` resolved to a version with no wheel matching
the installed torch, so pip fell back to the source distribution. Source builds
need `nvcc` and `CUDA_HOME`.

**Fix:** pin vLLM to a version whose wheel matches your torch. Do **not** set
`CUDA_HOME` and attempt the source build — it takes ~an hour and usually fails.

```bash
pip install vllm==0.8.5      # wheel; pins torch==2.6.0, pulls numpy
```

### 1.3 vLLM drags in a transformers that breaks the model

Installing `vllm==0.8.5` pulled `transformers 5.15.0`. vLLM 0.8.5 was built
against the 4.5x line, and the Qwen2.5-VL processor APIs changed in v5 — both
`qwen-vl-utils` and vLLM's model implementation break.

**Fix:** pin back *after* the vLLM install (order matters — installing
requirements first would re-upgrade it):

```bash
pip install "transformers==4.51.3"
```

### 1.4 protobuf: a structural conflict, not a cosmetic warning

```
wandb 0.28.1 requires protobuf>=5, but you have protobuf 4.25.9
```

vLLM's opentelemetry dependencies force `protobuf<5`; new wandb demands `>=5`.
Irreconcilable.

**Fix:** downgrade wandb, and remove `weave` (a wandb extra, unused here, source
of the otel conflicts):

```bash
pip install "wandb==0.19.11"
pip uninstall -y weave
```

### 1.5 ⚠️ The big one: inference and training need SEPARATE environments

**This is the most important item in this document.**

- `trl` (any recent version) requires `transformers>=4.56.2`
- `vllm 0.8.5` + `qwen-vl-utils` require the `transformers 4.5x` line

There is no single environment that satisfies both. Don't waste hours trying.

The pipeline splits cleanly because **the distill/eval path never imports TRL**
(`data/build_sft_traces.py` imports only `env.environment`, `training.common`,
`training.vllm_backend`), and the training path never imports vLLM:

| Env | Used by | Key pins |
|---|---|---|
| `vrr` (inference) | distill (Stage 0), eval (Stage 3) | `torch 2.6.0`, `transformers 4.51.3`, `vllm 0.8.5` |
| `vrr-train` (training) | SFT (Stage 1), GRPO (Stage 2) | `torch` cu124, `transformers>=4.56.2`, `trl`, `peft`, `deepspeed` — **no vLLM** |

Verify the TRL API actually matches the code before relying on it —
`training/grpo.py` passes `rollout_func=` to `GRPOTrainer`:

```bash
python -c "from trl import GRPOTrainer; import inspect; \
print('rollout_func' in inspect.signature(GRPOTrainer.__init__).parameters)"
```

### 1.6 A failed pip run leaves the env half-installed

After the vLLM source-build abort, **nothing** from `requirements.txt` had
installed — including CORE packages. This surfaced much later as a job dying on
`ModuleNotFoundError: No module named 'gymnasium'`.

**Fix:** after any pip failure, re-verify the *complete* import chain of the
script you're about to run, not just the package you were installing:

```bash
python -c "
import gymnasium, PIL, numpy, scipy, torch, transformers, vllm, qwen_vl_utils, wandb
from env.environment import InvestigationEnv
from training import common, vllm_backend
print('imports ok')"
```

### 1.6b A checkpoint is only portable to the library version that wrote it

The nastiest failure of the whole exercise, because **nothing errors**.

`peft` stores LoRA weights keyed by module *path*
(`base_model.model.model.language_model.layers.0.self_attn.q_proj`). Transformers
4.56 restructured Qwen2.5-VL so the language tower sits under `language_model`;
4.51 has no such node. Load a 4.56-written adapter under 4.51 and **zero keys
match** — peft emits a `UserWarning`, falls back to the bare base model, and the
job runs to completion producing numbers that look real.

Two full eval runs measured an untrained model this way.

```bash
# what the adapter actually contains
python -c "from safetensors.torch import load_file; \
  print(list(load_file('<ckpt>/adapter_model.safetensors').keys())[:3])"

# the assertion every eval should make
grep -c "missing adapter keys" logs/slurm/<job>.err   # must be 0
```

Rules: evaluate in the environment that trained the checkpoint, and treat a
non-zero missing-key count as a hard failure, not a warning.

### 1.7 Login-node import errors that are NOT real

Two tracebacks look fatal on a login node and are harmless — they're driver
absence, not broken installs:

- `Failed to import from vllm._C with ImportError('libcuda.so.1: cannot open shared object file')`
- deepspeed → triton → `RuntimeError: 0 active drivers ([]). There should only be one.`

Both resolve on a GPU node. Don't "fix" them.

---

## 2. Making the job script find and use the environment

### 2.1 `${BASH_SOURCE[0]}` does not point into your repo

**Symptom:**

```
/cm/local/apps/slurm/var/spool/job6997726/slurm_script: line 26:
  /cm/local/apps/slurm/var/spool/job6997726/arc_env.sh: No such file or directory
```

**Cause:** Slurm **copies the batch script into its spool directory** before
running it, so `dirname "${BASH_SOURCE[0]}"` resolves to the spool dir.

**Fix:** resolve against the submit directory:

```bash
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/arc_env.sh"
```

### 2.2 Conda envs from different condas live in different trees

`conda activate <name>` only finds envs known to the *currently loaded* conda:

- module `Miniforge3` → `~/.conda/envs/<name>`
- a personal miniconda → `~/miniconda3/envs/<name>`

We created `vrr` under one and `vrr-train` under the other, which produced
`EnvironmentNameNotFound: Could not find conda environment: vrr` inside jobs.

**Fix:** always activate by **full path**, and get the path from `conda info --envs`.
Better still, create with `-p` so the location is never a guess:

```bash
conda create -p /path/to/envs/<name> python=3.11
export CONDA_ENV=/path/to/envs/<name>
```

### 2.3 ⚠️ `conda activate` can silently no-op in a batch shell

The worst failure of the whole exercise. The job log showed `arc_env.sh` sourcing
fine, GPUs detected, no error from the activation line — and then Python failed
on `ModuleNotFoundError: gymnasium`, because `python` was still the *system* 3.9.
`conda activate` depends on shell hooks that may not be initialized in a
non-interactive shell; it returned success while changing nothing.

**Fix:** force `PATH` after the activation attempt. This needs no shell hooks and
cannot silently no-op:

```bash
source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV"
export PATH="$CONDA_ENV/bin:$PATH"
```

**And always print which interpreter you got.** This one line turns a class of
invisible failures into an obvious one:

```bash
echo "[arc_env] python=$(which python) $(python -V 2>&1)"
```

### 2.4 `set -e` + `[ -f x ] && source x` is a silent job killer

```bash
[ -f "$HOME/.config/vrr/secrets.env" ] && source "$HOME/.config/vrr/secrets.env"
```

Under `set -euo pipefail` (which the `.slurm` files set), if the file doesn't
exist the test returns 1, and as the last command of a top-level `&&` list that
**exits the job** — with no error message. Always terminate such lines:

```bash
[ -f "$X" ] && source "$X" || true
```

### 2.5 Storage paths in a shared script are assumptions, not facts

`arc_env.sh` defaulted `HF_HOME` and `WANDB_DIR` to `/projects/$USER/...`.
On TinkerCliffs there is no `/projects/$USER` — project storage is named after
the **allocation**, not the user. A nonexistent `HF_HOME` means the job
re-downloads or errors.

**Fix:** verify with `quota`, `df -h`, `ls -ld /projects/*` and set real paths.
Prefer, in order: `/projects/<allocation>` → `/globalscratch/$USER` → `$HOME`.

---

## 3. NCCL — the collective comms layer

### 3.1 Hardcoded `NCCL_SOCKET_IFNAME` breaks even single-node jobs

**Symptom:**

```
NCCL WARN Bootstrap : no socket interface found
RuntimeError: NCCL error: internal error
```

...during vLLM's `ncclGetUniqueId()`, before any weights load. This happened with
`tensor_parallel_size=2` on a **single node** — NCCL bootstrap needs a socket
interface even when all ranks are local.

**Cause:** `arc_env.sh` hardcoded `NCCL_SOCKET_IFNAME=ib0`.

**Two-stage lesson.** First fix checked whether `ib0` *existed* in
`/sys/class/net` — it did, and it still failed. The interface existed but had no
usable IPv4 address (IB without IPoIB). Existence is not usability.

**Fix that worked:** don't constrain NCCL at all; let it choose an interface that
actually has an IP.

```bash
unset NCCL_SOCKET_IFNAME
```

Fallbacks if that still fails on a single node: `export NCCL_SOCKET_IFNAME=lo`
(valid only when all ranks are on one node). For multi-node, get ground truth
first:

```bash
srun --gres=gpu:1 --time=00:02:00 bash -c 'ip -4 -o addr show; ls /sys/class/net'
```

**Log whatever you end up with** — `echo "[arc_env] nccl_ifname=${NCCL_SOCKET_IFNAME:-<auto>}"` —
so the next failure is diagnosable from the log alone.

---

## 4. Getting the scheduler to actually run you

### 4.1 `--mem=0` requests the ENTIRE node

The single biggest queue-time mistake. `--mem=0` means "all memory on the node"
— on a DGX that's ~2TB. Combined with `--gres=gpu:8`, the job can only start when
an entire node drains, and it cannot backfill into a partially-used node.

Observed: `ReqTRES=cpu=32,mem=2055848M,node=1,billing=1082`, pending **two days**
with `START_TIME=N/A` (the scheduler couldn't even predict a slot).

**Fix:** always pass an explicit `--mem`. Dropping to `--gres=gpu:2 --mem=200G
--time=02:00:00` moved the same job from "unschedulable" to a concrete
`START_TIME` on a named node, immediately.

### 4.2 Shrink the ask; backfill rewards small and short

Backfill scheduling fits jobs into gaps ahead of higher-priority work. A
2-GPU/2-hour job fits gaps an 8-GPU/8-hour job never will. Right-size all three
dimensions — GPUs, memory, walltime:

| Stage | Naive ask | What actually scheduled |
|---|---|---|
| distill | 8 GPU, `--mem=0`, 8h | 2 GPU, 200G, 4h (TP=2 holds a 32B on 2×80GB) |
| SFT | 8 GPU, `--mem=0`, 12h | 4 GPU, 400G, 12h |
| GRPO | 2 nodes × 8 GPU, 48h | 1 node × 8 GPU, 400G, 48h |

For GRPO specifically: single-node roughly doubles wall clock but avoids
multi-day pends, and with no checkpoint-resume wired up, a long pend is strictly
worse than a slower run.

### 4.3 Read `sinfo` state suffixes

```
tc-dgx[001-009]  mix-   gpu:a100
tc-dgx010        drain  gpu:a100
```

The trailing `-` means **the backfill scheduler has already planned that node for
a higher-priority job**. Every node showing `mix-` means the partition is
committed, not merely busy. `drain` means out of service. This is how you tell
"I'm in line" from "there is no line to be in".

### 4.4 Account is the Slurm accounting name, not your username

`--account` is the allocation you charge to (e.g. `ece-6474-spring2026`), not
your Linux user. Enumerate valid values:

```bash
sacctmgr show assoc user=$USER format=account%30,qos%30
```

Also check for allocations on *other clusters* — we found three allocations
valid on three clusters, ~6M unused core-hours, while fighting one saturated
partition. And prefer an allocation with no usage: fairshare priority is
usage-based.

### 4.5 Env vars beat retyping flags

```bash
export SBATCH_ACCOUNT=<acct>
export SBATCH_PARTITION=a100_normal_q
```

`sbatch` reads these natively.

⚠️ **Trap:** variables you define in a sourced env script (e.g. `ARC_ACCOUNT`,
`PARTITION` in `arc_env.sh`) are **dead code** unless something consumes them.
`arc_env.sh` is sourced *inside* the job, long after scheduling decisions were
made. Only `#SBATCH` directives, CLI flags, and `SBATCH_*` env vars at submit
time affect scheduling.

### 4.6 `logs/slurm/` must exist BEFORE the first submit

Slurm opens `--output`/`--error` paths before the script body runs, so a
`mkdir -p logs/slurm` inside the script is too late on the first submission.

### 4.7 Email notifications

Not on by default. At submit time:

```bash
--mail-user=<addr> --mail-type=BEGIN,END,FAIL
```

`BEGIN` is the "I got resources" ping. For **already-queued** jobs:

```bash
scontrol update jobid=<id> MailUser=<addr> MailType=BEGIN,END,FAIL
```

Some clusters only relay to on-campus addresses; prefer an institutional one.

---

## 5. Dependency chains

### 5.1 Chaining, and repairing a broken chain

```bash
D=$(sbatch --parsable stage0.slurm)
S=$(sbatch --parsable --dependency=afterok:$D stage1.slurm)
```

When a parent fails, children go to `DependencyNeverSatisfied` and pend forever.
You usually don't need to resubmit them — **re-point** the dependency:

```bash
scontrol update jobid=<child> Dependency=afterok:<new-parent>
```

### 5.2 `afterok` only checks the exit code

A stage can exit 0 and still produce garbage. Our distill exited clean with a
**13% keep rate** (134 usable traces from 1,033 images) — enough to let SFT start
on a near-empty dataset. Always gate on an artifact check, not just job state:

```bash
wc -l data/sft_traces.jsonl
```

### 5.3 `squeue --start` only works for non-dependency jobs

Dependency-blocked jobs show `START_TIME=N/A` — Slurm can't predict when the
parent finishes. You get estimates one stage at a time, for whichever job is at
the head of the chain.

### 5.4 The pipeline launcher is a submitter, not a job

`scripts/train_all.sh` is **run** (`./scripts/train_all.sh`), not `sbatch`ed. Each
stage needs a different shape (1 node vLLM / 1 node training / 2 node RL), so
one big allocation would idle most of its GPUs and blow past walltime.

---

## 6. Observability

### 6.1 W&B: not every stage opens a run

`training/sft.py` and `training/grpo.py` call `common.wandb_init`.
`data/build_sft_traces.py` and `eval/harness.py` **do not**. Expect no W&B run
during distill or eval — that's correct, not a failure. `[arc_env] wandb
mode=online` only reports the environment, not that a run exists.

### 6.2 W&B auth without an interactive login

```bash
mkdir -p ~/.config/vrr
echo 'export WANDB_API_KEY=<key>' > ~/.config/vrr/secrets.env
chmod 600 ~/.config/vrr/secrets.env
```

Sourced by `arc_env.sh`, outside the repo. No project needs to be created on the
website — the first `wandb.init()` creates it. If compute nodes can't reach
`api.wandb.ai`, `arc_env.sh` falls back to `WANDB_MODE=offline`; replay later
with `wandb sync $WANDB_DIR/offline-run-*`.

Check reachability *from a compute node*, not the login node — they often have
different egress rules. Test with `curl -s -o /dev/null -w '%{http_code}'`;
`000` is the "no route" signal (the API answers 404/405 on healthy endpoints, so
`curl -f` would give a false negative).

### 6.3 Monitoring commands

```bash
squeue -u $USER -o "%.10i %.12j %.2t %.10M %.10L %.6D %.20R"
squeue -j <id> --start
scontrol show job <id> | grep -E "JobState|Reason|TimeLimit|ReqTRES|NodeList"
sacct -j <id> -o JobID%14,State,Elapsed,ExitCode,MaxRSS,NodeList
sinfo -p <partition> -o "%.20N %.6t %.8G %.10m %.15C"
sprio -j <id>; sshare -U -u $USER
tail -f logs/slurm/<stage>-<id>.out
```

---

## 7. Prefetch model weights on the login node

A 32B is ~66GB. Downloading inside a time-limited job burns the allocation doing
network I/O. Download on the login node (no allocation needed), into the same
`HF_HOME` the jobs use:

```bash
nohup env HF_HOME=$HOME/hf_cache HF_HUB_ENABLE_HF_TRANSFER=1 \
  /path/to/env/bin/python -c \
  "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-VL-32B-Instruct'))" \
  > $HOME/hf_download.log 2>&1 &
```

Use the interpreter directly — it sidesteps `PATH` problems and the
`huggingface-cli` → `hf` rename in huggingface_hub 1.x. Check with
`du -sh $HOME/hf_cache` and `find $HOME/hf_cache -name "*.incomplete"`.

Note vLLM also caches its `torch.compile` artifacts (~2 min of startup) after the
first run, so reruns start meaningfully faster.

---

## 7b. Namespace your outputs by dataset

When a project outlives one substrate, unnamespaced paths (`data/manifest.jsonl`,
`checkpoints/sft-<model>`) mean run *n+1* silently overwrites run *n*, and a
stage can read a stale artifact from a different experiment without noticing.

This repo sets `VRR_DATASET` (see `training/common.py`) and derives every path
from it — manifest, traces, checkpoints, episode logs, results. `arc_env.sh`
exports it and prints it in the run banner, and each stage appends to
`logs/<ds>/runs.tsv` so there is an index mapping job ids to what they were.

Slurm's own `--output` cannot expand environment variables, so `logs/slurm/` stays
flat — job ids already make those unique, and `runs.tsv` supplies the meaning.

## 8. Preflight checklist

Run **all** of these before submitting anything long. Every one maps to a
failure above that cost a queue slot.

**On the login node (free):**

```bash
[ ] python -V                                  # 3.11
[ ] python -c "<full import chain of the target script>"
[ ] pytest tests/ -q                           # CPU-only invariants
[ ] python training/sft.py --dry-run
[ ] bash -n scripts/*.sh scripts/*.slurm       # shell syntax
[ ] mkdir -p logs/slurm checkpoints
[ ] quota; df -h                               # HF_HOME/WANDB_DIR are real & have room
[ ] du -sh $HF_HOME                            # weights prefetched
[ ] sacctmgr show assoc user=$USER format=account%30
[ ] sinfo -p <part> -o "%.20N %.6t %.8G"       # is the partition even available?
[ ] grep -n CONDA_ENV scripts/*.slurm scripts/arc_env.sh   # right env per stage
```

**In the job script, make these self-reporting (cheap, permanent):**

```bash
[ ] echo "[env] python=$(which python) $(python -V 2>&1)"
[ ] echo "[env] nccl_ifname=${NCCL_SOCKET_IFNAME:-<auto>}"
[ ] echo "[env] node=$(hostname) nodes=$SLURM_NNODES gpus=$GPUS_PER_NODE hf_home=$HF_HOME"
[ ] nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
```

**Before building a pipeline on a new dataset — the cheapest check of all:**

```bash
[ ] ceiling probe at full resolution   # can the model do this task AT ALL?
[ ] floor probe on the overview        # is it already solvable without inspecting?
```

A substrate needs a high ceiling and a floor at chance; the gap is the only space
a policy can learn in. Fifteen minutes here would have redirected a week of work
— see results/faces_negative_result.md.

**First submission of any new stage:**

```bash
[ ] Smallest shape that exercises the real code path (--limit N, few GPUs, short walltime)
[ ] No downstream dependents attached
[ ] Check the first 8 log lines within seconds of it starting
[ ] Verify the OUTPUT ARTIFACT, not just the exit code
```

---

## 9. Failure-mode quick reference

| Symptom | Root cause | Fix |
|---|---|---|
| `Could not find a version that satisfies gradio>=5.0` | Python 3.9 | Rebuild env on 3.11 |
| `AssertionError: CUDA_HOME is not set` during pip | vLLM building from sdist | Pin to a version with a wheel |
| `trl requires transformers>=4.56.2` vs vLLM needing 4.5x | Irreconcilable | Two envs, split by role |
| `.../spool/jobNNN/arc_env.sh: No such file` | `${BASH_SOURCE[0]}` is the spool copy | `${SLURM_SUBMIT_DIR:-$PWD}/scripts/...` |
| `EnvironmentNameNotFound: vrr` | Env belongs to a different conda | Activate by full path |
| `ModuleNotFoundError` though the package IS installed | `conda activate` silently no-op'd | `export PATH="$CONDA_ENV/bin:$PATH"` |
| Job exits instantly, no error | `set -e` + `[ -f x ] && source x` | Append `\|\| true` |
| `NCCL WARN Bootstrap : no socket interface found` | Hardcoded `NCCL_SOCKET_IFNAME` | `unset` it; let NCCL choose |
| `START_TIME=N/A`, pending for days | `--mem=0` + 8 GPUs = whole node | Explicit `--mem`, fewer GPUs, shorter time |
| `DependencyNeverSatisfied` | Parent failed | `scontrol update jobid=<child> Dependency=afterok:<new>` |
| Empty `logs/slurm/` while pending | Slurm writes the file at job *start* | Normal — wait |
| No W&B run appears | That stage never calls `wandb.init` | Check which stages log |
| `libcuda.so.1` / `0 active drivers` on login node | No GPU driver there | Ignore; works on compute nodes |

---

## 10. Generalizing: RL-specific setup order

For any RL post-training pipeline (rollout generation → SFT → policy
optimization → eval) on a scheduler, initialize and verify in this order. Later
items are worthless if earlier ones are wrong.

1. **Python + framework matrix.** Resolve the torch / transformers / vllm / trl /
   deepspeed compatibility set *first*. Accept that inference and training may
   need separate envs; that is normal, not a workaround. Record exact pins.
2. **Env discovery from inside a batch job.** Full paths, forced `PATH`,
   self-reporting interpreter. Never trust `conda activate` by name.
3. **Storage.** `HF_HOME`, `WANDB_DIR`, checkpoint dir — all must exist, be
   writable, and have room. Prefetch weights outside the allocation.
4. **Collectives.** NCCL interface selection, verified on a real compute node.
   Single-node TP needs this too, not just multi-node.
5. **Parallelism shape.** One strategy at a time (vLLM TP *or* torchrun rank
   sharding, never both). For rollout-heavy RL prefer ZeRO-2 over ZeRO-3 —
   ZeRO-3 re-gathers sharded params on every `generate()` call, and RL calls it
   every turn.
6. **Resource shape the scheduler can grant.** Explicit `--mem`; smallest GPU
   count that fits; walltime sized to a measured step time, not a guess.
7. **A smoke run of the real code path** at `--limit`/`--max-steps` scale, with
   no dependents attached.
8. **Artifact gates between stages**, since `afterok` only sees exit codes.
9. **Then** queue the full chain.

### RL-specific gotchas worth stating plainly

- **Rollouts dominate wall clock, not the optimizer step.** Each episode is
  `max_inspects + 1` *sequential* generates, and group-based methods (GRPO) need
  `num_generations` episodes per prompt. Estimate from generate latency, not
  from training throughput.
- **Bootstrap data quality gates everything.** A 13% distill keep rate means SFT
  trains on 134 examples and GRPO amplifies whatever that taught. Check keep
  rate and instrument *why* episodes are dropped (truncation vs. wrong answer vs.
  malformed) before spending days of GPU on downstream stages. `max_new_tokens`
  truncating before the final action line silently invalidates whole episodes.
- **No checkpoint-resume ⇒ never use preemptable queues for long stages.**
  Preemptable is fine for short, restartable work (rollout generation, eval).
- **Verifiable rewards are a debugging asset.** Because the reward is computed
  mechanically from structured fields, stage correctness is checkable on CPU
  (`pytest tests/`) with no GPU and no model. Use that.
