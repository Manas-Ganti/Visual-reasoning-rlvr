# ARC runbook — operational gotchas

Session notes from getting the GenImage/`genwukong` gate probes to actually run
on VT ARC (TinkerCliffs). Everything here cost at least one wasted GPU
allocation. Read this before debugging a job that "just won't start" or that
dies on an import.

Design docs live elsewhere — [`README.md`](../README.md),
[`.claude/CLAUDE.md`](../.claude/CLAUDE.md). This file is only about the
cluster.

---

## The known-good submit line

```bash
cd ~/ondemand/data/VLM-RL-aicontent-detection
git pull --ff-only origin rebuild/visual-reasoning-rlvr

HF_HOME=/home/manasganti/hf_cache CONDA_ENV=/home/manasganti/miniconda3/envs/vrr \
OVERVIEW_LONG_EDGE=64 VRR_DATASET=genwukong JOB=ceiling TP=1 \
sbatch --account=ece-6474-spring2026 --partition=h200_normal_q --qos=tc_h200_normal_short \
       --gres=gpu:h200:1 --cpus-per-task=8 --mem=96G --time=00:30:00 \
       --mail-user=manasganti@vt.edu scripts/arc_infer.slurm
```

`JOB=floor` for the other half of Gate 1. Every flag on that line is load-bearing;
the sections below say why.

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

### Activation: the silent failure (commit `cef7d8f`)

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

### HF cache

`HF_HOME=/home/manasganti/hf_cache` (Qwen2.5-VL-32B present: 68.3 GB, 32 files).
`arc_env.sh` still *defaults* to `/projects/$USER/hf_cache`, so **`HF_HOME` must
be passed or exported** — left deliberately explicit rather than silently
redirecting a 68 GB cache. `~/.bashrc` exports both `HF_HOME` and `CONDA_ENV`.

`huggingface-cli` / `hf` only exist inside the env; they are not on the base PATH.

---

## 4. Why the scheduling flags look like that

Both jobs pended a full day before any of this was understood.

* **`#SBATCH --mem=0` is still in the launcher headers** and means *all memory on
  the node* (~2 TB). It is correct for the 8-GPU full-node `distill`/`eval` jobs,
  and fatal for a small gate probe: it can only be satisfied on a node where
  nobody else holds a byte, so the job can never backfill. **Always pass an
  explicit `--mem` on gate submissions.**
* **`--qos=tc_h200_normal_short`** is the single biggest lever for a 30-minute
  job: priority went 1330 → 2312 and the pend reason went `Priority` →
  `Resources` (i.e. first in line). Check for an A100 equivalent before queueing
  there.
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

- [ ] `data/genwukong/manifest.jsonl` existence and class balance — never verified
- [ ] Gate results to `results/$VRR_DATASET/gate_*.json` + optional `wandb.init`
- [ ] `arc_env.sh` could probe for a usable `HF_HOME` the way
      `scripts/fetch_genimage.sh:72` probes for data storage
- [ ] Make `PARTITION` / `ACCOUNT` / `GRES` / `TIME` real env knobs, or delete the
      comments promising them
- [ ] `git remote set-url origin git@github.com:Manas-Ganti/Visual-reasoning-rlvr.git`
- [ ] Confirm `vrr-train` has a trl-compatible transformers before Stage 1
