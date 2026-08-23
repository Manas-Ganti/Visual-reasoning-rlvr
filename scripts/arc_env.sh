#!/usr/bin/env bash
# Shared environment for the VT ARC SLURM jobs (sourced, not executed).
#
# Everything site-specific lives here so the .slurm files stay about the run.
# Verify the two site facts before the first launch — they change between
# clusters and over time:
#
#   sinfo -o "%P %G %D %m %N"        # partitions, per-node GPUs, node names
#   quota                            # where to put HF_HOME (NOT your $HOME)
#
# Overridable from the submit line, e.g.:
#   ARC_ACCOUNT=myalloc PARTITION=h200_normal_q sbatch scripts/arc_sft.slurm
#
# No `set -e` here on purpose: the module/conda probes below are allowed to fail
# over to their alternatives. The .slurm files set it for the run itself.

# ---- site ----------------------------------------------------------------- #
export ARC_ACCOUNT="${ARC_ACCOUNT:-personal}"        # sbatch --account
export PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
# Dataset namespace: every artifact (manifest, traces, checkpoints, episode
# logs, results) lives under <root>/<VRR_DATASET>/, so substrates never
# overwrite each other. Override per submit: VRR_DATASET=faces sbatch ...
export VRR_DATASET="${VRR_DATASET:-genimage}"
# Overview resolution — the third difficulty axis, alongside degradation and
# budget. INSPECT's zoom factor is native_size/this. Every launcher passes it, so
# setting it here keeps a run's stages consistent with each other.
#
# Do NOT pick it by ratio. "~native/7" was calibrated when overview resolution and
# cell size were coupled — at 512px natives a harsher blur bought nothing, since
# the 4x4 cells were already too small for INSPECT to reveal anything. Above
# 1024px they are decoupled (the cell is native/4 regardless), so the blur can be
# pushed much harder for free. On synth1024, native/7 (=140) left a floor of
# 0.740; native/21 (=48) put it at chance. See results/substrate_synth1024.md.
#
# Choose it by sweeping the FLOOR gate and taking the mildest blur whose AUC
# confidence interval contains 0.5. The ceiling is measured at native resolution
# and is unaffected, so only the floor needs re-running.
export OVERVIEW_LONG_EDGE="${OVERVIEW_LONG_EDGE:-140}"
# Model weights are tens to hundreds of GB — keep the HF cache on project/scratch
# storage, never in $HOME (small quota, and it is not purged-but-fast storage).
export HF_HOME="${HF_HOME:-/projects/$USER/hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRANSFORMERS_VERBOSITY=warning
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1

# ---- W&B ------------------------------------------------------------------ #
# Credentials come from ONE of these, checked in order. Never put the key in
# this file — it is tracked in git.
#
#   1. `wandb login` on the login node        -> writes ~/.netrc (recommended;
#                                                $HOME is mounted on the compute
#                                                nodes, so jobs inherit it)
#   2. ~/.config/vrr/secrets.env              -> `export WANDB_API_KEY=...`,
#                                                chmod 600, outside the repo
#
export WANDB_PROJECT="${WANDB_PROJECT:-visual-reasoning-rlvr}"
# Run dirs hold the offline event log; keep them off the small $HOME quota for
# the same reason as HF_HOME.
export WANDB_DIR="${WANDB_DIR:-/projects/$USER/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WANDB_DIR/cache}"
mkdir -p "$WANDB_DIR" 2>/dev/null || true
# Never let a 48h job hang at startup on an unreachable W&B API.
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-60}"

[ -f "$HOME/.config/vrr/secrets.env" ] && source "$HOME/.config/vrr/secrets.env"

# Preflight: no credential, or no route to the API, means run OFFLINE rather
# than fail or block. Offline runs are written to $WANDB_DIR and replayed later
# with `wandb sync` from the login node — you lose live tracking, not the run.
if [ -z "${WANDB_MODE:-}" ]; then
  if [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
    echo "[arc_env] WARNING: no W&B credential (~/.netrc or WANDB_API_KEY) -> WANDB_MODE=offline"
    export WANDB_MODE=offline
  # Reachability, NOT HTTP status: api.wandb.ai answers 404 at / and 405 at
  # /graphql, so `curl -f` would call a perfectly healthy API "unreachable" and
  # force every run offline. http_code 000 is the real "no route" signal.
  elif [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 https://api.wandb.ai/graphql 2>/dev/null)" = "000" ]; then
    echo "[arc_env] WARNING: api.wandb.ai unreachable from $(hostname) -> WANDB_MODE=offline"
    echo "[arc_env]          after the job: wandb sync $WANDB_DIR/offline-run-*"
    export WANDB_MODE=offline
  else
    export WANDB_MODE=online
  fi
fi
echo "[arc_env] wandb mode=${WANDB_MODE} project=${WANDB_PROJECT} dir=${WANDB_DIR}"

# ---- modules / python ----------------------------------------------------- #
# ARC uses Lmod. Adjust the module names to what `module spider cuda` reports on
# the cluster you land on; the conda env is expected to hold the requirements.txt
# TRAINING profile (torch built against this CUDA).
#
# `module reset` swaps in the CLUSTER's conda, which cannot see a personal
# ~/miniconda3 root — so a bare env NAME that activates fine on the login node
# resolves to nothing inside the job. Give CONDA_ENV an ABSOLUTE PATH:
#
#     CONDA_ENV=/home/$USER/miniconda3/envs/vrr sbatch scripts/arc_infer.slurm
#
module reset >/dev/null 2>&1 || true
module load Miniforge3 >/dev/null 2>&1 || module load Anaconda3 >/dev/null 2>&1 || true

export CONDA_ENV="${CONDA_ENV:-vrr}"
if [ -x "$CONDA_ENV/bin/python" ]; then
  # PY is the interpreter every launcher must use. Resolving `python` through
  # PATH proved unreliable on ARC — a bare `python` here ran, printed nothing and
  # exited 0, silently turning a GPU job into a 2-second no-op. Call PY directly.
  export PY="$CONDA_ENV/bin/python"
  # Absolute path: put it on PATH directly. `source activate` can report success
  # in a non-interactive batch shell WITHOUT switching interpreters, so the ||
  # fallback never fires and the job runs in `base` — which surfaces much later
  # as a bare ModuleNotFoundError, after the GPU allocation is already spent.
  export CONDA_PREFIX="$CONDA_ENV"
  export PATH="$CONDA_ENV/bin:$PATH"
else
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda activate "$CONDA_ENV" && export PY="$(command -v python)" || {
    echo "[arc_env] FATAL: cannot activate conda env '$CONDA_ENV'." >&2
    echo "[arc_env]        Pass an absolute path: CONDA_ENV=/path/to/envs/vrr sbatch ..." >&2
    return 1 2>/dev/null || exit 1
  }
fi

# Verify rather than assume. This is the cheapest check in the pipeline and it
# guards the most expensive resource. Note "$PY" -c, never a bare `python`.
"$PY" -c '
import os, sys
env = os.environ.get("CONDA_ENV", "")
print("[arc_env] python=" + sys.executable, flush=True)
if os.path.isabs(env) and not sys.executable.startswith(os.path.realpath(env)) \
   and not sys.executable.startswith(env):
    sys.exit("[arc_env] FATAL: interpreter is not inside " + env)
try:
    import PIL  # cheap proxy for "requirements are installed here"
except ImportError as e:
    sys.exit("[arc_env] FATAL: %s in %s" % (e, sys.executable))
' || { echo "[arc_env] FATAL: python check failed (PY=$PY)" >&2; return 1 2>/dev/null || exit 1; }

cd "$PROJECT_DIR"

# ---- distributed rendezvous ----------------------------------------------- #
# One torchrun per node (srun --ntasks-per-node=1), torchrun spawns one process
# per GPU. MASTER_ADDR must be a routable hostname of the first allocated node.
export GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"

# ---- NCCL ------------------------------------------------------------------ #
# ARC's A100/H200 nodes are InfiniBand; let NCCL use IB and disable the Ethernet
# fallback that otherwise silently halves multi-node bandwidth. Flip
# NCCL_DEBUG=INFO when a multi-node job hangs at the first collective.
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# Intra-node NVLink topology is discovered automatically; P2P disable is only a
# debugging crutch (export NCCL_P2P_DISABLE=1) — it costs a lot of throughput.

echo "[arc_env] node=$(hostname) nodes=${SLURM_NNODES:-1} gpus/node=${GPUS_PER_NODE} \
master=${MASTER_ADDR}:${MASTER_PORT} hf_home=${HF_HOME}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# Launch helper: `arc_torchrun training/sft.py --model 32b`
arc_torchrun() {
  srun --ntasks-per-node=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
    torchrun \
      --nnodes "${SLURM_NNODES:-1}" \
      --nproc_per_node "${GPUS_PER_NODE}" \
      --rdzv_id "${SLURM_JOB_ID}" \
      --rdzv_backend c10d \
      --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
      "$@"
}
