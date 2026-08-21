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
# budget. INSPECT's zoom factor is native_size/this, so aim for ~native/7: 140
# suits 1024px substrates, 70 suits 512px ones. Every launcher passes it, so
# setting it here keeps a run's stages consistent with each other.
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
module reset >/dev/null 2>&1 || true
module load Miniforge3 >/dev/null 2>&1 || module load Anaconda3 >/dev/null 2>&1 || true
source activate "${CONDA_ENV:-vrr}" 2>/dev/null || conda activate "${CONDA_ENV:-vrr}"

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
