#!/usr/bin/env bash
# Stage 1 — SFT on ARC. Defaults: one 8-GPU node, Qwen2.5-VL-32B + LoRA, ZeRO-2.
#
#   sbatch scripts/arc_sft.slurm                        # 32B, 1 node
#   MODEL=72b DS=configs/deepspeed_zero3.json \
#     sbatch --nodes=2 scripts/arc_sft.slurm            # 72B, 2 nodes, ZeRO-3
#
# Partition/gres names differ per cluster (TinkerCliffs A100 vs Falcon H200) —
# confirm with `sinfo -o "%P %G %D %N"` and override PARTITION/--gres on submit.
#SBATCH --job-name=vrr-sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --partition=a100_normal_q
#SBATCH --output=logs/slurm/sft-%j.out
#SBATCH --error=logs/slurm/sft-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
# Mail address is deliberately NOT hardcoded — this file is tracked in git.
# SLURM delivers to $USER@<MailDomain> when the cluster sets one (check with
# `scontrol show config | grep -i mail`); otherwise pass --mail-user on submit:
#   sbatch --mail-user=you@vt.edu scripts/arc_infer.slurm

set -euo pipefail
mkdir -p logs/slurm "logs/${VRR_DATASET:-genimage}"
# SLURM copies the batch script to its spool dir, so ${BASH_SOURCE[0]} does NOT
# point into the repo — resolve against the submit dir instead.
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/arc_env.sh"

trap arc_notify_finish EXIT
arc_notify "<b>START</b> sft ${SLURM_JOB_ID:-?} · ${VRR_DATASET:-?} · ${SLURM_JOB_NODELIST:-?}"

MODEL="${MODEL:-32b}"
DS="${DS:-configs/deepspeed_zero2.json}"          # zero3(+_offload) for 72B / full FT
EPOCHS="${EPOCHS:-2}"
PER_DEVICE_BS="${PER_DEVICE_BS:-1}"               # multi-image contexts are big
GRAD_ACCUM="${GRAD_ACCUM:-8}"

# Effective batch = PER_DEVICE_BS × GRAD_ACCUM × (nodes × gpus/node).
arc_torchrun training/sft.py \
  --model "$MODEL" \
  --deepspeed "$DS" \
  --epochs "$EPOCHS" \
  --per-device-batch-size "$PER_DEVICE_BS" \
  --grad-accum "$GRAD_ACCUM" \
  --wandb-project "$WANDB_PROJECT" \
  --overview-long-edge "$OVERVIEW_LONG_EDGE" \
  "$@"
