#!/usr/bin/env bash
# Stage 2 — GRPO on ARC. Defaults: two 8-GPU nodes, 32B, ZeRO-2.
#
#   SFT_CKPT=checkpoints/sft-qwen2.5-vl-32b sbatch scripts/arc_grpo.slurm
#   sbatch --nodes=1 scripts/arc_grpo.slurm          # single node, ~2× the wall clock
#
# ZeRO-2 on purpose: every rollout turn calls generate(), and under ZeRO-3 that
# gathers the sharded parameters once per turn. Stay on ZeRO-2 unless the policy
# genuinely does not fit, then move to zero3(+offload) and expect it to be slower.
#
# Wall-clock here is dominated by rollouts, not by the optimizer step: each
# episode is `--max-inspects + 1` sequential generates, and GRPO needs
# `--num-generations` episodes per group. Scale by nodes, and keep the global
# rollout batch a multiple of the group size (the script checks this and tells
# you what to change).
#
# Why 2 nodes and 48h: rollout decode is HBM-bound (ZeRO-2 replicates the full
# 66GB bf16 32B policy per GPU, so each token is a full weight sweep — ~18s per
# turn on an A100-80, ~8s on an H200). One epoch over the ~1k-image train split
# is one optimizer step per unique prompt, so a single A100 node lands at 22–34h
# and does not fit a 24h limit. Nodes scale it near-linearly: world=16 keeps
# grad_accum at 1 and puts two groups in each step, halving the step count to
# ~515 for 11–17h on A100 / 5–8h on H200. The 48h limit also leaves --nodes=1
# runnable without editing this file. Calibrate with --max-steps 5 before
# committing a full allocation.
#SBATCH --job-name=vrr-grpo
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --partition=a100_normal_q
#SBATCH --output=logs/slurm/grpo-%j.out
#SBATCH --error=logs/slurm/grpo-%j.err

set -euo pipefail
mkdir -p logs/slurm "logs/${VRR_DATASET:-genimage}"
# SLURM copies the batch script to its spool dir, so ${BASH_SOURCE[0]} does NOT
# point into the repo — resolve against the submit dir instead.
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/arc_env.sh"

MODEL="${MODEL:-32b}"
SFT_CKPT="${SFT_CKPT:-checkpoints/${VRR_DATASET}/sft-qwen2.5-vl-32b}"
DS="${DS:-configs/deepspeed_zero2.json}"
NUM_GEN="${NUM_GEN:-8}"                            # GRPO group size G
DEGRADATION="${DEGRADATION:-clean}"

arc_torchrun training/grpo.py \
  --model "$MODEL" \
  --sft-checkpoint "$SFT_CKPT" \
  --deepspeed "$DS" \
  --num-generations "$NUM_GEN" \
  --degradation "$DEGRADATION" \
  --wandb-project "$WANDB_PROJECT" \
  --overview-long-edge "$OVERVIEW_LONG_EDGE" \
  "$@"
