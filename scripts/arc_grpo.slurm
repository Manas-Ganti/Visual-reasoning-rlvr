#!/usr/bin/env bash
# Stage 2 — GRPO on ARC. Defaults: two 8-GPU nodes, 32B, ZeRO-2.
#
#   SFT_CKPT=checkpoints/sft-qwen2.5-vl-32b sbatch scripts/arc_grpo.slurm
#   MAX_INSPECTS=6 SFT_CKPT=... sbatch scripts/arc_grpo.slurm   # match the gate
#   sbatch --nodes=2 scripts/arc_grpo.slurm          # two nodes, ~half the wall clock
#
# Defaults to ONE node. Multi-node here crosses an InfiniBand fabric, and ARC's
# GPU partitions are heterogeneous: a 2-node request was satisfied with
# tc-dgx001 + tc-gpu002, whose HCAs do not talk to each other. Every rank
# loaded, the trainer built, and the first BROADCAST died with
#   NET/IB : ... status=5 ... vendor err 249 (Recv)
# followed by a 600s watchdog timeout on the ranks still waiting. A single node
# keeps all collectives on NVLink and queues far better besides. If you do want
# two, pin them to one type with --constraint and be ready to set
# NCCL_IB_DISABLE=1 to fall back to TCP.
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
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --partition=a100_normal_q
#SBATCH --output=logs/slurm/grpo-%j.out
#SBATCH --error=logs/slurm/grpo-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
# Mail address is deliberately NOT hardcoded — this file is tracked in git.
# SLURM delivers to $USER@<MailDomain> when the cluster sets one (check with
# `scontrol show config | grep -i mail`); otherwise pass --mail-user on submit:
#   sbatch --mail-user=you@vt.edu scripts/arc_infer.slurm

set -euo pipefail
ARC_STAGE_ENV="$HOME/.conda/envs/vrr-train"   # default env for this stage; CONDA_ENV overrides
# SLURM copies the batch script to its spool dir, so ${BASH_SOURCE[0]} does NOT
# point into the repo — resolve against the submit dir instead.
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/arc_env.sh"
mkdir -p logs/slurm "logs/${VRR_DATASET}"
arc_require trl.trainer.grpo_trainer peft

trap arc_notify_finish EXIT
arc_notify "<b>START</b> grpo ${SLURM_JOB_ID:-?} · ${VRR_DATASET:-?} · ${SLURM_JOB_NODELIST:-?}"

MODEL="${MODEL:-32b}"
SFT_CKPT="${SFT_CKPT:-checkpoints/${VRR_DATASET}/sft-qwen2.5-vl-32b}"
# SFT_CKPT=none trains a fresh LoRA on the base model. That is what the 1-GPU
# smoke test uses, since the 32B SFT adapter cannot load onto the 7B:
#   MODEL=7b SFT_CKPT=none NUM_GEN=4 GROUPS_PER_STEP=1 scripts/arc_submit.sh grpo \
#     --gres=gpu:1 --mem=120G --time=01:00:00 -- --max-steps 2 --logp-parity
# It must print "[forward-probe] {... all True} images=N" and a
# "[logp-parity]" line, or the full run is not worth submitting.
[ "$SFT_CKPT" = none ] && SFT_CKPT=""
DS="${DS:-configs/deepspeed_zero2.json}"
NUM_GEN="${NUM_GEN:-8}"                            # GRPO group size G
DEGRADATION="${DEGRADATION:-clean}"
# Must match the budget the gates were measured at: group_variance_probe is run
# with --max-inspects, and a policy that clears the gate at 6 has not been shown
# to clear it at grpo.py's default of 4. Unset here once already: the launcher
# accepted MAX_INSPECTS=6 on the command line and silently ignored it.
MAX_INSPECTS="${MAX_INSPECTS:-6}"      # synth1024 gates and the 0.763 target were at 6
GRAD_ACCUM="${GRAD_ACCUM:-}"                       # default: one GRPO group/step
# Per-TURN generation budget. 640 was sized for the old prompt; the evidence
# asymmetry paragraph and the three tag definitions ask for more writing per
# turn, and the group-variance gate fell from a 99.8% answer rate to 75.8% at
# the same inspect budget. A turn cut off before its ACTION line is unparseable,
# so the episode burns a turn and can end with no verdict at -1.00.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
# Batched rollouts. GROUPS_PER_STEP distinct images per optimizer step; each
# rank rolls out (8 x GROUPS_PER_STEP / ranks) episodes in ONE lockstep batch.
# Run 3 was effectively GROUPS_PER_STEP=1: one image per step, one episode per
# rank, nothing to batch. ROLLOUT_BATCH caps concurrent episodes per GPU for KV
# memory: leave unset on H200 (8 fits), set 4 on A100-80. Calibrate first:
#   GROUPS_PER_STEP=8 sbatch -p h200_normal_q scripts/arc_grpo.slurm --max-steps 5
GROUPS_PER_STEP="${GROUPS_PER_STEP:-8}"
ROLLOUT_BATCH="${ROLLOUT_BATCH:-}"

arc_torchrun training/grpo.py \
  --model "$MODEL" \
  ${SFT_CKPT:+--sft-checkpoint "$SFT_CKPT"} \
  --deepspeed "$DS" \
  --num-generations "$NUM_GEN" \
  --degradation "$DEGRADATION" \
  --wandb-project "$WANDB_PROJECT" \
  --overview-long-edge "$OVERVIEW_LONG_EDGE" \
  ${MAX_INSPECTS:+--max-inspects "$MAX_INSPECTS"} \
  ${RESUME:+--resume ${RESUME}} \
  ${GRAD_ACCUM:+--grad-accum "$GRAD_ACCUM"} \
  ${MAX_NEW_TOKENS:+--max-new-tokens "$MAX_NEW_TOKENS"} \
  --groups-per-step "$GROUPS_PER_STEP" \
  ${ROLLOUT_BATCH:+--rollout-batch-size "$ROLLOUT_BATCH"} \
  "$@"
