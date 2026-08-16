#!/usr/bin/env bash
# Offline inference on ARC: trace distillation (Stage 0→1) and the eval harness.
# One node; vLLM holds a single tensor-parallel replica across its GPUs.
#
#   JOB=distill MODEL=72b sbatch scripts/arc_infer.slurm       # bigger teacher
#   JOB=eval ADAPTER=checkpoints/grpo-qwen2.5-vl-32b sbatch scripts/arc_infer.slurm
#   JOB=evidence sbatch scripts/arc_infer.slurm                # HF, rank-sharded
#
# vLLM tensor parallelism and torchrun rank-sharding are alternatives, never
# combined: the first puts one big model across the GPUs, the second puts one
# replica on each. Distill/eval default to vLLM; the evidence slice reads
# next-token logits, so it stays on the HF backend and shards by rank instead.
#SBATCH --job-name=vrr-infer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --partition=a100_normal_q
#SBATCH --output=logs/slurm/infer-%j.out
#SBATCH --error=logs/slurm/infer-%j.err

set -euo pipefail
mkdir -p logs/slurm "logs/${VRR_DATASET:-genimage}"
# SLURM copies the batch script to its spool dir, so ${BASH_SOURCE[0]} does NOT
# point into the repo — resolve against the submit dir instead.
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/arc_env.sh"

JOB="${JOB:-eval}"
MODEL="${MODEL:-32b}"
TP="${TP:-$GPUS_PER_NODE}"
BATCH_EPISODES="${BATCH_EPISODES:-32}"
LIMIT="${LIMIT:-}"

case "$JOB" in
  distill)
    python data/build_sft_traces.py \
      --model "$MODEL" --backend vllm --tensor-parallel-size "$TP" \
      --batch-episodes "$BATCH_EPISODES" ${LIMIT:+--limit "$LIMIT"} "$@"
    ;;
  eval)
    python eval/harness.py \
      --model "$MODEL" --backend vllm --tensor-parallel-size "$TP" \
      --batch-episodes "$BATCH_EPISODES" \
      ${ADAPTER:+--adapter "$ADAPTER"} ${ADAPTER:+--compare-base} \
      --budgets "${BUDGETS:-2,4}" \
      --degradations "${DEGRADATIONS:-clean,jpeg,blur_downscale}" \
      ${LIMIT:+--limit "$LIMIT"} \
      --trace-log "${TRACE_LOG:-logs/$VRR_DATASET/eval_episodes.jsonl}" "$@"
    ;;
  evidence)
    arc_torchrun data/build_evidence_slice.py \
      --model "$MODEL" --propose --limit "${LIMIT:-120}" --topk "${TOPK:-3}" "$@"
    ;;
  *)
    echo "JOB must be one of: distill | eval | evidence" >&2
    exit 2
    ;;
esac
