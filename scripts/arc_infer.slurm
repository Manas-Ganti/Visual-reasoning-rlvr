#!/usr/bin/env bash
# Offline inference on ARC: the two pre-training gates, trace distillation
# (Stage -1) and the eval harness. One node; vLLM holds a single tensor-parallel
# replica across its GPUs.
#
#   # --- gates: run these BEFORE spending a node on training -----------------
#   # Short and small — override the 8-GPU/8h defaults so they queue fast.
#   JOB=ceiling sbatch --gres=gpu:2 --time=00:30:00 scripts/arc_infer.slurm
#   JOB=floor   sbatch --gres=gpu:2 --time=00:30:00 scripts/arc_infer.slurm
#   # A failing ceiling is ambiguous: no signal, or signal the model won't commit
#   # to? AUC=1 scores the ranking instead of the sampled word and tells them apart.
#   JOB=ceiling AUC=1 sbatch --gres=gpu:1 --time=00:30:00 scripts/arc_infer.slurm
#   JOB=groupvar ADAPTER=checkpoints/genimage/sft-qwen2.5-vl-32b \
#     sbatch --gres=gpu:4 --time=02:00:00 scripts/arc_infer.slurm
#
#   # --- the expensive stages ------------------------------------------------
#   JOB=distill MODEL=72b sbatch scripts/arc_infer.slurm       # bigger teacher
#   JOB=eval ADAPTER=checkpoints/genimage/grpo-qwen2.5-vl-32b sbatch scripts/arc_infer.slurm
#   JOB=evidence sbatch scripts/arc_infer.slurm                # HF, rank-sharded
#
# Gate 1 (ceiling/floor) asks whether the SUBSTRATE is learnable: ceiling >=0.85
# at full resolution, floor ~chance through the env's overview. Gate 2 (groupvar)
# asks whether the POLICY sits where GRPO has a gradient — run it on the SFT
# checkpoint, want usable_groups >=0.40. Details in README "Before you train".
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

JOB="${JOB:-eval}"
MODEL="${MODEL:-32b}"
TP="${TP:-$GPUS_PER_NODE}"
BATCH_EPISODES="${BATCH_EPISODES:-32}"
LIMIT="${LIMIT:-}"

case "$JOB" in
  ceiling|floor)
    # Same probe, two conditions. The gap between them is the only room the
    # investigation has to work in, so both numbers are needed to read either.
    if [ "$JOB" = "ceiling" ]; then CONDITION=full; else CONDITION=overview; fi
    "$PY" tools/ceiling_probe.py \
      --model "$MODEL" --backend vllm --tensor-parallel-size "$TP" \
      --condition "$CONDITION" --limit "${LIMIT:-200}" \
      --overview-long-edge "$OVERVIEW_LONG_EDGE" \
      ${AUC:+--auc} \
      ${ADAPTER:+--adapter "$ADAPTER"} "$@"
    ;;
  groupvar)
    "$PY" tools/group_variance_probe.py \
      --model "$MODEL" --backend vllm --tensor-parallel-size "$TP" \
      --batch-episodes "$BATCH_EPISODES" \
      ${ADAPTER:+--adapter "$ADAPTER"} \
      --images "${IMAGES:-32}" -G "${G:-8}" \
      --degradation "${DEGRADATION:-clean}" \
      --max-inspects "${MAX_INSPECTS:-4}" \
      --overview-long-edge "$OVERVIEW_LONG_EDGE" "$@"
    ;;
  distill)
    # SEED/APPEND accumulate traces across passes: episodes are sampled, so a
    # second pass over the same images explores different trajectories. Needed
    # whenever the keep rate is low enough that one pass underfeeds SFT.
    "$PY" data/build_sft_traces.py \
      --model "$MODEL" --backend vllm --tensor-parallel-size "$TP" \
      --batch-episodes "$BATCH_EPISODES" ${LIMIT:+--limit "$LIMIT"} \
      --seed "${SEED:-0}" ${APPEND:+--append} \
      --overview-long-edge "$OVERVIEW_LONG_EDGE" "$@"
    ;;
  eval)
    "$PY" eval/harness.py \
      --model "$MODEL" --backend vllm --tensor-parallel-size "$TP" \
      --batch-episodes "$BATCH_EPISODES" \
      ${ADAPTER:+--adapter "$ADAPTER"} ${ADAPTER:+--compare-base} \
      --budgets "${BUDGETS:-2,4}" \
      --degradations "${DEGRADATIONS:-clean,jpeg,blur_downscale}" \
      ${LIMIT:+--limit "$LIMIT"} \
      --trace-log "${TRACE_LOG:-logs/$VRR_DATASET/eval_episodes.jsonl}" \
      --overview-long-edge "$OVERVIEW_LONG_EDGE" "$@"
    ;;
  evidence)
    arc_torchrun data/build_evidence_slice.py \
      --model "$MODEL" --propose --limit "${LIMIT:-120}" --topk "${TOPK:-3}" \
      --overview-long-edge "$OVERVIEW_LONG_EDGE" "$@"
    ;;
  *)
    echo "JOB must be one of: ceiling | floor | groupvar | distill | eval | evidence" >&2
    exit 2
    ;;
esac
