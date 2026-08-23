#!/usr/bin/env bash
# Submit the whole pipeline as one afterok chain: distill -> SFT -> GRPO -> eval.
# RUN this (./scripts/train_all.sh); do not sbatch it — it is a submitter.
#
#   VRR_DATASET=synth1024 OVERVIEW_LONG_EDGE=48 ./scripts/train_all.sh
#   MODEL=7b ./scripts/train_all.sh              # small tier
#   SKIP_DISTILL=1 ./scripts/train_all.sh        # reuse existing traces
#   DRY_RUN=1 ./scripts/train_all.sh             # print the sbatch lines, submit nothing
#
# Everything the stages must agree on is exported here rather than left to the
# submitting shell: a chain where SFT trains at one overview resolution and eval
# measures at another produces numbers that are quietly meaningless.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${SBATCH_ACCOUNT:?export SBATCH_ACCOUNT=<your-allocation> first}"

# ---- what every stage must share ------------------------------------------ #
export VRR_DATASET="${VRR_DATASET:-genimage}"
export OVERVIEW_LONG_EDGE="${OVERVIEW_LONG_EDGE:?set OVERVIEW_LONG_EDGE — it must match the value the floor gate passed at}"
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export CONDA_ENV="${CONDA_ENV:-vrr}"
export MODEL="${MODEL:-32b}"

# ---- site ------------------------------------------------------------------ #
# NOT via SBATCH_PARTITION: each launcher carries its own `#SBATCH --partition`
# line, and a script directive outranks the environment variable. The only thing
# that wins is --partition on the command line.
PARTITION="${PARTITION:-h200_normal_q}"
GRES="${GRES:-gpu:h200:8}"
# "short" caps at 24h and has the HIGHEST priority on the cluster (2000 vs base
# 1000); only GRPO's 48h needs to drop to base and queue slower.
QOS_SHORT="${QOS_SHORT:-tc_h200_normal_short}"
QOS_LONG="${QOS_LONG:-tc_h200_normal_base}"
MAIL="${MAIL_USER:+--mail-user=$MAIL_USER}"

TAG="qwen2.5-vl-${MODEL}"                      # matches common.model_tag()
SFT_CKPT="checkpoints/$VRR_DATASET/sft-$TAG"   # matches common.checkpoint_dir()
GRPO_CKPT="checkpoints/$VRR_DATASET/grpo-$TAG"
TRACES="data/$VRR_DATASET/sft_traces.jsonl"
mkdir -p logs/slurm checkpoints

# ---- the gates are not optional ------------------------------------------- #
# Skipping them is how the faces run burned a full pipeline, and how genwukong
# spent a day of GPU measuring image size. Both are cheap; this is not.
if [ -z "${GATES_OK:-}" ]; then
  cat >&2 <<MSG
refusing to submit: set GATES_OK=1 to confirm this substrate cleared its gates.

  python tools/manifest_stats.py --dataset $VRR_DATASET      # geometry ~0.5
  JOB=ceiling AUC=1 ... scripts/arc_infer.slurm              # want AUC >=0.85
  JOB=floor   AUC=1 ... scripts/arc_infer.slurm              # want AUC ~0.5
                                                             # at OVERVIEW_LONG_EDGE=$OVERVIEW_LONG_EDGE
MSG
  exit 2
fi

submit() {   # submit <qos> <time> <script> ; echoes the job id
  local qos="$1" time="$2" script="$3"; shift 3
  local cmd=(sbatch --parsable --partition="$PARTITION" --qos="$qos"
             --gres="$GRES" --time="$time" $MAIL "$@" "$script")
  if [ -n "${DRY_RUN:-}" ]; then echo "DRY ${cmd[*]}" >&2; echo "000000"; return; fi
  "${cmd[@]}"
}

echo "dataset=$VRR_DATASET  model=$MODEL  overview=$OVERVIEW_LONG_EDGE  partition=$PARTITION"
echo

dep=""
if [ -s "$TRACES" ] && [ -n "${SKIP_DISTILL:-}" ]; then
  echo "stage 0  distill   SKIPPED ($(wc -l < "$TRACES") traces on disk)"
else
  D=$(JOB=distill submit "$QOS_SHORT" "${T_DISTILL:-08:00:00}" scripts/arc_infer.slurm)
  echo "stage 0  distill   $D"
  dep="--dependency=afterok:$D"
fi

S=$(submit "$QOS_SHORT" "${T_SFT:-12:00:00}" scripts/arc_sft.slurm $dep)
echo "stage 1  sft       $S   -> $SFT_CKPT"

# GRPO's 48h exceeds the short QOS 24h cap, so it queues on base and waits longer.
G=$(SFT_CKPT="$SFT_CKPT" submit "$QOS_LONG" "${T_GRPO:-48:00:00}" scripts/arc_grpo.slurm \
      --dependency=afterok:$S)
echo "stage 2  grpo      $G   -> $GRPO_CKPT"

E=$(JOB=eval ADAPTER="$GRPO_CKPT" \
    BUDGETS="${BUDGETS:-2,4}" DEGRADATIONS="${DEGRADATIONS:-clean,jpeg,blur_downscale}" \
    submit "$QOS_SHORT" "${T_EVAL:-08:00:00}" scripts/arc_infer.slurm \
      --dependency=afterok:$G)
echo "stage 3  eval      $E"

echo
echo "NOTE: gate 2 (group variance) is NOT in this chain — it must run against the"
echo "      SFT checkpoint BEFORE grpo is worth its 48h. Cancel $G if it fails:"
echo "        JOB=groupvar ADAPTER=$SFT_CKPT ... scripts/arc_infer.slurm   # want >=0.40"
echo
echo "cancel all:  scancel ${D:-} $S $G $E"
squeue -u "$USER" -o "%.10i %.12j %.2t %.10M %.10L %.6D %R"
