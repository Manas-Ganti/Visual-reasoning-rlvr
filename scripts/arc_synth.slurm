#!/usr/bin/env bash
# Build a paired real/synthetic substrate on ARC (data/build_paired_synthetic.py).
#
# Four stages, submitted separately because they want different things: `caption`
# needs the VLM env, `generate` needs diffusers (which tends to pull a newer
# transformers than vLLM pins — keep them in separate envs), and `crop`/`assemble`
# need no GPU at all and are faster run on the login node.
#
#   # 1. reals: no GPU, run on the login node
#   python data/build_paired_synthetic.py --stage crop --dataset synth1024 \
#       --real-src <hf-repo-or-dir> --per-class 400
#
#   # 2. captions: the VLM
#   STAGE=caption DATASET=synth1024 sbatch --gres=gpu:h200:1 --time=01:00:00 \
#       --qos=tc_h200_normal_short scripts/arc_synth.slurm
#
#   # 3. generation: diffusers (CONDA_ENV=<gen env>)
#   STAGE=generate DATASET=synth1024 CONDA_ENV=/path/to/envs/vrr-gen \
#       sbatch --gres=gpu:h200:1 --time=02:00:00 scripts/arc_synth.slurm
#
#   # 4. assemble + verify: no GPU
#   python data/build_paired_synthetic.py --stage assemble --dataset synth1024
#   python tools/manifest_stats.py --dataset synth1024      # every predictor ~0.5
#
# Every stage is restartable — each skips what is already on disk — so a preempted
# or timed-out job is resumed by resubmitting the identical line.
#SBATCH --job-name=vrr-synth
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --partition=a100_normal_q
#SBATCH --output=logs/slurm/synth-%j.out
#SBATCH --error=logs/slurm/synth-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
# Mail address is deliberately NOT hardcoded — this file is tracked in git.
# SLURM delivers to $USER@<MailDomain> when the cluster sets one; otherwise pass
# --mail-user on submit. Partition likewise: the directive above is only a
# default, and --partition on the sbatch line overrides it.

set -euo pipefail
mkdir -p logs/slurm "logs/${VRR_DATASET:-genimage}"
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/arc_env.sh"

STAGE="${STAGE:-caption}"
DATASET="${DATASET:-$VRR_DATASET}"
MODEL="${MODEL:-32b}"
TP="${TP:-$GPUS_PER_NODE}"

case "$STAGE" in
  caption)
    # vLLM backend: one tensor-parallel replica, batched captioning.
    "$PY" data/build_paired_synthetic.py \
      --stage caption --dataset "$DATASET" \
      --model "$MODEL" --backend vllm --tensor-parallel-size "$TP" \
      --batch "${BATCH:-16}" \
      --generator "${GENERATOR:-sdxl}" "$@"
    ;;
  generate)
    "$PY" data/build_paired_synthetic.py \
      --stage generate --dataset "$DATASET" \
      --sdxl "${SDXL:-stabilityai/stable-diffusion-xl-base-1.0}" \
      ${CAPTIONS:+--captions "$CAPTIONS"} \
      --steps "${STEPS:-30}" --guidance "${GUIDANCE:-5.5}" \
      --size "${SIZE:-1024}" --generator "${GENERATOR:-sdxl}" "$@"
    ;;
  crop|assemble)
    # Allowed here for convenience, but neither needs a GPU — prefer the login node.
    "$PY" data/build_paired_synthetic.py \
      --stage "$STAGE" --dataset "$DATASET" \
      --size "${SIZE:-1024}" --per-class "${PER_CLASS:-400}" \
      --generator "${GENERATOR:-sdxl}" \
      ${REAL_SRC:+--real-src "$REAL_SRC"} ${REAL_DIR:+--real-dir "$REAL_DIR"} "$@"
    ;;
  *)
    echo "STAGE must be one of: crop | caption | generate | assemble" >&2
    exit 2
    ;;
esac
