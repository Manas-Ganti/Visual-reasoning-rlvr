#!/usr/bin/env bash
# Submit an ARC job with the scheduling flags filled in.
#
#   scripts/arc_submit.sh <grpo|sft|infer|synth> [--h200|--a100] [sbatch flags...] [-- python args...]
#
# Examples:
#   # 7B smoke test of the GRPO forward-pass fix: 1 A100, 2 steps
#   MODEL=7b SFT_CKPT=none NUM_GEN=4 GROUPS_PER_STEP=1 \
#     scripts/arc_submit.sh grpo --gres=gpu:1 --mem=120G --time=01:00:00 -- --max-steps 2 --logp-parity
#
#   # 32B calibration, one H200 node
#   GROUPS_PER_STEP=8 scripts/arc_submit.sh grpo --h200 --time=02:00:00 -- --max-steps 5 --logp-parity
#
#   # eval
#   JOB=eval ADAPTER=checkpoints/synth1024/grpo-qwen2.5-vl-32b-c3 scripts/arc_submit.sh infer --gres=gpu:2 --time=03:00:00
#
# What it adds, unless the same flag is already on the line:
#   --account    $ARC_ACCOUNT (default ece-6524-spring2026)
#   --partition  <gpu>_normal_q, gpu from --h200/--a100, else $ARC_GPU, else a100
#   --qos        tc_<gpu>_normal_short when the job is <= 24h (the highest-priority
#                QOS), tc_<gpu>_normal_base when longer. The time comes from your
#                --time, else from the launcher's own #SBATCH --time.
#   --mail-user  $ARC_MAIL_USER, if set
# and on H200 exports NCCL_SOCKET_IFNAME=eth0 (ib0 there has no IPv4: a silent
# 10-minute NCCL failure). Leave it unset on A100, where arc_env.sh probes.
#
# The job's environment (dataset, overview size, HF/W&B dirs, conda env) needs no
# variables either: arc_env.sh holds the project defaults and each launcher picks
# its env. Personal values go in ~/.config/vrr/site.env, e.g.
#   export ARC_MAIL_USER=you@vt.edu
#   export ARC_GPU=h200
#
# DRY_RUN=1 prints the sbatch line instead of submitting.

set -euo pipefail

usage() { sed -n '2,4p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-1}"; }
[ $# -ge 1 ] || usage
case "$1" in -h|--help) usage 0 ;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"     # #SBATCH --output paths and SLURM_SUBMIT_DIR are relative to here

stage="$1"; shift
script="scripts/arc_${stage}.slurm"
[ -f "$script" ] || { echo "arc_submit: no launcher $script (grpo|sft|infer|synth)" >&2; exit 1; }

[ -f "$HOME/.config/vrr/site.env" ] && source "$HOME/.config/vrr/site.env"
account="${ARC_ACCOUNT:-ece-6524-spring2026}"
gpu="${ARC_GPU:-a100}"

flags=(); pyargs=(); partition=""; qos=""; time=""; have_account=0; have_mail=0
while [ $# -gt 0 ]; do
  case "$1" in
    --) shift; pyargs=("$@"); break ;;
    --h200) gpu=h200 ;;
    --a100) gpu=a100 ;;
    --partition=*) partition="${1#*=}" ;;
    -p|--partition) partition="$2"; shift ;;
    --qos=*) qos="${1#*=}" ;;
    -q|--qos) qos="$2"; shift ;;
    --time=*) time="${1#*=}"; flags+=("$1") ;;
    -t|--time) time="$2"; flags+=("$1" "$2"); shift ;;
    --account=*|-A) have_account=1; flags+=("$1"); [ "$1" = -A ] && { flags+=("$2"); shift; } ;;
    --mail-user=*) have_mail=1; flags+=("$1") ;;
    *) flags+=("$1") ;;
  esac
  shift
done

if [ -n "$partition" ]; then
  case "$partition" in h200*) gpu=h200 ;; a100*) gpu=a100 ;; esac
else
  partition="${gpu}_normal_q"
fi

# Wall time in minutes, from any SLURM time format: MM, MM:SS, HH:MM:SS,
# D-HH, D-HH:MM, D-HH:MM:SS.
to_minutes() {
  local t="$1" d=0 h=0 m=0
  if [[ "$t" == *-* ]]; then
    d="${t%%-*}"; t="${t#*-}"
    IFS=: read -r h m _ <<< "$t"
  else
    local n; n=$(awk -F: '{print NF}' <<< "$t")
    case "$n" in
      1) m="$t" ;;
      2) IFS=: read -r m _ <<< "$t" ;;
      *) IFS=: read -r h m _ <<< "$t" ;;
    esac
  fi
  echo $(( 10#${d:-0} * 1440 + 10#${h:-0} * 60 + 10#${m:-0} ))
}

if [ -z "$qos" ]; then
  [ -n "$time" ] || time="$(sed -n 's/^#SBATCH --time=//p' "$script" | head -1)"
  if [ -n "$time" ] && [ "$(to_minutes "$time")" -le 1440 ]; then
    qos="tc_${gpu}_normal_short"
  else
    qos="tc_${gpu}_normal_base"
  fi
fi

cmd=(sbatch --partition="$partition" --qos="$qos")
[ "$have_account" = 1 ] || cmd+=(--account="$account")
[ "$have_mail" = 1 ] || [ -z "${ARC_MAIL_USER:-}" ] || cmd+=(--mail-user="$ARC_MAIL_USER")
cmd+=(${flags[@]+"${flags[@]}"} "$script" ${pyargs[@]+"${pyargs[@]}"})

if [ "$gpu" = h200 ] && [ -z "${NCCL_SOCKET_IFNAME:-}" ]; then
  export NCCL_SOCKET_IFNAME=eth0
fi

if [ "${DRY_RUN:-0}" = 1 ]; then
  [ "$gpu" = h200 ] && echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-}"
  printf '%q ' "${cmd[@]}"; echo
else
  exec "${cmd[@]}"
fi
