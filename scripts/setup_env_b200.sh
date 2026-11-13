#!/usr/bin/env bash
# Build the SFT/GRPO training env for Owl's B200 GPUs (Blackwell, sm_100).
#
#   bash scripts/setup_env_b200.sh            # on owl1, takes ~10 min
#
# Why a separate env: vrr-train is torch 2.6.0+cu126, which has no Blackwell
# kernels ("no kernel image is available for execution on the device").
# Blackwell needs CUDA 12.8 wheels, first shipped with torch 2.7. Everything
# else is pinned to the versions vrr-train runs with (docs/arc_runbook.md §3),
# so a B200 run differs from an A100/H200 run by the GPU and torch build only.
#
# No flash-attn: training/common.py falls back to SDPA without it, and building
# it for sm_100 is its own project.
#
# The env path is B200_ENV, never the generic CONDA_ENV: another project's
# ~/.bashrc exporting CONDA_ENV once made a setup script install into the wrong
# env (call centre: setup-script-installed-into-another-project-s-conda-env).

set -euo pipefail

ENV="${B200_ENV:-$HOME/.conda/envs/vrr-train-b200}"
CONDA="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
[ -x "$CONDA" ] || CONDA="$(command -v conda || true)"
[ -n "$CONDA" ] || { echo "setup_env_b200: no conda found (module load Miniforge3?)" >&2; exit 1; }

export PYTHONNOUSERSITE=1          # never install into ~/.local
# pip builds in $TMPDIR; /tmp and /home are different filesystems on ARC, which
# breaks some builds with "Invalid cross-device link".
export TMPDIR="$HOME/tmp/pip-b200"; mkdir -p "$TMPDIR"

echo "[setup] env=$ENV conda=$CONDA"
[ -x "$ENV/bin/python" ] || "$CONDA" create -y -p "$ENV" python=3.11
PY="$ENV/bin/python"
"$PY" -V | grep -q "Python 3.11" || { echo "[setup] $PY is not a working Python 3.11" >&2; exit 1; }

"$PY" -m pip install --upgrade pip
"$PY" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install \
  "transformers==5.15.0" "trl==1.9.2" "peft==0.20.0" "accelerate==1.14.0" "deepspeed==0.19.5" \
  "datasets>=4.7" wandb "numpy>=1.26,<2.3" "pillow>=10.0" "gymnasium>=1.0" "scipy>=1.11"

# Login-node check. Login nodes have no GPU, and importing deepspeed (which the
# TRL trainer modules pull in) starts triton, which dies there with
# "RuntimeError: 0 active drivers ([])". That is the node, not the env. So only
# versions are checked here; scripts/arc_check.slurm imports the trainer on a GPU.
"$PY" - <<'EOF'
from importlib.metadata import version
import torch, transformers, peft
print(f"[setup] torch {torch.__version__} (CUDA {torch.version.cuda}) · transformers "
      f"{transformers.__version__} · trl {version('trl')} · peft {peft.__version__} · "
      f"deepspeed {version('deepspeed')}")
assert torch.version.cuda and tuple(map(int, torch.version.cuda.split(".")[:2])) >= (12, 8), \
    "Blackwell needs a CUDA >= 12.8 torch build"
EOF
touch "$ENV/.vrr-train-b200"
echo "[setup] done. Verify on a B200 before training:"
echo "  scripts/arc_submit.sh check --b200"
