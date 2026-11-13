#!/usr/bin/env bash
# Short GPU sanity check for a training env on a new GPU type.
#
#   scripts/arc_submit.sh check --b200        # Owl B200, vrr-train-b200 env
#   scripts/arc_submit.sh check               # A100, vrr-train env
#
# Proves, on the actual hardware, what a 32B job would otherwise find out after
# loading 66 GB of weights: torch has kernels for this GPU (sm_100 on B200),
# bf16 matmul runs, NCCL initialises, and the trainer stack imports.
#SBATCH --job-name=vrr-check
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm/check-%j.out
#SBATCH --error=logs/slurm/check-%j.err

set -euo pipefail
ARC_STAGE_ENV="${ARC_STAGE_ENV:-$HOME/.conda/envs/vrr-train}"
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/arc_env.sh"
arc_require trl.trainer.grpo_trainer peft

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$PY" - <<'EOF'
import torch, torch.distributed as dist, os
assert torch.cuda.is_available(), "torch sees no GPU"
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
arch = torch.cuda.get_arch_list()
print(f"[check] {name} capability sm_{cap[0]}{cap[1]} · torch {torch.__version__} "
      f"(CUDA {torch.version.cuda}) · compiled for {arch}")
need = f"sm_{cap[0]}{cap[1]}"
assert need in arch, f"torch has no kernels for {need}: rebuild the env (scripts/setup_env_b200.sh)"
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); print(f"[check] bf16 matmul ok: {(x @ x).float().abs().mean().item():.3f}")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29511")
dist.init_process_group("nccl", rank=0, world_size=1)
t = torch.ones(1, device="cuda"); dist.all_reduce(t); dist.destroy_process_group()
print(f"[check] NCCL ok · free memory {torch.cuda.mem_get_info()[0] / 2**30:.0f} GiB")
print("[check] PASS")
EOF
