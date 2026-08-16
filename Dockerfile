# Reproducible training/inference image for visual-reasoning-rlvr
# (A100-80GB / H200, single or multi-node). CPU-only users (env + reward + tests
# + demo) don't need this — a plain `pip install -r requirements.txt` covers the
# CORE profile.
#
#   docker build -t visual-reasoning-rlvr .
#   docker run --gpus all -it -v $PWD:/workspace visual-reasoning-rlvr \
#       python eval/harness.py --backend vllm --tensor-parallel-size 8 --limit 50
#
# On ARC, run it through Apptainer instead of Docker:
#   apptainer build vrr.sif docker-daemon://visual-reasoning-rlvr:latest
#   srun --gres=gpu:8 apptainer exec --nv vrr.sif python training/sft.py --model 32b
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.hf_cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TOKENIZERS_PARALLELISM=false \
    NCCL_ASYNC_ERROR_HANDLING=1

# devel (not runtime) base: DeepSpeed and FlashAttention compile CUDA ops.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3-dev git build-essential ninja-build \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Torch first (CUDA 12.4 wheels), then the rest of the stack.
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt .
RUN pip install -r requirements.txt

# FlashAttention-2 — optional at runtime (common.py falls back to SDPA), but a
# large memory win for the multi-image contexts this env produces. Slow to build;
# tolerate failure so the image still ships if the wheel fights the toolchain.
RUN pip install flash-attn --no-build-isolation || \
    echo "flash-attn build failed; falling back to SDPA at runtime."

COPY . .

# Default: run the reward unit tests (also what CI runs). Override the command
# to build data / distill traces / train / evaluate.
CMD ["python", "-m", "pytest", "tests/", "-q"]
