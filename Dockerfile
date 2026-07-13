# Reproducible training/inference image for visual-reasoning-rlvr (A100 target).
# CPU-only users (env + reward + tests + demo) don't need this — a plain
# `pip install -r requirements.txt` covers the CORE profile.
#
#   docker build -t visual-reasoning-rlvr .
#   docker run --gpus all -it -v $PWD:/workspace visual-reasoning-rlvr \
#       python -m eval.harness --limit 50
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip git \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Torch first (CUDA 12.1 wheels), then the rest of the stack.
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Default: run the reward unit tests (also what CI runs). Override the command
# to build data / distill traces / train / evaluate.
CMD ["python", "-m", "pytest", "tests/", "-q"]
