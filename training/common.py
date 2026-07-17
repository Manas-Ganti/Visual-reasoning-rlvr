"""Shared helpers for the training + eval scripts: device/dtype resolution,
model loading, and the multi-turn rollout that drives the environment with a VLM.

Base model is Qwen2-VL-7B-Instruct (fallback 2B for compute). Kept import-light
at module load — torch/transformers are only imported inside functions — so the
CPU-only test/CI environment can import this module without a GPU stack.
"""

from __future__ import annotations

import os
import re

DEFAULT_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join(REPO_ROOT, "data", "manifest.jsonl")

_INDEX_RE = re.compile(r"index=(\d+)")
_DEGRADE_RE = re.compile(r"degradation=(\w+)")


# --------------------------------------------------------------------------- #
# Device / dtype
# --------------------------------------------------------------------------- #
def resolve_device(choice: str = "auto") -> str:
    import torch

    def _mps() -> bool:
        return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()

    if choice == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda requested but CUDA unavailable; using auto.")
        choice = "auto"
    if choice == "mps" and not _mps():
        print("WARNING: --device mps requested but MPS unavailable; using auto.")
        choice = "auto"
    if choice in ("cpu", "cuda", "mps"):
        return choice
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if _mps() else "cpu"


def resolve_dtype(device: str, use_bf16: bool = True):
    import torch

    if device == "cuda":
        return torch.bfloat16 if use_bf16 else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_policy(model_name: str, adapter: str | None, device: str, dtype):
    """Load a VLM policy (+ optional LoRA adapter) and its processor."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name, padding_side="left")
    model = AutoModelForImageTextToText.from_pretrained(model_name, dtype=dtype)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.to(device)
    model.eval()
    return model, processor


# --------------------------------------------------------------------------- #
# Seed dataset (GRPO consumes one row per sample; the env produces the real,
# image-grounded conversation at rollout time from the encoded index).
# --------------------------------------------------------------------------- #
def build_seed_dataset(indices, degradation: str = "clean"):
    from datasets import Dataset

    rows = [
        {"prompt": [{"role": "user", "content": f"index={i} degradation={degradation}"}]}
        for i in indices
    ]
    return Dataset.from_list(rows)


def seed_index(prompt) -> int:
    return int(_INDEX_RE.search(_flatten(prompt)).group(1))


def seed_degradation(prompt, default: str = "clean") -> str:
    m = _DEGRADE_RE.search(_flatten(prompt))
    return m.group(1) if m else default


def _flatten(prompt) -> str:
    if isinstance(prompt, list):
        return " ".join(
            part.get("content", "") if isinstance(part, dict) else str(part) for part in prompt
        )
    return str(prompt)


# --------------------------------------------------------------------------- #
# Multi-turn rollout
# --------------------------------------------------------------------------- #
def build_inputs(processor, obs, device):
    text = processor.apply_chat_template(
        obs["messages"], tokenize=False, add_generation_prompt=True
    )
    return processor(text=[text], images=obs["images"], return_tensors="pt", padding=True).to(device)


def run_episode(
    model,
    processor,
    env,
    index: int,
    max_turns: int,
    degradation: str = "clean",
    sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 320,
    collect_tokens: bool = True,
):
    """Drive one full episode with HF ``generate`` (one action per turn).

    When ``collect_tokens`` is set, also returns the flat token bookkeeping GRPO
    consumes: ``completion_ids`` / ``logprobs`` cover only the policy's generated
    tokens; the env's tool-result tokens ride in the context but not the gradient.
    Eval passes ``collect_tokens=False`` for a lighter greedy pass.
    """
    import torch

    device = next(model.parameters()).device
    obs, info = env.reset(options={"index": index, "degradation": degradation})

    prompt_ids = None
    completion_ids: list[int] = []
    logprobs: list[float] = []
    episode_return = 0.0

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=sample,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.1,
    )

    with torch.no_grad():
        for _ in range(max_turns):
            inputs = build_inputs(processor, obs, device)
            out = model.generate(
                **inputs, return_dict_in_generate=True, output_scores=collect_tokens, **gen_kwargs
            )
            gen_len = out.sequences.shape[1] - inputs["input_ids"].shape[1]
            new_tokens = out.sequences[0, -gen_len:]

            if collect_tokens:
                trans = model.compute_transition_scores(
                    out.sequences, out.scores, normalize_logits=True
                )[0]
                if prompt_ids is None:
                    prompt_ids = inputs["input_ids"][0].tolist()
                completion_ids.extend(new_tokens.tolist())
                logprobs.extend(trans.tolist())

            action_text = processor.decode(new_tokens, skip_special_tokens=True)
            obs, reward, terminated, truncated, info = env.step(action_text)
            episode_return += reward
            if terminated or truncated:
                break

    return {
        "prompt_ids": prompt_ids or [],
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "episode_reward": episode_return,
        "correct": bool(info.get("correct")),
        "answered": info.get("action_type") == "verdict",
        "steps": info.get("steps", 0),
        "inspects_used": info.get("inspects_used", 0),
        "confidence": info.get("confidence"),
        "ground_truth": info.get("ground_truth"),
        "predicted": info.get("predicted_verdict"),
    }
