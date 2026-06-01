"""GRPO fine-tuning of a VLM in the active-perception forgery-detection env.

This uses TRL's ``GRPOTrainer`` with a custom ``rollout_func`` so that each
"completion" is a full multi-turn episode (zoom / metadata / answer) driven by
``ForgeryDetectionEnv`` rather than a single forward generation. GRPO then forms
group-relative advantages from the per-episode returns and updates the policy.

Requirements (not installed by default in this repo):
    pip install "trl>=0.29" peft accelerate
A CUDA GPU (bf16) is the intended target. Apple Silicon (MPS) and CPU are
supported for small smoke runs via ``--device`` / ``--no-bf16`` / ``--max-steps``
(slow, and best-effort — a 3B VLM doing multi-turn rollouts on MPS is heavy).

Design notes
------------
* The dataset row is just a *seed*: a minimal prompt carrying the manifest index.
  rollout_func parses that index, resets the env to that sample, and runs the
  real (image-grounded) conversation. This cleanly decouples TRL's
  one-row-per-prompt expectation from our multi-turn, multi-image rollout.
* num_generations (the GRPO group size G) makes TRL repeat each seed G times;
  because rollouts sample (temperature > 0), each repeat is a different
  trajectory, giving GRPO the within-group reward spread it needs.
* MULTI-TURN TOKEN BOOKKEEPING is the part to validate against your installed
  TRL version. We treat the concatenation of the policy's generated tokens as
  the completion and drop the env's tool-result tokens from the gradient path.
  See TRL's openenv / browsergym GRPO examples if your version expects a
  different masking convention.

Smoke test (no training, no trl required):
    python train_grpo.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import re

import torch

from src.environment import ForgeryDetectionEnv, RewardConfig
from src import utils

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(HERE, "data", "metadata.jsonl")
OUTPUT_DIR = os.path.join(HERE, "checkpoints", "grpo-smolvlm-500m")
TRACE_LOG_PATH = os.path.join(HERE, "logs", "episodes.jsonl")

# Small image VLM (Idefics3) that trains on Apple Silicon / modest disk.
# Swap for Qwen/Qwen2.5-VL-3B-Instruct (or larger) on a CUDA box via --model.
MODEL_NAME = "HuggingFaceTB/SmolVLM-500M-Instruct"
MAX_TURNS = 6                # investigation budget per episode
HOLDOUT_FRACTION = 0.1       # tail of the manifest reserved for evaluate.py

_INDEX_RE = re.compile(r"index=(\d+)")


# --------------------------------------------------------------------------- #
# Device / dtype selection (shared with evaluate.py)
# --------------------------------------------------------------------------- #
def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def resolve_device(choice: str = "auto") -> str:
    """Resolve a device string. 'auto' prefers cuda > mps > cpu."""
    if choice == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda requested but CUDA is unavailable; using auto.")
        choice = "auto"
    if choice == "mps" and not _mps_available():
        print("WARNING: --device mps requested but MPS is unavailable; using auto.")
        choice = "auto"
    if choice in ("cpu", "cuda", "mps"):
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return "cpu"


def resolve_dtype(device: str, use_bf16: bool = True):
    """Pick a load dtype. bf16 only on CUDA; MPS uses fp16 (bf16 is partial
    there); CPU stays fp32 for correctness."""
    if device == "cuda":
        return torch.bfloat16 if use_bf16 else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


# --------------------------------------------------------------------------- #
# Dataset: one seed row per training sample
# --------------------------------------------------------------------------- #
def build_seed_dataset(num_records: int):
    """Build the TRL dataset. Each row's prompt only encodes the manifest index;
    the actual image-grounded conversation is produced by the env at rollout
    time. We hold out the tail of the manifest for evaluation."""
    from datasets import Dataset

    n_train = int(num_records * (1 - HOLDOUT_FRACTION))
    rows = [
        {"prompt": [{"role": "user", "content": f"index={i}"}]}
        for i in range(n_train)
    ]
    return Dataset.from_list(rows), n_train


def _seed_index(prompt) -> int:
    """Recover the manifest index from a seed prompt (str or chat list)."""
    if isinstance(prompt, list):
        prompt = " ".join(
            part.get("content", "") if isinstance(part, dict) else str(part)
            for part in prompt
        )
    m = _INDEX_RE.search(str(prompt))
    if not m:
        raise ValueError(f"Could not find index= in seed prompt: {prompt!r}")
    return int(m.group(1))


# --------------------------------------------------------------------------- #
# Multi-turn rollout
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_episode(model, processor, env, index, max_turns, sample=True,
                temperature=0.7, top_p=0.9):
    """Drive one full episode through the env, generating one action per turn.

    Returns the flat token bookkeeping GRPO consumes plus the scalar return.
    ``completion_ids`` / ``logprobs`` cover only the policy's generated tokens;
    env tool-result tokens are part of the running context but not the gradient.
    """
    device = next(model.parameters()).device
    obs, info = env.reset(options={"index": index})

    prompt_ids: list[int] | None = None
    completion_ids: list[int] = []
    logprobs: list[float] = []
    episode_return = 0.0

    # Small VLMs degenerate into unparseable actions at temperature=1.0; a
    # tighter nucleus + mild repetition penalty keeps actions valid while
    # leaving enough trajectory diversity for GRPO's within-group reward spread.
    gen_kwargs = dict(
        max_new_tokens=256, do_sample=sample,
        temperature=temperature, top_p=top_p, repetition_penalty=1.1,
    )

    for _ in range(max_turns):
        text = processor.apply_chat_template(
            obs["messages"], tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text], images=obs["images"], return_tensors="pt", padding=True
        ).to(device)

        out = model.generate(
            **inputs,
            return_dict_in_generate=True,
            output_scores=True,
            **gen_kwargs,
        )
        gen_len = out.sequences.shape[1] - inputs["input_ids"].shape[1]
        new_tokens = out.sequences[0, -gen_len:]

        # Per-token logprobs of exactly the tokens that were generated.
        transition = model.compute_transition_scores(
            out.sequences, out.scores, normalize_logits=True
        )[0]

        if prompt_ids is None:
            prompt_ids = inputs["input_ids"][0].tolist()
        completion_ids.extend(new_tokens.tolist())
        logprobs.extend(transition.tolist())

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
        "answered": info.get("action_type") == utils.ANSWER,
        "steps": info.get("steps", 0),
        "metadata_seen": bool(info.get("metadata_seen")),
    }


class EpisodeMetrics:
    """Accumulates per-episode outcomes for a terminal metrics readout."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.reward_sum = 0.0
        self.correct = 0
        self.answered = 0
        self.steps_sum = 0
        self.metadata = 0

    def add(self, ep: dict):
        self.n += 1
        self.reward_sum += ep["episode_reward"]
        self.correct += int(ep["correct"])
        self.answered += int(ep["answered"])
        self.steps_sum += ep["steps"]
        self.metadata += int(ep["metadata_seen"])

    def line(self, step: int) -> str:
        n = max(self.n, 1)
        return (
            f"[step {step:>4}] episodes={self.n} "
            f"reward={self.reward_sum / n:+.3f} "
            f"acc={self.correct / n:.3f} "
            f"avg_steps={self.steps_sum / n:.2f} "
            f"answer_rate={self.answered / n:.2f} "
            f"metadata_rate={self.metadata / n:.2f}"
        )


def make_rollout_func(env, processor, max_turns, trace_logger=None,
                      temperature=0.7, top_p=0.9):
    def rollout_func(prompts, trainer):
        model = trainer.model
        step = int(getattr(trainer.state, "global_step", 0))
        metrics = EpisodeMetrics()
        batch = {
            "prompt_ids": [],
            "completion_ids": [],
            "logprobs": [],
            "episode_reward": [],
        }
        for prompt in prompts:
            ep = run_episode(
                model, processor, env, _seed_index(prompt), max_turns,
                sample=True, temperature=temperature, top_p=top_p,
            )
            batch["prompt_ids"].append(ep["prompt_ids"])
            batch["completion_ids"].append(ep["completion_ids"])
            batch["logprobs"].append(ep["logprobs"])
            batch["episode_reward"].append(ep["episode_reward"])
            metrics.add(ep)
            if trace_logger is not None:
                trace_logger.log(env.get_trace(global_step=step, phase="train"))

        # Terminal metrics line, alongside TRL's own loss/KL logging.
        print(metrics.line(step), flush=True)
        return batch

    return rollout_func


def episode_reward_func(completions, episode_reward=None, **kwargs):
    """GRPO reward: the env return computed during rollout, passed straight
    through. (The signature matches TRL's reward-function contract; the
    ``episode_reward`` kwarg is the extra key returned by rollout_func.)"""
    return list(episode_reward)


# --------------------------------------------------------------------------- #
# Training entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--manifest", default=MANIFEST_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    parser.add_argument("--num-generations", type=int, default=4,
                        help="GRPO group size G (must divide global batch = "
                             "per_device_batch * grad_accum * num_devices = 4 here).")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                        help="Compute device. 'auto' prefers cuda > mps > cpu.")
    parser.add_argument("--no-bf16", action="store_true",
                        help="Disable bf16 (required on MPS/CPU; uses fp16/fp32 instead).")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Cap optimizer steps for a quick smoke run (-1 = full run).")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Rollout sampling temperature (1.0 makes small VLMs degenerate).")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Rollout nucleus sampling top-p.")
    parser.add_argument("--trace-log", default=TRACE_LOG_PATH,
                        help="JSONL file for episode reasoning traces (browser replay).")
    parser.add_argument("--trace-every", type=int, default=8,
                        help="Log every Nth rollout episode (GRPO emits many).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build env + dataset and exit (no model / no trl).")
    args = parser.parse_args()

    env = ForgeryDetectionEnv(
        manifest_path=args.manifest,
        max_steps=args.max_turns,
        reward_config=RewardConfig(),
        shuffle=False,  # rollout selects samples explicitly by index
    )
    dataset, n_train = build_seed_dataset(len(env.records))
    print(f"Loaded {len(env.records)} records; training on {n_train} seeds "
          f"(holdout tail reserved for evaluate.py).")

    if args.dry_run:
        print("Dry run: verifying one scripted episode wiring...")
        obs, info = env.reset(options={"index": 0})
        for action in ("ACTION: METADATA", "ACTION: ANSWER REAL"):
            obs, r, term, trunc, info = env.step(action)
            print(f"  {info['action_type']:<8} reward={r:+.2f}")
        print("Dry run OK. Install trl/peft/accelerate to train "
              "(CUDA for real runs; --device mps --no-bf16 --max-steps N for an MPS smoke).")
        return

    # Heavy imports kept inside main() so --dry-run works without them.
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from src.trace_logger import TraceLogger
    trace_logger = TraceLogger(args.trace_log, sample_every=args.trace_every)
    print(f"Logging reasoning traces every {args.trace_every} episodes to "
          f"{args.trace_log} (replay with: python visualize.py).")

    device = resolve_device(args.device)
    use_bf16 = device == "cuda" and not args.no_bf16
    dtype = resolve_dtype(device, use_bf16=use_bf16)
    print(f"Device: {device} | load dtype: {dtype} | bf16: {use_bf16}")

    processor = AutoProcessor.from_pretrained(args.model, padding_side="left")
    # No device_map: let the Trainer place the model (it auto-selects cuda/mps).
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype)

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    )
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=args.num_generations,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        bf16=use_bf16,
        fp16=device == "cuda" and args.no_bf16,
        use_cpu=device == "cpu",
        max_completion_length=256,
        log_completions=True,
        logging_steps=1,
        save_steps=100,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=episode_reward_func,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        rollout_func=make_rollout_func(env, processor, args.max_turns, trace_logger,
                                       temperature=args.temperature, top_p=args.top_p),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Saved policy adapters to {args.output_dir}")


if __name__ == "__main__":
    main()
