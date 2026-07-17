"""Stage 1 — SFT on distilled investigation traces.

Teaches the base VLM the pre/post predict-then-verify *format* and the two-action
tool use, so GRPO starts from a policy that already emits parseable, structured
trajectories (RL then sharpens the reasoning quality against the verifiable
reward).

Trace format (``data/sft_traces.jsonl``, produced by ``data/build_sft_traces.py``)
is intentionally compact — it stores only the seed and the assistant turns:

    {"index": 42, "degradation": "clean",
     "actions": ["OBSERVATION:...\\nACTION: INSPECT 6", "...VERDICT AI confidence=0.9"]}

We *replay* each trace through ``InvestigationEnv`` to reconstruct the exact
(messages, images) the agent would have seen — the env is the single source of
truth for image rendering, so distilled traces never need to ship pixels. Only
the assistant tokens contribute to the loss.

Requires: trl, peft, accelerate, wandb + a CUDA GPU (A100 target).

    python training/sft.py --traces data/sft_traces.jsonl --epochs 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import InvestigationEnv
from training import common

OUTPUT_DIR = os.path.join(common.REPO_ROOT, "checkpoints", "sft-qwen2vl")
DEFAULT_TRACES = os.path.join(common.REPO_ROOT, "data", "sft_traces.jsonl")


def load_traces(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def replay_to_conversation(env: InvestigationEnv, trace: dict) -> dict | None:
    """Replay a trace's actions to recover the full (messages, images). Returns
    ``None`` if the trace desyncs from the env (e.g. an over-budget action)."""
    env.reset(options={"index": trace["index"], "degradation": trace.get("degradation", "clean")})
    obs = None
    for action in trace["actions"]:
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    if obs is None:
        return None
    return {"messages": obs["messages"], "images": obs["images"]}


class VisionSFTCollator:
    """Apply the chat template, process images, and mask everything but the
    assistant tokens so the loss is on the model's own turns only. Assistant
    masking uses the processor's ``return_assistant_tokens_mask`` when the chat
    template supports it, and otherwise falls back to training on the full
    sequence (still valid, just less targeted)."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        import torch

        texts, images = [], []
        for ex in examples:
            texts.append(
                self.processor.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
            )
            images.append(ex["images"])
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        # Mask image placeholder tokens from the loss if the processor exposes them.
        image_token_id = getattr(self.processor, "image_token_id", None)
        if image_token_id is not None:
            labels[labels == image_token_id] = -100
        batch["labels"] = labels
        return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=common.DEFAULT_MODEL)
    ap.add_argument("--manifest", default=common.DEFAULT_MANIFEST)
    ap.add_argument("--traces", default=DEFAULT_TRACES)
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--max-inspects", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--wandb-project", default="visual-reasoning-rlvr")
    ap.add_argument("--dry-run", action="store_true", help="Replay traces + report, no model/trl.")
    args = ap.parse_args()

    env = InvestigationEnv(manifest_path=args.manifest, max_inspects=args.max_inspects, shuffle=False)
    traces = load_traces(args.traces)
    conversations = [c for t in traces if (c := replay_to_conversation(env, t))]
    print(f"Loaded {len(traces)} traces; {len(conversations)} replayed cleanly.")

    if args.dry_run:
        if conversations:
            c = conversations[0]
            print(f"Sample conversation: {len(c['messages'])} messages, {len(c['images'])} images.")
        print("Dry run OK. Install trl/peft/accelerate/wandb + CUDA to train.")
        return

    import wandb
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    wandb.init(project=args.wandb_project, name="sft", config=vars(args))

    device = common.resolve_device("auto")
    dtype = common.resolve_dtype(device, use_bf16=True)
    model, processor = common.load_policy(args.model, adapter=None, device=device, dtype=dtype)

    dataset = Dataset.from_list(conversations)
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM",
    )
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        bf16=device == "cuda",
        logging_steps=5,
        save_steps=100,
        report_to=["wandb"],
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
        data_collator=VisionSFTCollator(processor),
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Saved SFT checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
