"""Stage 1 — SFT on distilled investigation traces (multi-GPU / multi-node).

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

Parallelism: data-parallel across ranks, with DeepSpeed ZeRO sharding the
optimizer/param state inside the data-parallel group. One process per GPU.

    # single GPU (7B smoke run)
    python training/sft.py --model 7b

    # one 8-GPU A100/H200 node, 32B + LoRA (ZeRO-2 is enough: LoRA grads are tiny)
    accelerate launch --config_file configs/accelerate_ds_zero2.yaml \\
        training/sft.py --model 32b

    # multi-node 72B (ZeRO-3 shards the frozen base weights too)
    sbatch scripts/arc_sft.slurm            # see the script for ARC specifics
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import InvestigationEnv
from training import common

# Paths are dataset-namespaced (see training/common.py: VRR_DATASET).
DEFAULT_TRACES = common.DEFAULT_TRACES


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


def default_output_dir(model_name: str, dataset: str | None = None) -> str:
    return common.checkpoint_dir("sft", model_name, dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=common.DEFAULT_MODEL,
                    help="HF repo id or registry alias (7b | 32b | 72b | auto).")
    ap.add_argument("--dataset", default=common.DATASET,
                    help="Dataset namespace for manifest/traces/checkpoints/logs.")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--traces", default=None)
    ap.add_argument("--output-dir", default=None, help="Defaults to checkpoints/sft-<model>.")
    ap.add_argument("--max-inspects", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    # ---- scale / parallelism ----
    ap.add_argument("--per-device-batch-size", type=int, default=1,
                    help="Sequences per GPU. Multi-image contexts are large; 1-2 on A100-80.")
    ap.add_argument("--grad-accum", type=int, default=8,
                    help="Effective batch = per_device × grad_accum × world_size.")
    ap.add_argument("--deepspeed", default=None,
                    help="DeepSpeed JSON (configs/deepspeed_zero{2,3}.json). Omit when the "
                         "accelerate config already carries a DeepSpeed plugin.")
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                    action="store_false")
    ap.add_argument("--dataloader-workers", type=int, default=4)
    # ---- LoRA ----
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-target", choices=["attn", "all-linear"], default="all-linear")
    ap.add_argument("--full-finetune", action="store_true",
                    help="Skip LoRA. Needs ZeRO-3 (+ offload for 72B).")
    ap.add_argument("--wandb-project", default="visual-reasoning-rlvr")
    ap.add_argument("--dry-run", action="store_true", help="Replay traces + report, no model/trl.")
    args = ap.parse_args()

    dist = common.dist_info()
    model_name = common.resolve_model(args.model)
    args.manifest = args.manifest or common.manifest_path(args.dataset)
    args.traces = args.traces or common.traces_path(args.dataset)
    output_dir = args.output_dir or default_output_dir(model_name, args.dataset)
    common.record_run("sft", f"model={model_name} traces={args.traces} out={output_dir}",
                      args.dataset)

    env = InvestigationEnv(manifest_path=args.manifest, max_inspects=args.max_inspects,
                           shuffle=False, dataset=args.dataset)
    traces = load_traces(args.traces)
    conversations = [c for t in traces if (c := replay_to_conversation(env, t))]
    common.rank0_print(f"Loaded {len(traces)} traces; {len(conversations)} replayed cleanly.")

    if args.dry_run:
        if conversations:
            c = conversations[0]
            print(f"Sample conversation: {len(c['messages'])} messages, {len(c['images'])} images.")
        print(f"Model: {model_name} | world_size={dist.world_size} "
              f"(nodes={dist.num_nodes}) | effective batch="
              f"{args.per_device_batch_size * args.grad_accum * dist.world_size}")
        print("Dry run OK. Install trl/peft/accelerate/deepspeed/wandb + CUDA GPUs to train.")
        return

    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    common.warn_if_tight(model_name, training=True)
    wandb_run = common.wandb_init(
        args.wandb_project, "sft", model_name, args,
        extra={"stage": "sft", "num_traces": len(conversations)})

    device = common.resolve_device("auto")
    dtype = common.resolve_dtype(device, use_bf16=True)

    # SFTConfig FIRST, model SECOND: constructing TrainingArguments with a
    # DeepSpeed config installs the global HfDeepSpeedConfig, which is what makes
    # ZeRO-3 shard parameters *during* `from_pretrained` instead of materializing
    # the full 72B on every rank (an instant OOM).
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        bf16=common.is_cuda(device),
        tf32=common.is_cuda(device),
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=True,
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        seed=args.seed,
        report_to=["wandb"] if wandb_run is not None else [],
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    model, processor = common.load_policy(
        model_name, adapter=None, device=device, dtype=dtype, trainable=True
    )
    common.rank0_print(
        f"Policy: {model_name} | world_size={dist.world_size} (nodes={dist.num_nodes}) "
        f"| effective batch={args.per_device_batch_size * args.grad_accum * dist.world_size} "
        f"| {'full finetune' if args.full_finetune else f'LoRA r={args.lora_r} ({args.lora_target})'}"
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(conversations),
        processing_class=processor,
        data_collator=VisionSFTCollator(processor),
        peft_config=None if args.full_finetune else common.lora_config(
            r=args.lora_r, target=args.lora_target
        ),
    )
    trainer.add_callback(common.progress_callback())
    trainer.train()
    trainer.save_model(output_dir)  # Trainer writes from the main process only
    common.rank0_print(f"Saved SFT checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
