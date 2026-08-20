"""Stage 2 — GRPO against the environment's verifiable reward (multi-GPU / multi-node).

Uses TRL's ``GRPOTrainer`` with a custom ``rollout_func`` so each "completion" is
a full multi-turn investigation (inspect… → verdict) driven by
``InvestigationEnv`` rather than a single forward generation. GRPO forms
group-relative advantages from the per-episode returns (``env/reward.py``) and
updates the policy, KL-regularized (``--beta``) toward a frozen reference.

Rollouts use HF ``generate`` (we need per-token logprobs for the gradient);
offline eval + SFT-trace distillation use vLLM, where batched throughput matters
more than logprob bookkeeping.

Parallelism: accelerate shards the seed dataset across ranks, so every rank runs
its own environment and its own rollouts on a disjoint slice, then contributes
gradients data-parallel. ZeRO-2 is the recommended shape — under ZeRO-3 the
policy parameters must be gathered for every ``generate`` call, which is correct
here (see ``_generation_ctx``) but costs a collective per turn.

    # single GPU (7B smoke run)
    python training/grpo.py --model 7b --sft-checkpoint checkpoints/sft-qwen2.5-vl-7b

    # one 8-GPU node, 32B
    accelerate launch --config_file configs/accelerate_ds_zero2.yaml \\
        training/grpo.py --sft-checkpoint checkpoints/sft-qwen2.5-vl-32b --num-generations 8

    python training/grpo.py --dry-run          # wiring check, no model / no trl
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys

# Allow ``python training/grpo.py`` in addition to ``python -m training.grpo``.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import InvestigationEnv
from env.reward import RewardConfig
from training import common

# Paths are dataset-namespaced (see training/common.py: VRR_DATASET).


class EpisodeMetrics:
    """Accumulates per-episode outcomes for a terminal metrics readout / W&B."""

    def __init__(self):
        self.n = self.correct = self.answered = self.steps = self.inspects = 0
        self.reward_sum = 0.0

    def add(self, ep: dict):
        self.n += 1
        self.reward_sum += ep["episode_reward"]
        self.correct += int(ep["correct"])
        self.answered += int(ep["answered"])
        self.steps += ep["steps"]
        self.inspects += ep["inspects_used"]

    def counters(self) -> dict:
        return {
            "n": self.n, "reward_sum": self.reward_sum, "correct": self.correct,
            "answered": self.answered, "steps": self.steps, "inspects": self.inspects,
        }

    @staticmethod
    def summarize(counters: list[dict]) -> dict:
        """Fold per-rank counters into one global summary (rank 0 reports)."""
        tot = {k: sum(c[k] for c in counters) for k in counters[0]} if counters else {}
        n = max(tot.get("n", 0), 1)
        return {
            "rollout/reward": tot.get("reward_sum", 0.0) / n,
            "rollout/accuracy": tot.get("correct", 0) / n,
            "rollout/answer_rate": tot.get("answered", 0) / n,
            "rollout/avg_steps": tot.get("steps", 0) / n,
            "rollout/avg_inspects": tot.get("inspects", 0) / n,
            "rollout/episodes": tot.get("n", 0),
        }


def _generation_ctx(trainer):
    """Yield a model that is safe to call ``generate`` on.

    Under DDP the trainer's model is a wrapper without ``generate``; under ZeRO-3
    its parameters are sharded across ranks. TRL's
    ``unwrap_model_for_generation`` handles both (unwrapping, and gathering the
    ZeRO-3 shards for the duration of the call). Fall back to a plain unwrap on
    TRL versions that don't expose it.
    """
    try:
        from trl.models.utils import unwrap_model_for_generation

        return unwrap_model_for_generation(trainer.model, trainer.accelerator)
    except Exception:
        accel = getattr(trainer, "accelerator", None)
        model = accel.unwrap_model(trainer.model) if accel else trainer.model
        return contextlib.nullcontext(model)


def make_rollout_func(env, processor, max_turns, degradation, trace_logger, args):
    def rollout_func(prompts, trainer):
        step = int(getattr(trainer.state, "global_step", 0))
        metrics = EpisodeMetrics()
        batch = {"prompt_ids": [], "completion_ids": [], "logprobs": [], "episode_reward": []}

        with _generation_ctx(trainer) as model:
            policy = common.HFPolicy(model, processor, max_new_tokens=args.max_new_tokens)
            for prompt in prompts:
                ep = common.run_episode(
                    policy, env,
                    index=common.seed_index(prompt),
                    degradation=common.seed_degradation(prompt, degradation),
                    max_turns=max_turns, sample=True,
                    temperature=args.temperature, top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                )
                for k in batch:
                    batch[k].append(ep[k])
                metrics.add(ep)
                if trace_logger is not None:
                    trace_logger.log(env.get_trace(global_step=step, phase="train"))

        # Rank-local counters → one global summary, so the printed/logged rollout
        # metrics describe the whole data-parallel batch, not just this rank's slice.
        summary = EpisodeMetrics.summarize(common.gather_lists([metrics.counters()]))
        if common.is_main():
            print(f"[step {step:>4}] " + " ".join(
                f"{k.split('/')[-1]}={v:.3f}" for k, v in summary.items()), flush=True)
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(summary, step=step)
            except Exception:
                pass
        return batch

    return rollout_func


def episode_reward_func(completions, episode_reward=None, **kwargs):
    """GRPO reward = the env return computed during rollout, passed through."""
    return list(episode_reward)


def default_output_dir(model_name: str, dataset: str | None = None) -> str:
    return common.checkpoint_dir("grpo", model_name, dataset)


def check_batch_shape(args, world_size: int) -> None:
    """GRPO needs each group of ``num_generations`` samples inside one optimizer
    step, so the global rollout batch must be a multiple of the group size.
    Catch it here with a fixable message rather than deep inside TRL."""
    global_batch = args.per_device_batch_size * world_size * args.grad_accum
    if global_batch % args.num_generations:
        raise SystemExit(
            f"Global rollout batch {global_batch} (= per_device {args.per_device_batch_size} "
            f"× world {world_size} × accum {args.grad_accum}) is not a multiple of "
            f"--num-generations {args.num_generations}. Adjust --grad-accum "
            f"(e.g. {max(args.num_generations // max(args.per_device_batch_size * world_size, 1), 1)}) "
            f"or --num-generations."
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=common.DEFAULT_MODEL,
                    help="Base model: HF repo id or alias (7b | 32b | 72b | auto).")
    ap.add_argument("--sft-checkpoint", default=None,
                    help="Stage-1 output. A LoRA adapter dir is loaded on top of --model and "
                         "training continues on it; a merged/full checkpoint is used as the base.")
    ap.add_argument("--dataset", default=common.DATASET,
                    help="Dataset namespace for manifest/checkpoints/logs.")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--output-dir", default=None, help="Defaults to checkpoints/grpo-<model>.")
    ap.add_argument("--max-inspects", type=int, default=4)
    ap.add_argument("--degradation", default="clean")
    ap.add_argument("--num-generations", type=int, default=8, help="GRPO group size G.")
    ap.add_argument("--learning-rate", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.04, help="KL coefficient toward the reference.")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    # ---- scale / parallelism ----
    ap.add_argument("--per-device-batch-size", type=int, default=1,
                    help="Episodes rolled out per GPU per micro-step.")
    ap.add_argument("--grad-accum", type=int, default=None,
                    help="Default: num_generations / (per_device × world_size), i.e. one "
                         "GRPO group per optimizer step.")
    ap.add_argument("--deepspeed", default=None,
                    help="DeepSpeed JSON. ZeRO-2 recommended (ZeRO-3 gathers params per turn).")
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                    action="store_false")
    # ---- LoRA ----
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-target", choices=["attn", "all-linear"], default="all-linear")
    ap.add_argument("--trace-every", type=int, default=16)
    ap.add_argument("--wandb-project", default="visual-reasoning-rlvr")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dist = common.dist_info()
    model_name = common.resolve_model(args.model)
    args.manifest = args.manifest or common.manifest_path(args.dataset)
    output_dir = args.output_dir or default_output_dir(model_name, args.dataset)
    if args.grad_accum is None:
        per_step = max(args.per_device_batch_size * dist.world_size, 1)
        args.grad_accum = max(args.num_generations // per_step, 1)
    check_batch_shape(args, dist.world_size)
    common.record_run("grpo", f"model={model_name} sft={args.sft_checkpoint} "
                      f"out={output_dir} G={args.num_generations}", args.dataset)

    env = InvestigationEnv(
        manifest_path=args.manifest, max_inspects=args.max_inspects,
        reward_config=RewardConfig(), shuffle=False, default_degradation=args.degradation,
        seed=args.seed + dist.rank, dataset=args.dataset,
    )
    max_turns = args.max_inspects + 3
    train_idx = [i for i, r in enumerate(env.records) if r.get("split", "train") == "train"]
    common.rank0_print(f"Loaded {len(env.records)} records; {len(train_idx)} in train split.")

    if args.dry_run:
        obs, info = env.reset(options={"index": train_idx[0], "degradation": args.degradation})
        for action in ("HYPOTHESIS: h\nACTION: INSPECT 6",
                       "RECONCILIATION: CONFIRMED\nBELIEF_UPDATE: P(fake)=0.8\nACTION: VERDICT AI confidence=0.8"):
            obs, r, term, trunc, info = env.step(action)
            print(f"  {info['action_type']:<7} reward={r:+.3f}")
        print(f"Model: {model_name} | world_size={dist.world_size} (nodes={dist.num_nodes}) "
              f"| G={args.num_generations} | grad_accum={args.grad_accum}")
        print("Dry run OK. Install trl/peft/accelerate/deepspeed/wandb + CUDA GPUs to train.")
        return

    from trl import GRPOConfig, GRPOTrainer

    from env.trace_logger import TraceLogger

    common.warn_if_tight(model_name, training=True)
    wandb_run = common.wandb_init(
        args.wandb_project, "grpo", model_name, args,
        extra={"stage": "grpo", "grad_accum_resolved": args.grad_accum})

    device = common.resolve_device("auto")
    dtype = common.resolve_dtype(device, use_bf16=True)

    # Stage-1 output is normally a LoRA adapter: load it on top of the base and
    # keep training *that* adapter (peft_config stays None so PEFT doesn't wrap a
    # second time). Note the KL reference in the PEFT path is the adapter-disabled
    # base model, not SFT — to anchor to SFT exactly, merge the adapter first
    # (``training.vllm_backend.merge_adapter``) and train a fresh LoRA on it.
    is_adapter = bool(args.sft_checkpoint) and os.path.exists(
        os.path.join(args.sft_checkpoint, "adapter_config.json")
    )
    base_model = args.sft_checkpoint if (args.sft_checkpoint and not is_adapter) else model_name
    adapter = args.sft_checkpoint if is_adapter else None

    # GRPOConfig before the model: see the ZeRO-3 note in training/sft.py.
    training_args = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        learning_rate=args.learning_rate,
        beta=args.beta,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        bf16=common.is_cuda(device),
        tf32=common.is_cuda(device),
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=False,
        max_completion_length=args.max_new_tokens,
        log_completions=True,
        logging_steps=1,
        save_steps=100,
        save_total_limit=2,
        seed=args.seed,
        # Only rank 0 reports, and only if wandb actually came up — otherwise HF
        # would try to init it a second time and fail for the same reason.
        report_to=["wandb"] if wandb_run is not None else [],
    )

    model, processor = common.load_policy(
        base_model, adapter=adapter, device=device, dtype=dtype, trainable=True
    )
    common.rank0_print(
        f"Policy: {base_model}{f' + {adapter}' if adapter else ''} | device {device} | dtype {dtype} "
        f"| world_size={dist.world_size} (nodes={dist.num_nodes}) | G={args.num_generations} "
        f"| grad_accum={args.grad_accum}"
    )

    dataset = common.build_seed_dataset(train_idx, degradation=args.degradation)
    # Only rank 0 writes the trajectory log — the demo reads one file, and
    # concurrent appends from 8+ ranks would interleave mid-record.
    trace_logger = (
        TraceLogger(os.path.join(common.log_dir(args.dataset), "grpo_episodes.jsonl"),
                    sample_every=args.trace_every)
        if dist.is_main else None
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=episode_reward_func,
        args=training_args,
        train_dataset=dataset,
        peft_config=None if adapter else common.lora_config(r=args.lora_r, target=args.lora_target),
        rollout_func=make_rollout_func(env, processor, max_turns, args.degradation, trace_logger, args),
    )
    # ETA + walltime-margin to W&B, so a 10–34h run is followable from the
    # dashboard alone. logging_steps=1 above means one ETA point per step.
    trainer.add_callback(common.progress_callback())
    trainer.train()
    trainer.save_model(output_dir)
    common.rank0_print(f"Saved GRPO policy to {output_dir}")


if __name__ == "__main__":
    main()
