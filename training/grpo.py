"""Stage 2 — GRPO against the environment's verifiable reward.

Uses TRL's ``GRPOTrainer`` with a custom ``rollout_func`` so each "completion" is
a full multi-turn investigation (inspect… → verdict) driven by
``InvestigationEnv`` rather than a single forward generation. GRPO forms
group-relative advantages from the per-episode returns (``env/reward.py``) and
updates the policy, KL-regularized (``--beta``) toward the frozen SFT checkpoint
so RL sharpens the SFT behaviour instead of drifting off it.

Rollouts use HF ``generate`` (we need per-token logprobs for the gradient);
offline eval + SFT-trace distillation use vLLM, where batched throughput matters
more than logprob bookkeeping.

Requires: trl>=0.29, peft, accelerate, wandb. CUDA/bf16 (A100) is the target.

    python training/grpo.py --sft-checkpoint checkpoints/sft --degradation clean
    python training/grpo.py --dry-run          # wiring check, no model / no trl
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow ``python training/grpo.py`` in addition to ``python -m training.grpo``.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import InvestigationEnv
from env.reward import RewardConfig
from training import common

OUTPUT_DIR = os.path.join(common.REPO_ROOT, "checkpoints", "grpo-qwen2vl")
TRACE_LOG = os.path.join(common.REPO_ROOT, "logs", "grpo_episodes.jsonl")


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

    def summary(self) -> dict:
        n = max(self.n, 1)
        return {
            "rollout/reward": self.reward_sum / n,
            "rollout/accuracy": self.correct / n,
            "rollout/answer_rate": self.answered / n,
            "rollout/avg_steps": self.steps / n,
            "rollout/avg_inspects": self.inspects / n,
        }


def make_rollout_func(env, processor, max_turns, degradation, trace_logger, temperature, top_p):
    def rollout_func(prompts, trainer):
        model = trainer.model
        step = int(getattr(trainer.state, "global_step", 0))
        metrics = EpisodeMetrics()
        batch = {"prompt_ids": [], "completion_ids": [], "logprobs": [], "episode_reward": []}
        for prompt in prompts:
            ep = common.run_episode(
                model, processor, env,
                index=common.seed_index(prompt),
                degradation=common.seed_degradation(prompt, degradation),
                max_turns=max_turns, sample=True, temperature=temperature, top_p=top_p,
            )
            for k in batch:
                batch[k].append(ep[k])
            metrics.add(ep)
            if trace_logger is not None:
                trace_logger.log(env.get_trace(global_step=step, phase="train"))
        summary = metrics.summary()
        print(f"[step {step:>4}] " + " ".join(f"{k.split('/')[-1]}={v:.3f}" for k, v in summary.items()), flush=True)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=common.DEFAULT_MODEL,
                    help="Base model, OR pass --sft-checkpoint to start from SFT.")
    ap.add_argument("--sft-checkpoint", default=None,
                    help="SFT LoRA/merged checkpoint to initialize + KL-anchor to.")
    ap.add_argument("--manifest", default=common.DEFAULT_MANIFEST)
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--max-inspects", type=int, default=4)
    ap.add_argument("--degradation", default="clean")
    ap.add_argument("--num-generations", type=int, default=8, help="GRPO group size G.")
    ap.add_argument("--learning-rate", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.04, help="KL coefficient toward the ref (SFT) model.")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--trace-every", type=int, default=16)
    ap.add_argument("--wandb-project", default="visual-reasoning-rlvr")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    env = InvestigationEnv(
        manifest_path=args.manifest, max_inspects=args.max_inspects,
        reward_config=RewardConfig(), shuffle=False, default_degradation=args.degradation,
    )
    max_turns = args.max_inspects + 3
    train_idx = [i for i, r in enumerate(env.records) if r.get("split", "train") == "train"]
    print(f"Loaded {len(env.records)} records; {len(train_idx)} in train split.")

    if args.dry_run:
        obs, info = env.reset(options={"index": train_idx[0], "degradation": args.degradation})
        for action in ("HYPOTHESIS: h\nACTION: INSPECT 6",
                       "RECONCILIATION: CONFIRMED\nBELIEF_UPDATE: P(fake)=0.8\nACTION: VERDICT AI confidence=0.8"):
            obs, r, term, trunc, info = env.step(action)
            print(f"  {info['action_type']:<7} reward={r:+.3f}")
        print("Dry run OK. Install trl/peft/accelerate/wandb + a CUDA GPU to train.")
        return

    import wandb
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from env.trace_logger import TraceLogger

    wandb.init(project=args.wandb_project, name="grpo", config=vars(args))

    model_name = args.sft_checkpoint or args.model
    device = common.resolve_device("auto")
    dtype = common.resolve_dtype(device, use_bf16=True)
    model, processor = common.load_policy(model_name, adapter=None, device=device, dtype=dtype)
    print(f"Policy: {model_name} | device {device} | dtype {dtype}")

    dataset = common.build_seed_dataset(train_idx, degradation=args.degradation)
    trace_logger = TraceLogger(TRACE_LOG, sample_every=args.trace_every)

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM",
    )
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.num_generations,
        num_generations=args.num_generations,
        learning_rate=args.learning_rate,
        beta=args.beta,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        bf16=device == "cuda",
        max_completion_length=320,
        log_completions=True,
        logging_steps=1,
        save_steps=100,
        report_to=["wandb"],
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=episode_reward_func,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        rollout_func=make_rollout_func(
            env, processor, max_turns, args.degradation, trace_logger, args.temperature, args.top_p
        ),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Saved GRPO policy to {args.output_dir}")


if __name__ == "__main__":
    main()
