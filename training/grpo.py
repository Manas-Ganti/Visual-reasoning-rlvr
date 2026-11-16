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
        self.turns = self.retok_mismatch = self.turns_capped = 0
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
            "turns": self.turns, "retok_mismatch": self.retok_mismatch,
            "turns_capped": self.turns_capped,
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
            # Turns whose sampled ids differ from the template's re-encoding of
            # the same text (see common.assemble_episode). Near 0 is expected.
            "rollout/retok_mismatch_rate": tot.get("retok_mismatch", 0) / max(tot.get("turns", 0), 1),
            # Turns cut off at --max-new-tokens before their ACTION line. Run 3
            # needed 900 tokens for this to stay near 0; a rise means the cap is
            # eating answers (an unparseable turn scores as a format failure).
            "rollout/turn_capped_rate": tot.get("turns_capped", 0) / max(tot.get("turns", 0), 1),
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


def make_rollout_func(envs, processor, max_turns, degradation, trace_logger, args):
    """Roll out this rank's whole generation batch in lockstep.

    TRL hands ``rollout_func`` the rank's prompts for ``steps_per_generation``
    micro-steps at once (``per_device x steps_per_generation`` episodes). Run 3
    walked them one ``generate`` at a time. Here up to ``len(envs)`` episodes
    share every ``generate`` call, and a slot is refilled as soon as its episode
    ends. With one episode per rank (the old default shape) this degenerates to
    the sequential loop, so the batching only pays once ``--groups-per-step``
    gives each rank several episodes.
    """
    def rollout_func(prompts, trainer):
        step = int(getattr(trainer.state, "global_step", 0))
        metrics = EpisodeMetrics()
        jobs = [(common.seed_index(p), common.seed_degradation(p, degradation)) for p in prompts]

        def on_episode(env, ep):
            metrics.add(ep)
            if trace_logger is not None:
                trace_logger.log(env.get_trace(global_step=step, phase="train"))

        with _generation_ctx(trainer) as model:
            policy = common.HFPolicy(model, processor, max_new_tokens=args.max_new_tokens)
            episodes = common.run_episodes_batched(
                policy, envs, jobs, max_turns=max_turns, sample=True,
                temperature=args.temperature, top_p=args.top_p,
                max_new_tokens=args.max_new_tokens, on_episode=on_episode,
                collect_tokens=True, collect_logprobs=args.rollout_logprobs,
                full_episode=True, turn_end_id=policy.turn_end_id,
            )
        if len(episodes) != len(prompts):
            raise RuntimeError(f"rollout returned {len(episodes)} episodes for {len(prompts)} prompts")

        batch = {k: [ep[k] for ep in episodes]
                 for k in ("prompt_ids", "completion_ids", "logprobs", "episode_reward",
                           "env_mask", "images")}
        # TRL treats logprobs=None as "not collected" (its own HF path does the
        # same). An empty list per row would be read as zero-length logprobs.
        if not args.rollout_logprobs:
            batch["logprobs"] = None
        turns = sum(ep["turns"] for ep in episodes)
        metrics.retok_mismatch += sum(ep["retok_mismatch"] for ep in episodes)
        metrics.turns += turns
        metrics.turns_capped += sum(ep.get("turns_capped", 0) for ep in episodes)

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


def probe_forward_inputs(trainer) -> None:
    """Check, on the first training forward, that the policy gradient sees pixels.

    TRL 1.9.2 builds the training forward pass's images from the DATASET when
    ``rollout_func`` is used, and our seed dataset is text only. Runs 1-3 almost
    certainly trained on image placeholder tokens with no pixels and without
    the inspected cells (see ``VisionRolloutGRPOTrainer``). This wraps TRL's
    logprob helper and, on its first call, raises if ``pixel_values``,
    ``image_grid_thw`` or ``mm_token_type_ids`` are missing. Qwen2.5-VL needs all
    three: without ``mm_token_type_ids`` and the grid it silently falls back to
    1-D text positions. TRL also drops the images without a word when
    placeholder counts and image features disagree, so a missing key here means
    either the old bug or a count mismatch, and both must stop the run.
    """
    fn = getattr(trainer, "_get_per_token_logps_and_entropies", None)
    if fn is None:
        return
    seen = {"done": False}

    def wrapped(model, input_ids, *a, **kw):
        if not seen["done"]:
            seen["done"] = True
            have = {k: kw.get(k) is not None
                    for k in ("pixel_values", "image_grid_thw", "mm_token_type_ids")}
            n_img = (int(kw["image_grid_thw"].shape[0]) if have["image_grid_thw"] else 0)
            common.rank0_print(f"[forward-probe] {have} images={n_img} "
                               f"seq={tuple(input_ids.shape)}", flush=True)
            if not all(have.values()):
                raise RuntimeError(
                    f"[forward-probe] training forward is missing {[k for k, v in have.items() if not v]}: "
                    "the policy gradient would not see the images. Not training on that.")
        return fn(model, input_ids, *a, **kw)

    trainer._get_per_token_logps_and_entropies = wrapped


def make_trainer_class():
    """GRPOTrainer whose training forward sees every image of the episode.

    Why a subclass: on the ``rollout_func`` path TRL 1.9.2 takes images from the
    dataset. It does have a branch built for images INSIDE the completion,
    written for tool calls that return images: it runs the image processor on
    the flat image list, derives ``mm_token_type_ids`` from the actual token ids
    (Qwen2.5-VL needs these for its 3-D positions), and checks placeholder
    counts against image features. That is exactly our case. The branch is
    gated on ``self.tools``, and setting ``self.tools`` for the whole run would
    also switch on TRL's tool-call loop inside ``_generate``. So:

    * ``_generate`` runs TRL's own rollout_func path unchanged, then moves the
      episode images out of the extra fields into ``images``/``tool_images``;
    * only after that does it mark ``self.tools`` truthy, for the remainder of
      that one ``_generate_and_score_completions`` call (the forward-kwarg
      branch and nothing else), and the wrapper restores it in ``finally``.

    The probe above turns any silent failure of this into a crash on step 1.
    """
    from trl import GRPOTrainer

    def _rollout_images_marker():
        """Placeholder: makes TRL take its in-completion image branch."""

    class VisionRolloutGRPOTrainer(GRPOTrainer):
        logp_parity = False

        def _generate(self, prompts):
            out = list(super()._generate(prompts))
            extra = out[5]
            images = extra.pop("images", None)
            if images is not None:
                out[6] = images        # images: count per sample -> num_images
                out[7] = images        # tool_images: selects the in-completion branch
                self._saved_tools = self.tools
                self.tools = [_rollout_images_marker]
            return tuple(out)

        def _generate_and_score_completions(self, inputs):
            try:
                output = super()._generate_and_score_completions(inputs)
            finally:
                if hasattr(self, "_saved_tools"):
                    self.tools = self._saved_tools
                    del self._saved_tools
            if self.logp_parity:
                self._log_logp_parity(output)
            return output

        def _log_logp_parity(self, output):
            """Compare rollout logprobs with the training forward's, same tokens.

            If the training sequence reproduces what the policy saw while
            generating, the two agree to bf16 noise. Missing images, missing
            reveals, or a wrong temperature all show up as a gap. Diagnostic
            only (one extra forward per generation batch), behind
            --logp-parity.
            """
            import torch

            sampled = output.get("sampling_per_token_logps")
            if sampled is None:
                return
            ids = torch.cat([output["prompt_ids"], output["completion_ids"]], dim=1)
            attn = torch.cat([output["prompt_mask"], output["completion_mask"]], dim=1)
            mask = output["completion_mask"] * output.get("tool_mask", 1)
            fwd = {k: output[k] for k in ("pixel_values", "image_grid_thw", "num_images",
                                          "mm_token_type_ids") if k in output}
            was_training = self.model.training
            self.model.eval()
            try:
                with torch.no_grad():
                    logps, _, _ = self._get_per_token_logps_and_entropies(
                        self.model, ids, attn, output["completion_ids"].size(1),
                        self.args.per_device_train_batch_size, **fwd)
            finally:
                self.model.train(was_training)
            diff = (logps - sampled).abs()[mask.bool()]
            mean = self.accelerator.gather(diff.mean() if diff.numel() else diff.new_zeros(())).mean()
            mx = self.accelerator.gather(diff.max() if diff.numel() else diff.new_zeros(())).max()
            self._metrics["train"]["rollout/logp_parity_mean"].append(mean.item())
            self._metrics["train"]["rollout/logp_parity_max"].append(mx.item())
            common.rank0_print(f"[logp-parity] mean|dlogp|={mean.item():.4f} "
                               f"max={mx.item():.3f} over {int(mask.sum())} tokens", flush=True)

    return VisionRolloutGRPOTrainer


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
    ap.add_argument("--resume", nargs="?", const=True, default=None,
                    metavar="CHECKPOINT",
                    help="Resume from a checkpoint-N directory. Bare --resume picks the "
                         "latest inside --output-dir. GRPO writes one every save_steps, "
                         "and a run killed at hour 10 is otherwise a total loss.")
    ap.add_argument("--max-inspects", type=int, default=4)
    ap.add_argument("--overview-long-edge", type=int, default=140,
                    help="Overview resolution — how much the low-res view is blurred. "
                         "The zoom factor INSPECT buys is native_size/this, so ~native/7 "
                         "is the target: 140 suits 1024px images, 70 suits 512px. Too high "
                         "and the answer is readable from the overview; too low and the "
                         "agent cannot tell where to look. Find it with "
                         "tools/ceiling_probe.py --condition overview.")
    ap.add_argument("--degradation", default="clean")
    ap.add_argument("--num-generations", type=int, default=8, help="GRPO group size G.")
    ap.add_argument("--learning-rate", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.04, help="KL coefficient toward the reference.")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=640,
                    help="640, not 320: distilling synth1024 showed ~28% of turns exceeding a 320-token budget mid-block, which loses the ACTION line and makes the turn unparseable — 80% of episodes were rejected for it. eval/harness.py already used 640; the rollout paths did not, so the policy was being cut off during training and not during evaluation.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=25,
                    help="Checkpoint every N steps. 25, not 100: an epoch is ~157 steps at "
                         "~8 min each, so a run killed at the wall-time limit loses at most "
                         "~3.5 h, and steps 50/100/150 exist for held-out eval mid-run.")
    ap.add_argument("--save-total-limit", type=int, default=8,
                    help="Keep this many checkpoints (~2 GB each: LoRA + optimizer state). "
                         "8 holds a whole 157-step epoch at --save-steps 25.")
    # ---- scale / parallelism ----
    ap.add_argument("--per-device-batch-size", type=int, default=1,
                    help="Episodes rolled out per GPU per micro-step.")
    ap.add_argument("--grad-accum", type=int, default=None,
                    help="Default: num_generations / (per_device × world_size), i.e. one "
                         "GRPO group per optimizer step.")
    ap.add_argument("--groups-per-step", type=int, default=1,
                    help="GRPO groups (distinct images) per optimizer step. Sets the default "
                         "--grad-accum. At 1 on 8 ranks every rank rolls out ONE episode and "
                         "there is nothing to batch: run 3 saw one image per step. At 8, each "
                         "rank rolls out 8 episodes in one lockstep batch.")
    ap.add_argument("--rollout-batch-size", type=int, default=None,
                    help="Max episodes per lockstep generate on one rank. Default: all of the "
                         "rank's episodes. Lower it if the KV cache does not fit: each 32B "
                         "episode needs ~2 GB of KV at its last turn, on top of ~66 GB of "
                         "replicated bf16 weights under ZeRO-2. 8 fits an H200, ~4 an A100-80.")
    ap.add_argument("--rollout-logprobs", action="store_true",
                    help="Return sampling logprobs to TRL. Unused on this HF path (TRL reads them "
                         "only for vLLM importance sampling / the off-policy mask), and costs a "
                         "batch x steps x vocab score tensor.")
    ap.add_argument("--logp-parity", action="store_true",
                    help="Diagnostic: compare rollout logprobs with the training forward's on "
                         "the same tokens and log the gap. Implies --rollout-logprobs. Near 0 "
                         "means the gradient sees what the policy saw. Use for calibration "
                         "runs; it adds a forward pass per step.")
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

    if args.logp_parity:
        args.rollout_logprobs = True
    dist = common.dist_info()
    model_name = common.resolve_model(args.model)
    args.manifest = args.manifest or common.manifest_path(args.dataset)
    output_dir = args.output_dir or default_output_dir(model_name, args.dataset)
    if args.grad_accum is None:
        per_step = max(args.per_device_batch_size * dist.world_size, 1)
        args.grad_accum = max(args.num_generations * args.groups_per_step // per_step, 1)
    check_batch_shape(args, dist.world_size)
    common.record_run("grpo", f"model={model_name} sft={args.sft_checkpoint} "
                      f"out={output_dir} G={args.num_generations}", args.dataset)

    def env_factory():
        return InvestigationEnv(
            manifest_path=args.manifest, max_inspects=args.max_inspects,
            reward_config=RewardConfig(), shuffle=False, default_degradation=args.degradation,
            seed=args.seed + dist.rank, dataset=args.dataset,
            overview_long_edge=args.overview_long_edge,
        )

    env = env_factory()
    # Episodes this rank rolls out per generation: TRL sets steps_per_generation
    # to grad_accum by default, and per_device episodes come per micro-step.
    local_episodes = args.per_device_batch_size * args.grad_accum
    rollout_batch = min(args.rollout_batch_size or local_episodes, local_episodes)
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
              f"| G={args.num_generations} | grad_accum={args.grad_accum} "
              f"| episodes/rank/step={local_episodes} | rollout batch={rollout_batch} "
              f"| images/step={args.per_device_batch_size * dist.world_size * args.grad_accum // args.num_generations}")
        print("Dry run OK. Install trl/peft/accelerate/deepspeed/wandb + CUDA GPUs to train.")
        return

    from trl import GRPOConfig

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
    training_args = GRPOConfig(**common.supported_config_kwargs(GRPOConfig, dict(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        # Explicit, though TRL defaults it to grad_accum: one rollout_func call per
        # optimizer step, so the samples are always on-policy and TRL skips the
        # old-logprob forward pass.
        steps_per_generation=args.grad_accum,
        num_generations=args.num_generations,
        # TRL scores logprobs as logits / temperature. Runs 1-3 left this at
        # TRL's default 1.0 while rollouts sampled at 0.7, so the gradient was
        # taken from a different distribution than the one sampled from.
        temperature=args.temperature,
        top_p=args.top_p,
        # Would gather every episode image to every rank, every step, for a
        # W&B table. The trace logger already keeps the trajectories.
        log_multimodal=False,
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
        # Off: TRL prints every completion as a console table (13 MB of log for
        # a 5-step run). Trajectories already go to the rank-0 trace log.
        log_completions=False,
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        # Only rank 0 reports, and only if wandb actually came up — otherwise HF
        # would try to init it a second time and fail for the same reason.
        report_to=["wandb"] if wandb_run is not None else [],
    )))

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

    trainer_cls = make_trainer_class()
    trainer_cls.logp_parity = args.logp_parity
    trainer = trainer_cls(
        model=model,
        processing_class=processor,
        reward_funcs=episode_reward_func,
        args=training_args,
        train_dataset=dataset,
        peft_config=None if adapter else common.lora_config(r=args.lora_r, target=args.lora_target),
        rollout_func=make_rollout_func(
            common.make_env_pool(env_factory, rollout_batch), processor, max_turns,
            args.degradation, trace_logger, args),
    )
    common.rank0_print(f"Rollouts: {local_episodes} episodes/rank/step, lockstep batch {rollout_batch}")
    probe_forward_inputs(trainer)
    # ETA + walltime-margin to W&B, so a 10–34h run is followable from the
    # dashboard alone. logging_steps=1 above means one ETA point per step.
    trainer.add_callback(common.progress_callback())
    # A killed run is only recoverable if save_steps produced a checkpoint AND we
    # are told to pick it up: HF resumes optimizer, scheduler and step count, so
    # the rollouts already paid for are not repeated.
    if args.resume:
        common.rank0_print(f"Resuming from {args.resume}")
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(output_dir)
    common.rank0_print(f"Saved GRPO policy to {output_dir}")


if __name__ == "__main__":
    main()
