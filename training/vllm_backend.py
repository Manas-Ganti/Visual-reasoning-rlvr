"""vLLM inference backend — tensor-parallel, batched generation for the offline
paths (eval harness, SFT-trace distillation, evidence-slice saliency).

Why a second backend at all: GRPO rollouts must go through HF ``generate``
because we need per-token logprobs for the gradient. Everything *offline* only
needs text, and there the bottleneck is throughput — a 32B/72B policy running one
episode at a time leaves the GPUs idle between turns. vLLM gives us

* **tensor parallelism** (``--tensor-parallel-size 8``) so one 72B replica spans
  a whole A100/H200 node instead of needing device_map pipeline splits, and
* **continuous batching**, which ``common.run_episodes_batched`` feeds by
  advancing many episodes in lockstep — one batched generate per turn.

Parallelism note: use vLLM TP *or* the torchrun rank-sharding path, never both in
one process. Multi-node vLLM needs a Ray cluster; on ARC the practical shape is
one vLLM process per node (TP = GPUs on that node) with the manifest split by
``--num-shards/--shard-id``.

    python eval/harness.py --backend vllm --tensor-parallel-size 4 --adapter ckpt/grpo
"""

from __future__ import annotations

import contextlib

from training import common


class VLLMPolicy:
    """Batched vLLM policy exposing the same ``act`` / ``act_batch`` contract as
    ``common.HFPolicy`` (minus logprobs — offline use only)."""

    backend = "vllm"

    def __init__(
        self,
        model: str,
        *,
        adapter: str | None = None,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        max_model_len: int = 16384,
        gpu_memory_utilization: float = 0.90,
        max_images_per_prompt: int = 8,
        max_lora_rank: int = 32,
        min_pixels: int | None = common.DEFAULT_MIN_PIXELS,
        max_pixels: int | None = common.DEFAULT_MAX_PIXELS,
        seed: int = 0,
    ):
        from vllm import LLM

        self.model_name = common.resolve_model(model)
        self.processor = common.load_processor(
            self.model_name, min_pixels=min_pixels, max_pixels=max_pixels
        )
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            # An episode carries the overview + every reveal so far; cap generously.
            limit_mm_per_prompt={"image": max_images_per_prompt},
            mm_processor_kwargs=_pixel_kwargs(min_pixels, max_pixels),
            dtype="bfloat16",
            enable_lora=bool(adapter),
            max_lora_rank=max_lora_rank,
            seed=seed,
            trust_remote_code=True,
        )
        self.lora_request = None
        if adapter:
            from vllm.lora.request import LoRARequest

            self.lora_request = LoRARequest("policy", 1, adapter)

    # ------------------------------------------------------------------ #
    # Policy contract
    # ------------------------------------------------------------------ #
    def act(self, obs, **kwargs) -> dict:
        return self.act_batch([obs], **kwargs)[0]

    @contextlib.contextmanager
    def without_adapter(self):
        """Serve the base weights by dropping the LoRA request — no second engine,
        so ``--compare-base`` costs one model's worth of VRAM, not two."""
        saved, self.lora_request = self.lora_request, None
        try:
            yield self
        finally:
            self.lora_request = saved

    def act_batch(
        self,
        obs_list: list[dict],
        *,
        sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 320,
        collect_tokens: bool = False,  # accepted for signature parity; always empty
    ) -> list[dict]:
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=temperature if sample else 0.0,
            top_p=top_p if sample else 1.0,
            max_tokens=max_new_tokens,
            repetition_penalty=1.1,
        )
        prompts = [self._to_prompt(obs) for obs in obs_list]
        outputs = self.llm.generate(prompts, params, lora_request=self.lora_request)
        return [
            {"text": o.outputs[0].text, "prompt_ids": [], "token_ids": [], "logprobs": []}
            for o in outputs
        ]

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _to_prompt(self, obs: dict) -> dict:
        """Env observation → vLLM multimodal prompt.

        The chat template expands each ``{"type": "image"}`` placeholder into the
        model's image tokens, and ``multi_modal_data`` supplies the PIL images in
        the same order the env appended them (overview, then each reveal), so the
        alignment is positional and needs no URLs on disk.
        """
        text = self.processor.apply_chat_template(
            obs["messages"], tokenize=False, add_generation_prompt=True
        )
        return {"prompt": text, "multi_modal_data": {"image": list(obs["images"])}}


def _pixel_kwargs(min_pixels: int | None, max_pixels: int | None) -> dict:
    kw = {}
    if min_pixels:
        kw["min_pixels"] = min_pixels
    if max_pixels:
        kw["max_pixels"] = max_pixels
    return kw


def merge_adapter(base_model: str, adapter: str, out_dir: str) -> str:
    """Merge a LoRA adapter into the base weights and save.

    Escape hatch for vLLM builds whose LoRA support doesn't cover a given VLM:
    merge once, then serve ``out_dir`` as a plain model. Runs on CPU/fp16 — slow
    for 72B but it is a one-off.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    base_model = common.resolve_model(base_model)
    model = AutoModelForImageTextToText.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    model.save_pretrained(out_dir, safe_serialization=True)
    common.load_processor(base_model).save_pretrained(out_dir)
    print(f"Merged {adapter} into {base_model} -> {out_dir}")
    return out_dir


def build_policy(args, *, adapter: str | None = None):
    """Construct the backend selected by the standard CLI flags.

    Shared by ``eval/harness.py`` and ``data/build_sft_traces.py`` so both scripts
    take the same ``--backend / --tensor-parallel-size / --max-model-len`` flags.
    """
    if getattr(args, "backend", "hf") == "vllm":
        return VLLMPolicy(
            args.model,
            adapter=adapter,
            tensor_parallel_size=getattr(args, "tensor_parallel_size", 1),
            pipeline_parallel_size=getattr(args, "pipeline_parallel_size", 1),
            max_model_len=getattr(args, "max_model_len", 16384),
            gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.90),
            max_images_per_prompt=getattr(args, "max_inspects", 4) + 4,
        )
    device = common.resolve_device("auto")
    dtype = common.resolve_dtype(device, use_bf16=common.is_cuda(device))
    model, processor = common.load_policy(
        args.model,
        adapter,
        device,
        dtype,
        device_map=getattr(args, "device_map", None),
    )
    return common.HFPolicy(model, processor)


def add_backend_args(ap) -> None:
    """Register the inference-backend flags on an ``ArgumentParser``."""
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf",
                    help="hf: one replica per rank (torchrun-shardable). "
                         "vllm: one tensor-parallel replica, batched turns.")
    ap.add_argument("--tensor-parallel-size", type=int, default=1,
                    help="vLLM TP degree; set to the GPUs on the node for 32B/72B.")
    ap.add_argument("--pipeline-parallel-size", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--batch-episodes", type=int, default=16,
                    help="Episodes advanced in lockstep per batched generate (vLLM).")
    ap.add_argument("--device-map", default=None,
                    help="HF backend, single process only: 'auto' pipeline-shards one "
                         "model across the visible GPUs.")
