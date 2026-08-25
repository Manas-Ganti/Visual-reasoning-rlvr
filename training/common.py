"""Shared helpers for the training + eval scripts: the model registry, device /
dtype resolution, distributed (multi-GPU, multi-node) bookkeeping, model loading,
and the multi-turn rollout that drives the environment with a VLM.

Default base model is **Qwen2.5-VL-32B-Instruct**; ``--model 72b`` targets a full
8×(A100-80GB | H200) node and ``--model 7b`` is the small-GPU / smoke-test tier.
``--model auto`` picks the largest tier that fits the visible GPUs.

Two parallelism stories, deliberately kept separate:

* **Training** (SFT, GRPO) — data-parallel across ranks via 🤗 accelerate, with
  DeepSpeed ZeRO sharding optimizer/param state inside the data-parallel group.
  One process per GPU, launched by ``torchrun`` / ``accelerate launch``.
* **Inference** (eval, trace distillation, evidence slice) — either the same
  one-process-per-GPU sharding (HF backend; each rank evaluates a disjoint slice
  of the test split and the results are gathered on rank 0), or a single process
  with vLLM tensor parallelism across all GPUs of a node
  (``training/vllm_backend.py``), which is the faster path for large models.

Kept import-light at module load — torch/transformers are only imported inside
functions — so the CPU-only test/CI environment can import this module without a
GPU stack.
"""

from __future__ import annotations

import contextlib
import os
import re
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Model registry
#
# Short aliases so every script can take ``--model 32b`` instead of a full HF id;
# any HF repo id still passes through untouched. ``params_b`` drives the ``auto``
# tier pick and the VRAM sanity warnings — bf16 weights alone cost ~2 GB per
# billion params, and LoRA training needs roughly another 25% for activations,
# gradients and the KV cache during rollouts.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    params_b: float

    @property
    def weight_gb(self) -> float:
        return 2.0 * self.params_b  # bf16

    @property
    def train_gb(self) -> float:
        return 1.25 * self.weight_gb  # + LoRA state, activations, KV cache


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "7b": ModelSpec("Qwen/Qwen2.5-VL-7B-Instruct", 8.3),
    "32b": ModelSpec("Qwen/Qwen2.5-VL-32B-Instruct", 33.0),
    "72b": ModelSpec("Qwen/Qwen2.5-VL-72B-Instruct", 73.4),
    # Previous default, kept so the 7B→32B jump can be A/B'd against the old runs.
    "qwen2-vl-7b": ModelSpec("Qwen/Qwen2-VL-7B-Instruct", 8.3),
}

DEFAULT_MODEL = MODEL_REGISTRY["32b"].repo_id
LARGE_MODEL = MODEL_REGISTRY["72b"].repo_id
FALLBACK_MODEL = MODEL_REGISTRY["7b"].repo_id

# Visual-token budget. A single episode carries up to 1 overview + `max_inspects`
# reveals in context, so uncapped Qwen dynamic resolution can blow the context and
# the KV cache at 32B/72B. 256..1280 patch-tokens per image keeps a 4-inspect
# episode comfortably inside 8k tokens while leaving reveals legible.
DEFAULT_MIN_PIXELS = 256 * 28 * 28
DEFAULT_MAX_PIXELS = 1280 * 28 * 28

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------------- #
# Dataset namespacing
# --------------------------------------------------------------------------- #
# Every artifact a run produces — manifest, traces, checkpoints, episode logs,
# results — is namespaced by dataset, so substrates never overwrite each other's
# outputs and a stale file can't silently feed the next stage. Set once per shell
# (or per sbatch, which exports the submitting environment):
#
#     export VRR_DATASET=genimage
#
# Layout:
#     data/<ds>/{manifest,sft_traces}.jsonl · data/<ds>/images/
#     checkpoints/<ds>/{sft,grpo}-<model-tag>/
#     logs/<ds>/{eval,grpo}_episodes.jsonl · logs/<ds>/runs.tsv
#     results/<ds>/
DATASET = (os.environ.get("VRR_DATASET") or "genimage").strip()


def dataset_dir(dataset: str | None = None) -> str:
    return os.path.join(REPO_ROOT, "data", dataset or DATASET)


def manifest_path(dataset: str | None = None) -> str:
    return os.path.join(dataset_dir(dataset), "manifest.jsonl")


def traces_path(dataset: str | None = None) -> str:
    return os.path.join(dataset_dir(dataset), "sft_traces.jsonl")


def images_dir(dataset: str | None = None) -> str:
    return os.path.join(dataset_dir(dataset), "images")


def log_dir(dataset: str | None = None) -> str:
    return os.path.join(REPO_ROOT, "logs", dataset or DATASET)


def results_dir(dataset: str | None = None) -> str:
    return os.path.join(REPO_ROOT, "results", dataset or DATASET)


def model_tag(model_name: str) -> str:
    """Filesystem-safe short name, e.g. Qwen/Qwen2.5-VL-32B-Instruct -> qwen2.5-vl-32b."""
    return resolve_model(model_name).split("/")[-1].lower().replace("-instruct", "")


def checkpoint_dir(stage: str, model_name: str, dataset: str | None = None) -> str:
    """checkpoints/<dataset>/<stage>-<model-tag> — e.g. checkpoints/genimage/sft-qwen2.5-vl-32b."""
    return os.path.join(REPO_ROOT, "checkpoints", dataset or DATASET,
                        f"{stage}-{model_tag(model_name)}")


def record_run(stage: str, note: str = "", dataset: str | None = None) -> None:
    """Append one line to logs/<dataset>/runs.tsv — a registry of what ran, when,
    under which job id, writing where. Rank 0 only.

    Slurm's own logs are unique per job id but say nothing about which dataset or
    stage produced them; this is the index that makes a directory of
    ``infer-71xxxxx.out`` files navigable months later.
    """
    if not is_main():
        return
    import datetime

    sl = slurm_info()
    path = os.path.join(log_dir(dataset), "runs.tsv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new:
            f.write("timestamp\tdataset\tstage\tjob_id\tnode\tnote\n")
        f.write("\t".join([
            datetime.datetime.now().isoformat(timespec="seconds"),
            dataset or DATASET,
            stage,
            str(sl.get("slurm_job_id") or "-"),
            str(sl.get("slurm_nodelist") or os.uname().nodename),
            note.replace("\t", " "),
        ]) + "\n")


DEFAULT_MANIFEST = manifest_path()
DEFAULT_TRACES = traces_path()

_INDEX_RE = re.compile(r"index=(\d+)")
_DEGRADE_RE = re.compile(r"degradation=(\w+)")


def resolve_model(name: str) -> str:
    """Map a registry alias to an HF repo id; pass through anything else.

    ``auto`` picks the largest tier whose bf16 weights + training overhead fit in
    the aggregate VRAM of the current (possibly multi-node) job.
    """
    key = (name or "").strip().lower()
    if key == "auto":
        return autoselect_model()
    if key in MODEL_REGISTRY:
        return MODEL_REGISTRY[key].repo_id
    return name


def model_spec(repo_or_alias: str) -> ModelSpec | None:
    key = (repo_or_alias or "").strip()
    for alias, spec in MODEL_REGISTRY.items():
        if key.lower() == alias or key == spec.repo_id:
            return spec
    return None


def cluster_vram_gb() -> float:
    """Aggregate VRAM visible to the whole job (this node's GPUs × #nodes)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        per_node = sum(
            torch.cuda.get_device_properties(i).total_memory
            for i in range(torch.cuda.device_count())
        ) / 1024**3
    except Exception:
        return 0.0
    return per_node * max(dist_info().num_nodes, 1)


def autoselect_model() -> str:
    """Largest tier that fits, with headroom for ZeRO-sharded LoRA training."""
    vram = cluster_vram_gb()
    for alias in ("72b", "32b", "7b"):
        if vram >= MODEL_REGISTRY[alias].train_gb * 1.6:  # ZeRO/activation headroom
            return MODEL_REGISTRY[alias].repo_id
    return FALLBACK_MODEL


def warn_if_tight(repo_id: str, training: bool = True) -> None:
    """Print a rank-0 warning when the chosen model looks too big for the job."""
    spec = model_spec(repo_id)
    vram = cluster_vram_gb()
    if spec is None or vram <= 0:
        return
    need = spec.train_gb if training else spec.weight_gb
    if vram < need:
        rank0_print(
            f"WARNING: {repo_id} needs ~{need:.0f}GB {'training' if training else 'inference'} "
            f"VRAM but the job sees {vram:.0f}GB. Use more GPUs/nodes, enable "
            f"ZeRO-3 with CPU offload (configs/deepspeed_zero3_offload.json), or a smaller tier."
        )


# --------------------------------------------------------------------------- #
# Distributed bookkeeping
#
# Read straight from the launcher's environment so these work before (and
# without) ``torch.distributed`` being initialized — which matters because the
# scripts must stay runnable as a plain ``python foo.py`` on one GPU.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DistInfo:
    rank: int
    local_rank: int
    world_size: int
    local_world_size: int

    @property
    def num_nodes(self) -> int:
        return max(self.world_size // max(self.local_world_size, 1), 1)

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        """Global rank 0 — owns stdout, W&B, checkpoints and merged outputs."""
        return self.rank == 0

    @property
    def is_local_main(self) -> bool:
        """One per node — owns node-local work (e.g. HF cache warming)."""
        return self.local_rank == 0


def _env_int(*names: str, default: int = 0) -> int:
    for n in names:
        v = os.environ.get(n)
        if v is not None and v.strip():
            try:
                return int(v)
            except ValueError:
                pass
    return default


def dist_info() -> DistInfo:
    world = _env_int("WORLD_SIZE", "SLURM_NTASKS", default=1) or 1
    local_world = _env_int("LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE", default=0)
    return DistInfo(
        rank=_env_int("RANK", "SLURM_PROCID", default=0),
        local_rank=_env_int("LOCAL_RANK", "SLURM_LOCALID", default=0),
        world_size=world,
        local_world_size=local_world or world,
    )


def is_main() -> bool:
    return dist_info().is_main


def supported_config_kwargs(cls, desired: dict) -> dict:
    """Keep only the kwargs ``cls`` accepts, remapping obvious renames, loudly.

    TRL's SFTConfig/GRPOConfig inherit from transformers' TrainingArguments, and
    transformers 5.x moved several of its fields. A config written against 4.x
    then dies at construction with ONE unexpected keyword at a time — each
    discovered twenty minutes into a job holding a full node, and the next one
    only after the first is fixed.

    So adapt instead of guessing the new spelling. But never silently: a dropped
    knob changes training behaviour, and that must appear in the log rather than
    in a puzzling loss curve three hours later.
    """
    import dataclasses
    import inspect

    if dataclasses.is_dataclass(cls):
        accepted = {f.name for f in dataclasses.fields(cls)}
    else:
        try:
            accepted = set(inspect.signature(cls.__init__).parameters)
        except (TypeError, ValueError):
            return dict(desired)

    out, dropped, remapped = {}, [], []
    for k, v in desired.items():
        if k in accepted:
            out[k] = v
            continue
        # A rename usually keeps the distinctive stem: warmup_ratio -> lr_warmup_ratio.
        stem = k.replace("_", "")
        cand = [a for a in accepted if a not in desired and stem in a.replace("_", "")]
        if len(cand) == 1:
            out[cand[0]] = v
            remapped.append(f"{k} -> {cand[0]}")
        else:
            dropped.append(k)

    name = getattr(cls, "__name__", str(cls))
    if remapped:
        rank0_print(f"[{name}] remapped for this library version: {', '.join(remapped)}")
    if dropped:
        rank0_print(f"[{name}] WARNING dropped, not accepted by this version: "
                    f"{', '.join(dropped)} — training differs from the requested config")
    return out


def rank0_print(*args, **kwargs) -> None:
    if is_main():
        print(*args, **kwargs)


def init_distributed(backend: str = "nccl") -> DistInfo:
    """Initialize the process group + pin this rank to its GPU. No-op on 1 GPU.

    Used by the *inference* scripts (eval, distillation, evidence slice), which
    shard work by rank themselves. The training scripts leave this to accelerate.
    """
    info = dist_info()
    if not info.is_distributed:
        return info

    import torch
    import torch.distributed as dist

    if torch.cuda.is_available():
        torch.cuda.set_device(info.local_rank)
    elif backend == "nccl":
        backend = "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return info


def barrier() -> None:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        pass


def cleanup_distributed() -> None:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def shard(seq, info: DistInfo | None = None) -> list:
    """This rank's stride-slice of ``seq``.

    Strided (``seq[rank::world]``) rather than contiguous so that any ordering
    structure in the manifest (e.g. all fakes first) spreads evenly across ranks
    and every rank sees a comparable workload.
    """
    info = info or dist_info()
    return list(seq)[info.rank :: info.world_size] if info.is_distributed else list(seq)


def gather_lists(local: list) -> list:
    """All-gather per-rank lists and concatenate (identical on every rank).

    Returns ``local`` unchanged when not distributed, so callers have one path.
    """
    info = dist_info()
    if not info.is_distributed:
        return local
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return local
    buckets: list = [None] * info.world_size
    dist.all_gather_object(buckets, local)
    out: list = []
    for b in buckets:
        out.extend(b or [])
    return out


# --------------------------------------------------------------------------- #
# Run tracking — W&B identity + a progress/ETA callback
#
# The SLURM runs are long (GRPO is 10–34h depending on tier and node count), so
# the dashboard has to answer "is it healthy and when does it land?" without an
# SSH session on the compute node. Two pieces make that true:
#
#   * every run carries its SLURM identity in the W&B config, and resumes into
#     the *same* run id if SLURM requeues the job — so a preempted 48h run stays
#     one continuous chart instead of fragmenting into anonymous runs;
#   * every logging step publishes an ETA next to the metrics, compared against
#     the job's hard SLURM deadline, with a W&B alert when the run is projected
#     to be killed before it finishes.
#
# All of it is rank-0 only and best-effort: no W&B, no SLURM, or no scontrol
# degrades to fewer metrics, never to a crashed training run.
# --------------------------------------------------------------------------- #
def slurm_info() -> dict:
    """SLURM identity plus the job's hard deadline, resolved once.

    SLURM exports the job id but *not* the wall limit, so the deadline comes from
    ``scontrol``; when that is unavailable (local run, no SLURM) ``end_time`` is
    ``None`` and the walltime-margin metrics are simply omitted downstream.
    """
    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    info = {
        "slurm_job_id": job_id,
        "slurm_nodes": os.environ.get("SLURM_NNODES"),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "end_time": None,
    }
    if not job_id:
        return info
    try:
        import datetime
        import subprocess

        out = subprocess.run(
            ["scontrol", "show", "job", "-o", job_id],
            capture_output=True, text=True, timeout=10,
        ).stdout
        m = re.search(r"EndTime=(\S+)", out)
        if m and m.group(1) not in ("Unknown", "None"):
            info["end_time"] = datetime.datetime.strptime(
                m.group(1), "%Y-%m-%dT%H:%M:%S"
            ).timestamp()
    except Exception:
        pass  # not on SLURM, scontrol missing, or an unparseable EndTime
    return info


def wandb_init(project: str, stage: str, model_name: str, args, extra: dict | None = None):
    """Rank-0 ``wandb.init`` with SLURM identity and resume-on-requeue.

    ``stage`` is "sft"/"grpo". Keying the run id on the SLURM job id means a
    requeued job re-attaches to the same run rather than starting a new one.
    Returns the run (or ``None`` off rank 0 / when W&B is unavailable).
    """
    if not is_main():
        return None
    try:
        import wandb
    except ImportError:
        rank0_print("[wandb] wandb not installed; continuing untracked "
                    "(pip install wandb to enable the ETA dashboard)")
        os.environ["WANDB_MODE"] = "disabled"
        return None

    info = dist_info()
    sl = slurm_info()
    tag = model_name.split("/")[-1]
    name = f"{stage}-{tag}" + (f"-j{sl['slurm_job_id']}" if sl["slurm_job_id"] else "")
    config = {
        **vars(args),
        "world_size": info.world_size,
        "num_nodes": info.num_nodes,
        **{k: v for k, v in sl.items() if k != "end_time" and v is not None},
        **(extra or {}),
    }
    kwargs = dict(
        project=project,
        name=name,
        # Same job id → same run on requeue; `allow` still creates it the first time.
        id=f"{stage}-{sl['slurm_job_id']}" if sl["slurm_job_id"] else None,
        resume="allow" if sl["slurm_job_id"] else None,
        config=config,
    )
    # A logging failure must never take down a multi-hour training job, so this
    # degrades online → offline → disabled instead of raising. Offline runs land
    # in $WANDB_DIR and are replayed later with `wandb sync`.
    try:
        run = wandb.init(**kwargs)
    except Exception as e:
        rank0_print(f"[wandb] online init failed ({e}); retrying offline")
        try:
            os.environ["WANDB_MODE"] = "offline"
            run = wandb.init(**kwargs)
        except Exception as e2:
            rank0_print(f"[wandb] offline init failed too ({e2}); continuing untracked")
            os.environ["WANDB_MODE"] = "disabled"
            return None
    rank0_print(f"[wandb] {run.url}" if getattr(run, "url", None) else
                f"[wandb] run started (mode={os.environ.get('WANDB_MODE', 'online')})")
    return run


def progress_callback(ema_alpha: float = 0.2):
    """A ``TrainerCallback`` publishing progress + ETA to W&B and stdout.

    Step time is smoothed with an EMA and the **first step is excluded** — it
    carries model warmup, the initial generate, and dataloader spin-up, and would
    otherwise poison the ETA for the rest of a long run.

    Imported lazily (transformers is not available in the CPU-only CI env).
    """
    import time

    from transformers import TrainerCallback

    class _Progress(TrainerCallback):
        def __init__(self, alpha: float):
            self.alpha = alpha
            self.t0 = None
            self.last = None
            self.sec_per_step = None
            self.deadline = None
            self.alerted = False

        def on_train_begin(self, args, state, control, **kwargs):
            self.t0 = self.last = time.time()
            self.deadline = slurm_info()["end_time"]

        def on_step_end(self, args, state, control, **kwargs):
            now = time.time()
            dt = now - self.last
            self.last = now
            if state.global_step <= 1:
                return  # warmup step — measured, deliberately not folded in
            self.sec_per_step = (
                dt if self.sec_per_step is None
                else self.alpha * dt + (1 - self.alpha) * self.sec_per_step
            )

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not is_main() or self.t0 is None or self.sec_per_step is None:
                return
            total = state.max_steps or 0
            if total <= 0:
                return
            remaining = max(total - state.global_step, 0)
            eta_s = remaining * self.sec_per_step
            elapsed_s = time.time() - self.t0
            metrics = {
                "progress/global_step": state.global_step,
                "progress/total_steps": total,
                "progress/pct_complete": 100.0 * state.global_step / total,
                "progress/sec_per_step": self.sec_per_step,
                "progress/elapsed_hours": elapsed_s / 3600.0,
                "progress/eta_hours": eta_s / 3600.0,
                "progress/finish_unixtime": time.time() + eta_s,
            }
            # Compare the projection against the job's hard kill time, so the
            # dashboard shows *whether it will land* and not just when.
            if self.deadline:
                margin_s = (self.deadline - time.time()) - eta_s
                metrics["progress/walltime_left_hours"] = (self.deadline - time.time()) / 3600.0
                metrics["progress/walltime_margin_hours"] = margin_s / 3600.0
                metrics["progress/will_finish"] = int(margin_s > 0)

            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(metrics, step=state.global_step)
                    wandb.run.summary["progress/eta"] = time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(time.time() + eta_s)
                    )
                    # One alert, not one per step: the user is watching W&B, not
                    # the job's stdout, so this is how they hear about it.
                    if self.deadline and metrics["progress/walltime_margin_hours"] < 0 \
                            and not self.alerted:
                        self.alerted = True
                        wandb.alert(
                            title="GRPO/SFT will not finish inside its SLURM walltime",
                            text=(f"step {state.global_step}/{total}, "
                                  f"ETA {eta_s / 3600:.1f}h, "
                                  f"walltime left {(self.deadline - time.time()) / 3600:.1f}h. "
                                  f"Requeue with more nodes or cap --max-steps."),
                        )
            except Exception:
                pass

            eta_note = f" | eta {eta_s / 3600:.2f}h"
            if self.deadline:
                eta_note += f" | walltime margin {metrics['progress/walltime_margin_hours']:+.2f}h"
            print(f"[progress] step {state.global_step}/{total} "
                  f"({metrics['progress/pct_complete']:.1f}%) "
                  f"{self.sec_per_step:.1f}s/step{eta_note}", flush=True)

    return _Progress(ema_alpha)


# --------------------------------------------------------------------------- #
# Device / dtype
# --------------------------------------------------------------------------- #
def resolve_device(choice: str = "auto") -> str:
    """Resolve a device string. Under a distributed launch this returns this
    rank's own GPU (``cuda:<local_rank>``) so N processes don't pile onto cuda:0.
    """
    import torch

    info = dist_info()

    def _mps() -> bool:
        return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()

    if choice == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda requested but CUDA unavailable; using auto.")
        choice = "auto"
    if choice == "mps" and not _mps():
        print("WARNING: --device mps requested but MPS unavailable; using auto.")
        choice = "auto"
    if choice in ("cpu", "mps") or choice.startswith("cuda:"):
        return choice
    if choice == "cuda" or (choice == "auto" and torch.cuda.is_available()):
        return f"cuda:{info.local_rank}" if info.is_distributed else "cuda"
    return "mps" if _mps() else "cpu"


def is_cuda(device: str) -> bool:
    return str(device).startswith("cuda")


def resolve_dtype(device: str, use_bf16: bool = True):
    import torch

    if is_cuda(device):
        return torch.bfloat16 if use_bf16 else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def best_attn_implementation(device: str) -> str:
    """FlashAttention-2 when it's installed and we're on CUDA, else SDPA.

    Worth the check: at 32B/72B with several images in context, FA2 is a large
    memory win, but the wheel is fussy to build and must not be a hard dep.
    """
    if not is_cuda(device):
        return "eager"
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:
        return "sdpa"


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_processor(
    model_name: str,
    min_pixels: int | None = DEFAULT_MIN_PIXELS,
    max_pixels: int | None = DEFAULT_MAX_PIXELS,
):
    from transformers import AutoProcessor

    kwargs = {"padding_side": "left"}
    if min_pixels:
        kwargs["min_pixels"] = min_pixels
    if max_pixels:
        kwargs["max_pixels"] = max_pixels
    return AutoProcessor.from_pretrained(resolve_model(model_name), **kwargs)


def _zero3_active() -> bool:
    """True once a ZeRO-3 TrainingArguments/accelerate plugin has been created."""
    try:
        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

        return bool(is_deepspeed_zero3_enabled())
    except Exception:
        return False


def load_policy(
    model_name: str,
    adapter: str | None,
    device: str,
    dtype,
    *,
    attn_implementation: str | None = None,
    device_map: str | None = None,
    min_pixels: int | None = DEFAULT_MIN_PIXELS,
    max_pixels: int | None = DEFAULT_MAX_PIXELS,
    trainable: bool = False,
    gradient_checkpointing: bool = False,
):
    """Load a VLM policy (+ optional LoRA adapter) and its processor.

    ``device_map="auto"`` shards one model across the visible GPUs (pipeline
    style) — useful for single-process inference with a 72B, but never use it
    under a distributed launch, where each rank must own a full replica on its
    own GPU. ``trainable=True`` skips ``.eval()`` and leaves placement to
    accelerate/DeepSpeed.
    """
    from transformers import AutoModelForImageTextToText

    model_name = resolve_model(model_name)
    processor = load_processor(model_name, min_pixels=min_pixels, max_pixels=max_pixels)

    kwargs = {"attn_implementation": attn_implementation or best_attn_implementation(device)}
    if device_map:
        if dist_info().is_distributed:
            raise ValueError(
                "device_map is incompatible with a distributed launch: each rank must "
                "hold its own replica. Drop device_map (use DeepSpeed/FSDP to shard)."
            )
        kwargs["device_map"] = device_map

    try:
        model = AutoModelForImageTextToText.from_pretrained(model_name, dtype=dtype, **kwargs)
    except TypeError:  # transformers < 4.56 spells it torch_dtype
        model = AutoModelForImageTextToText.from_pretrained(model_name, torch_dtype=dtype, **kwargs)

    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)

    # Under ZeRO-3 the parameters are already sharded (and locally empty) by
    # deepspeed.zero.Init during from_pretrained — moving them by hand would
    # either fail or silently un-shard. DeepSpeed/Trainer owns placement there.
    if not device_map and not _zero3_active():
        model.to(device)

    if gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if trainable:
        model.train()
    else:
        model.eval()
    return model, processor


def lora_config(r: int = 32, alpha: int | None = None, dropout: float = 0.05, target: str = "all-linear"):
    """LoRA config shared by SFT and GRPO.

    ``target="all-linear"`` covers attention *and* MLP projections of the language
    tower; the Qwen2.5-VL vision blocks use different module names (``qkv``,
    ``proj``, ``fc1/fc2``) so they stay frozen, which is what we want — the visual
    encoder is not what needs to learn the investigation policy. Rank 32 (vs the
    old 16) is the cheap half of the 7B→32B upgrade: more capacity for the
    format + policy without touching optimizer memory much.
    """
    from peft import LoraConfig

    modules = {
        "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "all-linear": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }[target]
    return LoraConfig(
        r=r,
        lora_alpha=alpha if alpha is not None else 2 * r,
        lora_dropout=dropout,
        target_modules=modules,
        task_type="CAUSAL_LM",
    )


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
# Policies
#
# A policy is anything with ``act(obs, ...) -> {text, prompt_ids, token_ids,
# logprobs}``; ``run_episode`` below drives the env through that one method. Two
# implementations: HF ``generate`` (below — the training path, gives per-token
# logprobs) and vLLM (``training/vllm_backend.py`` — the batched inference path).
# --------------------------------------------------------------------------- #
def build_inputs(processor, obs, device):
    text = processor.apply_chat_template(
        obs["messages"], tokenize=False, add_generation_prompt=True
    )
    return processor(text=[text], images=obs["images"], return_tensors="pt", padding=True).to(device)


# The trajectory format puts ACTION last, so a turn is finished the moment that
# line is complete. Without a stop condition the model keeps generating to
# max_new_tokens — and a 32B asked to continue past a finished turn does not sit
# quietly: it apologises and regenerates the turn, repeatedly.
#
# Observed in GRPO rollouts: a well-reasoned two-turn episode followed by five
# near-identical "### Turn 2 (Final Correction)" blocks, scoring -0.87 for
# trailing noise the policy was given no way to avoid. Roughly two thirds of every
# turn's token budget went to that, which is also two thirds of the step time.
# Each alternative ends in \D — one character PAST the value — which is what
# proves the value is finished. Matching "INSPECT 1" at the end of the buffer
# would stop the turn on cell 1 when the model was about to write 12.
_ACTION_DONE = re.compile(
    r"ACTION:\s*(?:"
    r"INSPECT\s+\d+\D"
    r"|VERDICT\s+(?:AI|REAL)\b[^\n]*?confidence\s*=\s*[0-9]*\.?[0-9]+[^0-9.]"
    r")",
    re.IGNORECASE,
)


def _action_line_stopper(processor, prompt_len: int):
    """StoppingCriteria that ends the turn once its ACTION line is complete.

    Stops GENERATION rather than truncating text afterwards, so the token ids and
    logprobs GRPO differentiates stay aligned with the text the reward scores.
    Returns None when the transformers version does not expose StoppingCriteria,
    in which case behaviour is unchanged.
    """
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList
    except ImportError:
        return None

    class _Stop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs) -> bool:
            tail = input_ids[0, prompt_len:]
            if tail.numel() < 8:          # nothing plausible yet; skip the decode
                return False
            text = processor.decode(tail, skip_special_tokens=True)
            return bool(_ACTION_DONE.search(text))

    return StoppingCriteriaList([_Stop()])


class HFPolicy:
    """HF ``generate`` policy: one action per call, optional per-token logprobs.

    GRPO needs the logprobs for its gradient, so rollouts go through this even
    though vLLM would be faster; eval/distillation can use either backend.
    """

    backend = "hf"

    def __init__(self, model, processor, max_new_tokens: int = 640, repetition_penalty: float = 1.1):
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty

    @property
    def device(self):
        return next(self.model.parameters()).device

    @contextlib.contextmanager
    def without_adapter(self):
        """Temporarily evaluate the *base* model, adapter disabled.

        This is how ``--compare-base`` gets its baseline without a second copy of
        a 32B/72B in VRAM: PEFT just skips the LoRA deltas in the forward pass.
        A no-op (yields self) when no adapter is attached.
        """
        disable = getattr(self.model, "disable_adapter", None)
        if disable is None:
            yield self
            return
        with disable():
            yield self

    def act(
        self,
        obs,
        *,
        sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int | None = None,
        collect_tokens: bool = False,
    ) -> dict:
        import torch

        inputs = build_inputs(self.processor, obs, self.device)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "do_sample": sample,
            "repetition_penalty": self.repetition_penalty,
        }
        if sample:  # passing these under greedy decoding only earns warnings
            gen_kwargs.update(temperature=temperature, top_p=top_p)

        stopper = _action_line_stopper(self.processor, inputs["input_ids"].shape[1])
        if stopper is not None:
            gen_kwargs["stopping_criteria"] = stopper

        with torch.no_grad():
            out = self.model.generate(
                **inputs, return_dict_in_generate=True, output_scores=collect_tokens, **gen_kwargs
            )
        gen_len = out.sequences.shape[1] - inputs["input_ids"].shape[1]
        new_tokens = out.sequences[0, -gen_len:]

        logprobs: list[float] = []
        if collect_tokens:
            trans = self.model.compute_transition_scores(
                out.sequences, out.scores, normalize_logits=True
            )[0]
            logprobs = trans.tolist()

        return {
            "text": self.processor.decode(new_tokens, skip_special_tokens=True),
            "prompt_ids": inputs["input_ids"][0].tolist() if collect_tokens else [],
            "token_ids": new_tokens.tolist() if collect_tokens else [],
            "logprobs": logprobs,
        }


# --------------------------------------------------------------------------- #
# Multi-turn rollout
# --------------------------------------------------------------------------- #
def run_episode(
    policy,
    env,
    index: int,
    max_turns: int,
    degradation: str = "clean",
    sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 640,
    collect_tokens: bool = True,
    on_reset=None,
):
    """Drive one full episode through ``policy`` (one action per turn).

    When ``collect_tokens`` is set, also returns the flat token bookkeeping GRPO
    consumes: ``completion_ids`` / ``logprobs`` cover only the policy's generated
    tokens; the env's tool-result tokens ride in the context but not the gradient.
    Eval passes ``collect_tokens=False`` for a lighter greedy pass.

    ``on_reset(env)`` runs after ``reset`` and before the first action — the hook
    trace distillation uses to inject its teacher hint into the conversation.
    """
    obs, info = env.reset(options={"index": index, "degradation": degradation})
    if on_reset is not None:
        on_reset(env)
        obs = env._observation()

    prompt_ids = None
    completion_ids: list[int] = []
    logprobs: list[float] = []
    episode_return = 0.0

    for _ in range(max_turns):
        step_out = policy.act(
            obs,
            sample=sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            collect_tokens=collect_tokens,
        )
        if collect_tokens:
            if prompt_ids is None:
                prompt_ids = step_out["prompt_ids"]
            completion_ids.extend(step_out["token_ids"])
            logprobs.extend(step_out["logprobs"])

        obs, reward, terminated, truncated, info = env.step(step_out["text"])
        episode_return += reward
        if terminated or truncated:
            break

    return episode_result(info, episode_return, prompt_ids, completion_ids, logprobs)


def episode_result(info, episode_return, prompt_ids=None, completion_ids=None, logprobs=None) -> dict:
    return {
        "prompt_ids": prompt_ids or [],
        "completion_ids": completion_ids or [],
        "logprobs": logprobs or [],
        "episode_reward": episode_return,
        "correct": bool(info.get("correct")),
        "answered": info.get("action_type") == "verdict",
        "steps": info.get("steps", 0),
        "inspects_used": info.get("inspects_used", 0),
        "confidence": info.get("confidence"),
        "ground_truth": info.get("ground_truth"),
        "predicted": info.get("predicted_verdict"),
        "index": info.get("index"),
    }


def run_episodes_batched(
    policy,
    envs: list,
    jobs: list[tuple[int, str]],
    max_turns: int,
    sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 640,
    on_reset=None,
    on_episode=None,
) -> list[dict]:
    """Run many episodes in lockstep against a batching policy (``act_batch``).

    Multi-turn rollouts are latency-bound one episode at a time — the GPU sits
    idle between turns. Here ``len(envs)`` episodes advance together: every turn
    issues ONE batched generate over all still-live episodes, which is what makes
    vLLM worth using for eval and trace distillation. Episodes finish at different
    turns, so the live set shrinks as the batch progresses.

    ``on_episode(env, result)`` fires as each episode terminates, while that env's
    state is still intact (used for trace logging / inspected-cell readout).
    """
    results: list[dict] = [None] * len(jobs)  # type: ignore[list-item]
    queue = list(range(len(jobs)))
    slots: dict[int, dict] = {}  # env slot -> live episode bookkeeping

    while queue or slots:
        # Refill free env slots from the queue.
        for slot in range(len(envs)):
            if slot in slots or not queue:
                continue
            job_i = queue.pop(0)
            index, degradation = jobs[job_i]
            obs, _ = envs[slot].reset(options={"index": index, "degradation": degradation})
            if on_reset is not None:
                on_reset(envs[slot])
                obs = envs[slot]._observation()
            slots[slot] = {"job": job_i, "obs": obs, "turns": 0, "return": 0.0}
        if not slots:
            break

        live = sorted(slots)
        outs = policy.act_batch(
            [slots[s]["obs"] for s in live],
            sample=sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        for slot, out in zip(live, outs):
            st = slots[slot]
            obs, reward, terminated, truncated, info = envs[slot].step(out["text"])
            st["obs"], st["turns"] = obs, st["turns"] + 1
            st["return"] += reward
            if terminated or truncated or st["turns"] >= max_turns:
                res = episode_result(info, st["return"])
                results[st["job"]] = res
                if on_episode is not None:
                    on_episode(envs[slot], res)
                del slots[slot]

    return [r for r in results if r is not None]


def make_env_pool(env_factory, n: int) -> list:
    """``n`` independent env instances for lockstep batched rollouts."""
    return [env_factory() for _ in range(max(n, 1))]
