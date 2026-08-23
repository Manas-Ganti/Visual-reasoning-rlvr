"""Build a paired real/synthetic substrate to spec, instead of hoping one exists.

Every public AI-detection dataset surveyed for this project failed the same two
requirements, and they are the two the environment actually depends on:

* **native resolution >= 1024.** A 4x4 cell of a 512px image is 128 real pixels;
  of a 256px image, 64. ``INSPECT`` then upscales a blur and the central mechanic
  returns nothing. This is what killed the faces substrate (300px -> 75px cells).
* **pairing.** Fakes generated from unrelated prompts differ from the reals in
  *style*, and style survives any downsample — so the answer is free in the
  overview and the investigation is decoration. See
  ``results/geometry_confound.md``.

Generating the AI half from the reals' own captions gives both by construction,
plus identical geometry, which removes the confound that silently inflated every
gate number on GenImage.

Four stages, each independently restartable (they skip work already on disk):

    python data/build_paired_synthetic.py --stage crop     --dataset synth1024 \\
        --real-src <hf-repo-or-local-dir> --per-class 400
    python data/build_paired_synthetic.py --stage caption  --dataset synth1024 \\
        --backend vllm --tensor-parallel-size 1
    python data/build_paired_synthetic.py --stage generate --dataset synth1024
    python data/build_paired_synthetic.py --stage assemble --dataset synth1024

Then, before spending a GPU on gates::

    python tools/manifest_stats.py --dataset synth1024     # every predictor ~0.5

``caption`` needs the VLM (vLLM env); ``generate`` needs ``diffusers`` and may want
a separate env, since diffusers tends to pull a newer transformers than the one
vLLM pins. They are separate stages precisely so the two never have to coexist.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from build_manifest_hf import assign_splits  # noqa: E402
from training import common  # noqa: E402

# The generator sees roughly the first 77 CLIP tokens and silently discards the
# rest, so a caption's BUDGET matters as much as its density. Two consequences
# shape this prompt:
#
#   * ~50 words, not 60. Qwen overshoots any word limit it is given, and an
#     overshoot is not merely wasted — it is cut mid-clause by the text encoder.
#   * clutter FIRST. Left to itself the model opens with the subject and saves
#     background debris and imperfections for the final sentence, which is exactly
#     the part that gets truncated. That detail is the whole point: a tidy render
#     against a cluttered photograph makes "tidy = fake" readable straight from the
#     blurred overview, which is the shortcut this pairing exists to remove.
CAPTION_PROMPT_TEMPLATE = (
    "Describe this photograph in at most {words} words, as a single sentence "
    "fragment listing what is visible. Lead with the background clutter, "
    "incidental objects, and imperfections; mention the main subject and the "
    "lighting last. Be concrete and factual. Do not mention photography, image "
    "quality, or that this is a photo. Do not use full sentences."
)


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def class_dir(dataset: str, generator: str, cls: str) -> str:
    return os.path.join(common.images_dir(dataset), generator, cls)


def captions_path(dataset: str) -> str:
    return os.path.join(os.path.dirname(common.manifest_path(dataset)), "captions.jsonl")


def _listing(d: str) -> list[str]:
    return sorted(os.path.join(d, f) for f in os.listdir(d)) if os.path.isdir(d) else []


# --------------------------------------------------------------------------- #
# stage 1 — crop
# --------------------------------------------------------------------------- #
def iter_source(src: str, image_column: str | None):
    """Yield PIL images from a local directory or a streaming HF dataset."""
    if os.path.isdir(src):
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
        for root, _, files in os.walk(src):
            for f in sorted(files):
                if os.path.splitext(f)[1].lower() in exts:
                    try:
                        with Image.open(os.path.join(root, f)) as im:
                            yield im.convert("RGB")
                    except Exception:
                        continue
        return

    from datasets import load_dataset

    ds = load_dataset(src, split="train", streaming=True)
    col = image_column
    if col is None:
        feats = getattr(ds, "features", None) or {}
        col = next((k for k, v in feats.items()
                    if type(v).__name__ == "Image"), None) or "image"
    for row in ds:
        img = row.get(col)
        if img is None:
            continue
        try:
            yield img.convert("RGB")
        except Exception:
            continue


def center_crop_to(img: Image.Image, n: int) -> Image.Image:
    """Center-crop to n x n. Selects pixels; resamples nothing."""
    w, h = img.size
    left, top = (w - n) // 2, (h - n) // 2
    return img.crop((left, top, left + n, top + n))


def stage_crop(args) -> None:
    out = class_dir(args.dataset, args.generator, "real")
    os.makedirs(out, exist_ok=True)
    have = len(_listing(out))
    if have >= args.per_class:
        print(f"crop: {have} reals already on disk — nothing to do")
        return

    kept, seen, too_small = have, 0, 0
    for img in iter_source(args.real_src, args.image_column):
        if kept >= args.per_class:
            break
        seen += 1
        if min(img.size) < args.size:
            too_small += 1
            continue
        path = os.path.join(out, f"real_{kept:06d}.png")
        center_crop_to(img, args.size).save(path)
        kept += 1
        if kept == 1 or kept % 25 == 0:
            print(f"  kept {kept}/{args.per_class} (seen {seen})", flush=True)

    print(f"crop: kept {kept}, rejected {too_small} under {args.size}px, from {seen} read")
    if kept < args.per_class:
        print(f"  ! short of --per-class {args.per_class}. The source may not hold enough "
              f"images at >={args.size}px — check before lowering --size, since smaller "
              f"images shrink the 4x4 cell and are what make INSPECT useless.")


# --------------------------------------------------------------------------- #
# stage 2 — caption
# --------------------------------------------------------------------------- #
def stage_caption(args) -> None:
    from training import vllm_backend

    reals = _listing(class_dir(args.dataset, args.generator, "real"))
    if not reals:
        raise SystemExit("no cropped reals — run --stage crop first")

    done: dict[str, str] = {}
    cpath = captions_path(args.dataset)
    if os.path.exists(cpath):
        with open(cpath) as f:
            done = {r["file_name"]: r["caption"]
                    for r in (json.loads(l) for l in f if l.strip())}
    todo = [p for p in reals if os.path.relpath(p, common.REPO_ROOT) not in done]
    print(f"caption: {len(done)} done, {len(todo)} to do")
    if not todo:
        return

    policy = vllm_backend.build_policy(args)
    with open(cpath, "a") as f:
        for start in range(0, len(todo), args.batch):
            chunk = todo[start:start + args.batch]
            obs = []
            for p in chunk:
                with Image.open(p) as im:
                    obs.append({
                        "messages": [{"role": "user", "content": [
                            {"type": "image"},
                            {"type": "text",
                         "text": CAPTION_PROMPT_TEMPLATE.format(words=args.caption_words)},
                        ]}],
                        "images": [im.convert("RGB")],
                    })
            for p, o in zip(chunk, policy.act_batch(
                    obs, sample=False, max_new_tokens=args.max_caption_tokens)):
                caption = " ".join((o["text"] or "").split())
                f.write(json.dumps({"file_name": os.path.relpath(p, common.REPO_ROOT),
                                    "caption": caption}) + "\n")
            f.flush()      # append-and-flush per batch: a preempted job resumes
            print(f"  {min(start + args.batch, len(todo))}/{len(todo)}", flush=True)


# --------------------------------------------------------------------------- #
# stage 3 — generate
# --------------------------------------------------------------------------- #
def stage_generate(args) -> None:
    # Check inputs BEFORE importing torch/diffusers: loading them takes seconds and
    # fails with a library-path traceback that says nothing about the real problem.
    cpath = captions_path(args.dataset)
    if not os.path.exists(cpath):
        raise SystemExit(f"no captions at {cpath} — run --stage caption first")
    with open(cpath) as f:
        caps = [json.loads(l) for l in f if l.strip()]
    if not caps:
        raise SystemExit(f"{cpath} is empty — rerun --stage caption")

    import torch
    from diffusers import AutoPipelineForText2Image

    out = class_dir(args.dataset, args.generator, "ai")
    os.makedirs(out, exist_ok=True)

    pipe = AutoPipelineForText2Image.from_pretrained(
        args.sdxl, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    # Audit before spending an hour: the encoder's own tokenizer is the only
    # honest word-count, and a caption cut here loses its tail — which is where
    # the clutter was deliberately put.
    limit = getattr(pipe.tokenizer, "model_max_length", 77)
    over = [len(pipe.tokenizer(r["caption"]).input_ids) for r in caps if r["caption"]]
    n_cut = sum(1 for n in over if n > limit)
    if over:
        print(f"caption lengths: median {sorted(over)[len(over) // 2]} tokens, "
              f"max {max(over)}, encoder limit {limit}")
    if n_cut:
        print(f"  WARNING: {n_cut}/{len(over)} captions exceed {limit} tokens and will "
              f"be TRUNCATED — their tails are discarded. Re-caption with a smaller "
              f"--caption-words if that is most of them.")

    made = 0
    for i, row in enumerate(caps):
        path = os.path.join(out, f"ai_{i:06d}.png")
        if os.path.exists(path):          # restartable
            continue
        if not row["caption"]:
            continue
        # Seed from the index so a rerun reproduces the same substrate.
        gen = torch.Generator(device="cuda").manual_seed(args.seed + i)
        img = pipe(
            prompt=row["caption"],
            # No negative prompt on purpose. The usual "blurry, low quality, jpeg
            # artifacts" negatives push every generation toward clean, saturated
            # output — which widens exactly the global style gap the pairing
            # exists to close, and hands the floor a free answer.
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            height=args.size, width=args.size,
            generator=gen,
        ).images[0]
        img.save(path)
        made += 1
        if made == 1 or made % 25 == 0:
            print(f"  generated {made} (index {i}/{len(caps)})", flush=True)

    print(f"generate: wrote {made}, {len(_listing(out))} total on disk")


# --------------------------------------------------------------------------- #
# stage 4 — assemble
# --------------------------------------------------------------------------- #
def stage_assemble(args) -> None:
    reals = _listing(class_dir(args.dataset, args.generator, "real"))
    fakes = _listing(class_dir(args.dataset, args.generator, "ai"))
    if not reals or not fakes:
        raise SystemExit(f"one class is empty (real={len(reals)}, ai={len(fakes)})")

    # Equal counts, so no class-balance shortcut and no majority-baseline inflation.
    n = min(len(reals), len(fakes))
    if len(reals) != len(fakes):
        print(f"trimming to {n}/class (real={len(reals)}, ai={len(fakes)})")
    reals, fakes = reals[:n], fakes[:n]

    rows = []
    for label, paths in ((0, reals), (1, fakes)):
        sub = "ai" if label else "real"
        for i, p in enumerate(paths):
            rows.append({
                "id": f"{args.generator}_{sub}_{i:06d}",
                "file_name": os.path.relpath(p, common.REPO_ROOT),
                "label": label,
                "generator": args.generator,
            })
    assign_splits(rows, args.val, args.test, args.seed)

    out = common.manifest_path(args.dataset)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    import collections
    by = collections.Counter((r["split"], r["label"]) for r in rows)
    print(f"wrote {out}  ({len(rows)} rows)")
    for k in sorted(by):
        print(f"  split={k[0]:5} label={k[1]}  {by[k]}")
    print(f"\nnext:  python tools/manifest_stats.py --dataset {args.dataset}"
          f"   # every predictor should read ~0.5")


# --------------------------------------------------------------------------- #
def main() -> None:
    from training import vllm_backend

    ap = argparse.ArgumentParser()
    vllm_backend.add_backend_args(ap)
    ap.add_argument("--stage", required=True,
                    choices=["crop", "caption", "generate", "assemble", "all"])
    ap.add_argument("--dataset", required=True, help="Local namespace: data/<dataset>/...")
    ap.add_argument("--generator", default="sdxl",
                    help="Tag for the AI half. Reporting only — never a reward input. "
                         "Use a distinct tag per generator so eval/harness.py can break "
                         "results down and a held-out generator stays identifiable.")
    ap.add_argument("--real-src", default=None,
                    help="HF repo id OR a local directory of real photographs.")
    ap.add_argument("--image-column", default=None)
    ap.add_argument("--model", default=common.DEFAULT_MODEL, help="Captioning VLM.")
    ap.add_argument("--sdxl", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--per-class", type=int, default=400)
    ap.add_argument("--size", type=int, default=1024,
                    help="Both classes end up exactly SIZE x SIZE, so aspect, area and "
                         "long edge carry zero label information by construction. 1024 "
                         "gives a 256px 4x4 cell at --overview-long-edge 140 (~7x zoom).")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--caption-words", type=int, default=50,
                    help="Word budget asked of the captioner. SDXL's text encoders "
                         "truncate at 77 tokens (~55 words) and drop the remainder "
                         "silently, so aim under that rather than at it.")
    ap.add_argument("--max-caption-tokens", type=int, default=90,
                    help="Generation cap. Sized just above --caption-words so an "
                         "overshoot is visible as a cut caption rather than paid for "
                         "in tokens the generator will discard anyway.")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=5.5,
                    help="CFG. Deliberately lower than the usual 7.5+: high guidance "
                         "yields saturated, over-stylised output, which is a GLOBAL "
                         "difference that survives the overview downsample and hands "
                         "the floor a free answer. Lower CFG looks more like a snapshot.")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.stage in ("crop", "all") and not args.real_src:
        raise SystemExit("--real-src is required for --stage crop")

    stages = ["crop", "caption", "generate", "assemble"] if args.stage == "all" \
        else [args.stage]
    for s in stages:
        print(f"\n=== stage: {s} ===")
        {"crop": stage_crop, "caption": stage_caption,
         "generate": stage_generate, "assemble": stage_assemble}[s](args)


if __name__ == "__main__":
    main()
