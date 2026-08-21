"""Build ``data/genimage/manifest.jsonl`` from the GenImage benchmark.

Substrate: **GenImage** (Zhu et al., NeurIPS 2023 D&B) — real ImageNet photographs
paired against generations from eight models (ADM, BigGAN, GLIDE, Midjourney,
Stable Diffusion 1.4/1.5, VQDM, Wukong). Replaces the 300x300 StyleGAN2 face set,
which failed the ceiling probe: the base VLM scored 0.39 against a 0.57 majority
baseline and identified 0 of 57 fakes even at full resolution
(see ``results/faces_negative_result.md``).

Why this substrate instead:

* **Resolution.** A 4x4 cell of a 300px image is 75 real pixels — "revealing" it
  upscales a blur, so inspecting could never surface new evidence. ``--min-edge``
  enforces a floor (default 512, so each cell carries >=128px).
* **Artifact type.** Diffusion/GAN artifacts here are *semantic and localized* —
  malformed hands, garbled text, impossible geometry, inconsistent shadows —
  which VLMs actually detect, which survive downsampling, and which live in
  specific cells. That is what makes ``INSPECT <n>`` meaningful and what gives
  the eval-only evidence slice something real to score against.
* **A free difficulty axis.** ``generator`` rides along in each row, so pass rate
  can be reported per generator alongside degradation and budget.

GenImage ships one directory per generator, each with ``train``/``val`` splits and
``ai``/``nature`` class folders. The walk below is tolerant of the variations
mirrors introduce (``fake``/``real``, missing ``val``, extra nesting).

    python data/build_manifest_genimage.py --src /path/to/GenImage --per-generator 200
    python data/build_manifest_genimage.py --src /path/to/GenImage \\
        --generators sdv1.4,midjourney --per-generator 400 --copy

Emits the same minimal schema every other stage consumes, plus ``generator``::

    {"id", "file_name" (repo-relative), "label" (0 real / 1 AI), "split", "generator"}
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training import common  # noqa: E402

DATASET = "genimage"
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG"}
_AI_DIRS = {"ai", "fake", "generated", "synthetic"}
_REAL_DIRS = {"nature", "real", "natural", "imagenet"}


def _label_from_path(path: str):
    """1 for AI, 0 for real, None if no class folder appears in the path."""
    parts = {p.lower() for p in path.split(os.sep)}
    if parts & _AI_DIRS:
        return 1
    if parts & _REAL_DIRS:
        return 0
    return None


# Substring -> canonical generator name, longest first so "stable_diffusion_v_1_4"
# is not shadowed by a shorter partial match.
#
# The official release names its Stable Diffusion dirs `imagenet_ai_0419_sdv4`
# and `imagenet_ai_0424_sdv5` — note `sdv4`/`sdv5`, NOT `sdv1.4`. Mirrors also
# ship `stable_diffusion_v_1_4`. All spellings normalise to `sdv1.4` / `sdv1.5`
# so `--generators sdv1.4` selects the right thing regardless of the layout you
# happened to download.
_GENERATOR_ALIASES = (
    ("stable_diffusion_v_1_4", "sdv1.4"),
    ("stable_diffusion_v_1_5", "sdv1.5"),
    ("sdv1.4", "sdv1.4"),
    ("sdv1.5", "sdv1.5"),
    ("sdv4", "sdv1.4"),
    ("sdv5", "sdv1.5"),
    ("midjourney", "midjourney"),
    ("biggan", "biggan"),
    ("glide", "glide"),
    ("wukong", "wukong"),
    ("vqdm", "vqdm"),
    ("adm", "adm"),
)


def _generator_from_path(root: str, path: str) -> str:
    """First path component under the source root — GenImage's per-generator dir."""
    rel = os.path.relpath(path, root)
    head = rel.split(os.sep)[0].lower()
    for token, canonical in _GENERATOR_ALIASES:
        if token in head:
            return canonical
    return head


def scan(root: str, min_edge: int, verify: bool):
    """Walk the source tree once, grouping usable images by (generator, label)."""
    buckets = defaultdict(list)
    skipped = Counter()
    for dirpath, _dirs, files in os.walk(root):
        label = _label_from_path(dirpath)
        if label is None:
            continue
        gen = _generator_from_path(root, dirpath)
        for fn in files:
            if os.path.splitext(fn)[1] not in _IMG_EXTS:
                continue
            path = os.path.join(dirpath, fn)
            if verify:
                try:
                    with Image.open(path) as im:
                        if min(im.size) < min_edge:
                            skipped["too_small"] += 1
                            continue
                except Exception:
                    skipped["unreadable"] += 1
                    continue
            buckets[(gen, label)].append(path)
    return buckets, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Root of the unzipped GenImage tree.")
    ap.add_argument("--dataset", default=DATASET, help="Namespace under data/ and logs/.")
    ap.add_argument("--generators", default=None,
                    help="Comma list to keep (default: every generator found).")
    ap.add_argument("--per-generator", type=int, default=200,
                    help="Images PER CLASS per generator (so 2x this per generator).")
    ap.add_argument("--min-edge", type=int, default=512,
                    help="Reject images whose short edge is under this — a 4x4 cell "
                         "must still carry real detail after cropping.")
    ap.add_argument("--no-verify", action="store_true",
                    help="Skip opening every file to check size (much faster, less safe).")
    ap.add_argument("--copy", action="store_true",
                    help="Copy images instead of symlinking (portable, uses disk).")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="Manifest path (default: data/<dataset>/manifest.jsonl). Use this to "
                         "build one generator at a time on a tight disk — download, extract, "
                         "build to its own file, delete the source, repeat, then concatenate. "
                         "Splits are stratified by (generator, label), so per-generator runs "
                         "and one combined run produce identical splits.")
    args = ap.parse_args()

    out_dir = common.dataset_dir(args.dataset)
    img_dir = common.images_dir(args.dataset)
    os.makedirs(img_dir, exist_ok=True)

    print(f"Scanning {args.src} (min_edge={args.min_edge}, verify={not args.no_verify}) ...")
    buckets, skipped = scan(args.src, args.min_edge, not args.no_verify)
    if not buckets:
        raise SystemExit(
            f"No labelled images under {args.src}. Expected class folders named one of "
            f"{sorted(_AI_DIRS)} / {sorted(_REAL_DIRS)} somewhere in the path."
        )

    keep = {g.strip().lower() for g in args.generators.split(",")} if args.generators else None
    generators = sorted({g for (g, _l) in buckets if keep is None or g in keep})
    if not generators:
        raise SystemExit(f"--generators matched nothing. Found: "
                         f"{sorted({g for g, _ in buckets})}")

    rng = random.Random(args.seed)
    rows = []
    for gen in generators:
        for label in (0, 1):
            pool = buckets.get((gen, label), [])
            if not pool:
                print(f"  ! {gen} label={label}: no images, skipping")
                continue
            rng.shuffle(pool)
            chosen = pool[: args.per_generator]
            sub = "ai" if label == 1 else "real"
            os.makedirs(os.path.join(img_dir, gen, sub), exist_ok=True)
            for src_path in chosen:
                name = f"{len(rows):06d}{os.path.splitext(src_path)[1].lower()}"
                dst = os.path.join(img_dir, gen, sub, name)
                if not os.path.exists(dst):
                    if args.copy:
                        shutil.copy2(src_path, dst)
                    else:
                        os.symlink(os.path.abspath(src_path), dst)
                rows.append({
                    "id": f"{gen}_{sub}_{len(rows):06d}",
                    "file_name": os.path.relpath(dst, common.REPO_ROOT),
                    "label": label,
                    "generator": gen,
                })
            print(f"  {gen:24} {sub:4} {len(chosen):5} (pool {len(pool)})")

    # Deterministic split, stratified by (generator, label) so every generator and
    # class appears in test — otherwise a per-generator breakdown is unreadable.
    rng = random.Random(args.seed)
    by_stratum = defaultdict(list)
    for r in rows:
        by_stratum[(r["generator"], r["label"])].append(r)
    for stratum_rows in by_stratum.values():
        rng.shuffle(stratum_rows)
        n = len(stratum_rows)
        # Guarantee held-out items per stratum once it is big enough to spare
        # them — otherwise a small generator vanishes from the test split and the
        # per-generator breakdown silently loses a row.
        n_val = max(1, int(n * args.val)) if n >= 10 else 0
        n_test = max(1, int(n * args.test)) if n >= 10 else 0
        for i, r in enumerate(stratum_rows):
            r["split"] = "val" if i < n_val else ("test" if i < n_val + n_test else "train")

    rows.sort(key=lambda r: r["id"])
    manifest = args.out or common.manifest_path(args.dataset)
    os.makedirs(os.path.dirname(os.path.abspath(manifest)), exist_ok=True)
    with open(manifest, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    splits = Counter(r["split"] for r in rows)
    labels = Counter(r["label"] for r in rows)
    print(f"\nWrote {len(rows)} rows -> {manifest}")
    print(f"  splits {dict(splits)}")
    print(f"  labels AI={labels[1]} REAL={labels[0]} "
          f"(majority baseline {max(labels.values()) / max(len(rows), 1):.3f})")
    print(f"  generators {generators}")
    if skipped:
        print(f"  skipped {dict(skipped)}")
    print(f"  images {'copied' if args.copy else 'symlinked'} under {img_dir}")
    print("\nNext: measure the ceiling and the floor before training anything —")
    print(f"  VRR_DATASET={args.dataset} python tools/ceiling_probe.py "
          "--backend vllm --tensor-parallel-size 2 --domain image --condition full")
    print(f"  VRR_DATASET={args.dataset} python tools/ceiling_probe.py "
          "--backend vllm --tensor-parallel-size 2 --domain image --condition overview")


if __name__ == "__main__":
    main()
