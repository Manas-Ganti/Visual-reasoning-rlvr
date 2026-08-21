"""Build a manifest by STREAMING a paired real/fake dataset off the Hugging Face Hub.

No archives, no Drive quota, no unzipping. ``load_dataset(..., streaming=True)``
pulls images one at a time and stops as soon as enough have been kept, so a
400-per-class build transfers well under a gigabyte regardless of how large the
source repo is.

    python data/build_manifest_hf.py \\
        --real bitmind/MS-COCO-unique \\
        --fake bitmind/MS-COCO-unique___FLUX.1-dev \\
        --dataset cocoflux --per-class 400

Prefer **paired** repos — bitmind's ``<real>___<generator>`` convention means the
fakes were generated from that real set's captions, so subject matter is matched
by construction. Pairing an arbitrary generator against an arbitrary photo set
adds a content-style shortcut ("arty-looking = fake") on top of whatever else is
being measured.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It never crops, resizes, or re-encodes lossily. Images are written exactly as
decoded, PNG by default. The only levers are *selection* (``--min-edge``,
``--aspect-range``) — pick different images, never alter them.

That matters because the failure mode here is a shortcut, not a lack of signal:
if every fake is a 1024x1024 square and every real is a 500x375 rectangle, aspect
ratio alone predicts the label, survives the overview downsample untouched, and
the agent never needs to investigate at all. So the build prints a per-class
dimension and aspect summary and warns when the two halves look separable
without opening the image. Fix that by choosing better-matched sources or by
filtering, and confirm with ``tools/ceiling_probe.py --condition overview``.

Emits the same schema every other stage consumes::

    {"id", "file_name" (repo-relative), "label" (0 real / 1 AI), "split", "generator"}
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training import common  # noqa: E402


def pick_image_column(ds, override: str | None) -> str:
    """Find the PIL-image column. Most of these repos call it ``image``."""
    if override:
        return override
    features = getattr(ds, "features", None) or {}
    for name, feat in features.items():
        if type(feat).__name__ == "Image":
            return name
    for name in ("image", "img", "png", "jpg"):
        if name in features:
            return name
    raise SystemExit(
        f"Could not find an image column. Columns: {sorted(features)}. "
        f"Pass --image-column."
    )


def stream_class(repo, split, config, image_col, want, min_edge, aspect_range,
                 out_dir, prefix, fmt):
    """Stream ``repo`` and save the first ``want`` images that pass the filters.

    Returns (paths, dims, rejected) where dims is a list of (w, h). Stops reading
    as soon as ``want`` images are kept — the whole point of streaming.
    """
    from datasets import load_dataset

    ds = load_dataset(repo, name=config, split=split, streaming=True)
    col = pick_image_column(ds, image_col)

    os.makedirs(out_dir, exist_ok=True)
    paths, dims = [], []
    rejected = Counter()
    seen = 0

    for row in ds:
        if len(paths) >= want:
            break
        seen += 1
        img = row.get(col)
        if img is None:
            rejected["no_image"] += 1
            continue
        try:
            img = img.convert("RGB")
        except Exception:
            rejected["unreadable"] += 1
            continue

        w, h = img.size
        if min(w, h) < min_edge:
            rejected["too_small"] += 1
            continue
        if aspect_range is not None:
            lo, hi = aspect_range
            if not lo <= (w / h) <= hi:
                rejected["aspect"] += 1
                continue

        name = f"{prefix}_{len(paths):06d}.{fmt}"
        path = os.path.join(out_dir, name)
        # Lossless by default: whatever the source encoder did is preserved in
        # the decoded pixels, and both classes are written identically, so
        # storage adds no signal either way.
        img.save(path, quality=95) if fmt == "jpg" else img.save(path)
        paths.append(path)
        dims.append((w, h))

        if len(paths) % 100 == 0:
            print(f"    kept {len(paths)}/{want} (seen {seen})", flush=True)

    if len(paths) < want:
        print(f"  ! only kept {len(paths)}/{want} from {repo} after {seen} rows "
              f"— rejected {dict(rejected)}")
    return paths, dims, rejected


def summarize_dims(dims: list[tuple[int, int]]) -> dict:
    """Short-edge and aspect-ratio distribution for one class."""
    if not dims:
        return {}
    short = sorted(min(w, h) for w, h in dims)
    aspect = sorted(w / h for w, h in dims)
    mid = lambda xs: statistics.median(xs)  # noqa: E731
    return {
        "n": len(dims),
        "short_min": short[0], "short_med": int(mid(short)), "short_max": short[-1],
        "aspect_min": round(aspect[0], 3),
        "aspect_med": round(mid(aspect), 3),
        "aspect_max": round(aspect[-1], 3),
        "square_frac": round(sum(0.95 <= a <= 1.05 for a in aspect) / len(aspect), 3),
    }


def shortcut_warnings(real: dict, fake: dict) -> list[str]:
    """Ways the two halves might be separable WITHOUT looking at the image.

    Anything here survives the environment's overview downsample, which is what
    makes it fatal rather than merely untidy: the agent could answer correctly
    from the blurred overview and never spend an inspect.
    """
    out = []
    if not real or not fake:
        return out

    if abs(real["aspect_med"] - fake["aspect_med"]) > 0.15:
        out.append(
            f"aspect ratio: real median {real['aspect_med']} vs fake "
            f"{fake['aspect_med']} — shape alone may predict the label")
    if abs(real["square_frac"] - fake["square_frac"]) > 0.5:
        out.append(
            f"squareness: {real['square_frac']:.0%} of real vs "
            f"{fake['square_frac']:.0%} of fake are square — a near-perfect giveaway")
    # Disjoint alone is not enough: a dataset whose images are all exactly 1.25
    # and one that is all exactly 1.28 technically never overlap, and nothing is
    # learnable from 0.03. Require a real margin between the two ranges.
    gap = max(real["aspect_min"], fake["aspect_min"]) - min(real["aspect_max"], fake["aspect_max"])
    if gap > 0.10:
        out.append(f"aspect ranges are disjoint by {gap:.2f} — trivially separable")

    lo, hi = sorted((real["short_med"], fake["short_med"]))
    if hi >= 2 * lo:
        out.append(
            f"resolution: real median short edge {real['short_med']}px vs fake "
            f"{fake['short_med']}px — partly hidden by the VLM's token budget, "
            f"but worth checking with the floor probe")
    return out


def assign_splits(rows, val: float, test: float, seed: int) -> None:
    """Stratified by (generator, label), matching build_manifest_genimage."""
    rng = random.Random(seed)
    by_stratum = defaultdict(list)
    for r in rows:
        by_stratum[(r["generator"], r["label"])].append(r)
    for stratum in by_stratum.values():
        rng.shuffle(stratum)
        n = len(stratum)
        n_val = max(1, int(n * val)) if n >= 10 else 0
        n_test = max(1, int(n * test)) if n >= 10 else 0
        for i, r in enumerate(stratum):
            r["split"] = "val" if i < n_val else ("test" if i < n_val + n_test else "train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, help="HF dataset id holding REAL images.")
    ap.add_argument("--fake", required=True, help="HF dataset id holding AI images.")
    ap.add_argument("--dataset", required=True,
                    help="Local namespace: data/<dataset>/, checkpoints/<dataset>/, ...")
    ap.add_argument("--generator", default=None,
                    help="Generator tag for the manifest rows (default: inferred from --fake).")
    ap.add_argument("--per-class", type=int, default=400, help="Images kept PER CLASS.")
    ap.add_argument("--real-split", default="train")
    ap.add_argument("--fake-split", default="train")
    ap.add_argument("--real-config", default=None)
    ap.add_argument("--fake-config", default=None)
    ap.add_argument("--image-column", default=None,
                    help="Override the auto-detected PIL image column.")
    ap.add_argument("--min-edge", type=int, default=512,
                    help="Reject images whose short edge is under this — a 4x4 cell "
                         "must still carry real detail after cropping.")
    ap.add_argument("--aspect-range", default=None,
                    help="Keep only images with LO<=w/h<=HI, e.g. 0.9,1.1. A filter, "
                         "not a transform: it selects matching images rather than "
                         "cropping anything. Use when the shortcut warnings fire.")
    ap.add_argument("--format", choices=["png", "jpg"], default="png",
                    help="png (default) is lossless. jpg is ~8x smaller and, applied "
                         "identically to both classes, also flattens any difference "
                         "in the sources' compression history.")
    ap.add_argument("--out", default=None, help="Manifest path (default: data/<ds>/manifest.jsonl).")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    generator = args.generator or args.fake.split("/")[-1].split("___")[-1].lower()
    aspect_range = None
    if args.aspect_range:
        lo, hi = (float(x) for x in args.aspect_range.split(","))
        aspect_range = (lo, hi)

    img_root = os.path.join(common.images_dir(args.dataset), generator)
    fmt = args.format

    print(f"Streaming real={args.real} fake={args.fake} -> generator={generator!r}")
    print(f"  per-class={args.per_class} min_edge={args.min_edge} "
          f"aspect_range={aspect_range} format={fmt}")

    print(f"  [real] {args.real}")
    real_paths, real_dims, _ = stream_class(
        args.real, args.real_split, args.real_config, args.image_column,
        args.per_class, args.min_edge, aspect_range,
        os.path.join(img_root, "real"), "real", fmt)
    print(f"  [ai]   {args.fake}")
    fake_paths, fake_dims, _ = stream_class(
        args.fake, args.fake_split, args.fake_config, args.image_column,
        args.per_class, args.min_edge, aspect_range,
        os.path.join(img_root, "ai"), "ai", fmt)

    if not real_paths or not fake_paths:
        raise SystemExit("One class came back empty — check --min-edge, the splits, "
                         "and that both repos actually hold images.")

    rows = []
    for label, paths in ((0, real_paths), (1, fake_paths)):
        sub = "ai" if label else "real"
        for i, p in enumerate(paths):
            rows.append({
                "id": f"{generator}_{sub}_{i:06d}",
                "file_name": os.path.relpath(p, common.REPO_ROOT),
                "label": label,
                "generator": generator,
            })

    assign_splits(rows, args.val, args.test, args.seed)
    rows.sort(key=lambda r: r["id"])

    manifest = args.out or common.manifest_path(args.dataset)
    os.makedirs(os.path.dirname(os.path.abspath(manifest)), exist_ok=True)
    with open(manifest, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    labels = Counter(r["label"] for r in rows)
    splits = Counter(r["split"] for r in rows)
    print(f"\nWrote {len(rows)} rows -> {manifest}")
    print(f"  splits {dict(splits)}")
    print(f"  labels AI={labels[1]} REAL={labels[0]} "
          f"(majority baseline {max(labels.values()) / len(rows):.3f})")

    # ---- the check that decides whether this substrate is honest ----------- #
    r_sum, f_sum = summarize_dims(real_dims), summarize_dims(fake_dims)
    print("\n=== dimensions (is the label predictable WITHOUT looking?) ===")
    print(f"  {'class':<6}{'n':>6}{'short edge min/med/max':>26}{'aspect min/med/max':>24}{'square':>8}")
    for name, s in (("real", r_sum), ("ai", f_sum)):
        edge = "{}/{}/{}".format(s["short_min"], s["short_med"], s["short_max"])
        asp = "{}/{}/{}".format(s["aspect_min"], s["aspect_med"], s["aspect_max"])
        print(f"  {name:<6}{s['n']:>6}{edge:>26}{asp:>24}{s['square_frac']:>8.0%}")

    warnings = shortcut_warnings(r_sum, f_sum)
    if warnings:
        print("\n  ⚠ SHORTCUT RISK — these survive the overview downsample:")
        for w in warnings:
            print(f"    - {w}")
        print("    Any of these lets the agent answer from the blurred overview without")
        print("    investigating, which makes a high pass rate meaningless. Fix by")
        print("    choosing better-matched sources, or filter with --aspect-range.")
    else:
        print("\n  no obvious shortcut: the two classes look alike in shape and size.")

    print(f"\nNext — confirm it empirically before training anything:")
    print(f"  VRR_DATASET={args.dataset} python tools/ceiling_probe.py "
          "--backend vllm --tensor-parallel-size 2 --condition full     # want >=0.85")
    print(f"  VRR_DATASET={args.dataset} python tools/ceiling_probe.py "
          "--backend vllm --tensor-parallel-size 2 --condition overview  # want ~chance")


if __name__ == "__main__":
    main()
