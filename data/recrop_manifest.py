"""Center-crop an existing substrate to a fixed size, in place of rebuilding it.

Why this exists: on GenImage the generated half is uniformly 512x512 while the
real half is varied and slightly smaller, which makes AREA a ~0.85-AUC predictor
of the label with no model and no content (``tools/manifest_stats.py``). Size
survives the overview downsample untouched — ``env/grid.make_overview`` scales
width and height by one factor — and Qwen bins images to a patch grid, so
different sizes even reach the policy as different token counts. Both gates then
measure the shortcut instead of the task.

``build_manifest_hf.py --center-crop`` fixes this at build time, but re-streaming
re-downloads everything AND resamples a different set of images, which throws
away comparability with the run that found the problem. This crops the images
already on disk: same substrate, same selection, one channel removed.

    python data/recrop_manifest.py --src genwukong --dst genwukong448 --size 448

Cropping selects pixels and invents nothing, so the artifacts the task depends on
survive. That is why it is allowed here where resizing is not: resampling would
manufacture frequency content the source never had, from two different starting
sizes — a new shortcut in place of the old one.

Verify before spending a GPU on it::

    python tools/manifest_stats.py --dataset genwukong448   # every predictor ~0.5
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training import common  # noqa: E402


def center_crop_to(img: Image.Image, n: int) -> Image.Image:
    """Center-crop to ``n`` x ``n``. Selects pixels; resamples nothing."""
    w, h = img.size
    left, top = (w - n) // 2, (h - n) // 2
    return img.crop((left, top, left + n, top + n))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source dataset namespace.")
    ap.add_argument("--dst", required=True, help="Destination namespace (must differ).")
    ap.add_argument("--size", type=int, default=448,
                    help="Crop to SIZE x SIZE. A multiple of 28 (Qwen's patch size) "
                         "leaves no rounding difference between images. Default 448.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be kept or dropped; write nothing.")
    args = ap.parse_args()

    if args.src == args.dst:
        raise SystemExit("--dst must differ from --src: cropping in place would "
                         "destroy the substrate the current gate numbers describe.")

    src_manifest = common.manifest_path(args.src)
    dst_manifest = common.manifest_path(args.dst)
    rows = [json.loads(l) for l in open(src_manifest) if l.strip()]
    print(f"src={src_manifest}  {len(rows)} rows -> {args.size}x{args.size}")

    kept: list[dict] = []
    dropped = collections.Counter()
    dims_before = collections.Counter()

    for r in rows:
        src_path = r["file_name"]
        if not os.path.isabs(src_path):
            src_path = os.path.join(common.REPO_ROOT, src_path)
        try:
            with Image.open(src_path) as im:
                im = im.convert("RGB")
                w, h = im.size
                dims_before[(w, h)] += 1
                if min(w, h) < args.size:
                    dropped["too_small"] += 1
                    continue
                out = center_crop_to(im, args.size)
        except Exception as e:
            dropped[f"unreadable:{type(e).__name__}"] += 1
            continue

        # Mirror the source layout under the new namespace so the two substrates
        # sit side by side and neither can overwrite the other.
        rel = os.path.relpath(src_path, common.REPO_ROOT)
        rel = rel.replace(f"data/{args.src}/", f"data/{args.dst}/", 1)
        dst_path = os.path.join(common.REPO_ROOT, rel)
        if not args.dry_run:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            out.save(dst_path)                     # PNG in, PNG out: lossless
        kept.append({**r, "file_name": rel})

    n_small = dropped.get("too_small", 0)
    if n_small:
        common_dims = ", ".join(f"{w}x{h}x{c}" for (w, h), c in dims_before.most_common(4))
        print(f"\n{n_small} image(s) smaller than {args.size}px were dropped. "
              f"Most common source sizes: {common_dims}")
        print("  If that is a large share of one class, lower --size — dropping "
              "unevenly across classes replaces a size shortcut with a sampling one.")
    if dropped:
        print(f"dropped: {dict(dropped)}")

    by = collections.Counter((r["split"], r["label"]) for r in kept)
    print(f"\nkept {len(kept)}/{len(rows)}")
    for k in sorted(by):
        print(f"  split={k[0]:5} label={k[1]}  {by[k]}")

    labels = collections.Counter(r["label"] for r in kept)
    if len(labels) == 2:
        skew = max(labels.values()) / max(min(labels.values()), 1)
        if skew > 1.10:
            print(f"  WARNING: classes now differ by {skew:.2f}x — the crop dropped "
                  f"one class harder than the other, which is itself a shortcut.")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    os.makedirs(os.path.dirname(dst_manifest), exist_ok=True)
    with open(dst_manifest, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {dst_manifest}")
    print(f"next:  python tools/manifest_stats.py --dataset {args.dst}   "
          f"# every predictor should read ~0.5")


if __name__ == "__main__":
    main()
