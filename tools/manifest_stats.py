"""Is the label readable from the FILE rather than the picture?

The floor gate asks whether the env's downsampled overview withholds the answer.
When it does not, the usual cause on GenImage is not that the model sees
generation artifacts through the blur — blur destroys those — but that the two
classes were *encoded* differently. Real ImageNet photos arrive as heavily
compressed JPEGs; generated images arrive at whatever quality the pipeline saved.
That difference is a global statistic which survives any amount of downsampling,
so a policy can score well without ever looking at content, and the investigation
becomes decoration.

This reports per-class encoding stats and, more usefully, the AUC of
bytes-per-pixel as a *standalone* predictor of the label. No model, no GPU:

    python tools/manifest_stats.py                     # $VRR_DATASET
    python tools/manifest_stats.py --dataset genwukong --per-class 200

Read the bytes/px AUC as the confound's size:

    ~0.50   encoding carries no class information — the floor is not from here
    ~0.65   a real leak worth equalising
    ~0.80+  the label is largely readable from file size alone; fix the data
            before trusting any floor or ceiling number
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ceiling_probe import auc  # noqa: E402  (same directory; keep one implementation)
from training import common  # noqa: E402


def summarise(label: str, rows: list[dict]) -> None:
    n = len(rows)
    if not n:
        print(f"  {label}: no readable images")
        return
    fmts = collections.Counter(r["format"] for r in rows)
    modes = collections.Counter(r["mode"] for r in rows)
    mean = lambda k: sum(r[k] for r in rows) / n
    print(f"  {label}: n={n}  formats={dict(fmts)}  modes={dict(modes)}")
    print(f"      mean {mean('w'):.0f}x{mean('h'):.0f}px   "
          f"mean {mean('bytes') / 1024:.1f} KB   "
          f"mean {mean('bpp'):.4f} bytes/px")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=common.DATASET)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--split", default=None, help="Default: every split.")
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    manifest = args.manifest or common.manifest_path(args.dataset)
    rows = [json.loads(l) for l in open(manifest) if l.strip()]
    if args.split:
        rows = [r for r in rows if r.get("split") == args.split]

    # Sample per class across the WHOLE manifest: it is written one class at a
    # time (--only real / --only ai), so any head slice is single-class.
    by_label = collections.defaultdict(list)
    for r in rows:
        by_label[int(r["label"])].append(r)
    rng = random.Random(args.seed)
    picked = {k: rng.sample(v, min(args.per_class, len(v))) for k, v in by_label.items()}

    print(f"manifest={manifest}  split={args.split or 'all'}  "
          f"rows={len(rows)}  sampling {args.per_class}/class")

    stats: dict[int, list[dict]] = {0: [], 1: []}
    errors: list[str] = []
    for label, sample in picked.items():
        for r in sample:
            path = r["file_name"]
            if not os.path.isabs(path):
                path = os.path.join(common.REPO_ROOT, path)
            try:
                with Image.open(path) as im:
                    w, h = im.size
                    fmt, mode = im.format, im.mode
                size = os.path.getsize(path)
            except Exception as e:  # surfaced, never swallowed
                errors.append(f"{path}: {type(e).__name__}: {e}")
                continue
            stats[label].append({"format": fmt, "mode": mode, "w": w, "h": h,
                                 "bytes": size, "bpp": size / max(w * h, 1)})

    if errors:
        print(f"\n{len(errors)} image(s) unreadable — first few:")
        for e in errors[:5]:
            print(f"  {e}")
        if not any(stats.values()):
            raise SystemExit("no images could be read; check file_name paths")

    print("\nper-class encoding:")
    summarise("REAL", stats[0])
    summarise("AI  ", stats[1])

    scores = [r["bpp"] for r in stats[0]] + [r["bpp"] for r in stats[1]]
    labels = [0] * len(stats[0]) + [1] * len(stats[1])
    if stats[0] and stats[1]:
        a = auc(scores, labels)
        # Direction-free: a confound that runs the other way is just as damaging.
        strength = max(a, 1 - a)
        print(f"\nbytes/px as a standalone label predictor: AUC={a:.3f} "
              f"(strength {strength:.3f}, 0.5 = no leak)")
        if strength >= 0.80:
            print("  VERDICT: the label is largely readable from file size alone. Any "
                  "floor above chance is explained by this, not by the picture. "
                  "Re-encode both classes identically before trusting the gates.")
        elif strength >= 0.65:
            print("  VERDICT: a real encoding leak. Worth equalising, though it may "
                  "not account for the whole floor on its own.")
        else:
            print("  VERDICT: encoding does not carry the label. A high floor is "
                  "coming from the image content, not the container.")

        dims = {lab: collections.Counter((r["w"], r["h"]) for r in stats[lab])
                for lab in (0, 1)}
        if dims[0].most_common(1) != dims[1].most_common(1):
            print(f"  NOTE: most common dimensions differ — "
                  f"REAL {dims[0].most_common(1)[0]} vs AI {dims[1].most_common(1)[0]}. "
                  f"Resolution alone can carry the label.")


if __name__ == "__main__":
    main()
