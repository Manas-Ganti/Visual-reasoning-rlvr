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
    if not os.path.exists(manifest):
        root = os.path.join(common.REPO_ROOT, "data")
        built = sorted(d for d in os.listdir(root)
                       if os.path.exists(os.path.join(root, d, "manifest.jsonl")))
        raise SystemExit(
            f"no manifest at {manifest}\n"
            f"  built datasets: {', '.join(built) or '(none)'}\n"
            f"  build one first:   python data/build_manifest_hf.py --dataset "
            f"{args.dataset} --real <repo> --fake <repo> --per-class 400\n"
            f"  or screen a pair:  python data/build_manifest_hf.py --survey "
            f"--dataset {args.dataset} --real <repo> --fake <repo> --per-class 100")
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

    if not (stats[0] and stats[1]):
        return
    labels = [0] * len(stats[0]) + [1] * len(stats[1])
    rows_all = stats[0] + stats[1]

    # Each of these is a scalar the model can perceive WITHOUT looking at content.
    # Geometry matters most: env/grid.make_overview scales width and height by the
    # same factor and restores uniformly, so aspect ratio and relative size reach
    # the policy through the overview completely untouched. An encoding leak at
    # least dies when the pixels are resampled; a geometry leak never does.
    features = {
        "bytes/px      ": [r["bpp"] for r in rows_all],
        "aspect (w/h)  ": [r["w"] / max(r["h"], 1) for r in rows_all],
        "long edge     ": [max(r["w"], r["h"]) for r in rows_all],
        "area (px)     ": [r["w"] * r["h"] for r in rows_all],
    }
    print("\nstandalone label predictors — no model, no content, just the file:")
    worst = ("", 0.5)
    strengths = {}
    for name, vals in features.items():
        a = auc(vals, labels)
        # Direction-free: a confound running the other way is just as damaging.
        strength = max(a, 1 - a)
        strengths[name.strip()] = strength
        flag = "  <-- LEAK" if strength >= 0.65 else ""
        print(f"  {name} AUC={a:.3f}  strength={strength:.3f}{flag}")
        if strength > worst[1]:
            worst = (name.strip(), strength)

    for lab, tag in ((0, "REAL"), (1, "AI  ")):
        v = stats[lab]
        sq = sum(1 for r in v if r["w"] == r["h"]) / len(v)
        top, cnt = collections.Counter((r["w"], r["h"]) for r in v).most_common(1)[0]
        print(f"  {tag}: {sq:.0%} square, modal size {top} at {cnt}/{len(v)}")

    print(f"\nstrongest shortcut: {worst[0]} at strength {worst[1]:.3f}")
    # bytes/px is not a shortcut in the same sense as the geometric ones: the model
    # is shown pixels, never the file. A raised value means one class compresses
    # better — i.e. is smoother — which is a real visual property a detector should
    # be using, not a container artifact to be engineered away.
    if worst[0].startswith("bytes"):
        geo = max(strengths[k] for k in strengths if not k.startswith("bytes"))
        print("  NOTE: the strongest signal is bytes/px, which the model never sees. "
              "It indicates one class is smoother and so more compressible — a genuine "
              "visual difference, not a container confound. Do not engineer it away.")
        print(f"  Geometry, which the model DOES see through the overview, is at "
              f"{geo:.3f}.")
        if geo < 0.65:
            print("  VERDICT: container is clean.")
            return
    if worst[1] >= 0.80:
        print("  VERDICT: the label is largely readable from the file alone. No gate "
              "number means anything until this is equalised — a policy can score "
              "well without ever looking at the picture.")
    elif worst[1] >= 0.65:
        print("  VERDICT: a real shortcut. If it is geometric it survives the overview "
              "intact and explains a floor above chance; fix it by SELECTION "
              "(build_manifest_hf --aspect-range / --min-edge) so the two classes "
              "match in shape and size.")
    else:
        print("  VERDICT: no strong shortcut in the container. A high floor is coming "
              "from image content — the honest fix is a different generator.")


if __name__ == "__main__":
    main()
