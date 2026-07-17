"""Build ``data/manifest.jsonl`` from the Fake-Vs-Real-Faces (Hard) dataset.

Substrate: ``hamzaboulahia/hardfakevsrealfaces`` on Kaggle — 1,288 300×300 JPEGs
(700 StyleGAN2 fakes from thispersondoesnotexist, 589 real from Unsplash), with a
``data.csv`` mapping image id → label. Image-level labels only: there are no
tiers, no per-region artifact annotations, and no metadata channel — by design,
the reward is anchored on the agent's own documented trajectory, not region
labels (those appear only in the eval-only evidence slice; see ``curation.md``).

This script is deliberately tolerant about the on-disk layout (Kaggle mirrors
reorganize), resolving images by either the CSV or a fake/real subfolder
convention, copies them into ``data/images/`` so the manifest is portable and
reproducible (rerun on the A100 to regenerate identical splits), and writes rows::

    {"id", "file_name" (repo-relative), "label" (0 real / 1 AI), "split"}

Usage:
    python data/build_manifest.py                      # download via kagglehub
    python data/build_manifest.py --src /path/to/unzipped/dataset
    python data/build_manifest.py --val 0.1 --test 0.1 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import defaultdict

from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(REPO_ROOT, "data", "images")
MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "manifest.jsonl")

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_FAKE_WORDS = ("fake", "ai", "synthetic", "gan", "1")
_REAL_WORDS = ("real", "authentic", "genuine", "0")


def _download() -> str:
    """Fetch the dataset via kagglehub, returning the local root path."""
    try:
        import kagglehub
    except ImportError as e:  # pragma: no cover - env-dependent
        raise SystemExit(
            "kagglehub not installed. `pip install kagglehub` and set Kaggle "
            "credentials (KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json), "
            "or pass --src pointing at an already-unzipped copy."
        ) from e
    print("Downloading hamzaboulahia/hardfakevsrealfaces via kagglehub ...")
    return kagglehub.dataset_download("hamzaboulahia/hardfakevsrealfaces")


def _label_from_text(text: str) -> int | None:
    t = str(text).strip().lower()
    if t in _REAL_WORDS or "real" in t:
        return 0
    if t in _FAKE_WORDS or "fake" in t:
        return 1
    return None


def _index_images(root: str) -> dict[str, str]:
    """Map a bare filename (and its stem) to its absolute path, recursively."""
    index: dict[str, str] = {}
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in _IMG_EXTS:
                p = os.path.join(dirpath, fn)
                index.setdefault(fn, p)
                index.setdefault(os.path.splitext(fn)[0], p)
    return index


def _from_csv(root: str) -> list[tuple[str, int]] | None:
    """Parse a ``*.csv`` label file if present. Returns (abs_path, label) pairs."""
    csv_path = next(
        (os.path.join(dp, fn) for dp, _, fs in os.walk(root) for fn in fs if fn.lower().endswith(".csv")),
        None,
    )
    if not csv_path:
        return None
    print(f"Using label CSV: {csv_path}")
    images = _index_images(root)
    rows: list[tuple[str, int]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fields = [c.lower() for c in (reader.fieldnames or [])]
        id_col = next((c for c in (reader.fieldnames or []) if c.lower() in ("id", "images_id", "image_id", "filename", "image", "name")), None)
        label_col = next((c for c in (reader.fieldnames or []) if "label" in c.lower() or "class" in c.lower()), None)
        if id_col is None or label_col is None:
            print(f"  CSV columns {fields} not recognized; falling back to folder layout.")
            return None
        for r in reader:
            key = os.path.basename(str(r[id_col]))
            label = _label_from_text(r[label_col])
            path = images.get(key) or images.get(os.path.splitext(key)[0])
            if path and label is not None:
                rows.append((path, label))
    return rows or None


def _from_folders(root: str) -> list[tuple[str, int]]:
    """Fallback: infer label from a fake/real component anywhere in the path."""
    rows: list[tuple[str, int]] = []
    for dirpath, _, files in os.walk(root):
        parts = dirpath.lower().replace(os.sep, "/").split("/")
        label = None
        if any("fake" in p for p in parts):
            label = 1
        elif any("real" in p for p in parts):
            label = 0
        if label is None:
            continue
        for fn in files:
            if os.path.splitext(fn)[1].lower() in _IMG_EXTS:
                rows.append((os.path.join(dirpath, fn), label))
    return rows


def _stratified_split(rows, val_frac, test_frac, seed):
    import random

    rng = random.Random(seed)
    by_label: dict[int, list] = defaultdict(list)
    for r in rows:
        by_label[r[1]].append(r)
    assigned = []
    for label, items in by_label.items():
        rng.shuffle(items)
        n = len(items)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        for i, (path, lab) in enumerate(items):
            split = "test" if i < n_test else "val" if i < n_test + n_val else "train"
            assigned.append((path, lab, split))
    rng.shuffle(assigned)
    return assigned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="Local path to an unzipped dataset (skips download).")
    ap.add_argument("--out", default=MANIFEST_PATH)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = args.src or _download()
    print(f"Dataset root: {root}")

    rows = _from_csv(root) or _from_folders(root)
    if not rows:
        raise SystemExit(
            f"Found no labeled images under {root}. Expected a label CSV or "
            "fake/real subfolders."
        )
    print(f"Discovered {len(rows)} labeled images "
          f"({sum(l == 1 for _, l in rows)} fake / {sum(l == 0 for _, l in rows)} real).")

    os.makedirs(IMAGES_DIR, exist_ok=True)
    assigned = _stratified_split(rows, args.val, args.test, args.seed)

    counts: dict[str, int] = defaultdict(int)
    with open(args.out, "w") as out:
        for i, (src_path, label, split) in enumerate(assigned):
            sub = "fake" if label == 1 else "real"
            os.makedirs(os.path.join(IMAGES_DIR, sub), exist_ok=True)
            dst = os.path.join(IMAGES_DIR, sub, f"{sub}_{i:05d}.png")
            try:
                Image.open(src_path).convert("RGB").save(dst)  # normalize to PNG
            except Exception as e:  # skip unreadable files rather than abort
                print(f"  skip {src_path}: {e}")
                continue
            rel = os.path.relpath(dst, REPO_ROOT)
            out.write(json.dumps({"id": f"face_{i:05d}", "file_name": rel, "label": label, "split": split}) + "\n")
            counts[split] += 1

    print(f"Wrote {sum(counts.values())} rows to {args.out}")
    print("  splits: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
