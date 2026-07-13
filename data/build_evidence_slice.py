"""Build the eval-only evidence slice: fakes with human-verified artifact cells.

Two-phase, because the honest version keeps a human in the loop:

1. PROPOSE (``--propose``): for each fake TEST image, compute a per-cell saliency
   map — an occlusion proxy for GradCAM that is model-agnostic and maps cleanly
   onto the 4×4 grid: gray-box each cell in turn, ask the base VLM "AI or REAL?",
   and score the cell by how much hiding it *lowers* P(AI). Cells whose occlusion
   most reduces the fake signal are the ones carrying the artifact. Writes the
   top-k proposed cells per image to a review file.

2. FINALIZE (``--finalize``): a human opens the review file and, per row, edits
   ``artifact_cells`` to the cells where a real, nameable artifact is actually
   visible at zoom (and sets it to ``[]`` for genuinely indistinguishable fakes,
   which are then excluded from the inspect-hit metric but still counted in
   verdict accuracy). This step produces ``evidence_slice.jsonl``.

The slice is used ONLY by ``eval/harness.py`` — never in any reward. See
``data/curation.md`` for the scoping rationale.

    python data/build_evidence_slice.py --propose --limit 120 --topk 3
    #  ... hand-edit data/evidence_slice_review.jsonl ...
    python data/build_evidence_slice.py --finalize
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw

from env import grid
from env.environment import InvestigationEnv
from training import common

REVIEW_PATH = os.path.join(common.REPO_ROOT, "data", "evidence_slice_review.jsonl")
FINAL_PATH = os.path.join(common.REPO_ROOT, "data", "evidence_slice.jsonl")

CLASSIFY_PROMPT = (
    "Is this face a real photograph or AI-generated? Answer with a single word: "
    "AI or REAL."
)


def _p_ai(model, processor, image) -> float:
    """P(next token == 'AI') under a single-shot classification prompt."""
    import torch

    device = next(model.parameters()).device
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": CLASSIFY_PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    probs = torch.softmax(logits, dim=-1)
    tok = processor.tokenizer
    # Sum the probability mass over the plausible 'AI' spellings.
    ai_ids = {tok(t, add_special_tokens=False)["input_ids"][0] for t in (" AI", "AI", " ai", "Fake", " Fake")}
    return float(sum(probs[i] for i in ai_ids if i < probs.shape[0]))


def _occlude(image: Image.Image, cell: int, grid_size: int) -> Image.Image:
    out = image.copy()
    box = grid.cell_bbox(image.width, image.height, cell, grid_size)
    ImageDraw.Draw(out).rectangle(box, fill=(127, 127, 127))
    return out


def propose(args):
    env = InvestigationEnv(manifest_path=args.manifest, max_inspects=args.max_inspects, shuffle=False)
    fakes = [
        (i, r) for i, r in enumerate(env.records)
        if r.get("split") == "test" and int(r["label"]) == 1
    ]
    if args.limit:
        fakes = fakes[: args.limit]

    device = common.resolve_device("auto")
    dtype = common.resolve_dtype(device, use_bf16=device == "cuda")
    model, processor = common.load_policy(args.model, None, device, dtype)
    print(f"Proposing artifact cells for {len(fakes)} fake test images.")

    n_cells = grid.num_cells(args.grid)
    with open(REVIEW_PATH, "w") as f:
        for n, (idx, rec) in enumerate(fakes):
            image = Image.open(env._resolve_path(rec)).convert("RGB")
            base = _p_ai(model, processor, image)
            drops = {c: base - _p_ai(model, processor, _occlude(image, c, args.grid)) for c in range(1, n_cells + 1)}
            proposed = sorted(drops, key=drops.get, reverse=True)[: args.topk]
            f.write(json.dumps({
                "index": idx, "id": rec.get("id"),
                "proposed_cells": proposed,
                "saliency": {str(c): round(drops[c], 4) for c in proposed},
                "artifact_cells": proposed,  # human edits this in FINALIZE
            }) + "\n")
            if (n + 1) % 20 == 0:
                print(f"  {n + 1}/{len(fakes)}", flush=True)
    print(f"Wrote proposals to {REVIEW_PATH}. Hand-verify 'artifact_cells' per row, "
          f"then run --finalize.")


def finalize(args):
    kept = 0
    with open(REVIEW_PATH) as fin, open(FINAL_PATH, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            fout.write(json.dumps({
                "index": row["index"], "id": row.get("id"),
                "artifact_cells": row.get("artifact_cells", []),
            }) + "\n")
            kept += 1
    print(f"Wrote {kept} rows to {FINAL_PATH} "
          f"({sum(1 for _ in open(FINAL_PATH))} total; empty artifact_cells = indistinguishable).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=common.DEFAULT_MODEL)
    ap.add_argument("--manifest", default=common.DEFAULT_MANIFEST)
    ap.add_argument("--max-inspects", type=int, default=4)
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--propose", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()

    if args.propose:
        propose(args)
    elif args.finalize:
        finalize(args)
    else:
        ap.error("pass --propose or --finalize")


if __name__ == "__main__":
    main()
