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
   verdict accuracy). This step produces ``data/<dataset>/evidence_slice.jsonl``.

The slice is used ONLY by ``eval/harness.py`` — never in any reward. See
``data/curation.md`` for the scoping rationale.

Phase 1 is 17 forward passes per image (original + 16 occlusions), which is why
it is both batched and rank-shardable — HF backend only, since it reads next-token
logits rather than generating.

    torchrun --nproc_per_node 8 data/build_evidence_slice.py --propose --limit 120 --topk 3
    #  ... hand-edit data/<dataset>/evidence_slice_review.jsonl ...
    python data/build_evidence_slice.py --finalize
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw

from env import grid, prompts
from env.environment import InvestigationEnv
from training import common

def review_path(dataset: str | None = None) -> str:
    return os.path.join(common.dataset_dir(dataset), "evidence_slice_review.jsonl")


def final_path(dataset: str | None = None) -> str:
    return os.path.join(common.dataset_dir(dataset), "evidence_slice.jsonl")


def _p_ai(model, processor, images, prompt: str, batch_size: int = 8) -> list[float]:
    """P(next token == 'AI') per image, under a single-shot classification prompt.

    Batched: one image plus its 16 occlusions is 17 forward passes per row, and at
    32B/72B that dominates the runtime. The processor pads left, so ``[:, -1]`` is
    the true final position for every row in the batch.
    """
    import torch

    device = next(model.parameters()).device
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tok = processor.tokenizer
    # Sum the probability mass over the plausible 'AI' spellings.
    ai_ids = {tok(t, add_special_tokens=False)["input_ids"][0] for t in (" AI", "AI", " ai", "Fake", " Fake")}

    out: list[float] = []
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        inputs = processor(
            text=[text] * len(chunk), images=chunk, return_tensors="pt", padding=True
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1]
        probs = torch.softmax(logits.float(), dim=-1)
        for row in probs:
            out.append(float(sum(row[i] for i in ai_ids if i < row.shape[0])))
    return out


def _occlude(image: Image.Image, cell: int, grid_size: int) -> Image.Image:
    out = image.copy()
    box = grid.cell_bbox(image.width, image.height, cell, grid_size)
    ImageDraw.Draw(out).rectangle(box, fill=(127, 127, 127))
    return out


def propose(args):
    """Occlusion saliency over the fake test images, strided across ranks.

    Pure per-image work with no shared state, so ``torchrun --nproc_per_node 8``
    scales it almost linearly; the per-rank proposals are gathered and written by
    rank 0 as one review file.
    """
    dist = common.init_distributed()
    env = InvestigationEnv(manifest_path=args.manifest, max_inspects=args.max_inspects,
                           shuffle=False, dataset=args.dataset,
                           overview_long_edge=args.overview_long_edge)
    fakes = [
        (i, r) for i, r in enumerate(env.records)
        if r.get("split") == "test" and int(r["label"]) == 1
    ]
    if args.limit:
        fakes = fakes[: args.limit]

    device = common.resolve_device("auto")
    dtype = common.resolve_dtype(device, use_bf16=common.is_cuda(device))
    common.warn_if_tight(common.resolve_model(args.model), training=False)
    model, processor = common.load_policy(args.model, None, device, dtype)
    common.rank0_print(
        f"Proposing artifact cells for {len(fakes)} fake test images "
        f"(world_size={dist.world_size})."
    )

    n_cells = grid.num_cells(args.grid)
    rows = []
    local = common.shard(fakes)
    for n, (idx, rec) in enumerate(local):
        image = Image.open(env._resolve_path(rec)).convert("RGB")
        cells = list(range(1, n_cells + 1))
        # [original, occlude(1), ..., occlude(16)] in one batched pass.
        scores = _p_ai(
            model, processor,
            [image] + [_occlude(image, c, args.grid) for c in cells],
            prompt=prompts.classify_prompt(dataset=args.dataset),
            batch_size=args.batch_size,
        )
        base, drops = scores[0], {c: scores[0] - s for c, s in zip(cells, scores[1:])}
        proposed = sorted(drops, key=drops.get, reverse=True)[: args.topk]
        rows.append({
            "index": idx, "id": rec.get("id"),
            "p_ai": round(base, 4),
            "proposed_cells": proposed,
            "saliency": {str(c): round(drops[c], 4) for c in proposed},
            "artifact_cells": proposed,  # human edits this in FINALIZE
        })
        if (n + 1) % 20 == 0:
            common.rank0_print(f"  rank0: {n + 1}/{len(local)}", flush=True)

    rows = common.gather_lists(rows)
    if dist.is_main:
        rows.sort(key=lambda r: r["index"])
        with open(review_path(args.dataset), "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"Wrote {len(rows)} proposals to {review_path(args.dataset)}. "
              f"Hand-verify 'artifact_cells' "
              f"per row, then run --finalize.")
    common.cleanup_distributed()


def finalize(args):
    kept = 0
    review, final = review_path(args.dataset), final_path(args.dataset)
    with open(review) as fin, open(final, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            fout.write(json.dumps({
                "index": row["index"], "id": row.get("id"),
                "artifact_cells": row.get("artifact_cells", []),
            }) + "\n")
            kept += 1
    print(f"Wrote {kept} rows to {final} "
          f"({sum(1 for _ in open(final))} total; empty artifact_cells = indistinguishable).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=common.DEFAULT_MODEL,
                    help="HF repo id or registry alias (7b | 32b | 72b | auto).")
    ap.add_argument("--dataset", default=common.DATASET,
                    help="Dataset namespace: selects the manifest, the output paths, "
                         "and the domain-specific classification question.")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--max-inspects", type=int, default=4)
    ap.add_argument("--overview-long-edge", type=int, default=140,
                    help="Overview resolution — how much the low-res view is blurred. "
                         "The zoom factor INSPECT buys is native_size/this, so ~native/7 "
                         "is the target: 140 suits 1024px images, 70 suits 512px. Too high "
                         "and the answer is readable from the overview; too low and the "
                         "agent cannot tell where to look. Find it with "
                         "tools/ceiling_probe.py --condition overview.")
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Occluded variants scored per forward pass (lower it at 72B).")
    ap.add_argument("--propose", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    args.manifest = args.manifest or common.manifest_path(args.dataset)

    if args.propose:
        propose(args)
    elif args.finalize:
        finalize(args)
    else:
        ap.error("pass --propose or --finalize")


if __name__ == "__main__":
    main()
