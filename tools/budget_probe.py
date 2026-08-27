"""The ceiling the AGENT actually faces — AUC as a function of inspect budget.

``ceiling_probe --condition full`` scores the whole image at native resolution.
The agent never sees that. It sees a blurred overview plus ``max_inspects`` cells
of a 4x4 grid — a quarter of the image at the default budget of 4. On synth1024
the gap between those two conditions was the entire story of GRPO run 1:

    full image, no environment          AUC 0.930
    inside the environment, 4 inspects  ~0.55 accuracy, indistinguishable from chance

The gate certified a substrate the policy could not reach. This measures the
condition that matters instead, and reports it as a MARGIN OVER THE FLOOR rather
than against a fixed threshold — because for a genuinely hard task the absolute
number matters far less than whether investigating buys anything at all.

    python tools/budget_probe.py --backend vllm --auc --budgets 0,2,4,6,8

Budget 0 IS the floor (overview only), so one run yields floor, ceiling and gap
together, all on the same images with the same scoring.

Cells are chosen at RANDOM, and nested across budgets (the budget-4 set contains
the budget-2 set), so the curve is comparable point to point and the variance from
resampling is removed. Random selection deliberately strips out the policy's
cell-choosing skill: this measures whether the INFORMATION is available at a given
budget, which is a lower bound on what a policy that chooses well could reach.

Reading it:
    gap >= 0.15-0.20   investigation earns its place; the environment works
    gap ~ 0            the agent that spends its whole budget lands where the
                       agent that spent none would have — no reward for looking
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ceiling_probe import auc, score_ai  # noqa: E402  (one implementation, shared)
from env import grid, prompts  # noqa: E402
from training import common, vllm_backend  # noqa: E402


def nested_cells(rng: random.Random, n_cells: int, budgets: list[int]) -> dict[int, list[int]]:
    """One shuffled cell order per image, sliced by budget.

    Nesting matters: if each budget drew its own sample, a dip at budget 6 could
    be luck of the draw rather than the curve. Here budget 6 is budget 4 plus two
    more cells, so any change is attributable to the extra evidence.
    """
    order = list(range(1, n_cells + 1))
    rng.shuffle(order)
    return {b: sorted(order[:b]) for b in budgets}


def main() -> None:
    ap = argparse.ArgumentParser()
    vllm_backend.add_backend_args(ap)
    ap.add_argument("--model", default=common.DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None,
                    help="Optional LoRA. Default probes the BASE model, which is the "
                         "information ceiling training has to work under.")
    ap.add_argument("--dataset", default=common.DATASET)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--budgets", default="0,2,4,6,8",
                    help="Inspect counts to measure. 0 is the floor (overview only).")
    ap.add_argument("--grid-size", type=int, default=4)
    ap.add_argument("--reveal-size", type=int, default=336,
                    help="Must match InvestigationEnv.reveal_size, or this measures a "
                         "zoom factor the agent never gets.")
    ap.add_argument("--overview-long-edge", type=int, default=56,
                    help="Must match the value every training stage used.")
    ap.add_argument("--domain", choices=sorted(prompts.DOMAINS), default=None)
    ap.add_argument("--question", default=None)
    ap.add_argument("--auc", action="store_true",
                    help="Score the first-token ranking. Strongly recommended: argmax "
                         "accuracy cannot separate 'no signal' from 'signal the policy "
                         "will not commit to', and this policy will not commit.")
    ap.add_argument("--logprob-top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.manifest = args.manifest or common.manifest_path(args.dataset)
    budgets = sorted({int(b) for b in args.budgets.split(",") if b.strip() != ""})
    domain = args.domain or prompts.resolve_domain(args.dataset)
    question = args.question or prompts.classify_prompt(domain)

    rows = [json.loads(l) for l in open(args.manifest) if l.strip()]
    items = [r for r in rows if r.get("split") == args.split][: args.limit]
    if not items:
        raise SystemExit(f"no rows with split={args.split} in {args.manifest}")

    n_ai = sum(int(r["label"]) == 1 for r in items)
    print(f"[budget] dataset={args.dataset} domain={domain} — {len(items)} {args.split} "
          f"images ({n_ai} AI / {len(items) - n_ai} REAL)")
    print(f"  overview_long_edge={args.overview_long_edge}  grid={args.grid_size}x{args.grid_size}"
          f"  reveal_size={args.reveal_size}  budgets={budgets}")
    print(f"  question: {question}")

    policy = vllm_backend.build_policy(args, adapter=args.adapter)
    if args.auc and not hasattr(policy, "first_token_logprobs"):
        raise SystemExit("--auc needs first-token logprobs: use --backend vllm")

    n_cells = grid.num_cells(args.grid_size)
    results: dict[int, dict] = {b: {"scores": [], "labels": [], "uncovered": 0} for b in budgets}

    for start in range(0, len(items), args.batch):
        chunk = items[start : start + args.batch]
        loaded = []
        for r in chunk:
            img = Image.open(os.path.join(common.REPO_ROOT, r["file_name"])).convert("RGB")
            rng = random.Random(f"{args.seed}:{r['id']}")     # per-image, reproducible
            loaded.append((r, img, nested_cells(rng, n_cells, budgets)))

        for b in budgets:
            obs_list = []
            for r, img, picks in loaded:
                overview = grid.make_overview(
                    img, long_edge=args.overview_long_edge, restore_to=img.size[0])
                # Same shape the env hands the policy: overview first, then reveals.
                images = [overview] + [
                    grid.crop_cell(img, c, args.grid_size, upscale_to=args.reveal_size)
                    for c in picks[b]
                ]
                content = [{"type": "image"} for _ in images]
                content.append({"type": "text", "text": question})
                obs_list.append({"messages": [{"role": "user", "content": content}],
                                 "images": images})

            if args.auc:
                lps = policy.first_token_logprobs(obs_list, top_k=args.logprob_top_k)
                for (r, _, _), lp in zip(loaded, lps):
                    s = score_ai(lp)
                    if s is None:
                        results[b]["uncovered"] += 1
                        continue
                    results[b]["scores"].append(s)
                    results[b]["labels"].append(int(r["label"]))
            else:
                from ceiling_probe import parse_answer
                outs = policy.act_batch(obs_list, sample=False, max_new_tokens=8)
                for (r, _, _), o in zip(loaded, outs):
                    pred = parse_answer(o["text"])
                    if pred is None:
                        results[b]["uncovered"] += 1
                        continue
                    results[b]["scores"].append(1.0 if pred == "AI" else 0.0)
                    results[b]["labels"].append(int(r["label"]))
        print(f"  {min(start + args.batch, len(items))}/{len(items)}", flush=True)

    def best_accuracy(scores, labels):
        """Accuracy at the best threshold — what the substrate offers once the
        decision is calibrated, which is the quantity training could reach."""
        if not scores:
            return float("nan")
        cand = sorted(set(scores))
        cand = [0.0] + [(a + b) / 2 for a, b in zip(cand, cand[1:])] + [1.0]
        n, pos = len(scores), sum(labels)
        return max(
            (sum(1 for s, l in zip(scores, labels) if l == 1 and s >= t)
             + sum(1 for s, l in zip(scores, labels) if l == 0 and s < t)) / n
            for t in cand
        ) if n and 0 < pos < n else float("nan")

    print("\n=== ceiling by inspect budget ===")
    print(f"  {'budget':>7} {'image seen':>11} {'AUC':>7} {'best acc':>9} {'gap vs b=0':>11}")
    floor = None
    table = {}
    for b in budgets:
        r = results[b]
        a = auc(r["scores"], r["labels"]) if r["scores"] else float("nan")
        acc = best_accuracy(r["scores"], r["labels"])
        if floor is None:
            floor = a
        table[b] = (a, acc)
        seen = f"{b / n_cells:.0%}"
        gap = "" if b == budgets[0] else f"{a - floor:+.3f}"
        note = "  <- floor" if b == budgets[0] else ""
        print(f"  {b:>7} {seen:>11} {a:>7.3f} {acc:>9.3f} {gap:>11}{note}")
        if r["uncovered"]:
            print(f"          ({r['uncovered']} item(s) unscored — no answer token in top-k)")

    top = budgets[-1]
    gap = table[top][0] - floor
    print(f"\n  gap at budget {top}: {gap:+.3f}")
    if gap >= 0.15:
        print("  VERDICT: investigation earns its place. Train at this budget.")
    elif gap >= 0.08:
        print("  VERDICT: thin. Investigation buys something, but a policy will struggle "
              "to find it through sampling noise. Raise the budget further, or make the "
              "artifacts easier to localise, before spending a training run.")
    else:
        print("  VERDICT: investigation buys nothing measurable — the agent that spends "
              "its whole budget lands where the agent that spent none would have. Raising "
              "the budget will not fix this; the grid shape or the substrate is wrong.")
    print("\n  Cells were chosen at RANDOM, so this is a LOWER bound: a policy that picks "
          "cells well should beat it. If the gap is zero here, no amount of cell-choosing "
          "skill can rescue it — there is nothing to choose between.")


if __name__ == "__main__":
    main()
