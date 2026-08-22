"""Upper bound: can the model do this task AT ALL, under best-case conditions?

The investigative environment deliberately withholds information — blurry
overview, a small inspect budget, a strict output format. If the policy fails
there, it could be the environment or it could be the model. This strips all of
that away: full-resolution image, one question, one word.

    python tools/ceiling_probe.py --backend vllm --tensor-parallel-size 2 --limit 100

Interpretation:
  ~0.85+     the model sees the artifacts; the collapse is environmental → fixable
  ~0.50-0.60 the model cannot separate these classes at all; no RL on top of the
             env will help, and the honest move is a different substrate/teacher
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import grid, prompts  # noqa: E402
from training import common, vllm_backend  # noqa: E402


def parse_answer(text: str):
    t = (text or "").strip().upper()
    has_ai, has_real = "AI" in t, "REAL" in t
    if has_ai and not has_real:
        return "AI"
    if has_real and not has_ai:
        return "REAL"
    return None  # empty, hedged, or both words present


# --------------------------------------------------------------------------- #
# Ranking mode (--auc)
# --------------------------------------------------------------------------- #
# Accuracy on a forced one-word answer measures the model's PRIOR as much as its
# eyesight: a policy that ranks every AI image above every real one still scores
# at the majority baseline if it never crosses its own threshold for saying the
# rarer word. That is not a hypothetical — the first genwukong ceiling answered
# "REAL" on 70 of 80 images yet was right on 9 of the 10 it called "AI".
#
# AUC asks the question the gate actually cares about: does P(AI) run higher on
# the AI images than on the real ones? It is threshold-free, so it separates "no
# signal" (AUC ~0.5, the substrate is dead) from "signal the model will not act
# on" (AUC high, fixable by calibration or by RL).

def _bucket(token: str) -> str | None:
    """Which answer a first token votes for, or None if it votes for neither."""
    t = (token or "").strip().upper()
    if not t:
        return None
    if t.startswith("AI") or t == "A":
        return "AI"
    if t.startswith("REA") or t == "R":
        return "REAL"
    return None


def score_ai(logprobs: dict[str, float]) -> float | None:
    """P(AI) / (P(AI) + P(REAL)) from the first-token distribution.

    Renormalising over just the two answers discards the mass the model spends on
    preambles, which is noise for a ranking question. None means the top-k held
    no recognisable answer token — reported as coverage rather than silently
    scored, because a low-coverage AUC is not measuring what it claims to.
    """
    import math

    mass = {"AI": 0.0, "REAL": 0.0}
    for tok, lp in logprobs.items():
        b = _bucket(tok)
        if b:
            mass[b] += math.exp(lp)
    total = mass["AI"] + mass["REAL"]
    return None if total <= 0.0 else mass["AI"] / total


def auc(scores: list[float], labels: list[int]) -> float:
    """Mann-Whitney U / rank AUC with tie-averaged ranks. No scipy dependency."""
    pairs = sorted(zip(scores, labels))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    n_pos = sum(l for _, l in pairs)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_pos = sum(r for r, (_, l) in zip(ranks, pairs) if l == 1)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def report_ranking(scores: list[float], labels: list[int], uncovered: int) -> None:
    n = len(scores)
    if n == 0:
        print("\nno scored items — every top-k lacked an AI/REAL token; "
              "raise --logprob-top-k or check the question")
        return
    a = auc(scores, labels)
    print(f"\nn={n}  AUC={a:.3f}   (0.5 = no signal, 1.0 = perfect ranking)")
    if uncovered:
        print(f"  WARNING: {uncovered} item(s) had no AI/REAL token in the top-k "
              f"and were excluded — treat the AUC as provisional")

    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    print(f"  mean P(AI): on AI images {mean(pos):.3f} | on REAL images {mean(neg):.3f}")

    # Sweep the OBSERVED scores, not a fixed grid. When a model is reluctant the
    # separation collapses into a narrow band near zero (here: AI mean 0.249 vs
    # REAL mean 0.034, most of both under 0.05), and a linear 0.05 grid steps
    # straight over the operating point — reporting a "best accuracy" far below
    # what the ranking actually supports.
    cand = sorted(set(scores))
    cand = [0.0] + [(a + b) / 2 for a, b in zip(cand, cand[1:])] + [1.0]
    n_pos, n_neg = sum(labels), n - sum(labels)
    swept = []
    for thr in cand:
        tp = sum(1 for s, l in zip(scores, labels) if l == 1 and s >= thr)
        tn = sum(1 for s, l in zip(scores, labels) if l == 0 and s < thr)
        swept.append(((tp + tn) / n, thr, tp, tn))
    best_acc, best_thr, best_tp, best_tn = max(swept)

    qs = sorted(scores)
    pct = lambda q: qs[min(len(qs) - 1, int(q * len(qs)))]
    print(f"  score percentiles: p10={pct(.10):.4f} p50={pct(.50):.4f} "
          f"p90={pct(.90):.4f} max={qs[-1]:.4f}")
    print("  threshold sweep (accuracy if we called AI at or above the threshold):")
    for frac in (0.1, 0.25, 0.5, 0.75, 0.9):
        acc, thr, tp, tn = swept[min(len(swept) - 1, int(frac * len(swept)))]
        print(f"    thr={thr:.4f}  acc={acc:.3f}  "
              f"AI recall={tp / max(n_pos, 1):.3f}  REAL recall={tn / max(n_neg, 1):.3f}")
    print(f"  BEST accuracy {best_acc:.3f} at threshold {best_thr:.4f} "
          f"(AI recall {best_tp / max(n_pos, 1):.3f}, "
          f"REAL recall {best_tn / max(n_neg, 1):.3f})")
    print("  ^ what this substrate offers once the decision is calibrated; compare "
          "this to the 0.85 gate, NOT the raw accuracy above.")


def main():
    ap = argparse.ArgumentParser()
    vllm_backend.add_backend_args(ap)
    ap.add_argument("--model", default=common.DEFAULT_MODEL)
    ap.add_argument("--dataset", default=common.DATASET,
                    help="Dataset namespace: selects the manifest and, unless --domain is\n"
                         "given, the domain-specific question.")
    ap.add_argument("--adapter", default=None, help="Optional LoRA to probe instead of base.")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--domain", choices=sorted(prompts.DOMAINS), default=None,
                    help="Which domain's question to ask. Default: whatever the dataset "
                         "maps to in env/prompts.DATASET_DOMAIN — so this is right by "
                         "default and only needs setting to probe a substrate as if it "
                         "were another.")
    ap.add_argument("--question", default=None, help="Override the prompt entirely.")
    ap.add_argument("--condition", choices=["full", "overview"], default="full",
                    help="full: native resolution (the CEILING — what the model could know). "
                         "overview: the env's downsampled view (the FLOOR — what it knows "
                         "for free). A usable substrate needs ceiling high and floor at chance; "
                         "the gap is the room investigation has to work in.")
    ap.add_argument("--overview-long-edge", type=int, default=140,
                    help="Must match env/grid.make_overview to measure the real floor.")
    ap.add_argument("--auc", action="store_true",
                    help="Score the FIRST-TOKEN ranking instead of the sampled word. "
                         "Reports AUC + a threshold sweep, which separates 'no signal' "
                         "from 'signal the model will not commit to'. vLLM backend only.")
    ap.add_argument("--logprob-top-k", type=int, default=20,
                    help="How many first-token logprobs to pull for --auc.")
    args = ap.parse_args()
    args.manifest = args.manifest or common.manifest_path(args.dataset)

    domain = args.domain or prompts.resolve_domain(args.dataset)
    question = args.question or prompts.classify_prompt(domain)

    rows = [json.loads(l) for l in open(args.manifest) if l.strip()]
    items = [r for r in rows if r.get("split") == args.split][: args.limit]
    truths = collections.Counter("AI" if int(r["label"]) == 1 else "REAL" for r in items)
    print(f"[{args.condition}] dataset={args.dataset} domain={domain} — {len(items)} "
          f"{args.split} images {dict(truths)}; "
          f"majority baseline {max(truths.values()) / len(items):.3f}")
    print(f"  question: {question}")

    policy = vllm_backend.build_policy(args, adapter=args.adapter)

    if args.auc and not hasattr(policy, "first_token_logprobs"):
        raise SystemExit("--auc needs first-token logprobs: use --backend vllm")

    cm = collections.Counter()
    scores: list[float] = []
    labels: list[int] = []
    uncovered = 0
    for start in range(0, len(items), args.batch):
        chunk = items[start : start + args.batch]
        obs_list = []
        for r in chunk:
            img = Image.open(os.path.join(common.REPO_ROOT, r["file_name"])).convert("RGB")
            if args.condition == "overview":
                img = grid.make_overview(img, long_edge=args.overview_long_edge,
                                         restore_to=img.size[0])
            obs_list.append({
                "messages": [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ]}],
                "images": [img],
            })
        if args.auc:
            for r, lp in zip(chunk, policy.first_token_logprobs(
                    obs_list, top_k=args.logprob_top_k)):
                s = score_ai(lp)
                if s is None:
                    uncovered += 1
                    continue
                scores.append(s)
                labels.append(int(r["label"]))
        else:
            outs = policy.act_batch(obs_list, sample=False, max_new_tokens=8)
            for r, o in zip(chunk, outs):
                truth = "AI" if int(r["label"]) == 1 else "REAL"
                cm[(truth, parse_answer(o["text"]))] += 1
        print(f"  {min(start + args.batch, len(items))}/{len(items)}", flush=True)

    if args.auc:
        report_ranking(scores, labels, uncovered)
        return

    n = sum(cm.values())
    correct = cm[("AI", "AI")] + cm[("REAL", "REAL")]
    answered = n - sum(cm[(t, None)] for t in ("AI", "REAL"))
    print(f"\nn={n}  accuracy={correct / max(n, 1):.3f}  answer_rate={answered / max(n, 1):.3f}")
    for t in ("AI", "REAL"):
        print(f"  truth {t:4} " + str({p: cm[(t, p)] for p in ("AI", "REAL", None)}))
    ai_pred = cm[("AI", "AI")] + cm[("REAL", "AI")]
    print(f"  predicted AI on {ai_pred}/{max(answered, 1)} answered "
          f"({ai_pred / max(answered, 1):.1%}) — class balance check")


if __name__ == "__main__":
    main()
