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

from env import grid  # noqa: E402
from training import common, vllm_backend  # noqa: E402

QUESTIONS = {
    "face": ("Is this face a real photograph or AI-generated (StyleGAN2)? "
             "Answer with exactly one word: REAL or AI."),
    "image": ("Is this image a real photograph or AI-generated? "
              "Answer with exactly one word: REAL or AI."),
}


def parse_answer(text: str):
    t = (text or "").strip().upper()
    has_ai, has_real = "AI" in t, "REAL" in t
    if has_ai and not has_real:
        return "AI"
    if has_real and not has_ai:
        return "REAL"
    return None  # empty, hedged, or both words present


def main():
    ap = argparse.ArgumentParser()
    vllm_backend.add_backend_args(ap)
    ap.add_argument("--model", default=common.DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None, help="Optional LoRA to probe instead of base.")
    ap.add_argument("--manifest", default=common.DEFAULT_MANIFEST)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--domain", choices=sorted(QUESTIONS), default="face",
                    help="Which prompt to ask; add entries to QUESTIONS for new substrates.")
    ap.add_argument("--question", default=None, help="Override the prompt entirely.")
    ap.add_argument("--condition", choices=["full", "overview"], default="full",
                    help="full: native resolution (the CEILING — what the model could know). "
                         "overview: the env's downsampled view (the FLOOR — what it knows "
                         "for free). A usable substrate needs ceiling high and floor at chance; "
                         "the gap is the room investigation has to work in.")
    ap.add_argument("--overview-long-edge", type=int, default=140,
                    help="Must match env/grid.make_overview to measure the real floor.")
    args = ap.parse_args()

    question = args.question or QUESTIONS[args.domain]

    rows = [json.loads(l) for l in open(args.manifest) if l.strip()]
    items = [r for r in rows if r.get("split") == args.split][: args.limit]
    truths = collections.Counter("AI" if int(r["label"]) == 1 else "REAL" for r in items)
    print(f"[{args.condition}] {len(items)} {args.split} images — {dict(truths)}; "
          f"majority baseline {max(truths.values()) / len(items):.3f}")

    policy = vllm_backend.build_policy(args, adapter=args.adapter)

    cm = collections.Counter()
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
        outs = policy.act_batch(obs_list, sample=False, max_new_tokens=8)
        for r, o in zip(chunk, outs):
            truth = "AI" if int(r["label"]) == 1 else "REAL"
            cm[(truth, parse_answer(o["text"]))] += 1
        print(f"  {min(start + args.batch, len(items))}/{len(items)}", flush=True)

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
