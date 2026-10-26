"""Convert distilled SFT traces into the demo's episode-log schema.

``data/build_sft_traces.py`` writes ``{index, degradation, actions[]}`` — raw
assistant completions. ``demo/app.py`` reads ``TraceLogger`` records, which need
an ``image_path`` and per-turn *parsed* fields. This bridges the two so Stage 0
output can be eyeballed in the viewer.

    python tools/traces_to_demo.py                       # -> logs/distill_episodes.jsonl
    python demo/app.py --log logs/distill_episodes.jsonl

Reward fields are left empty: the reward is a property of an *environment*
episode (budget, executed/rejected inspects), not of a replayed transcript.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.trajectory import parse_turn  # noqa: E402
from training import common  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_manifest(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=common.DATASET)
    ap.add_argument("--traces", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--grid", type=int, default=4)
    args = ap.parse_args()
    args.traces = args.traces or common.traces_path(args.dataset)
    args.manifest = args.manifest or common.manifest_path(args.dataset)
    args.out = args.out or os.path.join(common.log_dir(args.dataset),
                                        "distill_episodes.jsonl")

    records = load_manifest(args.manifest)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    n = 0
    with open(args.traces) as fin, open(args.out, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            rec = records[row["index"]]
            truth = "AI" if int(rec["label"]) == 1 else "REAL"

            turns, prediction, confidence = [], None, None
            for text in row["actions"]:
                t = parse_turn(text)
                cell = t.cell if t.action_type == "inspect" else None
                turns.append({
                    "action_type": t.action_type,
                    "cell": cell,
                    # Replayed transcripts carry no rejection info; a cell inside
                    # the grid is treated as having been executed.
                    "executed": bool(cell and 1 <= cell <= args.grid ** 2),
                    "verdict": t.verdict,
                    "confidence": t.confidence,
                    "p_fake": t.p_fake,
                    "reconciliation": t.reconciliation,
                    "observation": t.observation,
                    "reasoning": t.reasoning,
                    "hypothesis": t.hypothesis,
                })
                if t.action_type == "verdict":
                    prediction, confidence = t.verdict, t.confidence

            fout.write(json.dumps({
                "episode_id": f"distill-{row['index']:05d}",
                "image_path": os.path.join(REPO, rec["file_name"]),
                "grid": args.grid,
                "degradation": row.get("degradation", "clean"),
                "ground_truth": truth,
                "prediction": prediction,
                "confidence": confidence,
                "correct": prediction == truth,
                "total_reward": None,
                "reward_breakdown": {},
                "turns": turns,
            }) + "\n")
            n += 1

    print(f"Wrote {n} episodes to {args.out}")


if __name__ == "__main__":
    main()
