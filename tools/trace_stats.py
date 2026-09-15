"""Does a freshly distilled trace set teach what we wanted it to?

Run this BEFORE SFT. Distillation is cheap; SFT and GRPO are not, and cycle 2
spent ~20 GPU-hours training on traces whose behaviour was wrong in ways the
aggregate keep rate could not show.

The three checks correspond to the three failures measured in GRPO run 2 (see
results/reasoning_trace_analysis.md):

1. BELIEF RATCHET — run 2's median P(fake) slid 0.50 -> 0.20 -> 0.10 -> 0.05 ->
   0.01 over five inspections of a 16-cell grid, asserting 100-to-1 odds having
   seen under a third of the image. The prompt now says a clean cell is worth
   less than 0.1. If the median still bottoms out near 0.01, it did not land.
2. TAG DIRECTION — 60.5% of CONFIRMED tags preceded a FALLING belief, because
   the policy read CONFIRMED as "I checked this". The prompt now defines it as
   "the predicted artifact IS there, P(fake) goes UP".
3. UNCLEAR PRESENT — a cell that does not show the predicted feature is evidence
   of nothing. UNCLEAR did not appear in run 2 at all; misses were being scored
   as evidence for REAL.

    python tools/trace_stats.py                          # $VRR_DATASET's traces
    python tools/trace_stats.py --traces path/to.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import trajectory as tj  # noqa: E402
from training import common  # noqa: E402

_PFAKE = re.compile(r"P\(fake\)\s*=\s*([0-9]*\.?[0-9]+)", re.I)


def beliefs(turns: list[str]) -> list[float]:
    out = []
    for t in turns:
        for x in _PFAKE.findall(t):
            try:
                out.append(float(x))
            except ValueError:      # the teacher writes prose: "0.7." and friends
                pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=None)
    ap.add_argument("--dataset", default=common.DATASET)
    args = ap.parse_args()

    path = args.traces or os.path.join("data", args.dataset, "sft_traces.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"no traces at {path} — run data/build_sft_traces.py first")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    print(f"{len(rows)} traces from {path}\n")

    # --- 1. the ratchet ----------------------------------------------------
    paths = [beliefs(r["actions"]) for r in rows]
    paths = [p for p in paths if len(p) >= 3]
    print("1. BELIEF RATCHET   median P(fake) after N updates   (starts at 0.50)")
    print("   run 2 was        0.200  0.100  0.100  0.050  0.010   <- the failure")
    meds = []
    for k in range(5):
        vals = [p[k] for p in paths if len(p) > k]
        if len(vals) < 10:
            break
        meds.append(st.median(vals))
    print("   this set is      " + "  ".join(f"{m:.3f}" for m in meds))
    finals = [p[-1] for p in paths]
    frac_floor = sum(1 for v in finals if v <= 0.05) / max(len(finals), 1)
    print(f"   final P(fake) <= 0.05 in {frac_floor:.0%} of traces "
          f"(run 2: 65% below 0.10)")
    print(f"   {'FAIL — still ratcheting to certainty' if frac_floor > 0.40 else 'OK'}\n")

    # --- 2. tag direction --------------------------------------------------
    up = down = flat = 0
    tags = {"CONFIRMED": 0, "REFUTED": 0, "UNCLEAR": 0}
    for r in rows:
        prev = 0.5
        for turn in r["actions"]:
            e = tj.parse_turn(turn)
            if e.reconciliation in tags:
                tags[e.reconciliation] += 1
            if e.p_fake is None:
                continue
            d, prev = e.p_fake - prev, e.p_fake
            if e.reconciliation != tj.CONFIRMED:
                continue
            up += d > 0.02
            down += d < -0.02
            flat += abs(d) <= 0.02
    tot = max(up + down + flat, 1)
    print("2. TAG DIRECTION    CONFIRMED should be followed by P(fake) RISING")
    print(f"   up {up / tot:.1%}   down {down / tot:.1%}   flat {flat / tot:.1%}   (n={tot})")
    print(f"   run 2 was        up 24.8%   down 60.5%   <- the failure")
    print(f"   {'FAIL — CONFIRMED still means I-checked-this' if down / tot > 0.35 else 'OK'}\n")

    # --- 3. unclear --------------------------------------------------------
    tt = max(sum(tags.values()), 1)
    print("3. UNCLEAR PRESENT  mis-aimed inspections need somewhere to go")
    for k, v in tags.items():
        print(f"   {k:<10} {v:>6}  {v / tt:>6.1%}")
    print(f"   {'FAIL — UNCLEAR unused, misses still count as evidence' if tags['UNCLEAR'] / tt < 0.02 else 'OK'}")


if __name__ == "__main__":
    main()
