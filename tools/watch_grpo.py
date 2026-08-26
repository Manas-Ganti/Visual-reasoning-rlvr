"""Health metrics for a running GRPO job, read from its completions parquet.

Reads only files the trainer has already written, so it is safe to run against a
live job. The reward curve alone hides the failures that matter here: a policy
can hold its mean reward while collapsing onto one class, saturating confidence,
or spending its whole budget every episode.

    python tools/watch_grpo.py --dataset synth1024
    python tools/watch_grpo.py --dataset synth1024 --last 20   # recent steps only
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training import common  # noqa: E402

_VERDICT = re.compile(r"VERDICT\s+(AI|REAL)", re.I)
_CONF = re.compile(r"confidence\s*=\s*([01](?:\.\d+)?|\.\d+)", re.I)
_PFAKE = re.compile(r"P\(fake\)\s*=\s*([01](?:\.\d+)?|\.\d+)", re.I)
_RECON = re.compile(r"RECONCILIATION[:\*\s]*([A-Za-z]+)")
_INSPECT = re.compile(r"\bINSPECT\s+\d+", re.I)
_LOOP = re.compile(r"apolog|misunderstand|corrected version|Final Correction|let me clarify", re.I)


def row_metrics(text: str) -> dict:
    v = _VERDICT.findall(text or "")
    conf = _CONF.findall(text or "")
    pf = [float(x) for x in _PFAKE.findall(text or "")]
    recs = [r.upper() for r in _RECON.findall(text or "")]
    coherent = bad = 0
    prev = 0.5
    for i, r in enumerate(recs):
        if i >= len(pf):
            break
        want = 1 if r.startswith("CONFIRM") else (-1 if r.startswith("REFUT") else 0)
        d = pf[i] - prev
        prev = pf[i]
        if want == 0:
            continue
        (coherent := coherent + 1) if (d > 0) == (want > 0) and abs(d) > 1e-9 else (bad := bad + 1)
    return {
        "answered": bool(v),
        "verdict": v[-1].upper() if v else None,
        "conf": float(conf[-1]) if conf else None,
        "inspects": len(_INSPECT.findall(text or "")),
        "pfake_final": pf[-1] if pf else None,
        "saturated": bool(pf) and (pf[-1] <= 0.02 or pf[-1] >= 0.98),
        "n_confirmed": sum(r.startswith("CONFIRM") for r in recs),
        "n_refuted": sum(r.startswith("REFUT") for r in recs),
        "coherent": coherent,
        "incoherent": bad,
        "loop": bool(_LOOP.search(text or "")),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=common.DATASET)
    ap.add_argument("--model", default="32b")
    ap.add_argument("--last", type=int, default=0, help="Only the N most recent step files.")
    args = ap.parse_args()

    import pandas as pd

    ckpt = common.checkpoint_dir("grpo", args.model, args.dataset)
    files = sorted(glob.glob(os.path.join(ckpt, "completions", "*.parquet")))
    if not files:
        raise SystemExit(f"no completions under {ckpt}/completions — has the job written a step yet?")
    if args.last:
        files = files[-args.last:]

    frames = []
    for f in files:
        try:
            d = pd.read_parquet(f)
        except Exception:
            continue          # the newest file may be mid-write
        step = int(re.search(r"(\d+)", os.path.basename(f)).group(1))
        col = "completion" if "completion" in d.columns else d.columns[1]
        m = pd.DataFrame([row_metrics(t) for t in d[col].fillna("")])
        m["step"] = step
        for c in ("episode_reward_func", "reward", "advantage"):
            if c in d.columns:
                m[c] = d[c].values
        frames.append(m)
    a = pd.concat(frames, ignore_index=True)

    rew = next((c for c in ("episode_reward_func", "reward") if c in a.columns), None)
    n = len(a)
    print(f"{len(files)} step file(s), {n} rollouts, steps {a.step.min()}–{a.step.max()}\n")

    ai = (a.verdict == "AI").sum()
    real = (a.verdict == "REAL").sum()
    print(f"  verdict balance   AI {ai}  REAL {real}   "
          f"({ai / max(ai + real, 1):.0%} AI)   <- 50% on a balanced split")
    print(f"  answered          {a.answered.mean():.0%}")
    print(f"  mean confidence   {a.conf.dropna().mean():.2f}   <- 0.97 means it never hedges")
    print(f"  belief saturated  {a.saturated.mean():.0%}   (P(fake) at 0.0 or 1.0)")
    print(f"  inspects/episode  {a.inspects.mean():.2f}   <- 4.0 means the budget is always spent")
    print(f"  CONFIRMED share   {a.n_confirmed.sum() / max(a.n_confirmed.sum() + a.n_refuted.sum(), 1):.0%}")
    print(f"  belief coherent   {a.coherent.sum() / max(a.coherent.sum() + a.incoherent.sum(), 1):.0%}")
    print(f"  loops             {a.loop.mean():.0%}")
    if rew:
        print(f"  mean reward       {a[rew].mean():+.3f}   positive {(a[rew] > 0).mean():.0%}")

    if len(files) > 3 and rew:
        print("\n  trend (equal thirds):")
        a["b"] = pd.qcut(a.step, 3, labels=["early", "mid", "late"], duplicates="drop")
        t = a.groupby("b", observed=True).agg(
            reward=(rew, "mean"),
            pct_AI=("verdict", lambda s: (s == "AI").mean()),
            conf=("conf", "mean"),
            inspects=("inspects", "mean"),
        ).round(3)
        print(t.to_string())
        print("\n  What to look for: pct_AI moving toward 0.50 (collapse correcting),")
        print("  conf falling below ~0.9 (learning to hedge), inspects below 4.0")
        print("  (learning efficiency). Reward alone can rise while all three stay stuck.")


if __name__ == "__main__":
    main()
