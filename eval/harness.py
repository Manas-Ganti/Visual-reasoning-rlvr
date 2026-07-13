"""Evaluation + Stage-3 verification.

Runs the (greedy) policy over the held-out TEST split and reports the headline
deliverables:

1. **Pass rate × degradation × budget** — the difficulty-calibration grid. Lets
   you state "to target this model at ~50% pass, use degradation Y at budget k".
2. **Calibration curve** — bucket verdicts by stated confidence; does 0.8 mean
   right ~80% of the time?
3. **Base vs policy** — accuracy delta + McNemar significance on paired items.
4. **Evidence slice** (optional, honest-scoped) — on human-verified fake images
   with known artifact cells, does the policy INSPECT true-artifact cells more
   often than the baseline? Indistinguishable images are excluded from this
   metric but still counted in verdict accuracy.

Stage-3 "prove it's real" checks map onto the flags here: hold out an unseen
`--degradation` level; run `--ablation no_coherence` GRPO elsewhere and compare;
`--evidence-slice` for the artifact-targeting signal; the trajectory logs (via
`--trace-log`) are the audit surface.

Greedy rollouts via HF generate keep eval deterministic; pass `--adapter` to
evaluate a trained checkpoint against the base model.

    python eval/harness.py --adapter checkpoints/grpo-qwen2vl \\
        --budgets 2,4 --degradations clean,jpeg,blur_downscale --limit 200
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import InvestigationEnv
from env.reward import RewardConfig
from training import common


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(results: list[dict]) -> dict:
    """Binary metrics with AI as the positive class."""
    tp = sum(r["predicted"] == "AI" and r["ground_truth"] == "AI" for r in results)
    fp = sum(r["predicted"] == "AI" and r["ground_truth"] == "REAL" for r in results)
    fn = sum(r["predicted"] == "REAL" and r["ground_truth"] == "AI" for r in results)
    n = len(results) or 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "n": len(results),
        "accuracy": sum(r["correct"] for r in results) / n,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0,
        "avg_inspects": sum(r["inspects_used"] for r in results) / n,
        "answer_rate": sum(r["answered"] for r in results) / n,
    }


def mcnemar(base: list[dict], policy: list[dict]) -> dict:
    b = sum(x["correct"] and not y["correct"] for x, y in zip(base, policy))
    c = sum(not x["correct"] and y["correct"] for x, y in zip(base, policy))
    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    try:
        from scipy.stats import chi2

        p = float(chi2.sf(stat, df=1))
    except ImportError:
        p = None
    return {"b": b, "c": c, "statistic": stat, "p_value": p}


def calibration_curve(results: list[dict], bins=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01)) -> list[dict]:
    """Empirical accuracy per stated-confidence bucket."""
    rows = []
    for lo, hi in zip(bins, bins[1:]):
        bucket = [r for r in results if r["confidence"] is not None and lo <= r["confidence"] < hi]
        if bucket:
            rows.append(
                {
                    "range": f"[{lo:.2f},{hi:.2f})",
                    "n": len(bucket),
                    "mean_conf": sum(r["confidence"] for r in bucket) / len(bucket),
                    "accuracy": sum(r["correct"] for r in bucket) / len(bucket),
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# Rollouts
# --------------------------------------------------------------------------- #
def eval_over(model, processor, env, indices, degradation, max_turns, logger=None, phase="eval"):
    results = []
    for i in indices:
        ep = common.run_episode(
            model, processor, env, index=i, degradation=degradation,
            max_turns=max_turns, sample=False, collect_tokens=False,
        )
        ep["inspected_cells"] = list(env.state["inspected_cells"])
        results.append(ep)
        if logger is not None:
            logger.log(env.get_trace(phase=phase), force=True)
    return results


def evidence_slice_hit_rate(model, processor, env, slice_rows, degradation, max_turns) -> dict:
    """Fraction of episodes whose inspected cells intersect the human-verified
    artifact cells. Higher ⇒ the policy looks where the real artifacts are."""
    hits, n = 0, 0
    for row in slice_rows:
        artifact_cells = set(row["artifact_cells"])
        if not artifact_cells:
            continue  # indistinguishable image: excluded from this metric
        common.run_episode(
            model, processor, env, index=row["index"], degradation=degradation,
            max_turns=max_turns, sample=False, collect_tokens=False,
        )
        inspected = set(env.state["inspected_cells"])
        hits += bool(inspected & artifact_cells)
        n += 1
    return {"n": n, "inspect_hit_rate": hits / n if n else 0.0}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _print_grid(title, grid):
    print(f"\n=== {title}: pass rate × degradation × budget ===")
    degs = sorted({d for d, _ in grid})
    budgets = sorted({b for _, b in grid})
    header = "  budget " + "".join(f"{d:>16}" for d in degs)
    print(header)
    for b in budgets:
        cells = "".join(f"{grid[(d, b)]['accuracy']:>16.3f}" for d in degs)
        print(f"  {b:>6} {cells}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=common.DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None, help="Trained LoRA adapter (policy).")
    ap.add_argument("--manifest", default=common.DEFAULT_MANIFEST)
    ap.add_argument("--budgets", default="4", help="Comma list of inspect budgets, e.g. 2,4")
    ap.add_argument("--degradations", default="clean", help="Comma list of degradation levels.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--compare-base", action="store_true", help="Also run the base model + McNemar.")
    ap.add_argument("--evidence-slice", default=None, help="Path to evidence_slice.jsonl.")
    ap.add_argument("--trace-log", default=None)
    args = ap.parse_args()

    budgets = [int(x) for x in args.budgets.split(",")]
    degradations = [x.strip() for x in args.degradations.split(",")]

    device = common.resolve_device("auto")
    dtype = common.resolve_dtype(device, use_bf16=device == "cuda")

    # Test indices are shared across all configs (same env, re-created per budget).
    probe = InvestigationEnv(manifest_path=args.manifest, max_inspects=max(budgets), shuffle=False)
    test_idx = [i for i, r in enumerate(probe.records) if r.get("split") == "test"]
    if args.limit:
        test_idx = test_idx[: args.limit]
    print(f"Evaluating on {len(test_idx)} held-out test images "
          f"(budgets={budgets}, degradations={degradations}).")

    logger = None
    if args.trace_log:
        from env.trace_logger import TraceLogger

        logger = TraceLogger(args.trace_log)

    model, processor = common.load_policy(args.model, args.adapter, device, dtype)
    base_model = None
    if args.compare_base and args.adapter:
        base_model, _ = common.load_policy(args.model, None, device, dtype)

    policy_grid, base_grid = {}, {}
    paired_base, paired_policy = [], []
    for budget in budgets:
        env = InvestigationEnv(
            manifest_path=args.manifest, max_inspects=budget,
            reward_config=RewardConfig(), shuffle=False,
        )
        max_turns = budget + 3
        for deg in degradations:
            pol = eval_over(model, processor, env, test_idx, deg, max_turns, logger, "eval-policy")
            policy_grid[(deg, budget)] = compute_metrics(pol)
            if base_model is not None:
                bas = eval_over(base_model, processor, env, test_idx, deg, max_turns, logger, "eval-base")
                base_grid[(deg, budget)] = compute_metrics(bas)
                if deg == degradations[0] and budget == budgets[0]:
                    paired_base, paired_policy = bas, pol

    _print_grid("Policy", policy_grid)
    if base_grid:
        _print_grid("Base", base_grid)
        test = mcnemar(paired_base, paired_policy)
        p = f"{test['p_value']:.4f}" if test["p_value"] is not None else "n/a (install scipy)"
        d0 = degradations[0]
        b0 = budgets[0]
        print(f"\n=== Base vs Policy (McNemar @ {d0}, budget {b0}) ===")
        print(f"  base-only-correct={test['b']} policy-only-correct={test['c']} p={p}")
        print(f"  accuracy delta: "
              f"{policy_grid[(d0, b0)]['accuracy'] - base_grid[(d0, b0)]['accuracy']:+.3f}")

    # Calibration on the first config.
    ref_env = InvestigationEnv(manifest_path=args.manifest, max_inspects=budgets[0], shuffle=False)
    cal = eval_over(model, processor, ref_env, test_idx, degradations[0], budgets[0] + 3)
    print(f"\n=== Calibration curve ({degradations[0]}, budget {budgets[0]}) ===")
    for row in calibration_curve(cal):
        print(f"  conf {row['range']} n={row['n']:>3} "
              f"mean_conf={row['mean_conf']:.2f} acc={row['accuracy']:.3f}")

    if args.evidence_slice:
        import json

        with open(args.evidence_slice) as f:
            slice_rows = [json.loads(line) for line in f if line.strip()]
        env = InvestigationEnv(manifest_path=args.manifest, max_inspects=budgets[0], shuffle=False)
        pol_hit = evidence_slice_hit_rate(model, processor, env, slice_rows, degradations[0], budgets[0] + 3)
        print(f"\n=== Evidence slice (inspect-hit) ===\n  policy: {pol_hit}")
        if base_model is not None:
            base_hit = evidence_slice_hit_rate(base_model, processor, env, slice_rows, degradations[0], budgets[0] + 3)
            print(f"  base:   {base_hit}")


if __name__ == "__main__":
    main()
