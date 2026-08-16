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

Greedy rollouts keep eval deterministic; pass `--adapter` to evaluate a trained
checkpoint against the base model (the baseline runs with the adapter *disabled*
rather than as a second model, so comparing a 72B costs one replica, not two).

Two ways to spend a multi-GPU node:

    # vLLM, one tensor-parallel replica over 8 GPUs, episodes batched in lockstep
    python eval/harness.py --backend vllm --tensor-parallel-size 8 \\
        --batch-episodes 32 --adapter checkpoints/grpo-qwen2.5-vl-32b --compare-base

    # HF, one replica per GPU, test split strided across ranks and gathered
    torchrun --nproc_per_node 8 eval/harness.py --adapter checkpoints/grpo-qwen2.5-vl-32b \\
        --budgets 2,4 --degradations clean,jpeg,blur_downscale
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import InvestigationEnv
from env.reward import RewardConfig
from training import common
from training import vllm_backend


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
    """Paired significance test. Pairs on the manifest index rather than list
    position — with rank-sharded or batched rollouts, completion order is not
    submission order."""
    b_by_idx = {r["index"]: r for r in base}
    p_by_idx = {r["index"]: r for r in policy}
    shared = sorted(set(b_by_idx) & set(p_by_idx))
    b = sum(b_by_idx[i]["correct"] and not p_by_idx[i]["correct"] for i in shared)
    c = sum(not b_by_idx[i]["correct"] and p_by_idx[i]["correct"] for i in shared)
    if b + c == 0:
        return {"n_pairs": len(shared), "b": b, "c": c, "statistic": 0.0, "p_value": 1.0}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    try:
        from scipy.stats import chi2

        p = float(chi2.sf(stat, df=1))
    except ImportError:
        p = None
    return {"n_pairs": len(shared), "b": b, "c": c, "statistic": stat, "p_value": p}


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
def rollout_set(policy, env_factory, jobs, max_turns, batch_episodes=1, logger=None,
                phase="eval", max_new_tokens=640):
    """Run ``jobs`` (list of ``(index, degradation)``) and return every rank's
    results.

    Two execution shapes behind one call: a batching policy (vLLM) advances
    ``batch_episodes`` episodes in lockstep against a pool of envs; otherwise one
    env runs them sequentially. Either way the job list is strided across ranks
    first and the per-rank results are all-gathered afterwards, so every rank
    ends up holding the full set and rank 0 can report.
    """

    def on_episode(env, res):
        res["inspected_cells"] = sorted(env.state["inspected_cells"])
        if logger is not None:
            logger.log(env.get_trace(phase=phase), force=True)

    local_jobs = common.shard(jobs)
    if getattr(policy, "act_batch", None) is not None and batch_episodes > 1:
        envs = common.make_env_pool(env_factory, batch_episodes)
        results = common.run_episodes_batched(
            policy, envs, local_jobs, max_turns, sample=False, on_episode=on_episode,
            max_new_tokens=max_new_tokens,
        )
    else:
        env = env_factory()
        results = []
        for index, degradation in local_jobs:
            ep = common.run_episode(
                policy, env, index=index, degradation=degradation,
                max_turns=max_turns, sample=False, collect_tokens=False,
                max_new_tokens=max_new_tokens,
            )
            on_episode(env, ep)
            results.append(ep)
    return common.gather_lists(results)


def evidence_slice_hit_rate(results: list[dict], slice_rows: list[dict]) -> dict:
    """Fraction of episodes whose inspected cells intersect the human-verified
    artifact cells. Higher ⇒ the policy looks where the real artifacts are."""
    cells_by_index = {r["index"]: set(r["artifact_cells"]) for r in slice_rows if r["artifact_cells"]}
    scored = [r for r in results if r["index"] in cells_by_index]
    hits = sum(bool(set(r["inspected_cells"]) & cells_by_index[r["index"]]) for r in scored)
    n = len(scored)
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
    ap.add_argument("--model", default=common.DEFAULT_MODEL,
                    help="HF repo id or registry alias (7b | 32b | 72b | auto).")
    ap.add_argument("--adapter", default=None, help="Trained LoRA adapter (policy).")
    ap.add_argument("--dataset", default=common.DATASET,
                    help="Dataset namespace for manifest and episode logs.")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--budgets", default="4", help="Comma list of inspect budgets, e.g. 2,4")
    ap.add_argument("--degradations", default="clean", help="Comma list of degradation levels.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--compare-base", action="store_true",
                    help="Also evaluate the base model (adapter disabled) + McNemar.")
    ap.add_argument("--evidence-slice", default=None, help="Path to evidence_slice.jsonl.")
    ap.add_argument("--max-new-tokens", type=int, default=640,
                    help="Per-turn generation budget. The default backend value (320) "
                         "truncates verbose completions before their ACTION line, which "
                         "scores as an invalid turn and loses the episode.")
    ap.add_argument("--trace-log", default=None)
    vllm_backend.add_backend_args(ap)
    args = ap.parse_args()
    args.manifest = args.manifest or common.manifest_path(args.dataset)

    budgets = [int(x) for x in args.budgets.split(",")]
    degradations = [x.strip() for x in args.degradations.split(",")]
    args.max_inspects = max(budgets)  # image-per-prompt cap for the vLLM engine

    dist = common.init_distributed()
    common.record_run("eval", f"model={args.model} adapter={args.adapter} "
                      f"budgets={args.budgets} degradations={args.degradations}",
                      args.dataset)
    if dist.is_distributed and args.backend == "vllm":
        raise SystemExit(
            "Pick one: vLLM tensor parallelism OR torchrun rank sharding. Run the vLLM "
            "backend as a single process (--tensor-parallel-size N), or drop --backend vllm."
        )

    # Test indices are shared across all configs (same env, re-created per budget).
    probe = InvestigationEnv(manifest_path=args.manifest, max_inspects=max(budgets), shuffle=False)
    test_idx = [i for i, r in enumerate(probe.records) if r.get("split") == "test"]
    if args.limit:
        test_idx = test_idx[: args.limit]
    common.rank0_print(
        f"Evaluating {common.resolve_model(args.model)} on {len(test_idx)} held-out test images "
        f"(budgets={budgets}, degradations={degradations}, backend={args.backend}, "
        f"world_size={dist.world_size})."
    )

    logger = None
    if args.trace_log and dist.is_main:  # single writer: ranks would interleave records
        from env.trace_logger import TraceLogger

        logger = TraceLogger(args.trace_log)

    common.warn_if_tight(common.resolve_model(args.model), training=False)
    policy = vllm_backend.build_policy(args, adapter=args.adapter)
    compare_base = args.compare_base and bool(args.adapter)
    if args.compare_base and not args.adapter:
        common.rank0_print("--compare-base needs --adapter (the base IS the policy otherwise); skipping.")

    def env_factory(budget):
        return lambda: InvestigationEnv(
            manifest_path=args.manifest, max_inspects=budget,
            reward_config=RewardConfig(), shuffle=False,
        )

    policy_grid, base_grid = {}, {}
    paired_base, paired_policy = [], []
    reference = []  # policy rollouts of the first (degradation, budget) config
    for budget in budgets:
        factory = env_factory(budget)
        max_turns = budget + 3
        for deg in degradations:
            jobs = [(i, deg) for i in test_idx]
            pol = rollout_set(policy, factory, jobs, max_turns, args.batch_episodes, logger, "eval-policy",
                              max_new_tokens=args.max_new_tokens)
            policy_grid[(deg, budget)] = compute_metrics(pol)
            if deg == degradations[0] and budget == budgets[0]:
                # Calibration reads the same rollouts rather than re-running the
                # split — a full extra pass is expensive at 32B/72B.
                reference = pol
            if compare_base:
                with policy.without_adapter() as base_policy:
                    bas = rollout_set(base_policy, factory, jobs, max_turns, args.batch_episodes,
                                      logger, "eval-base",
                                      max_new_tokens=args.max_new_tokens)
                base_grid[(deg, budget)] = compute_metrics(bas)
                if deg == degradations[0] and budget == budgets[0]:
                    paired_base, paired_policy = bas, pol

    if args.evidence_slice:
        import json

        with open(args.evidence_slice) as f:
            slice_rows = [json.loads(line) for line in f if line.strip()]
        slice_jobs = [(r["index"], degradations[0]) for r in slice_rows if r.get("artifact_cells")]
        pol_slice = rollout_set(policy, env_factory(budgets[0]), slice_jobs, budgets[0] + 3,
                                args.batch_episodes, phase="eval-evidence")
        base_slice = None
        if compare_base:
            with policy.without_adapter() as base_policy:
                base_slice = rollout_set(base_policy, env_factory(budgets[0]), slice_jobs,
                                         budgets[0] + 3, args.batch_episodes, phase="eval-evidence-base")
    else:
        slice_rows = pol_slice = base_slice = None

    if not dist.is_main:
        common.cleanup_distributed()
        return

    _print_grid("Policy", policy_grid)
    if base_grid:
        _print_grid("Base", base_grid)
        test = mcnemar(paired_base, paired_policy)
        p = f"{test['p_value']:.4f}" if test["p_value"] is not None else "n/a (install scipy)"
        d0, b0 = degradations[0], budgets[0]
        print(f"\n=== Base vs Policy (McNemar @ {d0}, budget {b0}, n={test['n_pairs']}) ===")
        print(f"  base-only-correct={test['b']} policy-only-correct={test['c']} p={p}")
        print(f"  accuracy delta: "
              f"{policy_grid[(d0, b0)]['accuracy'] - base_grid[(d0, b0)]['accuracy']:+.3f}")

    print(f"\n=== Calibration curve ({degradations[0]}, budget {budgets[0]}) ===")
    for row in calibration_curve(reference):
        print(f"  conf {row['range']} n={row['n']:>3} "
              f"mean_conf={row['mean_conf']:.2f} acc={row['accuracy']:.3f}")

    if pol_slice is not None:
        print("\n=== Evidence slice (inspect-hit) ===")
        print(f"  policy: {evidence_slice_hit_rate(pol_slice, slice_rows)}")
        if base_slice is not None:
            print(f"  base:   {evidence_slice_hit_rate(base_slice, slice_rows)}")

    common.cleanup_distributed()


if __name__ == "__main__":
    main()
