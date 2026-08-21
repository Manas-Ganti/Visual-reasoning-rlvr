"""Will GRPO actually learn anything? Measure group variance before you launch it.

GRPO has no critic. It scores a *group* of ``G`` rollouts on the same image and
uses each rollout's advantage relative to that group's mean. If every rollout in
a group lands on the same verdict, the dominant term of the reward
(``w_correct``, ±1.0) is constant inside the group and contributes **nothing** to
the advantage — no matter how wrong the whole group is.

That is not a harmless no-op. The process terms (belief coherence, verdict
consistency, prediction tracking — 0.70 of the weight put together) still vary,
so the gradient does not vanish; it *redirects*. A run whose groups are unanimous
spends its entire budget teaching the policy to write internally-coherent
investigations, decoupled from whether they reach the right answer. On the face
substrate every group would have been unanimously wrong (AI-class recall 0.00),
and 24 hours of GRPO would have optimised exactly that pathology.
See ``results/faces_negative_result.md``.

The ceiling probe (``tools/ceiling_probe.py``) asks whether the *substrate* is
usable. This asks the next question: whether the *policy* sits in the band where
GRPO has signal. Run it on the SFT checkpoint — the actual GRPO starting point —
not on the base model.

    VRR_DATASET=genimage python tools/group_variance_probe.py \\
        --backend vllm --tensor-parallel-size 8 --batch-episodes 32 \\
        --adapter checkpoints/genimage/sft-qwen2.5-vl-32b --images 32 -G 8

Reading the result — ``usable_groups`` is the number that matters:

  >=0.40   healthy. Most groups contain both a right and a wrong rollout, so the
           outcome term drives the advantage. Launch GRPO.
  0.15-0.40  thin but workable. Expect slow, noisy progress; consider moving the
           difficulty knobs (budget, degradation) toward a ~50% pass rate first.
  <0.15    do not launch. Check ``all_wrong`` vs ``all_correct``: all-wrong means
           the task is too hard for this policy (ease the degradation, raise the
           budget, or revisit the substrate); all-correct means it is too easy
           (tighten them). Either way GRPO would train on process only.
"""

from __future__ import annotations

import argparse
import collections
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import InvestigationEnv  # noqa: E402
from env.reward import RewardConfig  # noqa: E402
from training import common, vllm_backend  # noqa: E402


def outcome_component(correct: bool, cfg: RewardConfig) -> float:
    """The part of the reward GRPO is *supposed* to be driven by: ±w_correct.

    Isolated from the total so the two variances can be compared directly — a
    group with total-reward spread but zero outcome spread is the failure this
    tool exists to catch.
    """
    return cfg.w_correct if correct else -cfg.w_correct


def summarize(groups: dict[int, list[dict]], cfg: RewardConfig) -> dict:
    """Per-group unanimity + the variance split between outcome and process."""
    n_groups = len(groups)
    all_correct = all_wrong = 0
    outcome_sd, total_sd = [], []
    pass_rates = []

    for rows in groups.values():
        correct = [bool(r["correct"]) for r in rows]
        pass_rates.append(sum(correct) / len(correct))
        if all(correct):
            all_correct += 1
        elif not any(correct):
            all_wrong += 1
        if len(rows) > 1:
            outcome_sd.append(statistics.pstdev([outcome_component(c, cfg) for c in correct]))
            total_sd.append(statistics.pstdev([r["episode_reward"] for r in rows]))

    unanimous = all_correct + all_wrong
    return {
        "groups": n_groups,
        "rollouts": sum(len(v) for v in groups.values()),
        "mean_pass_rate": sum(pass_rates) / max(n_groups, 1),
        "all_correct": all_correct / max(n_groups, 1),
        "all_wrong": all_wrong / max(n_groups, 1),
        "unanimous": unanimous / max(n_groups, 1),
        "usable_groups": (n_groups - unanimous) / max(n_groups, 1),
        "mean_outcome_sd": sum(outcome_sd) / max(len(outcome_sd), 1),
        "mean_total_sd": sum(total_sd) / max(len(total_sd), 1),
    }


def verdict_line(s: dict) -> str:
    u = s["usable_groups"]
    if u >= 0.40:
        return "HEALTHY — the outcome term drives the advantage. Launch GRPO."
    if u >= 0.15:
        return ("THIN — GRPO will be slow and noisy. Consider tuning budget / "
                "degradation toward a ~50% pass rate first.")
    skew = "too hard (all-wrong)" if s["all_wrong"] >= s["all_correct"] else "too easy (all-correct)"
    return (f"DO NOT LAUNCH — {skew}. Nearly every group is unanimous, so GRPO "
            f"would optimise the process terms alone.")


def main():
    ap = argparse.ArgumentParser()
    vllm_backend.add_backend_args(ap)
    ap.add_argument("--model", default=common.DEFAULT_MODEL,
                    help="HF repo id or registry alias (7b | 32b | 72b | auto).")
    ap.add_argument("--adapter", default=None,
                    help="LoRA to probe. Use the SFT checkpoint — that is where GRPO starts.")
    ap.add_argument("--dataset", default=common.DATASET)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--split", default="train",
                    help="GRPO trains on train, so that is what this measures by default.")
    ap.add_argument("--images", type=int, default=32, help="Number of distinct images (= groups).")
    ap.add_argument("-G", "--num-generations", type=int, default=8,
                    help="Group size. Match your GRPO --num-generations.")
    ap.add_argument("--max-inspects", type=int, default=4)
    ap.add_argument("--overview-long-edge", type=int, default=140,
                    help="Overview resolution — how much the low-res view is blurred. "
                         "The zoom factor INSPECT buys is native_size/this, so ~native/7 "
                         "is the target: 140 suits 1024px images, 70 suits 512px. Too high "
                         "and the answer is readable from the overview; too low and the "
                         "agent cannot tell where to look. Find it with "
                         "tools/ceiling_probe.py --condition overview.")
    ap.add_argument("--degradation", default="clean")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="Match your GRPO rollout temperature — variance depends on it.")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=640,
                    help="Too low truncates the ACTION line and scores real answers as "
                         "failures — the bug that hid the face-substrate collapse.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.manifest = args.manifest or common.manifest_path(args.dataset)

    dist = common.init_distributed()
    if dist.is_distributed and args.backend == "vllm":
        raise SystemExit(
            "Pick one: vLLM tensor parallelism OR torchrun rank sharding. Run the vLLM "
            "backend as a single process (--tensor-parallel-size N), or drop --backend vllm."
        )

    def env_factory():
        return InvestigationEnv(
            manifest_path=args.manifest, max_inspects=args.max_inspects,
            reward_config=RewardConfig(), shuffle=False, dataset=args.dataset,
            overview_long_edge=args.overview_long_edge,
        )

    probe = env_factory()
    idx = [i for i, r in enumerate(probe.records) if r.get("split", "train") == args.split]
    if not idx:
        raise SystemExit(f"No records in split {args.split!r} of {args.manifest}")
    idx = idx[: args.images]

    # G rollouts per image. All G share the image's manifest index, so the groups
    # survive rank-sharding and gathering without extra bookkeeping.
    jobs = [(i, args.degradation) for i in idx for _ in range(args.num_generations)]
    max_turns = args.max_inspects + 3

    common.record_run("group_variance", f"model={args.model} adapter={args.adapter} "
                      f"images={len(idx)} G={args.num_generations}", args.dataset)
    common.warn_if_tight(common.resolve_model(args.model), training=False)
    policy = vllm_backend.build_policy(args, adapter=args.adapter)
    common.rank0_print(
        f"Probing group variance: {len(idx)} images x G={args.num_generations} = "
        f"{len(jobs)} rollouts (adapter={args.adapter or 'none'}, "
        f"temperature={args.temperature}, world_size={dist.world_size})."
    )

    local_jobs = common.shard(jobs)
    if getattr(policy, "act_batch", None) is not None and args.batch_episodes > 1:
        envs = common.make_env_pool(env_factory, args.batch_episodes)
        results = common.run_episodes_batched(
            policy, envs, local_jobs, max_turns, sample=True, temperature=args.temperature,
            top_p=args.top_p, max_new_tokens=args.max_new_tokens,
        )
    else:
        env = env_factory()
        results = []
        for n, (index, degradation) in enumerate(local_jobs):
            results.append(common.run_episode(
                policy, env, index=index, degradation=degradation, max_turns=max_turns,
                sample=True, temperature=args.temperature, top_p=args.top_p,
                max_new_tokens=args.max_new_tokens, collect_tokens=False,
            ))
            if (n + 1) % 25 == 0:
                common.rank0_print(f"  rank0: {n + 1}/{len(local_jobs)} rollouts", flush=True)

    results = common.gather_lists(results)
    if not dist.is_main:
        common.cleanup_distributed()
        return

    groups: dict[int, list[dict]] = collections.defaultdict(list)
    for r in results:
        groups[r["index"]].append(r)

    cfg = RewardConfig()
    s = summarize(groups, cfg)

    print(f"\n=== Group variance (G={args.num_generations}, {s['groups']} groups, "
          f"{s['rollouts']} rollouts) ===")
    print(f"  mean pass rate         {s['mean_pass_rate']:.3f}   (want ~0.2-0.8)")
    print(f"  groups all-correct     {s['all_correct']:.3f}")
    print(f"  groups all-wrong       {s['all_wrong']:.3f}")
    print(f"  groups unanimous       {s['unanimous']:.3f}   <- dead outcome signal")
    print(f"  groups usable          {s['usable_groups']:.3f}   <- THE NUMBER")
    print(f"\n  mean within-group sd, outcome term   {s['mean_outcome_sd']:.4f}")
    print(f"  mean within-group sd, total reward   {s['mean_total_sd']:.4f}")
    if s["mean_total_sd"] > 1e-9:
        share = s["mean_outcome_sd"] / s["mean_total_sd"]
        print(f"  outcome share of reward spread       {share:.3f}   "
              f"(near 0 = GRPO trains on process only)")

    answered = sum(r["answered"] for r in results)
    print(f"\n  answer rate {answered / max(len(results), 1):.3f} "
          f"(low = format failures or truncation, not difficulty — raise --max-new-tokens)")
    print(f"\n  {verdict_line(s)}")

    common.cleanup_distributed()


if __name__ == "__main__":
    main()
