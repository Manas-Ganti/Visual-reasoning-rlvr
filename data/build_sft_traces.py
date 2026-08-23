"""Distill Stage-1 SFT traces from the base VLM by teacher-forcing the label.

For each training image we tell the base model the correct answer up front and
ask it to *conduct* a convincing, well-formatted investigation that lands on it —
committing a hypothesis before each inspect and reconciling after. The model
still drives the real environment (it must choose cells and read the actual
reveals), so the demonstrations are grounded, not fabricated. We keep a trace
only if every turn parses into the required format AND the final verdict matches
the ground-truth label; everything else is discarded. The teacher hint is used
ONLY here — the saved trace stores just the assistant turns, so SFT never sees the
label leak.

Trace quality is what Stage 1 inherits, so this is the one place worth spending a
*bigger* teacher than the student: distilling with ``--model 72b`` into a 32B
student is a legitimate (and cheap, one-off) upgrade.

Output rows (consumed by ``training/sft.py``):
    {"index", "degradation", "actions": [assistant turn text, ...]}

Two ways to spend a multi-GPU node:

    # vLLM, one tensor-parallel replica, 32 episodes advancing in lockstep
    python data/build_sft_traces.py --backend vllm --tensor-parallel-size 8 \\
        --model 72b --batch-episodes 32 --limit 1200

    # HF, one replica per GPU, images strided across ranks
    torchrun --nproc_per_node 8 data/build_sft_traces.py --model 32b --limit 1200
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import InvestigationEnv
from training import common
from training import vllm_backend

# Paths are dataset-namespaced (see training/common.py: VRR_DATASET).

TEACHER_HINT = (
    "INSTRUCTOR NOTE (not part of the record): the ground-truth label for this "
    "image is {truth}. Produce a *genuine* investigation that a careful analyst "
    "would run to reach {truth}: inspect the cells most likely to be decisive, "
    "commit a testable HYPOTHESIS before each inspect, and RECONCILE honestly "
    "after each reveal. Keep P(fake) moving consistently with what you observe, "
    "and finish with ACTION: VERDICT {truth} confidence=<your calibrated value>. "
    "Follow the required labelled format exactly."
)


def inject_teacher_hint(env) -> None:
    """Append the label hint as an extra user message. Never saved to the trace —
    we persist only the assistant turns."""
    truth = env._ground_truth()
    env.state["messages"].append(
        {"role": "user", "content": [{"type": "text", "text": TEACHER_HINT.format(truth=truth)}]}
    )


def harvest(env, result: dict) -> None:
    """Pull the assistant turns + a format verdict off the finished env.

    Recovered from the conversation rather than tracked alongside it, so the
    sequential and batched runners need no extra bookkeeping.
    """
    result["actions"] = [
        m["content"][0]["text"] for m in env.state["messages"] if m["role"] == "assistant"
    ]
    result["well_formed"] = all(t["action_type"] != "invalid" for t in env.state["trace"])
    result["degradation"] = env.state["degradation"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=common.DEFAULT_MODEL,
                    help="Teacher: HF repo id or alias (7b | 32b | 72b | auto). May be larger "
                         "than the student — distillation is a one-off cost.")
    ap.add_argument("--dataset", default=common.DATASET,
                    help="Dataset namespace for manifest and trace output.")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-inspects", type=int, default=4)
    ap.add_argument("--overview-long-edge", type=int, default=140,
                    help="Overview resolution — how much the low-res view is blurred. "
                         "The zoom factor INSPECT buys is native_size/this, so ~native/7 "
                         "is the target: 140 suits 1024px images, 70 suits 512px. Too high "
                         "and the answer is readable from the overview; too low and the "
                         "agent cannot tell where to look. Find it with "
                         "tools/ceiling_probe.py --condition overview.")
    ap.add_argument("--degradation", default="clean")
    ap.add_argument("--limit", type=int, default=None, help="Cap #train images attempted.")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0,
                    help="Engine seed. Episodes are SAMPLED, so a second pass at a "
                         "different seed yields different trajectories over the same "
                         "images — the cheapest way to raise trace count when the keep "
                         "rate is low. Combine with --append.")
    ap.add_argument("--dump-failures", default=None, metavar="PATH",
                    help="Write rejected episodes here (capped at --dump-limit) with the "
                         "reason and their raw turns. A keep rate is a symptom; the "
                         "malformed turns themselves say whether the model ran past "
                         "--max-new-tokens, dropped a field, or invented an action.")
    ap.add_argument("--dump-limit", type=int, default=40)
    ap.add_argument("--append", action="store_true",
                    help="Append to --out instead of overwriting, to accumulate passes. "
                         "Repeated images with different trajectories are fine for SFT "
                         "and add demonstration diversity.")
    ap.add_argument("--max-new-tokens", type=int, default=320)
    vllm_backend.add_backend_args(ap)
    args = ap.parse_args()
    args.manifest = args.manifest or common.manifest_path(args.dataset)
    args.out = args.out or common.traces_path(args.dataset)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    dist = common.init_distributed()
    if dist.is_distributed and args.backend == "vllm":
        raise SystemExit(
            "Pick one: vLLM tensor parallelism OR torchrun rank sharding. Run the vLLM "
            "backend as a single process (--tensor-parallel-size N), or drop --backend vllm."
        )

    def env_factory():
        return InvestigationEnv(
            manifest_path=args.manifest, max_inspects=args.max_inspects, shuffle=False,
            dataset=args.dataset, overview_long_edge=args.overview_long_edge,
        )

    probe = env_factory()
    max_turns = args.max_inspects + 3
    train_idx = [i for i, r in enumerate(probe.records) if r.get("split", "train") == "train"]
    if args.limit:
        train_idx = train_idx[: args.limit]
    jobs = [(i, args.degradation) for i in train_idx]

    common.record_run("distill", f"model={args.model} out={args.out} "
                      f"degradation={args.degradation}", args.dataset)
    common.warn_if_tight(common.resolve_model(args.model), training=False)
    policy = vllm_backend.build_policy(args)
    common.rank0_print(
        f"Distilling with {common.resolve_model(args.model)} over {len(jobs)} images "
        f"(backend={args.backend}, world_size={dist.world_size})."
    )

    local_jobs = common.shard(jobs)
    if getattr(policy, "act_batch", None) is not None and args.batch_episodes > 1:
        envs = common.make_env_pool(env_factory, args.batch_episodes)
        results = common.run_episodes_batched(
            policy, envs, local_jobs, max_turns, sample=True, temperature=args.temperature,
            max_new_tokens=args.max_new_tokens, on_reset=inject_teacher_hint, on_episode=harvest,
        )
    else:
        env = env_factory()
        results = []
        for n, (index, degradation) in enumerate(local_jobs):
            ep = common.run_episode(
                policy, env, index=index, degradation=degradation, max_turns=max_turns,
                sample=True, temperature=args.temperature, max_new_tokens=args.max_new_tokens,
                collect_tokens=False, on_reset=inject_teacher_hint,
            )
            harvest(env, ep)
            results.append(ep)
            if (n + 1) % 25 == 0:
                common.rank0_print(f"  rank0: {n + 1}/{len(local_jobs)} attempted", flush=True)

    # Keep only faithful, correct, well-formed demonstrations.
    rows = [
        {"index": r["index"], "degradation": r["degradation"], "actions": r["actions"]}
        for r in results
        if r["correct"] and r["well_formed"] and r["actions"]
    ]
    rows = common.gather_lists(rows)
    # A bare keep rate says a substrate is expensive to distil but not why. The
    # two failures want opposite fixes: bad FORMAT calls for a bigger teacher or
    # fewer turns to get right, while a wrong VERDICT under a revealed label means
    # the teacher hint is not landing.
    flags = common.gather_lists([
        {"c": bool(r["correct"]), "w": bool(r["well_formed"]), "a": bool(r["actions"])}
        for r in results
    ])
    fails = common.gather_lists([
        {"index": r["index"],
         "reason": "malformed" if not r["well_formed"]
                   else ("wrong_verdict" if not r["correct"] else "empty"),
         "n_turns": len(r["actions"]),
         "last_turn_words": len(r["actions"][-1].split()) if r["actions"] else 0,
         "actions": r["actions"]}
        for r in results
        if not (r["correct"] and r["well_formed"] and r["actions"])
    ][: max(args.dump_limit, 0)] if args.dump_failures else [])

    if dist.is_main:
        rows.sort(key=lambda r: r["index"])
        mode = "a" if args.append else "w"
        with open(args.out, mode) as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        total = sum(1 for _ in open(args.out))
        print(f"Wrote {len(rows)} SFT traces to {args.out} "
              f"(keep rate {len(rows) / max(len(jobs), 1):.1%}"
              f"{f'; {total} total after append' if args.append else ''}).")

        n = max(len(flags), 1)
        bad_fmt = sum(1 for x in flags if not x["w"])
        bad_verdict = sum(1 for x in flags if x["w"] and not x["c"])
        empty = sum(1 for x in flags if x["w"] and x["c"] and not x["a"])
        print(f"  dropped: {bad_fmt} malformed ({bad_fmt / n:.1%}), "
              f"{bad_verdict} wrong verdict ({bad_verdict / n:.1%}), {empty} empty")
        if bad_fmt > bad_verdict:
            print("  -> FORMAT is the bottleneck: try --model 72b (a bigger teacher is "
                  "worth it here) or fewer --max-inspects, since every turn must parse.")
        elif bad_verdict:
            print("  -> VERDICT is the bottleneck: the teacher hint is not landing, so a "
                  "bigger teacher will not obviously help. Check TEACHER_HINT and whether "
                  "the substrate is simply hard at this overview resolution.")
        if args.dump_failures:
            bad = [r for r in fails][: args.dump_limit]
            with open(args.dump_failures, "w") as f:
                for r in bad:
                    f.write(json.dumps(r) + "\n")
            print(f"  wrote {len(bad)} rejected episode(s) to {args.dump_failures}")
        if len(rows) / max(len(jobs), 1) < 0.30:
            print("  -> low yield: rerun with --seed <n> --append to accumulate passes; "
                  "episodes are sampled, so each pass explores different trajectories.")
    common.cleanup_distributed()


if __name__ == "__main__":
    main()
