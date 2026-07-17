"""Distill Stage-1 SFT traces from the base Qwen2-VL by teacher-forcing the label.

For each training image we tell the base model the correct answer up front and
ask it to *conduct* a convincing, well-formatted investigation that lands on it —
committing a hypothesis before each inspect and reconciling after. The model
still drives the real environment (it must choose cells and read the actual
reveals), so the demonstrations are grounded, not fabricated. We keep a trace
only if every turn parses into the required format AND the final verdict matches
the ground-truth label; everything else is discarded. The teacher hint is used
ONLY here — the saved trace stores just the assistant turns, so SFT never sees the
label leak.

Output rows (consumed by ``training/sft.py``):
    {"index", "degradation", "actions": [assistant turn text, ...]}

Requires transformers + a GPU (A100). Run before ``training/sft.py``:
    python data/build_sft_traces.py --limit 800 --out data/sft_traces.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import InvestigationEnv
from env.trajectory import parse_turn
from training import common

DEFAULT_OUT = os.path.join(common.REPO_ROOT, "data", "sft_traces.jsonl")

TEACHER_HINT = (
    "INSTRUCTOR NOTE (not part of the record): the ground-truth label for this "
    "image is {truth}. Produce a *genuine* investigation that a careful analyst "
    "would run to reach {truth}: inspect the cells most likely to be decisive, "
    "commit a testable HYPOTHESIS before each inspect, and RECONCILE honestly "
    "after each reveal. Keep P(fake) moving consistently with what you observe, "
    "and finish with ACTION: VERDICT {truth} confidence=<your calibrated value>. "
    "Follow the required labelled format exactly."
)


def distill_one(model, processor, env, index, degradation, max_turns, max_new_tokens=320) -> list[str] | None:
    import torch

    device = next(model.parameters()).device
    obs, info = env.reset(options={"index": index, "degradation": degradation})
    # Inject the teacher hint as an extra user message (never saved to the trace).
    env.state["messages"].append(
        {"role": "user", "content": [{"type": "text", "text": TEACHER_HINT.format(truth=info["ground_truth"])}]}
    )

    actions: list[str] = []
    with torch.no_grad():
        for _ in range(max_turns):
            obs = env._observation()  # includes the injected hint on turn 1
            inputs = common.build_inputs(processor, obs, device)
            out = model.generate(
                **inputs, do_sample=True, temperature=0.6, top_p=0.9,
                repetition_penalty=1.1, max_new_tokens=max_new_tokens,
            )
            new_tokens = out[0, inputs["input_ids"].shape[1]:]
            action = processor.decode(new_tokens, skip_special_tokens=True).strip()
            if parse_turn(action).action_type == "invalid":
                return None  # reject malformed demonstrations outright
            actions.append(action)
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

    # Keep only faithful, correct demonstrations.
    if info.get("correct"):
        return actions
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=common.DEFAULT_MODEL)
    ap.add_argument("--manifest", default=common.DEFAULT_MANIFEST)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--max-inspects", type=int, default=4)
    ap.add_argument("--degradation", default="clean")
    ap.add_argument("--limit", type=int, default=None, help="Cap #train images attempted.")
    args = ap.parse_args()

    env = InvestigationEnv(manifest_path=args.manifest, max_inspects=args.max_inspects, shuffle=False)
    max_turns = args.max_inspects + 3
    train_idx = [i for i, r in enumerate(env.records) if r.get("split", "train") == "train"]
    if args.limit:
        train_idx = train_idx[: args.limit]

    device = common.resolve_device("auto")
    dtype = common.resolve_dtype(device, use_bf16=True)
    model, processor = common.load_policy(args.model, adapter=None, device=device, dtype=dtype)
    print(f"Distilling with {args.model} on {device} over {len(train_idx)} images.")

    kept = 0
    with open(args.out, "w") as f:
        for n, idx in enumerate(train_idx):
            actions = distill_one(model, processor, env, idx, args.degradation, max_turns)
            if actions:
                f.write(json.dumps({"index": idx, "degradation": args.degradation, "actions": actions}) + "\n")
                kept += 1
            if (n + 1) % 25 == 0:
                print(f"  {n + 1}/{len(train_idx)} attempted, {kept} kept", flush=True)
    print(f"Wrote {kept} SFT traces to {args.out} (keep rate {kept / max(len(train_idx), 1):.1%}).")


if __name__ == "__main__":
    main()
