"""Benchmark the base VLM against the GRPO-trained policy on the held-out tail
of the manifest, using the same active-perception environment.

Reports accuracy / precision / recall / F1 (AI = positive class), mean
investigation steps, and a McNemar test on the paired correct/incorrect
outcomes so improvements over the base model can be judged for significance.

Self-contained: needs only transformers + torch (+ peft if loading an adapter).

Examples:
    python evaluate.py --limit 50                       # base model only
    python evaluate.py --adapter checkpoints/grpo-...   # base vs policy
"""

from __future__ import annotations

import argparse

import torch

from src.environment import ForgeryDetectionEnv
from src import utils
# Reuse the training split constants so eval never touches a trained-on sample.
from train_grpo import (
    HOLDOUT_FRACTION, MODEL_NAME, MANIFEST_PATH, MAX_TURNS,
    resolve_device, resolve_dtype,
)


@torch.no_grad()
def run_eval_episode(model, processor, env, index, max_turns):
    """Greedy multi-turn episode. Returns prediction bookkeeping for one image."""
    device = next(model.parameters()).device
    obs, info = env.reset(options={"index": index})

    for _ in range(max_turns):
        text = processor.apply_chat_template(
            obs["messages"], tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text], images=obs["images"], return_tensors="pt", padding=True
        ).to(device)
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        action_text = processor.decode(new_tokens, skip_special_tokens=True)
        obs, _, terminated, truncated, info = env.step(action_text)
        if terminated or truncated:
            break

    return {
        "id": info["id"],
        "truth": info["ground_truth"],
        "pred": info.get("predicted_verdict"),
        "correct": bool(info.get("correct")),
        "steps": info["steps"],
        "metadata_seen": info["metadata_seen"],
        "n_zooms": len(info["viewed_cells"]),
    }


def compute_metrics(results: list[dict]) -> dict:
    """Binary metrics with AI as the positive class."""
    tp = sum(r["pred"] == "AI" and r["truth"] == "AI" for r in results)
    fp = sum(r["pred"] == "AI" and r["truth"] == "REAL" for r in results)
    fn = sum(r["pred"] == "REAL" and r["truth"] == "AI" for r in results)
    n = len(results)

    acc = sum(r["correct"] for r in results) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    no_answer = sum(r["pred"] is None for r in results)

    return {
        "n": n,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_steps": sum(r["steps"] for r in results) / n if n else 0.0,
        "metadata_rate": sum(r["metadata_seen"] for r in results) / n if n else 0.0,
        "no_answer": no_answer,
    }


def mcnemar(base: list[dict], policy: list[dict]) -> dict:
    """McNemar test on paired correctness over the same items."""
    b = sum(bz["correct"] and not pz["correct"] for bz, pz in zip(base, policy))
    c = sum(not bz["correct"] and pz["correct"] for bz, pz in zip(base, policy))
    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0}

    stat = (abs(b - c) - 1) ** 2 / (b + c)  # with continuity correction
    try:
        from scipy.stats import chi2
        p = float(chi2.sf(stat, df=1))
    except ImportError:
        p = None
    return {"b": b, "c": c, "statistic": stat, "p_value": p}


def _load_model(model_name: str, adapter: str | None, device: str, dtype):
    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(model_name, padding_side="left")
    model = AutoModelForImageTextToText.from_pretrained(model_name, dtype=dtype)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.to(device)
    model.eval()
    return model, processor


def _print_metrics(title: str, m: dict):
    print(f"\n=== {title} (n={m['n']}) ===")
    print(f"  accuracy : {m['accuracy']:.3f}")
    print(f"  precision: {m['precision']:.3f}   recall: {m['recall']:.3f}   "
          f"f1: {m['f1']:.3f}")
    print(f"  avg steps: {m['avg_steps']:.2f}   metadata used: "
          f"{m['metadata_rate']:.0%}   no-answer: {m['no_answer']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--adapter", default=None,
                        help="Path to trained LoRA adapter for the RL policy.")
    parser.add_argument("--manifest", default=MANIFEST_PATH)
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of held-out samples (for quick runs).")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                        help="Compute device. 'auto' prefers cuda > mps > cpu.")
    parser.add_argument("--no-bf16", action="store_true",
                        help="Disable bf16 (required on MPS/CPU; uses fp16/fp32 instead).")
    parser.add_argument("--trace-log", default=None,
                        help="If set, write per-episode reasoning traces here "
                             "(browser replay with: python visualize.py).")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(device, use_bf16=device == "cuda" and not args.no_bf16)
    print(f"Device: {device} | load dtype: {dtype}")

    logger = None
    if args.trace_log:
        from src.trace_logger import TraceLogger
        logger = TraceLogger(args.trace_log)

    env = ForgeryDetectionEnv(
        manifest_path=args.manifest, max_steps=args.max_turns, shuffle=False
    )
    num_records = len(env.records)
    start = int(num_records * (1 - HOLDOUT_FRACTION))
    indices = list(range(start, num_records))
    if args.limit:
        indices = indices[: args.limit]
    print(f"Evaluating on {len(indices)} held-out samples "
          f"(indices {indices[0]}..{indices[-1]}).")

    def evaluate_over(model, phase):
        results = []
        for i in indices:
            results.append(run_eval_episode(model, processor, env, i, args.max_turns))
            if logger is not None:
                logger.log(env.get_trace(phase=phase), force=True)
        return results

    base_model, processor = _load_model(args.model, None, device, dtype)
    base_results = evaluate_over(base_model, phase="eval-base")
    base_metrics = compute_metrics(base_results)
    _print_metrics("Base VLM", base_metrics)

    if args.adapter:
        policy_model, _ = _load_model(args.model, args.adapter, device, dtype)
        policy_results = evaluate_over(policy_model, phase="eval-policy")
        policy_metrics = compute_metrics(policy_results)
        _print_metrics("RL Policy", policy_metrics)

        test = mcnemar(base_results, policy_results)
        p_str = f"{test['p_value']:.4f}" if test["p_value"] is not None else "n/a (install scipy)"
        print("\n=== Base vs RL Policy (McNemar) ===")
        print(f"  base-only-correct: {test['b']}   policy-only-correct: {test['c']}")
        print(f"  statistic: {test['statistic']:.3f}   p-value: {p_str}")
        print(f"  accuracy delta: {policy_metrics['accuracy'] - base_metrics['accuracy']:+.3f}")


if __name__ == "__main__":
    main()
