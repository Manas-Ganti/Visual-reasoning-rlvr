"""Recover readable trajectories from a GRPO SLURM log.

TRL prints a sample of each step's rollouts as a rich table, so the completions
are in the ``.out`` file — wrapped into a 20-character column and interleaved
with the prompt, which makes them invisible to grep and unreadable by eye. The
run-2 log held 701 complete episodes that went unread until this existed.

The completions parquet under ``checkpoints/<ds>/<run>/completions/`` is the
better source when you have it (``tools/watch_grpo.py`` reads it). This is for
when you have a log and not the checkpoint — a killed run, or a job whose
output directory was lost.

    python tools/traces_from_log.py logs/slurm/grpo-123.out            # all
    python tools/traces_from_log.py grpo-123.out --step 40 --limit 3   # a few
    python tools/traces_from_log.py grpo-123.out --stats               # summary
"""

from __future__ import annotations

import argparse
import re
import statistics as st

# rich draws each row as: │ │ <prompt 20> │ <completion 20> │ <reward 20> │ <adv 11> │ │
_ROW = re.compile(r"^│ │(.{20})│(.{20})│(.{20})│(.{11})│ │$")
_SEP = re.compile(r"^│ ├.*┤ │$")
_STEP = re.compile(r"Step (\d+) ")
_PFAKE = re.compile(r"P\(fake\)\s*=\s*([0-9]*\.?[0-9]+)", re.I)
_VERDICT = re.compile(r"VERDICT\s+(AI|REAL)", re.I)
_RECON = re.compile(r"RECONCILIATION:\s*(CONFIRMED|REFUTED|UNCLEAR)", re.I)


def parse_log(path: str) -> list[dict]:
    """Every rollout the log rendered, oldest first."""
    out: list[dict] = []
    cur: list[str] = []
    rew = adv = step = None
    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if "╭" in line and (m := _STEP.search(line)):
                step = int(m.group(1))
            if _SEP.match(line):
                if cur:
                    out.append({"step": step, "reward": rew, "advantage": adv,
                                "text": "".join(cur)})
                cur, rew, adv = [], None, None
                continue
            if m := _ROW.match(line):
                cur.append(m.group(2).rstrip())
                if rew is None and m.group(3).strip():
                    rew = m.group(3).strip()
                if adv is None and m.group(4).strip():
                    adv = m.group(4).strip()
    if cur:
        out.append({"step": step, "reward": rew, "advantage": adv, "text": "".join(cur)})
    return out


def beliefs(text: str) -> list[float]:
    vals = []
    for x in _PFAKE.findall(text):
        try:
            vals.append(float(x))
        except ValueError:      # "0.7." and friends — the policy writes prose
            pass
    return vals


def stats(eps: list[dict]) -> None:
    """The two numbers that explained run 2: how fast belief falls, and whether
    the CONFIRMED tag means what the reward assumes."""
    paths = [beliefs(e["text"]) for e in eps]
    paths = [p for p in paths if len(p) >= 4]
    print(f"{len(eps)} rollouts, {len(paths)} with a belief path\n")

    print("median P(fake) after N updates   (0.50 at the start)")
    for k in range(6):
        vals = [p[k] for p in paths if len(p) > k]
        if len(vals) < 20:
            break
        print(f"  {k + 1:>2}  n={len(vals):>4}   {st.median(vals):.3f}")

    up = down = flat = 0
    for e in eps:
        p = beliefs(e["text"])
        tags = _RECON.findall(e["text"])
        prev = 0.5
        for tag, val in zip(tags, p):
            d, prev = val - prev, val
            if tag.upper() != "CONFIRMED":
                continue
            up += d > 0.02
            down += d < -0.02
            flat += abs(d) <= 0.02
    tot = max(up + down + flat, 1)
    print(f"\nCONFIRMED followed by belief UP {up / tot:.1%} / DOWN {down / tot:.1%} "
          f"/ flat {flat / tot:.1%}  (n={tot})")
    print("  The reward reads CONFIRMED as 'artifact found, P(fake) rises'. A large")
    print("  DOWN share means the policy reads it as 'I checked this' instead.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--step", type=int, help="Only this optimizer step.")
    ap.add_argument("--limit", type=int, default=0, help="At most N episodes.")
    ap.add_argument("--stats", action="store_true", help="Summary instead of text.")
    args = ap.parse_args()

    eps = parse_log(args.log)
    if not eps:
        raise SystemExit(f"no rollout tables found in {args.log} — is it a GRPO log?")
    if args.step is not None:
        eps = [e for e in eps if e["step"] == args.step]
    if args.limit:
        eps = eps[: args.limit]

    if args.stats:
        stats(eps)
        return
    for e in eps:
        print(f"\n{'=' * 78}\nstep {e['step']}  reward={e['reward']}  "
              f"advantage={e['advantage']}\n{'=' * 78}")
        print(e["text"])


if __name__ == "__main__":
    main()
