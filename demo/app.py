"""Gradio trajectory viewer: step through a logged investigation turn by turn.

Training / eval write one ``InvestigationEnv.get_trace`` dict per episode to a
JSONL log (``env/trace_logger.py``). This app replays them so you can watch the
agent's reasoning unfold — the low-res overview, the 4×4 grid, the inspect path
numbered in visit order, and for each turn the predict-then-verify block
(hypothesis committed *before* the reveal, reconciliation and belief update
*after*), ending in the verdict vs ground truth and the reward breakdown.

The predict-then-verify structure makes SFT-vs-RL contrast legible: a shortcutting
policy narrates after looking; a genuine one commits falsifiable predictions
first. That contrast is the point of the demo.

Rendering is Pillow-only (unit-testable); Gradio is imported lazily in
``build_app``.

    python demo/app.py                          # reads logs/grpo_episodes.jsonl
    python demo/app.py --log logs/eval.jsonl --port 7861
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFont

from env import grid
from env.trace_logger import TraceLogger

DEFAULT_LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "grpo_episodes.jsonl")

GRID_COLOR = (190, 190, 190)
PAST_COLOR = (255, 196, 0)      # amber: previously inspected
CURRENT_COLOR = (255, 64, 64)   # red: inspected this turn
RENDER_SIZE = 384


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _badge(draw, xy, text, font, color):
    x, y = xy
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    pad = 3
    draw.rectangle([x, y, x + (r - l) + 2 * pad, y + (b - t) + 2 * pad], fill=color)
    draw.text((x + pad - l, y + pad - t), text, fill=(0, 0, 0), font=font)


def render_turn(episode: dict, turn_no: int):
    """Annotated image + a Markdown panel for turn ``turn_no`` (1-indexed)."""
    turns = episode.get("turns", [])
    grid_size = episode.get("grid", 4)
    path = episode.get("image_path")
    try:
        base = Image.open(path).convert("RGB")
    except Exception:
        base = Image.new("RGB", (300, 300), (40, 40, 40))
    base = base.resize((RENDER_SIZE, RENDER_SIZE))
    draw = ImageDraw.Draw(base)
    font = _font(16)

    # Grid lines.
    for k in range(1, grid_size):
        draw.line([(k * RENDER_SIZE // grid_size, 0), (k * RENDER_SIZE // grid_size, RENDER_SIZE)], fill=GRID_COLOR)
        draw.line([(0, k * RENDER_SIZE // grid_size), (RENDER_SIZE, k * RENDER_SIZE // grid_size)], fill=GRID_COLOR)

    # Inspect badges up to this turn, in visit order.
    order = 0
    for i, turn in enumerate(turns[:turn_no], start=1):
        if turn.get("action_type") == "inspect" and turn.get("executed") and turn.get("cell"):
            order += 1
            row, col = grid.cell_rowcol(turn["cell"], grid_size)
            x = col * RENDER_SIZE // grid_size + 4
            y = row * RENDER_SIZE // grid_size + 4
            color = CURRENT_COLOR if i == turn_no else PAST_COLOR
            _badge(draw, (x, y), str(order), font, color)

    turn = turns[turn_no - 1] if 0 < turn_no <= len(turns) else {}
    md = _panel(episode, turn, turn_no, len(turns))
    return base, md


def _panel(episode, turn, turn_no, total) -> str:
    lines = [f"### Turn {turn_no} / {total}"]
    if turn.get("reconciliation") and turn["reconciliation"] != "unclear":
        lines.append(f"**Reconciliation (post-reveal):** {turn['reconciliation']}")
    if turn.get("p_fake") is not None:
        lines.append(f"**Belief:** P(fake) = {turn['p_fake']}")
    if turn.get("observation"):
        lines.append(f"**Observation:** {turn['observation']}")
    if turn.get("reasoning"):
        lines.append(f"**Reasoning:** {turn['reasoning']}")
    if turn.get("hypothesis"):
        lines.append(f"**Hypothesis (pre-reveal):** {turn['hypothesis']}")
    act = turn.get("action_type", "?")
    if act == "inspect":
        act = f"INSPECT {turn.get('cell')}" + ("" if turn.get("executed") else " (rejected)")
    elif act == "verdict":
        act = f"VERDICT {turn.get('verdict')} (confidence {turn.get('confidence')})"
    lines.append(f"**Action:** {act}")
    return "\n\n".join(lines)


def _header(episode) -> str:
    ok = "✅" if episode.get("correct") else "❌"
    bd = episode.get("reward_breakdown", {})
    bd_str = " · ".join(f"{k}={v:+.2f}" for k, v in bd.items()) if bd else "—"
    return (
        f"**{episode.get('episode_id','?')}** — truth **{episode.get('ground_truth')}**, "
        f"predicted **{episode.get('prediction')}** {ok} "
        f"(conf {episode.get('confidence')}, degradation `{episode.get('degradation')}`)  \n"
        f"reward **{episode.get('total_reward')}** · {bd_str}"
    )


def build_app(log_path: str):
    import gradio as gr

    episodes = TraceLogger.load(log_path)
    if not episodes:
        episodes = [{"episode_id": "no-data", "turns": [], "image_path": None,
                     "ground_truth": "?", "prediction": "?", "grid": 4}]

    def labels():
        return [f"{i}: {e.get('episode_id')} ({'✓' if e.get('correct') else '✗'})" for i, e in enumerate(episodes)]

    def show(ep_choice, turn_no):
        idx = int(ep_choice.split(":")[0]) if ep_choice else 0
        ep = episodes[idx]
        total = max(len(ep.get("turns", [])), 1)
        turn_no = max(1, min(turn_no, total))
        img, md = render_turn(ep, turn_no)
        return img, md, _header(ep), gr.update(maximum=total, value=turn_no)

    with gr.Blocks(title="visual-reasoning-rlvr — trajectory viewer") as app:
        gr.Markdown("# 🔎 Investigation trajectory viewer")
        with gr.Row():
            ep_dd = gr.Dropdown(labels(), value=labels()[0], label="Episode")
            turn_sl = gr.Slider(1, 8, value=1, step=1, label="Turn")
        header = gr.Markdown(_header(episodes[0]))
        with gr.Row():
            img = gr.Image(label="Overview + inspect path", height=RENDER_SIZE)
            panel = gr.Markdown()
        for control in (ep_dd, turn_sl):
            control.change(show, [ep_dd, turn_sl], [img, panel, header, turn_sl])
        app.load(show, [ep_dd, turn_sl], [img, panel, header, turn_sl])
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    print(f"Loading traces from {args.log}")
    build_app(args.log).launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
