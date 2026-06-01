"""Browser replay of logged reasoning traces (post-hoc, no model required).

Training/evaluation write per-episode traces to a JSONL log
(``logs/episodes.jsonl`` by default). This app loads that log and lets you step
through any episode turn by turn, showing how the VLM investigated the image:
the 3x3 grid, the zoom path (numbered in visit order), revealed metadata, the
per-turn THOUGHT/ACTION, reward, and the final verdict vs ground truth.

Run:
    python visualize.py                         # reads logs/episodes.jsonl
    python visualize.py --log path/to/log.jsonl --port 7861

The rendering functions below depend only on Pillow, so they are unit-testable
without launching Gradio (which is imported lazily inside build_app).
"""

from __future__ import annotations

import argparse
import os

from PIL import Image, ImageDraw, ImageFont

from src import utils
from src.trace_logger import TraceLogger

DEFAULT_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "episodes.jsonl")

GRID_COLOR = (190, 190, 190)
PAST_ZOOM_COLOR = (255, 196, 0)     # amber: previously inspected cells
CURRENT_ZOOM_COLOR = (255, 64, 64)  # red: cell inspected on the current turn


# --------------------------------------------------------------------------- #
# Rendering (Pillow only — no Gradio)
# --------------------------------------------------------------------------- #
def _load_font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _format_action(turn: dict) -> str:
    t = turn.get("action_type")
    if t == utils.ZOOM:
        return f"ZOOM {turn.get('cell')}"
    if t == utils.METADATA:
        return "METADATA"
    if t == utils.ANSWER:
        return f"ANSWER {turn.get('verdict')}"
    return "INVALID"


def _draw_label(draw: ImageDraw.ImageDraw, xy, text: str, font, color):
    """Draw a small number badge with a solid background for legibility."""
    x, y = xy
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    pad = 3
    draw.rectangle([x, y, x + (r - l) + 2 * pad, y + (b - t) + 2 * pad], fill=color)
    draw.text((x + pad - l, y + pad - t), text, fill=(0, 0, 0), font=font)


def render_turn(episode: dict, turn_no: int):
    """Render the image annotated up to ``turn_no`` (1-indexed) + a Markdown
    panel describing that turn. Returns ``(PIL.Image, markdown_str)``."""
    turns = episode.get("turns", [])
    n = len(turns)
    turn_no = max(1, min(turn_no, n)) if n else 1

    path = episode.get("image_path", "")
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        img = Image.new("RGB", (512, 512), (40, 40, 40))
        d = ImageDraw.Draw(img)
        d.text((20, 20), f"image not found:\n{path}", fill=(255, 120, 120),
               font=_load_font(16))
        return img, f"**Image not found:** `{path}`"

    w, h = img.size
    grid = episode.get("grid", 3)
    draw = ImageDraw.Draw(img)
    lw = max(2, w // 300)
    font = _load_font(max(14, w // 36))

    # 3x3 grid lines.
    for i in range(1, grid):
        x = round(w * i / grid)
        y = round(h * i / grid)
        draw.line([(x, 0), (x, h)], fill=GRID_COLOR, width=1)
        draw.line([(0, y), (w, y)], fill=GRID_COLOR, width=1)

    # Zoom path: number cells in visit order; bold the current turn's cell.
    zoom_order = 0
    for idx in range(turn_no):
        t = turns[idx]
        if t.get("action_type") != utils.ZOOM or not t.get("bbox"):
            continue
        zoom_order += 1
        is_current = idx == turn_no - 1
        color = CURRENT_ZOOM_COLOR if is_current else PAST_ZOOM_COLOR
        box = t["bbox"]
        draw.rectangle(box, outline=color, width=lw * (2 if is_current else 1))
        _draw_label(draw, (box[0] + 2, box[1] + 2), str(zoom_order), font, color)

    return img, _reasoning_markdown(episode, turn_no)


def _reasoning_markdown(episode: dict, turn_no: int) -> str:
    turns = episode.get("turns", [])
    n = len(turns)
    cur = turns[turn_no - 1] if n else {}

    # Render the thought as a multi-line blockquote so it stays prominent even
    # when the model emits several lines of reasoning.
    thought = (cur.get("thought") or "").strip()
    if thought:
        thought_md = "\n".join(f"> {ln}" for ln in thought.splitlines())
    else:
        thought_md = "> _(none)_"

    lines = [
        f"### `{episode.get('episode_id')}` · phase: `{episode.get('phase')}`",
        f"**Ground truth:** {episode.get('ground_truth')}  ·  "
        f"**global step:** {episode.get('global_step')}",
        f"**Turn {turn_no} / {n}**",
        "",
        "#### 💭 Thought",
        thought_md,
        "",
        f"**ACTION:** `{_format_action(cur)}`",
        f"**Reward:** {cur.get('reward', 0):+.3f}  ·  "
        f"**Cumulative:** {cur.get('cum_reward', 0):+.3f}",
    ]

    if cur.get("action_type") == utils.METADATA and cur.get("metadata_shown"):
        lines.append("\n**Metadata revealed:**")
        for k, v in cur["metadata_shown"].items():
            lines.append(f"- *{k.replace('_', ' ')}*: {v}")

    if cur.get("action_type") == utils.ANSWER:
        ok = episode.get("correct")
        mark = "✓ **correct**" if ok else "✗ **incorrect**"
        lines.append(
            f"\n> **Verdict: {cur.get('verdict')}** vs truth "
            f"**{episode.get('ground_truth')}** — {mark}  \n"
            f"> **Total reward: {episode.get('total_reward', 0):+.3f}**"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Reward distribution histogram (Pillow only)
# --------------------------------------------------------------------------- #
HIST_BG = (24, 24, 28)
HIST_BAR = (90, 160, 255)        # blue bars
HIST_AXIS = (150, 150, 150)
HIST_ZERO = (255, 120, 120)      # red marker at reward = 0


def render_reward_distribution(episodes: list[dict], phase: str = "all",
                               bins: int = 20, size=(640, 360)) -> Image.Image:
    """Histogram of per-episode ``total_reward`` for the given phase.

    Pillow-only so it stays unit-testable without Gradio/matplotlib, matching
    the other renderers in this module. Returns a PIL image.
    """
    rewards = [
        float(e.get("total_reward", 0.0))
        for e in episodes
        if phase == "all" or e.get("phase") == phase
    ]

    W, H = size
    img = Image.new("RGB", (W, H), HIST_BG)
    draw = ImageDraw.Draw(img)
    font = _load_font(14)
    title_font = _load_font(16)

    pad_l, pad_r, pad_t, pad_b = 52, 16, 40, 44
    plot_w = W - pad_l - pad_r
    plot_h = H - pad_t - pad_b

    title = f"Reward distribution · phase: {phase} · n={len(rewards)}"
    draw.text((pad_l, 12), title, fill=(230, 230, 230), font=title_font)

    if not rewards:
        draw.text((pad_l, pad_t + plot_h // 2), "no episodes for this phase",
                  fill=HIST_AXIS, font=font)
        return img

    lo, hi = min(rewards), max(rewards)
    if hi - lo < 1e-9:          # all identical → widen so the single bar shows
        lo, hi = lo - 0.5, hi + 0.5
    width = (hi - lo) / bins

    counts = [0] * bins
    for r in rewards:
        idx = min(int((r - lo) / width), bins - 1)
        counts[idx] += 1
    max_count = max(counts) or 1

    # Axes.
    x0, y0 = pad_l, pad_t + plot_h
    draw.line([(x0, pad_t), (x0, y0)], fill=HIST_AXIS, width=1)        # y
    draw.line([(x0, y0), (x0 + plot_w, y0)], fill=HIST_AXIS, width=1)  # x

    # Bars.
    bw = plot_w / bins
    for i, c in enumerate(counts):
        if c == 0:
            continue
        bh = (c / max_count) * plot_h
        bx0 = x0 + i * bw
        draw.rectangle([bx0 + 1, y0 - bh, bx0 + bw - 1, y0], fill=HIST_BAR)

    # Red marker at reward = 0 if it falls inside the range.
    if lo <= 0 <= hi:
        zx = x0 + (0 - lo) / (hi - lo) * plot_w
        draw.line([(zx, pad_t), (zx, y0)], fill=HIST_ZERO, width=1)

    # x-axis labels (min, 0?, max) and y-axis max count.
    draw.text((x0 - 4, y0 + 6), f"{lo:+.2f}", fill=HIST_AXIS, font=font)
    draw.text((x0 + plot_w - 32, y0 + 6), f"{hi:+.2f}", fill=HIST_AXIS, font=font)
    draw.text((6, pad_t - 6), str(max_count), fill=HIST_AXIS, font=font)
    mean = sum(rewards) / len(rewards)
    draw.text((x0 + plot_w / 2 - 40, y0 + 24), f"mean = {mean:+.3f}",
              fill=(230, 230, 230), font=font)
    return img


def episode_label(ep: dict, idx: int) -> str:
    mark = "✓" if ep.get("correct") else "✗"
    pred = ep.get("prediction") or "—"
    return (f"[{idx}] {ep.get('episode_id')} · {ep.get('phase')} · "
            f"pred {pred}/{ep.get('ground_truth')} {mark} · "
            f"R={ep.get('total_reward', 0):+.2f} · step {ep.get('global_step')}")


# --------------------------------------------------------------------------- #
# Gradio app (lazy import)
# --------------------------------------------------------------------------- #
def build_app(log_path: str):
    import gradio as gr

    state = {"episodes": TraceLogger.load(log_path)}

    def phases() -> list[str]:
        return ["all"] + sorted({e.get("phase", "?") for e in state["episodes"]})

    def choices(phase: str):
        out = []
        for i, ep in enumerate(state["episodes"]):
            if phase != "all" and ep.get("phase") != phase:
                continue
            out.append((episode_label(ep, i), i))
        return out

    def refresh(phase: str):
        chs = choices(phase)
        first = chs[0][1] if chs else None
        slider_up, img, md = select_episode(first)
        hist = render_reward_distribution(state["episodes"], phase)
        return gr.update(choices=chs, value=first), slider_up, img, md, hist

    def reload_and_refresh(phase: str):
        state["episodes"] = TraceLogger.load(log_path)
        return (gr.update(choices=phases()), *refresh("all"))

    def select_episode(ep_idx):
        if ep_idx is None:
            return gr.update(maximum=1, value=1), None, "_No episode selected._"
        ep = state["episodes"][ep_idx]
        n = max(ep.get("num_turns", len(ep.get("turns", []))), 1)
        img, md = render_turn(ep, 1)
        # Slider requires maximum > minimum; render_turn clamps overshoot to n.
        return gr.update(minimum=1, maximum=max(n, 2), value=1), img, md

    def show_turn(ep_idx, turn):
        if ep_idx is None:
            return None, "_No episode selected._"
        return render_turn(state["episodes"][ep_idx], int(turn))

    with gr.Blocks(title="VLM Reasoning Replay") as demo:
        gr.Markdown(
            "# VLM Active-Perception Reasoning Replay\n"
            f"Replaying traces from `{log_path}`. "
            "Pick an episode and scrub turns to see how the model investigated."
        )
        with gr.Row():
            phase_dd = gr.Dropdown(phases(), value="all", label="Phase", scale=2)
            reload_btn = gr.Button("Reload log", scale=1)
        episode_dd = gr.Dropdown(choices("all"), label="Episode")
        turn_slider = gr.Slider(1, 6, value=1, step=1, label="Turn")
        with gr.Row():
            image_out = gr.Image(type="pil", label="Image + zoom path", scale=2)
            md_out = gr.Markdown()
        with gr.Accordion("Reward distribution", open=True):
            hist_out = gr.Image(type="pil", label="Per-episode total reward",
                                show_label=False)

        refresh_out = [episode_dd, turn_slider, image_out, md_out, hist_out]
        phase_dd.change(refresh, [phase_dd], refresh_out)
        reload_btn.click(reload_and_refresh, [phase_dd], [phase_dd, *refresh_out])
        episode_dd.change(select_episode, [episode_dd], [turn_slider, image_out, md_out])
        turn_slider.change(show_turn, [episode_dd, turn_slider], [image_out, md_out])
        demo.load(refresh, [phase_dd], refresh_out)

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=DEFAULT_LOG, help="Episode trace JSONL.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    n = len(TraceLogger.load(args.log))
    print(f"Loaded {n} episode trace(s) from {args.log}")
    build_app(args.log).launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
