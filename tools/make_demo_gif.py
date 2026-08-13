"""Render one episode into an animated GIF for the README.

Shows the actual mechanic the environment implements: the agent starts on a
low-resolution OVERVIEW where artifacts are not resolvable, and each INSPECT
sharpens one 4x4 cell. Left panel is what the model can see at that turn; right
panel is the structured block it emitted.

    python tools/make_demo_gif.py                       # auto-pick a good episode
    python tools/make_demo_gif.py --episode-id distill-00042
    python tools/make_demo_gif.py --list                # show candidates

Input is the demo episode log (see tools/traces_to_demo.py).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import grid  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

W = 1360
IMG = 512                      # image panel edge
PAD = 24
TEXT_X = IMG + 2 * PAD + 10
TEXT_W = W - TEXT_X - PAD      # pixels available for wrapped prose
LEAD = 21                      # line leading for body text

BG = (18, 20, 24)
FG = (232, 234, 238)
DIM = (150, 156, 166)
ACCENT = (255, 186, 66)        # current inspect
PAST = (86, 168, 255)          # earlier inspects
GRID_LINE = (255, 255, 255, 70)
OK, BAD = (94, 214, 130), (240, 110, 110)

_FONTS = [
    "/System/Library/Fonts/Supplemental/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
_BOLDS = [
    "/System/Library/Fonts/Supplemental/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def font(size: int, bold: bool = False):
    for path in (_BOLDS if bold else _FONTS):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def visible_image(base: Image.Image, revealed: list[int], grid_n: int) -> Image.Image:
    """Overview with every already-inspected cell restored to full resolution."""
    view = grid.make_overview(base, long_edge=140, restore_to=base.size[0])
    for cell in revealed:
        box = grid.cell_bbox(base.size[0], base.size[1], cell, grid_n)
        view.paste(base.crop(box), box[:2])
    return view


def draw_panel(img: Image.Image, turns, upto: int, grid_n: int) -> Image.Image:
    """Grid lines + numbered badges for inspects up to (and including) `upto`."""
    img = img.convert("RGB").resize((IMG, IMG), Image.LANCZOS)
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    step = IMG // grid_n
    for k in range(1, grid_n):
        d.line([(k * step, 0), (k * step, IMG)], fill=GRID_LINE, width=1)
        d.line([(0, k * step), (IMG, k * step)], fill=GRID_LINE, width=1)
    img = Image.alpha_composite(img.convert("RGBA"), over).convert("RGB")

    d = ImageDraw.Draw(img)
    f = font(19, bold=True)
    order = 0
    for i, t in enumerate(turns[:upto], start=1):
        if t.get("action_type") != "inspect" or not t.get("cell"):
            continue
        order += 1
        row, col = grid.cell_rowcol(t["cell"], grid_n)
        x, y = col * step + 6, row * step + 6
        color = ACCENT if i == upto else PAST
        d.rectangle([x, y, x + 26, y + 24], fill=color)
        d.text((x + 8, y + 3), str(order), font=f, fill=(15, 15, 18))
    return img


def wrap_px(text: str, f, max_px: int) -> list[str]:
    """Greedy wrap on measured pixel width — never truncates."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if f.getbbox(trial) and f.getbbox(trial)[2] > max_px and cur:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines or [""]


def blocks(ep, t) -> list[tuple]:
    """The panel as (kind, caption, payload) blocks, so height can be measured
    before anything is drawn. Every field is rendered in full."""
    out = [("title", None, f"Turn ?"), ("sub", None, f"ground truth: {ep['ground_truth']}")]
    if t.get("reconciliation") and t["reconciliation"] != "unclear":
        out.append(("recon", "RECONCILIATION", t["reconciliation"].upper()))
    if t.get("p_fake") is not None:
        out.append(("belief", "BELIEF", float(t["p_fake"])))
    for label in ("observation", "reasoning", "hypothesis"):
        if t.get(label):
            cap = {"hypothesis": "HYPOTHESIS (pre-reveal)"}.get(label, label.upper())
            out.append(("prose", cap, t[label]))
    act = t.get("action_type", "?")
    if act == "inspect":
        out.append(("action", "ACTION", (f"INSPECT {t.get('cell')}", ACCENT)))
    elif act == "verdict":
        col = OK if t.get("verdict") == ep["ground_truth"] else BAD
        out.append(("action", "ACTION",
                    (f"VERDICT {t.get('verdict')}  confidence={t.get('confidence')}", col)))
    else:
        out.append(("action", "ACTION", (act, DIM)))
    return out


def panel_height(bl, body) -> int:
    """Pixel height the text column needs, measured not guessed."""
    y = PAD + 20
    for kind, _cap, payload in bl:
        if kind == "title":
            y += 38
        elif kind == "sub":
            y += 32
        elif kind == "recon":
            y += 18 + 26
        elif kind == "belief":
            y += 18 + 28
        elif kind == "prose":
            y += 18 + LEAD * len(wrap_px(payload, body, TEXT_W)) + 10
        elif kind == "action":
            y += 20 + 18 + 26
    return y + PAD


def frame(ep, turn_no: int, base: Image.Image, grid_n: int, height: int) -> Image.Image:
    turns = ep["turns"]
    t = turns[turn_no - 1]
    revealed = [x["cell"] for x in turns[:turn_no]
                if x.get("action_type") == "inspect" and x.get("cell")]
    # The cell chosen THIS turn is only revealed after the action executes, so
    # the visible image lags the badge by one cell.
    view = visible_image(base, revealed[:-1] if t.get("action_type") == "inspect" else revealed, grid_n)

    canvas = Image.new("RGB", (W, height), BG)
    canvas.paste(draw_panel(view, turns, turn_no, grid_n), (PAD, PAD + 26))
    d = ImageDraw.Draw(canvas)
    d.text((PAD, 14), "visual-reasoning-rlvr — investigative episode",
           font=font(17, bold=True), fill=DIM)

    small, body = font(14, bold=True), font(15)
    x, y = TEXT_X, PAD + 20

    for kind, cap, payload in blocks(ep, t):
        if kind == "title":
            d.text((x, y), f"Turn {turn_no} / {len(turns)}", font=font(24, bold=True), fill=FG)
            y += 38
        elif kind == "sub":
            d.text((x, y), payload, font=font(15), fill=DIM)
            y += 32
        elif kind == "recon":
            col = OK if payload == "CONFIRMED" else BAD
            d.text((x, y), cap, font=small, fill=DIM); y += 18
            d.text((x, y), payload, font=font(16, bold=True), fill=col); y += 26
        elif kind == "belief":
            d.text((x, y), cap, font=small, fill=DIM); y += 18
            d.text((x, y), f"P(fake) = {payload:.2f}", font=font(16, bold=True), fill=FG)
            bx, bw = x + 155, 300
            d.rectangle([bx, y + 4, bx + bw, y + 14], fill=(48, 52, 60))
            d.rectangle([bx, y + 4, bx + int(bw * payload), y + 14], fill=ACCENT)
            y += 28
        elif kind == "prose":
            d.text((x, y), cap, font=small, fill=DIM); y += 18
            for ln in wrap_px(payload, body, TEXT_W):
                d.text((x, y), ln, font=body, fill=FG); y += LEAD
            y += 10
        elif kind == "action":
            text, col = payload
            y += 20
            d.text((x, y), cap, font=small, fill=DIM)
            d.text((x, y + 18), text, font=font(19, bold=True), fill=col)
            y += 44
    return canvas


def _clean_field(s: str) -> bool:
    """Did this field parse onto a sentence start?

    ``env/trajectory.py`` anchors on the FIRST occurrence of a label token, so a
    completion whose prose contains the word "observation"/"hypothesis" before
    the real labelled field captures from the wrong place and lands mid-sentence.
    Those still train fine but read badly in a demo, so prefer clean ones.
    """
    s = (s or "").strip()
    return bool(s) and (s[0].isupper() or s[0].isdigit())


def score(ep) -> tuple:
    """Rank episodes: prefer a full, cleanly-parsed investigation."""
    turns = ep["turns"]
    ins = sum(1 for t in turns if t["action_type"] == "inspect")
    recon = sum(1 for t in turns if t.get("reconciliation") in ("confirmed", "refuted"))
    beliefs = [t["p_fake"] for t in turns if t.get("p_fake") is not None]
    moved = len(set(beliefs)) > 1
    clean = sum(
        _clean_field(t.get(k, "")) for t in turns for k in ("observation", "reasoning", "hypothesis")
    )
    total = sum(bool(t.get(k)) for t in turns for k in ("observation", "reasoning", "hypothesis"))
    all_clean = clean == total and total > 0
    return (ep["correct"], ins >= 3, all_clean, recon >= 2, moved, clean, ins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.path.join(REPO, "logs", "distill_episodes.jsonl"))
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "trajectory_demo.gif"))
    ap.add_argument("--episode-id", default=None)
    ap.add_argument("--truth", default="AI", choices=["AI", "REAL", "any"])
    ap.add_argument("--ms", type=int, default=2600, help="ms per turn frame")
    ap.add_argument("--hold", type=int, default=4200, help="ms on the verdict frame")
    ap.add_argument("--colors", type=int, default=128)
    ap.add_argument("--frames", action="store_true",
                    help="Also write per-turn PNGs next to the GIF.")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    eps = [json.loads(l) for l in open(args.log) if l.strip()]
    if args.truth != "any":
        eps = [e for e in eps if e["ground_truth"] == args.truth] or eps

    if args.list:
        for e in sorted(eps, key=score, reverse=True)[:15]:
            ins = sum(1 for t in e["turns"] if t["action_type"] == "inspect")
            print(f"{e['episode_id']}  truth={e['ground_truth']}  turns={len(e['turns'])} "
                  f"inspects={ins}  correct={e['correct']}")
        return

    ep = (next(e for e in eps if e["episode_id"] == args.episode_id)
          if args.episode_id else max(eps, key=score))

    base = Image.open(ep["image_path"]).convert("RGB")
    grid_n = ep.get("grid", 4)

    # GIF frames must share one size, so take the tallest turn — every field is
    # rendered in full, nothing is clipped or elided.
    body = font(15)
    height = max(
        max(panel_height(blocks(ep, t), body) for t in ep["turns"]),
        IMG + 2 * PAD + 26,
    )
    frames = [frame(ep, i, base, grid_n, height) for i in range(1, len(ep["turns"]) + 1)]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.frames:
        fdir = os.path.join(os.path.dirname(args.out), "frames")
        os.makedirs(fdir, exist_ok=True)
        for i, f in enumerate(frames, start=1):
            f.save(os.path.join(fdir, f"turn_{i}.png"))
        print(f"wrote {len(frames)} PNG frames to {fdir}")
    durations = [args.ms] * (len(frames) - 1) + [args.hold]
    frames = [f.quantize(colors=args.colors, method=Image.MEDIANCUT) for f in frames]
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)

    kb = os.path.getsize(args.out) / 1024
    print(f"{ep['episode_id']}: {len(frames)} frames -> {args.out} ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
