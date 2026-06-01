"""Active-perception helpers: 3x3 grid crop math and parsers for the VLM's
text actions.

The VLM interacts with the environment purely through text. Every turn it must
emit a short reasoning trace followed by exactly one action on its own line:

    THOUGHT: <free-form reasoning>
    ACTION: ZOOM <n>        # n in 1..9, row-major over a 3x3 grid
    ACTION: METADATA
    ACTION: ANSWER <AI|REAL>

The parsers below are deliberately tolerant (case-insensitive, tolerate
parentheses like ``ZOOM(5)``) so that a slightly malformed completion is still
usable instead of throwing the whole rollout away.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from PIL import Image

# --------------------------------------------------------------------------- #
# Grid crop math
# --------------------------------------------------------------------------- #


def grid_cell_bbox(width: int, height: int, cell: int, grid: int = 3):
    """Return the ``(left, upper, right, lower)`` pixel box for ``cell``.

    Cells are numbered row-major starting at 1 (top-left)::

        1 2 3
        4 5 6
        7 8 9
    """
    if not 1 <= cell <= grid * grid:
        raise ValueError(f"cell {cell} out of range for {grid}x{grid} grid")

    idx = cell - 1
    row, col = divmod(idx, grid)

    # Use rounding rather than integer floor so the rightmost / bottommost
    # cells reach the true edge of the image even when it isn't divisible.
    left = round(col * width / grid)
    right = round((col + 1) * width / grid)
    upper = round(row * height / grid)
    lower = round((row + 1) * height / grid)
    return left, upper, right, lower


def crop_grid_cell(
    image: Image.Image,
    cell: int,
    grid: int = 3,
    upscale_to: Optional[int] = 512,
) -> Image.Image:
    """Crop ``cell`` out of ``image`` and optionally upscale it.

    Upscaling the small crop back up to ``upscale_to`` on its long edge is what
    makes this a "zoom": the model receives the same region at higher effective
    resolution, surfacing generation artifacts (warped textures, fused edges)
    that are invisible in the downsampled full frame.
    """
    box = grid_cell_bbox(image.width, image.height, cell, grid)
    crop = image.crop(box)

    if upscale_to:
        w, h = crop.size
        if max(w, h) < upscale_to and max(w, h) > 0:
            scale = upscale_to / max(w, h)
            crop = crop.resize(
                (max(1, round(w * scale)), max(1, round(h * scale))),
                Image.Resampling.LANCZOS,
            )
    return crop


# --------------------------------------------------------------------------- #
# Action parsing
# --------------------------------------------------------------------------- #

ZOOM = "zoom"
METADATA = "metadata"
ANSWER = "answer"
INVALID = "invalid"

_ACTION_LINE = re.compile(r"ACTION\s*:?\s*(.+)", re.IGNORECASE)
_ZOOM = re.compile(r"\bZOOM\b\s*\(?\s*([1-9])\s*\)?", re.IGNORECASE)
_METADATA = re.compile(r"\bMETADATA\b", re.IGNORECASE)
_ANSWER = re.compile(r"\bANSWER\b\s*\(?\s*([A-Za-z]+)", re.IGNORECASE)
_THOUGHT = re.compile(r"THOUGHT\s*:?\s*(.*?)(?:\n\s*ACTION|\Z)", re.IGNORECASE | re.DOTALL)


@dataclass
class Action:
    type: str  # ZOOM | METADATA | ANSWER | INVALID
    cell: Optional[int] = None  # set when type == ZOOM
    verdict: Optional[str] = None  # "AI" | "REAL", set when type == ANSWER
    thought: str = ""
    raw: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.type == ANSWER


def normalize_verdict(token: str) -> Optional[str]:
    """Map a free-form verdict word onto the canonical ``"AI"`` / ``"REAL"``."""
    t = token.strip().lower()
    if t in {"ai", "fake", "synthetic", "generated", "artificial", "cgi"}:
        return "AI"
    if t in {"real", "authentic", "genuine", "photo", "photograph", "camera"}:
        return "REAL"
    return None


def extract_thought(text: str) -> str:
    m = _THOUGHT.search(text)
    if m:
        return m.group(1).strip()
    return ""


def is_substantive_thought(
    thought: str, previous: str = "", min_words: int = 4
) -> bool:
    """Heuristic guard for Option-A reasoning reward.

    Returns True when ``thought`` looks like a genuine, non-degenerate reason:
    it has at least ``min_words`` words and is not a verbatim repeat of the
    ``previous`` turn's thought (the cheapest form of boilerplate gaming). This
    is deliberately a *presence/non-degeneracy* floor, not a quality judge --
    that is Option C (see ``ForgeryDetectionEnv._judge_reasoning``).
    """
    t = (thought or "").strip()
    if len(t.split()) < min_words:
        return False
    if previous and t.lower() == previous.strip().lower():
        return False
    return True


def parse_action(text: str) -> Action:
    """Parse a model completion into a single :class:`Action`.

    Only the *last* ACTION line is honored, so a model that narrates a plan
    ("first I'll ZOOM, then ANSWER") before committing isn't misread. If no
    well-formed action is found the action is ``INVALID`` and the environment
    will return corrective feedback instead of advancing.
    """
    thought = extract_thought(text)

    action_lines = _ACTION_LINE.findall(text)
    candidate = action_lines[-1] if action_lines else text

    answer = _ANSWER.search(candidate)
    if answer:
        verdict = normalize_verdict(answer.group(1))
        if verdict:
            return Action(ANSWER, verdict=verdict, thought=thought, raw=text)
        return Action(INVALID, thought=thought, raw=text)

    zoom = _ZOOM.search(candidate)
    if zoom:
        return Action(ZOOM, cell=int(zoom.group(1)), thought=thought, raw=text)

    if _METADATA.search(candidate):
        return Action(METADATA, thought=thought, raw=text)

    return Action(INVALID, thought=thought, raw=text)


def label_to_verdict(label: int) -> str:
    """Manifest label (0=Real, 1=AI) -> canonical verdict string."""
    return "AI" if int(label) == 1 else "REAL"
