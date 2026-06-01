"""System grounding, tool descriptions, output-format spec and the templates
used to render tool results back to the VLM.

These strings are the only place the agent's "rules of the game" live, so the
environment, training rollout and evaluation all import from here to stay in
sync.
"""

from __future__ import annotations

import json
from typing import Any

# --------------------------------------------------------------------------- #
# System prompt: who the agent is + the rules of the environment
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """\
You are a forensic image analyst. Your job is to decide whether an image is a \
real photograph (REAL) or AI-generated synthetic media (AI).

You investigate actively before deciding. You are shown the full image first \
and may gather more evidence using tools. The image is divided into a 3x3 grid \
of cells numbered row-major:

    1 2 3
    4 5 6
    7 8 9

TOOLS (use exactly one per turn):
- ZOOM <n>   Inspect grid cell n (1-9) at higher resolution. Look for \
generation artifacts: warped textures, fused or extra fingers, garbled text, \
inconsistent lighting, melted backgrounds, asymmetric eyes/teeth.
- METADATA   Reveal forensic metadata for the image (camera EXIF, software \
signature, color space). Genuine photos usually carry camera hardware \
signatures; synthetic images usually do not. Metadata can be missing or \
stripped, so treat it as one clue among several, not proof.
- ANSWER <AI|REAL>   Commit to a final verdict and end the investigation.

You have a limited number of turns, so investigate efficiently: gather the \
evidence you need, then answer. Answering correctly with fewer steps is better \
than answering after wasting turns. YOU MUST ALWAYS GIVE A REASON FOR WHY YOU PICK A CHOICE \

OUTPUT FORMAT (every turn, exactly):
THOUGHT: <one or two sentences of reasoning about what you see and what to do next>
ACTION: <ZOOM n | METADATA | ANSWER AI | ANSWER REAL>

Do not output anything after the ACTION line."""


# --------------------------------------------------------------------------- #
# Initial user turn
# --------------------------------------------------------------------------- #

INITIAL_USER_TEXT = (
    "Here is the full image under investigation. Begin your analysis. "
    "Remember to respond with a THOUGHT line and a single ACTION line."
)


# --------------------------------------------------------------------------- #
# Tool-result templates (the environment's reply to the agent)
# --------------------------------------------------------------------------- #

ZOOM_RESULT_TEXT = (
    "Zoomed view of cell {cell}. Examine it for synthetic artifacts, then "
    "decide your next action."
)

METADATA_NOTICE = (
    "Forensic metadata report:\n{report}\n"
    "Factor this into your reasoning, then decide your next action."
)

INVALID_ACTION_FEEDBACK = (
    "Your previous response did not contain a valid ACTION. Respond with a "
    "THOUGHT line and exactly one ACTION line, where ACTION is one of: "
    "ZOOM <1-9>, METADATA, or ANSWER <AI|REAL>."
)

REPEATED_ZOOM_FEEDBACK = (
    "You have already zoomed into cell {cell}. Choose a different cell, request "
    "METADATA, or ANSWER."
)

OUT_OF_TURNS_FEEDBACK = (
    "You have run out of investigation turns and must decide now. Respond with "
    "ACTION: ANSWER AI or ACTION: ANSWER REAL."
)


def format_metadata(metadata: dict[str, Any]) -> str:
    """Render the manifest's metadata dict into a readable forensic report."""
    if not metadata:
        return "  (no metadata available)"
    lines = []
    for key, value in metadata.items():
        label = key.replace("_", " ").title()
        if isinstance(value, dict):
            value = json.dumps(value)
        lines.append(f"  - {label}: {value}")
    return "\n".join(lines)
