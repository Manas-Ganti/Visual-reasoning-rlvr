"""System grounding and the pre/post documentation format.

These strings are the single source of truth for the "rules of the game": the
environment, the SFT trace distiller, the GRPO rollout, and the eval harness all
import from here so the agent is graded against exactly the format it was told to
produce. The verifiable reward reads structured fields out of the trajectory
(``env/trajectory.py``), so the format spec below and the parser must stay in
lockstep.

**Domains.** Everything about the *format* is substrate-independent, but the
subject line and the artifact checklist are not: telling a model to look for
malformed irises and melted earrings on a photograph of a golden retriever is
worse than telling it nothing, because it names a checklist that cannot be
satisfied. So the substrate-specific half lives in ``DOMAINS`` and is selected by
dataset — the same ``VRR_DATASET`` namespace that keys data, checkpoints, logs
and results (``training/common.py``). Adding substrate #3 is a dict entry here,
not another sweep through the prompt text.

``VRR_DATASET`` is read directly rather than imported from ``training.common``:
``training`` imports ``env``, so the reverse import would be circular. The
default is mirrored deliberately and asserted in ``tests/test_prompts.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Domains — the substrate-specific half of the system prompt
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Domain:
    """What changes between substrates. Everything else in the prompt is shared."""

    subject: str              # how the image is referred to, e.g. "a face image"
    artifacts: str            # what generation artifacts to hunt for
    hypothesis_example: str   # a concrete HYPOTHESIS line for the format spec


DOMAINS: dict[str, Domain] = {
    # ImageNet-category photographs vs. eight generators (GenImage: ADM, BigGAN,
    # GLIDE, Midjourney, SD 1.4/1.5, VQDM, Wukong). Diffusion artifacts are
    # semantic and spatially localized — which is what makes INSPECT <n> a real
    # decision rather than an arbitrary one.
    "image": Domain(
        subject="an image",
        artifacts=(
            "malformed or extra fingers, hands and limbs; garbled or nonsensical "
            "text on signs, labels and packaging; impossible geometry where "
            "structures fail to connect or straight lines warp; lighting that "
            "disagrees with itself — shadows falling in conflicting directions, "
            "missing shadows, reflections that do not match their source; melted "
            "or repeated textures and implausibly uniform surfaces; incoherent "
            "boundaries where two objects meet; background detail that dissolves "
            "into nonsense; anatomically wrong animal features"
        ),
        hypothesis_example=(
            "if AI, the lettering on the sign in this cell will be unreadable"
        ),
    ),
    # 300x300 StyleGAN2 faces vs. Unsplash photographs. Retired as a training
    # substrate (see results/faces_negative_result.md) but kept configured so the
    # negative result stays reproducible.
    "face": Domain(
        subject="a face image",
        artifacts=(
            "asymmetric or malformed eyes/irises, teeth/gum blending, mismatched "
            "or melted earrings, unnatural ear or hairline structure, over-smooth "
            "or waxy skin texture, background warping near the head"
        ),
        hypothesis_example="if AI, the left iris in this cell will be misshapen",
    ),
}

# Which domain each dataset namespace investigates.
DATASET_DOMAIN: dict[str, str] = {
    "genimage": "image",
    "faces": "face",
}

DEFAULT_DOMAIN = "image"
# Mirrors training.common.DATASET — see the module docstring for why it is not
# imported.
_DEFAULT_DATASET = "genimage"


def resolve_domain(dataset: str | None = None) -> str:
    """Domain name for a dataset namespace (default: ``$VRR_DATASET``).

    Unknown datasets fall back to ``image`` — the generic prompt is a safe
    default for a new substrate, whereas the face checklist is actively wrong
    everywhere except faces.
    """
    ds = (dataset or os.environ.get("VRR_DATASET") or _DEFAULT_DATASET).strip()
    return DATASET_DOMAIN.get(ds, DEFAULT_DOMAIN)


def get_domain(domain: str | None = None, dataset: str | None = None) -> Domain:
    """Resolve a ``Domain``, by explicit name or via the dataset namespace."""
    name = domain or resolve_domain(dataset)
    if name not in DOMAINS:
        raise KeyError(f"Unknown domain {name!r}. Known: {sorted(DOMAINS)}")
    return DOMAINS[name]


# --------------------------------------------------------------------------- #
# System prompt: identity + rules of the environment
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT_TEMPLATE = """\
You are an investigative image analyst. You must decide whether {subject} is a \
real photograph (REAL) or AI-generated (AI).

You do NOT get a clear picture up front. You are shown only a low-resolution \
OVERVIEW where fine details are blurred away. The correct answer is not reachable \
from the overview alone: you must actively inspect regions to sharpen them before \
you can be sure.

The image is divided into a 4x4 grid of cells, numbered row-major from 1:

     1  2  3  4
     5  6  7  8
     9 10 11 12
    13 14 15 16

You have exactly TWO actions:
- INSPECT <n>       Reveal grid cell n (1-16) at high resolution. Costs one unit \
of your limited budget. Use it to test a specific prediction, not to look around \
randomly. Look for generation artifacts: {artifacts}.
- VERDICT <AI|REAL> confidence=<0.0-1.0>   Commit your final answer and END the \
episode. State a calibrated confidence: use a high confidence only when your \
inspections genuinely settled the question.

THE INVESTIGATION IS SEQUENTIAL AND YOU MUST PREDICT BEFORE YOU LOOK. Each turn \
you write a structured block. Before an INSPECT you commit a testable HYPOTHESIS \
about what the cell will show; after the reveal, on your NEXT turn, you first \
RECONCILE what you actually saw against that hypothesis and update your belief. \
This predict-then-verify discipline is mandatory.

You are NOT graded on how eloquent or detailed your writing is. You are graded on \
reaching the right verdict efficiently, and on whether your beliefs move \
sensibly given what you actually observed. Spend inspects only when a hypothesis \
justifies them; a correct verdict with fewer inspects beats a correct verdict \
after exhausting the budget."""


# --------------------------------------------------------------------------- #
# Output format spec (appended to the system prompt so it is unmissable)
# --------------------------------------------------------------------------- #

FORMAT_SPEC_TEMPLATE = """\
OUTPUT FORMAT — every turn is exactly one block with these labelled fields.

On the FIRST turn (nothing has been revealed yet), omit RECONCILIATION and \
BELIEF_UPDATE and start at OBSERVATION:

    OBSERVATION: <what you can perceive at the current resolution>
    REASONING: <why the region you are about to inspect matters / your uncertainty>
    HYPOTHESIS: <a testable prediction, e.g. "{hypothesis_example}">
    ACTION: INSPECT <n>

On EVERY LATER turn, first reconcile the previous reveal, then continue:

    RECONCILIATION: <CONFIRMED or REFUTED — did the reveal match your last \
hypothesis, and how>
    BELIEF_UPDATE: P(fake)=<0.0-1.0> because <what moved it>
    OBSERVATION: <what the last reveal showed / what you now perceive>
    REASONING: <why your next step matters>
    HYPOTHESIS: <your next testable prediction>
    ACTION: INSPECT <n>

To finish, replace the ACTION line with a verdict (you may still write \
RECONCILIATION / BELIEF_UPDATE first):

    ACTION: VERDICT <AI|REAL> confidence=<0.0-1.0>

Emit exactly one ACTION line and write nothing after it."""


def system_prompt(domain: str | None = None, dataset: str | None = None) -> str:
    """The full system prompt (grounding + format spec) for one domain.

    This is the only way to build the prompt — there is deliberately no
    module-level ``SYSTEM_PROMPT_FULL`` constant, because a global resolved at
    import time is exactly how the face checklist survived a substrate switch.
    """
    d = get_domain(domain, dataset)
    return (
        SYSTEM_PROMPT_TEMPLATE.format(subject=d.subject, artifacts=d.artifacts)
        + "\n\n"
        + FORMAT_SPEC_TEMPLATE.format(hypothesis_example=d.hypothesis_example)
    )


def classify_prompt(domain: str | None = None, dataset: str | None = None) -> str:
    """One-shot classification question — no environment, no format, one word.

    Used by ``tools/ceiling_probe.py`` (ceiling/floor measurement) and
    ``data/build_evidence_slice.py`` (occlusion saliency). Shares the domain
    registry so a substrate switch cannot leave these asking about faces.
    """
    d = get_domain(domain, dataset)
    return (
        f"Is {d.subject} a real photograph or AI-generated? "
        "Answer with exactly one word: REAL or AI."
    )


# --------------------------------------------------------------------------- #
# User turns: initial overview + environment replies (substrate-independent)
# --------------------------------------------------------------------------- #

INITIAL_USER_TEXT = (
    "Here is the low-resolution OVERVIEW of the image under investigation. "
    "Fine details are intentionally blurred. Begin your predict-then-verify "
    "investigation. You have a budget of {budget} inspects."
)

INSPECT_RESULT_TEXT = (
    "High-resolution reveal of cell {cell}. Inspects remaining: {remaining}. "
    "On your next turn, RECONCILE this against your hypothesis before continuing."
)

BUDGET_EXHAUSTED_TEXT = (
    "You have used your entire inspect budget. You must now commit a verdict. "
    "Reconcile your last reveal, then respond with "
    "ACTION: VERDICT <AI|REAL> confidence=<0.0-1.0>."
)

INVALID_ACTION_FEEDBACK = (
    "Your previous response had no valid ACTION line. Respond with the labelled "
    "format and exactly one ACTION line: either INSPECT <1-16> or "
    "VERDICT <AI|REAL> confidence=<0.0-1.0>."
)

REPEATED_INSPECT_FEEDBACK = (
    "You already inspected cell {cell}; re-revealing it wastes budget. Choose a "
    "different cell or commit a verdict."
)

VERDICT_ACK_TEXT = "Verdict recorded: {verdict} (confidence {confidence}). Investigation complete."
