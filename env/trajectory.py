"""Parsing and accumulation of the pre/post predict-then-verify trajectory.

The agent speaks only in text. Every turn it emits one labelled block (see
``env/prompts.FORMAT_SPEC``) carrying, at most:

    RECONCILIATION  did the previous reveal match the previous hypothesis?
    BELIEF_UPDATE   P(fake)=<0..1> after that reveal
    OBSERVATION     what it perceives now
    REASONING       why the next region matters
    HYPOTHESIS      a testable prediction about the cell it will inspect
    ACTION          INSPECT <n>  |  VERDICT <AI|REAL> confidence=<c>

``parse_turn`` turns one completion into a :class:`TurnEntry`; :class:`Trajectory`
accumulates the episode. The reward (``env/reward.py``) reads *only* the
structured fields extracted here — never the free prose — which is what keeps the
reward mechanically verifiable and free of any LLM judge.

Parsing is deliberately tolerant (case-insensitive labels, optional colons,
``INSPECT(5)`` / ``VERDICT: AI`` variants) so a slightly malformed but usable
completion is scored on its content rather than thrown away.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# Canonical action / label / reconciliation tokens.
INSPECT = "inspect"
VERDICT = "verdict"
INVALID = "invalid"

AI = "AI"
REAL = "REAL"

CONFIRMED = "confirmed"
REFUTED = "refuted"
UNCLEAR = "unclear"

# --------------------------------------------------------------------------- #
# Field regexes. Each label captures lazily up to the next known label or EOS.
# --------------------------------------------------------------------------- #
_LABELS = r"RECONCILIATION|BELIEF_UPDATE|BELIEF|OBSERVATION|REASONING|HYPOTHESIS|ACTION"


def _field(label: str) -> re.Pattern:
    # ``label`` may itself be an alternation (e.g. "BELIEF_UPDATE|BELIEF"), so it
    # must be wrapped in a non-capturing group or the alternation would swallow
    # the capture group in one of its branches.
    return re.compile(
        rf"(?:{label})\s*:?\s*(.*?)(?=\n\s*(?:{_LABELS})\b|\Z)",
        re.IGNORECASE | re.DOTALL,
    )


_RE_RECONCILE = _field("RECONCILIATION")
_RE_OBSERVATION = _field("OBSERVATION")
_RE_REASONING = _field("REASONING")
_RE_HYPOTHESIS = _field("HYPOTHESIS")
_RE_BELIEF_FIELD = _field(r"BELIEF_UPDATE|BELIEF")

_RE_ACTION_LINE = re.compile(r"ACTION\s*:?\s*(.+)", re.IGNORECASE)
_RE_INSPECT = re.compile(r"\bINSPECT\b\s*\(?\s*(\d{1,2})\s*\)?", re.IGNORECASE)
_RE_VERDICT = re.compile(r"\bVERDICT\b\s*:?\s*\(?\s*([A-Za-z]+)", re.IGNORECASE)
# P(fake) probability, and a standalone confidence=... value.
_RE_PFAKE = re.compile(r"P\s*\(?\s*fake\s*\)?\s*[:=]\s*([01](?:\.\d+)?|\.\d+)", re.IGNORECASE)
_RE_PROB_ANY = re.compile(r"\b([01]?\.\d+|[01]\.0+|0|1)\b")
_RE_CONFIDENCE = re.compile(r"conf(?:idence)?\s*[:=]?\s*([01](?:\.\d+)?|\.\d+)", re.IGNORECASE)

# Reconciliation direction cues. REFUTED wins ties (an explicit "not confirmed"
# should read as refuted, not confirmed).
_REFUTE_CUES = (
    "refut", "not confirm", "did not", "didn't", "does not", "doesn't",
    "no artifact", "no anomal", "not malformed", "looks natural", "appears real",
    "consistent with real", "clean", "failed", "rejected", "unfounded", "wrong",
)
_CONFIRM_CUES = (
    "confirm", "as predicted", "as hypothesiz", "matched", "match ", "held",
    "artifact present", "anomal", "malformed", "asymmetr", "warp", "consistent with fake",
    "supports fake", "supported",
)


def _clean(text: Optional[str]) -> str:
    return (text or "").strip()


def _search(pat: re.Pattern, text: str) -> str:
    m = pat.search(text)
    return _clean(m.group(1)) if m else ""


def classify_reconciliation(text: str) -> str:
    """Map reconciliation prose onto ``confirmed`` / ``refuted`` / ``unclear``.

    Convention (matches the system prompt, which frames hypotheses as fake-tests
    like "if AI, this iris will be malformed"): CONFIRMED ⇒ the predicted artifact
    was present ⇒ evidence toward fake; REFUTED ⇒ it was absent ⇒ evidence toward
    real. The reward uses this direction to check that the belief moved sensibly.
    """
    t = text.lower()
    if not t:
        return UNCLEAR
    if any(cue in t for cue in _REFUTE_CUES):
        return REFUTED
    if any(cue in t for cue in _CONFIRM_CUES):
        return CONFIRMED
    return UNCLEAR


def _parse_pfake(belief_text: str) -> Optional[float]:
    """Pull P(fake) out of a BELIEF_UPDATE field. Prefers an explicit
    ``P(fake)=..`` token; falls back to the first bare probability present."""
    m = _RE_PFAKE.search(belief_text)
    if not m:
        m = _RE_PROB_ANY.search(belief_text)
    if not m:
        return None
    try:
        return max(0.0, min(1.0, float(m.group(1))))
    except ValueError:
        return None


def normalize_verdict(token: str) -> Optional[str]:
    t = token.strip().lower()
    if t in {"ai", "fake", "synthetic", "generated", "artificial", "gan"}:
        return AI
    if t in {"real", "authentic", "genuine", "photo", "photograph", "camera"}:
        return REAL
    return None


def label_to_verdict(label: int) -> str:
    """Manifest label (0=Real, 1=AI) -> canonical verdict string."""
    return AI if int(label) == 1 else REAL


@dataclass
class TurnEntry:
    """One parsed assistant turn."""

    action_type: str = INVALID            # inspect | verdict | invalid
    cell: Optional[int] = None            # set for inspect (1..grid^2, unclamped here)
    verdict: Optional[str] = None         # AI | REAL, set for verdict
    confidence: Optional[float] = None    # set for verdict
    p_fake: Optional[float] = None        # from BELIEF_UPDATE
    reconciliation: str = UNCLEAR         # confirmed | refuted | unclear
    observation: str = ""
    reasoning: str = ""
    hypothesis: str = ""
    raw: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.action_type == VERDICT


def parse_turn(text: str) -> TurnEntry:
    """Parse one completion into a :class:`TurnEntry`.

    Only the *last* ACTION line is honored, so a model that narrates a plan
    ("I will INSPECT then VERDICT") before committing is not misread.
    """
    recon_text = _search(_RE_RECONCILE, text)
    entry = TurnEntry(
        raw=text,
        observation=_search(_RE_OBSERVATION, text),
        reasoning=_search(_RE_REASONING, text),
        hypothesis=_search(_RE_HYPOTHESIS, text),
        reconciliation=classify_reconciliation(recon_text),
        p_fake=_parse_pfake(_search(_RE_BELIEF_FIELD, text)),
    )

    action_lines = _RE_ACTION_LINE.findall(text)
    candidate = action_lines[-1] if action_lines else text

    verdict_m = _RE_VERDICT.search(candidate)
    if verdict_m:
        verdict = normalize_verdict(verdict_m.group(1))
        if verdict:
            entry.action_type = VERDICT
            entry.verdict = verdict
            conf_m = _RE_CONFIDENCE.search(candidate) or _RE_CONFIDENCE.search(text)
            if conf_m:
                try:
                    entry.confidence = max(0.0, min(1.0, float(conf_m.group(1))))
                except ValueError:
                    entry.confidence = None
            return entry
        return entry  # INVALID: unrecognized verdict token

    inspect_m = _RE_INSPECT.search(candidate)
    if inspect_m:
        entry.action_type = INSPECT
        entry.cell = int(inspect_m.group(1))
        return entry

    return entry  # INVALID


@dataclass
class Trajectory:
    """Ordered accumulation of an episode's turns plus derived signals the reward
    consumes. ``prior`` is the belief before any evidence (a neutral 0.5)."""

    entries: list[TurnEntry] = field(default_factory=list)
    prior: float = 0.5

    def add(self, entry: TurnEntry) -> None:
        self.entries.append(entry)

    # -- derived views used by env/reward.py -------------------------------- #
    @property
    def final(self) -> Optional[TurnEntry]:
        return self.entries[-1] if self.entries else None

    @property
    def answered(self) -> bool:
        return bool(self.final and self.final.is_terminal)

    @property
    def final_verdict(self) -> Optional[str]:
        return self.final.verdict if self.answered else None

    @property
    def final_confidence(self) -> Optional[float]:
        return self.final.confidence if self.answered else None

    def belief_series(self) -> list[float]:
        """P(fake) values in turn order, skipping turns that carried none."""
        return [e.p_fake for e in self.entries if e.p_fake is not None]

    def final_belief(self) -> Optional[float]:
        series = self.belief_series()
        return series[-1] if series else None

    def belief_steps(self) -> list[tuple[str, float, float]]:
        """Aligned ``(reconciliation, prev_belief, new_belief)`` triples for every
        turn that recorded a belief. ``prev_belief`` seeds from ``prior`` for the
        first such turn, so a single belief update is still checkable against the
        neutral starting point."""
        steps = []
        prev = self.prior
        for e in self.entries:
            if e.p_fake is None:
                continue
            steps.append((e.reconciliation, prev, e.p_fake))
            prev = e.p_fake
        return steps

    def reconciliations(self) -> list[str]:
        """Reconciliation flags, one per turn that actually recorded one (i.e.
        every non-first turn). ``unclear`` is included so gaming by omission is
        visible to the reward."""
        return [e.reconciliation for e in self.entries[1:]]

    def num_inspects(self) -> int:
        return sum(e.action_type == INSPECT for e in self.entries)
