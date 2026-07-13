"""The verifiable reward.

Every term below is a mechanical function of the structured trajectory
(``env/trajectory.py``) and the ground-truth label. There is **no LLM judge**
anywhere, and the *prose* the agent writes is never scored for quality — only the
labelled fields (belief numbers, reconciliation direction, the committed verdict
and confidence) are read. Eloquence, detail, and length earn nothing.

The design intent, and the tension worth understanding before touching the
weights:

* ``verdict_correct`` carries the dominant weight so a right answer always beats a
  wrong one no matter how pretty the trajectory around it.
* ``belief_coherence`` and ``verdict_consistency`` reward the *process* — that the
  agent's stated P(fake) moved in the direction its own reconciliations imply, and
  that the final call follows from the accumulated evidence rather than
  contradicting it. These are ungated (they apply even to wrong answers) on
  purpose: they shape *how* the agent reasons. They are deliberately weaker than
  ``verdict_correct`` so they can never make a confidently-wrong episode look good.
* ``prediction_tracking`` is intentionally SOFT (small weight): it is the most
  game-able term (an agent can just always write "CONFIRMED"), so it only nudges.
  The honest cross-check on whether predictions land on real artifacts lives in
  the eval evidence slice (GradCAM + human-verified cells), never in this hot loop.
* ``action_cost`` makes exhaustive looking unprofitable, forcing hypothesis-driven
  inspection. ``confident_wrong`` penalizes miscalibration so the agent learns to
  hedge on genuinely indistinguishable images instead of guessing loudly.

The documented failure history that motivated this shape lives in
``results/reward_failure_history.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from env.trajectory import AI, CONFIRMED, REFUTED, UNCLEAR, Trajectory

_EPS = 0.02  # belief movements smaller than this count as "held steady"


@dataclass
class RewardConfig:
    """Reward weights. ``verdict_correct`` intentionally dominates; the process
    terms shape reasoning without ever overturning outcome."""

    # Positive terms
    w_correct: float = 1.0             # ±this on a right / wrong verdict
    w_belief_coherence: float = 0.30   # beliefs move sensibly given reconciliations
    w_verdict_consistency: float = 0.30  # final call follows accumulated evidence
    w_prediction_tracking: float = 0.10  # SOFT: fraction of hypotheses confirmed

    # Cost / penalty terms (subtracted)
    c_action: float = 0.05             # per inspect — budget pressure
    c_confident_wrong: float = 0.50    # scaled by stated confidence on a wrong call
    no_answer_penalty: float = 1.00    # ran out of budget without committing

    # Confidence assumed when a verdict omits one (keeps confident_wrong defined).
    default_confidence: float = 0.5


# --------------------------------------------------------------------------- #
# Individual, independently-testable term scorers. Each returns a raw score
# (unweighted); ``compute_episode_reward`` applies the weights.
# --------------------------------------------------------------------------- #

def _expected_direction(reconciliation: str) -> int:
    """Belief direction a reconciliation implies. Hypotheses are framed as
    fake-tests (see the system prompt), so a CONFIRMED artifact pushes P(fake)
    up (+1), a REFUTED one pushes it down (-1), and UNCLEAR implies no move (0)."""
    if reconciliation == CONFIRMED:
        return +1
    if reconciliation == REFUTED:
        return -1
    return 0


def belief_coherence_score(traj: Trajectory) -> float:
    """[0,1]: did P(fake) move in the direction the agent's own reconciliations
    imply? Full credit when the sign of the belief change matches the expected
    direction; half credit when the belief held steady despite a directional
    reconciliation; zero when it moved the *wrong* way (incoherent) or drifted
    with no reconciled reason. Returns 0 when the agent recorded no beliefs at
    all (there is nothing coherent to reward)."""
    steps = traj.belief_steps()
    if not steps:
        return 0.0
    total = 0.0
    for reconciliation, prev, new in steps:
        delta = new - prev
        expected = _expected_direction(reconciliation)
        if expected != 0:
            if (delta > _EPS and expected > 0) or (delta < -_EPS and expected < 0):
                total += 1.0
            elif abs(delta) <= _EPS:
                total += 0.5              # held steady — weakly coherent
            # else: moved against its own evidence -> 0
        else:  # UNCLEAR reconciliation: belief should not lurch without a reason
            total += 1.0 if abs(delta) <= _EPS else 0.0
    return total / len(steps)


def verdict_consistency_score(traj: Trajectory) -> float:
    """[0,1]: does the final verdict follow from the trajectory rather than
    contradict it? Two mechanical checks, averaged over whichever are available:
    (a) the final belief agrees with the verdict (P(fake)>0.5 ⇒ AI); (b) the net
    reconciliation evidence agrees with the verdict (more CONFIRMED-than-REFUTED
    ⇒ AI). If neither check has signal, an answered episode gets a neutral 0.5;
    an unanswered one gets 0."""
    if not traj.answered:
        return 0.0
    verdict_is_ai = traj.final_verdict == AI
    checks: list[bool] = []

    final_belief = traj.final_belief()
    if final_belief is not None and abs(final_belief - 0.5) > _EPS:
        checks.append((final_belief > 0.5) == verdict_is_ai)

    net = sum(_expected_direction(r) for r in traj.reconciliations())
    if net != 0:
        checks.append((net > 0) == verdict_is_ai)

    if not checks:
        return 0.5
    return sum(1.0 for c in checks if c) / len(checks)


def prediction_tracking_score(traj: Trajectory) -> float:
    """[0,1] SOFT: fraction of reconciliations the agent marked CONFIRMED — i.e.
    how often its pre-action hypotheses were borne out. Deliberately game-able
    (always-CONFIRMED maxes it), which is why its weight is small and the real
    check lives in the eval evidence slice. UNCLEAR reconciliations count against
    it, discouraging vague non-predictions."""
    recons = traj.reconciliations()
    if not recons:
        return 0.0
    return sum(r == CONFIRMED for r in recons) / len(recons)


def confident_wrong_penalty(traj: Trajectory, ground_truth: str, cfg: RewardConfig) -> float:
    """<=0: a wrong verdict is penalized in proportion to the confidence it was
    stated with, so the agent learns to hedge on indistinguishable images instead
    of guessing loudly. Right answers and non-answers incur nothing here."""
    if not traj.answered or traj.final_verdict == ground_truth:
        return 0.0
    conf = traj.final_confidence
    if conf is None:
        conf = cfg.default_confidence
    return -cfg.c_confident_wrong * conf


# --------------------------------------------------------------------------- #
# Episode aggregation
# --------------------------------------------------------------------------- #

def compute_episode_reward(
    traj: Trajectory, ground_truth: str, cfg: RewardConfig | None = None
) -> tuple[float, dict[str, float]]:
    """The authoritative episode reward. Returns ``(total, breakdown)`` where
    ``breakdown`` maps each term to its weighted contribution — the env credits
    ``total`` at episode end and logs ``breakdown`` for the trace/eval. Summing
    the breakdown reproduces ``total`` exactly.

    ``ground_truth`` is the canonical ``"AI"`` / ``"REAL"`` label.
    """
    cfg = cfg or RewardConfig()
    b: dict[str, float] = {}

    if traj.answered:
        b["verdict_correct"] = cfg.w_correct if traj.final_verdict == ground_truth else -cfg.w_correct
    else:
        b["verdict_correct"] = 0.0

    b["belief_coherence"] = cfg.w_belief_coherence * belief_coherence_score(traj)
    b["verdict_consistency"] = cfg.w_verdict_consistency * verdict_consistency_score(traj)
    b["prediction_tracking"] = cfg.w_prediction_tracking * prediction_tracking_score(traj)
    b["action_cost"] = -cfg.c_action * traj.num_inspects()
    b["confident_wrong"] = confident_wrong_penalty(traj, ground_truth, cfg)
    b["no_answer"] = 0.0 if traj.answered else -cfg.no_answer_penalty

    total = float(sum(b.values()))
    return total, b
