"""Edge-case tests for the verifiable reward. These are the tests CI runs."""

import pytest

from env.reward import (
    RewardConfig,
    belief_brier_score,
    belief_coherence_score,
    compute_episode_reward,
    confident_wrong_penalty,
    prediction_tracking_score,
    verdict_consistency_score,
)
from env.trajectory import AI, REAL, Trajectory, TurnEntry, parse_turn


# --------------------------------------------------------------------------- #
# Trajectory builders
# --------------------------------------------------------------------------- #
def traj_from_texts(*texts) -> Trajectory:
    t = Trajectory()
    for x in texts:
        t.add(parse_turn(x))
    return t


def traj_from_entries(*entries) -> Trajectory:
    t = Trajectory()
    for e in entries:
        t.add(e)
    return t


# --------------------------------------------------------------------------- #
# belief_coherence
# --------------------------------------------------------------------------- #
def test_coherence_confirmed_up_gets_full_credit():
    t = traj_from_entries(
        TurnEntry(action_type="inspect"),
        TurnEntry(action_type="inspect", reconciliation="confirmed", p_fake=0.8),
    )
    assert belief_coherence_score(t) == 1.0  # 0.5 prior -> 0.8, confirmed => up


def test_coherence_refuted_down_gets_full_credit():
    t = traj_from_entries(
        TurnEntry(action_type="inspect"),
        TurnEntry(action_type="inspect", reconciliation="refuted", p_fake=0.2),
    )
    assert belief_coherence_score(t) == 1.0


def test_coherence_wrong_direction_zero():
    # confirmed (expect up) but belief went DOWN => incoherent
    t = traj_from_entries(
        TurnEntry(action_type="inspect"),
        TurnEntry(action_type="inspect", reconciliation="confirmed", p_fake=0.2),
    )
    assert belief_coherence_score(t) == 0.0


def test_coherence_held_steady_half_credit():
    t = traj_from_entries(
        TurnEntry(action_type="inspect"),
        TurnEntry(action_type="inspect", reconciliation="confirmed", p_fake=0.5),
    )
    assert belief_coherence_score(t) == 0.5


def test_coherence_unclear_move_is_incoherent():
    t = traj_from_entries(
        TurnEntry(action_type="inspect"),
        TurnEntry(action_type="inspect", reconciliation="unclear", p_fake=0.9),
    )
    assert belief_coherence_score(t) == 0.0


def test_coherence_unclear_steady_is_fine():
    t = traj_from_entries(
        TurnEntry(action_type="inspect"),
        TurnEntry(action_type="inspect", reconciliation="unclear", p_fake=0.5),
    )
    assert belief_coherence_score(t) == 1.0


def test_coherence_no_beliefs_zero():
    t = traj_from_entries(TurnEntry(action_type="inspect"))
    assert belief_coherence_score(t) == 0.0


def test_coherence_averages_multiple_steps():
    t = traj_from_entries(
        TurnEntry(action_type="inspect"),
        TurnEntry(action_type="inspect", reconciliation="confirmed", p_fake=0.8),  # 1.0
        TurnEntry(action_type="inspect", reconciliation="confirmed", p_fake=0.6),  # went down -> 0.0
    )
    assert belief_coherence_score(t) == 0.5


# --------------------------------------------------------------------------- #
# verdict_consistency
# --------------------------------------------------------------------------- #
def test_consistency_belief_and_evidence_agree():
    t = traj_from_entries(
        TurnEntry(action_type="inspect"),
        TurnEntry(action_type="inspect", reconciliation="confirmed", p_fake=0.85),
        TurnEntry(action_type="verdict", verdict=AI, confidence=0.9, p_fake=0.85),
    )
    assert verdict_consistency_score(t) == 1.0


def test_consistency_contradiction_zero():
    # final belief says fake, but verdict says REAL, and evidence was confirmed(fake)
    t = traj_from_entries(
        TurnEntry(action_type="inspect"),
        TurnEntry(action_type="inspect", reconciliation="confirmed", p_fake=0.85),
        TurnEntry(action_type="verdict", verdict=REAL, confidence=0.9, p_fake=0.85),
    )
    assert verdict_consistency_score(t) == 0.0


def test_consistency_unanswered_zero():
    t = traj_from_entries(TurnEntry(action_type="inspect", p_fake=0.85))
    assert verdict_consistency_score(t) == 0.0


def test_consistency_no_signal_is_neutral():
    # answered, but belief exactly 0.5 and no directional reconciliations
    t = traj_from_entries(
        TurnEntry(action_type="verdict", verdict=AI, confidence=0.5, p_fake=0.5),
    )
    assert verdict_consistency_score(t) == 0.5


# --------------------------------------------------------------------------- #
# prediction_tracking
# --------------------------------------------------------------------------- #
def test_prediction_tracking_fraction_confirmed():
    t = traj_from_entries(
        TurnEntry(action_type="inspect"),
        TurnEntry(action_type="inspect", reconciliation="confirmed"),
        TurnEntry(action_type="inspect", reconciliation="refuted"),
        TurnEntry(action_type="verdict", verdict=AI, reconciliation="confirmed"),
    )
    # reconciliations exclude turn 1: [confirmed, refuted, confirmed] -> 2/3
    assert prediction_tracking_score(t) == pytest.approx(2 / 3)


def test_prediction_tracking_empty_zero():
    t = traj_from_entries(TurnEntry(action_type="inspect"))
    assert prediction_tracking_score(t) == 0.0


# --------------------------------------------------------------------------- #
# confident_wrong
# --------------------------------------------------------------------------- #
def test_confident_wrong_scaled_by_confidence():
    cfg = RewardConfig()
    t = traj_from_entries(TurnEntry(action_type="verdict", verdict=AI, confidence=0.8))
    assert confident_wrong_penalty(t, REAL, cfg) == pytest.approx(-0.5 * 0.8)


def test_confident_wrong_zero_when_correct():
    cfg = RewardConfig()
    t = traj_from_entries(TurnEntry(action_type="verdict", verdict=AI, confidence=0.9))
    assert confident_wrong_penalty(t, AI, cfg) == 0.0


def test_confident_wrong_uses_default_confidence():
    cfg = RewardConfig()
    t = traj_from_entries(TurnEntry(action_type="verdict", verdict=AI, confidence=None))
    assert confident_wrong_penalty(t, REAL, cfg) == pytest.approx(-0.5 * cfg.default_confidence)


# --------------------------------------------------------------------------- #
# compute_episode_reward — integration
# --------------------------------------------------------------------------- #
def test_breakdown_sums_to_total():
    t = traj_from_texts(
        "OBSERVATION: blur\nHYPOTHESIS: iris malformed\nACTION: INSPECT 6",
        "RECONCILIATION: CONFIRMED artifact present\nBELIEF_UPDATE: P(fake)=0.8\n"
        "OBSERVATION: warped iris\nHYPOTHESIS: earring asymmetric\nACTION: INSPECT 8",
        "RECONCILIATION: CONFIRMED again\nBELIEF_UPDATE: P(fake)=0.9\n"
        "ACTION: VERDICT AI confidence=0.85",
    )
    total, b = compute_episode_reward(t, AI)
    assert total == pytest.approx(sum(b.values()))


def test_correct_beats_wrong_all_else_equal():
    good = traj_from_texts(
        "OBSERVATION: blur\nHYPOTHESIS: h\nACTION: INSPECT 6",
        "RECONCILIATION: CONFIRMED\nBELIEF_UPDATE: P(fake)=0.85\nACTION: VERDICT AI confidence=0.9",
    )
    correct_total, _ = compute_episode_reward(good, AI)
    wrong_total, _ = compute_episode_reward(good, REAL)  # same traj, opposite truth
    assert correct_total > wrong_total
    assert correct_total > 0 > wrong_total


def test_confident_wrong_is_net_negative():
    t = traj_from_texts(
        "OBSERVATION: blur\nHYPOTHESIS: h\nACTION: INSPECT 6",
        "RECONCILIATION: CONFIRMED\nBELIEF_UPDATE: P(fake)=0.9\nACTION: VERDICT AI confidence=0.95",
    )
    total, b = compute_episode_reward(t, REAL)  # confidently AI but truth REAL
    assert total < 0
    assert b["verdict_correct"] == -1.0
    assert b["confident_wrong"] < 0


def test_no_answer_penalized():
    t = traj_from_texts(
        "OBSERVATION: blur\nHYPOTHESIS: h\nACTION: INSPECT 6",
        "RECONCILIATION: CONFIRMED\nBELIEF_UPDATE: P(fake)=0.8\nACTION: INSPECT 8",
    )
    total, b = compute_episode_reward(t, AI)
    assert b["no_answer"] == -RewardConfig().no_answer_penalty
    assert b["verdict_correct"] == 0.0
    assert total < 0


def test_action_cost_scales_with_inspects():
    cfg = RewardConfig()
    one = traj_from_texts(
        "OBSERVATION: x\nHYPOTHESIS: h\nACTION: INSPECT 1",
        "RECONCILIATION: CONFIRMED\nBELIEF_UPDATE: P(fake)=0.8\nACTION: VERDICT AI confidence=0.7",
    )
    _, b1 = compute_episode_reward(one, AI, cfg)
    assert b1["action_cost"] == pytest.approx(-cfg.c_action * 1)


def test_reward_hacking_guard_incoherent_scores_low():
    """An adversarial 'confident but incoherent' trajectory (beliefs lurch against
    their own reconciliations, wrong verdict) must score below a faithful one."""
    faithful = traj_from_texts(
        "OBSERVATION: x\nHYPOTHESIS: h\nACTION: INSPECT 6",
        "RECONCILIATION: REFUTED, looks natural\nBELIEF_UPDATE: P(fake)=0.2\n"
        "ACTION: VERDICT REAL confidence=0.8",
    )
    incoherent = traj_from_texts(
        "OBSERVATION: x\nHYPOTHESIS: h\nACTION: INSPECT 6",
        "RECONCILIATION: REFUTED, looks natural\nBELIEF_UPDATE: P(fake)=0.9\n"
        "ACTION: VERDICT AI confidence=0.95",
    )
    faithful_total, _ = compute_episode_reward(faithful, REAL)
    incoherent_total, _ = compute_episode_reward(incoherent, REAL)
    assert faithful_total > incoherent_total


# --------------------------------------------------------------------------- #
# belief_brier — a proper scoring rule on the final P(fake)
# --------------------------------------------------------------------------- #
# GRPO run 1 stated ~0.95 confidence on essentially every episode while scoring
# 53%, because nothing in the reward read the probability the agent had already
# written down. These pin the properties that make that expensive.

def _traj_with_belief(p_fake: float, verdict: str) -> Trajectory:
    return traj_from_texts(
        "OBSERVATION: a scene\nHYPOTHESIS: if AI, cell 6 warps\nACTION: INSPECT 6",
        f"RECONCILIATION: CONFIRMED\nBELIEF_UPDATE: P(fake)={p_fake} because evidence\n"
        f"ACTION: VERDICT {verdict} confidence=0.9",
    )


def test_brier_rewards_a_confident_correct_belief():
    assert belief_brier_score(_traj_with_belief(1.0, AI), AI) == pytest.approx(1.0)


def test_brier_punishes_a_confident_wrong_belief():
    assert belief_brier_score(_traj_with_belief(1.0, AI), REAL) == pytest.approx(-1.0)


def test_brier_gives_partial_credit_for_an_honest_hedge():
    # 0.5 scores +0.5 either way — better than being confidently wrong, worse
    # than being confidently right. That asymmetry is the whole point.
    assert belief_brier_score(_traj_with_belief(0.5, AI), AI) == pytest.approx(0.5)
    assert belief_brier_score(_traj_with_belief(0.5, REAL), REAL) == pytest.approx(0.5)


def test_brier_hedge_beats_confident_wrong():
    hedge = belief_brier_score(_traj_with_belief(0.5, REAL), AI)
    loud = belief_brier_score(_traj_with_belief(0.02, REAL), AI)
    assert hedge > loud


def test_brier_is_zero_without_a_recorded_belief():
    t = traj_from_texts("OBSERVATION: a scene\nACTION: VERDICT AI confidence=0.9")
    assert belief_brier_score(t, AI) == 0.0


def test_brier_appears_in_the_breakdown_and_the_sum_still_reconciles():
    t = _traj_with_belief(0.9, AI)
    total, b = compute_episode_reward(t, AI)
    assert "belief_brier" in b
    assert total == pytest.approx(sum(b.values()))


# --------------------------------------------------------------------------- #
# confident_wrong is asymmetric by verdict
# --------------------------------------------------------------------------- #
# Finding an artifact proves AI; finding none across cells that were never opened
# proves very little. A wrong REAL is therefore the more culpable error.

def test_wrong_real_costs_more_than_wrong_ai_at_equal_confidence():
    cfg = RewardConfig()
    wrong_real = confident_wrong_penalty(_traj_with_belief(0.1, REAL), AI, cfg)
    wrong_ai = confident_wrong_penalty(_traj_with_belief(0.9, AI), REAL, cfg)
    assert wrong_real < wrong_ai < 0
    assert wrong_real == pytest.approx(wrong_ai * cfg.c_confident_wrong_real_mult)


def test_correct_verdicts_still_pay_no_confidence_penalty():
    cfg = RewardConfig()
    assert confident_wrong_penalty(_traj_with_belief(0.9, AI), AI, cfg) == 0.0
    assert confident_wrong_penalty(_traj_with_belief(0.1, REAL), REAL, cfg) == 0.0


# --------------------------------------------------------------------------- #
# prediction_tracking is retired
# --------------------------------------------------------------------------- #

def test_always_confirmed_no_longer_earns_anything():
    """Run 1 wrote CONFIRMED on 91% of reconciliations. The scorer still exists
    for the eval breakdown, but its weight is 0 so it cannot be farmed."""
    assert RewardConfig().w_prediction_tracking == 0.0
    t = _traj_with_belief(0.9, AI)
    _, b = compute_episode_reward(t, AI)
    assert b["prediction_tracking"] == 0.0
