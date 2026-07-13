"""Edge-case tests for the verifiable reward. These are the tests CI runs."""

import pytest

from env.reward import (
    RewardConfig,
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
