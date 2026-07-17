"""Edge-case tests for the pre/post trajectory parser."""

from env.trajectory import (
    AI,
    CONFIRMED,
    INSPECT,
    INVALID,
    REAL,
    REFUTED,
    UNCLEAR,
    VERDICT,
    Trajectory,
    classify_reconciliation,
    normalize_verdict,
    parse_turn,
)


# --------------------------------------------------------------------------- #
# Field parsing
# --------------------------------------------------------------------------- #
def test_first_turn_inspect_no_post_fields():
    e = parse_turn(
        "OBSERVATION: A blurry face, roughly centered.\n"
        "REASONING: Eyes are where GAN artifacts show first.\n"
        "HYPOTHESIS: If AI, the left iris in cell 6 will be misshapen.\n"
        "ACTION: INSPECT 6"
    )
    assert e.action_type == INSPECT
    assert e.cell == 6
    assert e.reconciliation == UNCLEAR  # nothing to reconcile on turn 1
    assert e.p_fake is None
    assert "iris" in e.hypothesis
    assert "centered" in e.observation


def test_later_turn_reconcile_belief_inspect():
    e = parse_turn(
        "RECONCILIATION: CONFIRMED - the iris really is malformed as predicted.\n"
        "BELIEF_UPDATE: P(fake)=0.75 because the artifact is clear.\n"
        "OBSERVATION: The reveal showed a warped iris edge.\n"
        "REASONING: Check the ear for a second independent artifact.\n"
        "HYPOTHESIS: If AI, the earring in cell 8 will be asymmetric.\n"
        "ACTION: INSPECT 8"
    )
    assert e.action_type == INSPECT and e.cell == 8
    assert e.reconciliation == CONFIRMED
    assert e.p_fake == 0.75


def test_verdict_with_confidence():
    e = parse_turn(
        "RECONCILIATION: CONFIRMED once more.\n"
        "BELIEF_UPDATE: P(fake)=0.9\n"
        "OBSERVATION: Multiple artifacts.\n"
        "REASONING: Enough evidence.\n"
        "HYPOTHESIS: none needed.\n"
        "ACTION: VERDICT AI confidence=0.9"
    )
    assert e.is_terminal and e.action_type == VERDICT
    assert e.verdict == AI
    assert e.confidence == 0.9


def test_tolerant_action_variants():
    assert parse_turn("ACTION: inspect(11)").cell == 11
    assert parse_turn("action inspect 3").cell == 3
    v = parse_turn("ACTION: VERDICT: REAL confidence 0.4")
    assert v.verdict == REAL and v.confidence == 0.4


def test_only_last_action_line_honored():
    e = parse_turn(
        "REASONING: first I will INSPECT then maybe VERDICT.\n"
        "ACTION: INSPECT 2\n"
        "ACTION: VERDICT AI confidence=0.6"
    )
    assert e.action_type == VERDICT and e.verdict == AI


def test_invalid_when_no_action():
    e = parse_turn("OBSERVATION: I am unsure what to do.")
    assert e.action_type == INVALID
    assert e.cell is None and e.verdict is None


def test_unrecognized_verdict_token_is_invalid():
    assert parse_turn("ACTION: VERDICT maybe confidence=0.5").action_type == INVALID


def test_confidence_defaults_to_none_when_absent():
    assert parse_turn("ACTION: VERDICT REAL").confidence is None


# --------------------------------------------------------------------------- #
# Reconciliation classification
# --------------------------------------------------------------------------- #
def test_classify_reconciliation():
    assert classify_reconciliation("CONFIRMED, the iris is malformed") == CONFIRMED
    assert classify_reconciliation("REFUTED, it looks natural") == REFUTED
    assert classify_reconciliation("The cell was uninformative.") == UNCLEAR
    assert classify_reconciliation("") == UNCLEAR


def test_refute_beats_confirm_on_negation():
    # "not confirmed" must read as refuted, not confirmed.
    assert classify_reconciliation("this did not confirm my hypothesis") == REFUTED


# --------------------------------------------------------------------------- #
# P(fake) extraction
# --------------------------------------------------------------------------- #
def test_pfake_variants():
    assert parse_turn("BELIEF_UPDATE: P(fake)=0.3\nACTION: INSPECT 1").p_fake == 0.3
    assert parse_turn("BELIEF_UPDATE: P(fake): .8\nACTION: INSPECT 1").p_fake == 0.8
    # bare probability fallback
    assert parse_turn("BELIEF_UPDATE: now around 0.6 fake\nACTION: INSPECT 1").p_fake == 0.6


def test_pfake_clamped():
    assert parse_turn("BELIEF_UPDATE: P(fake)=1.0\nACTION: INSPECT 1").p_fake == 1.0


def test_normalize_verdict():
    assert normalize_verdict("fake") == AI
    assert normalize_verdict("Genuine") == REAL
    assert normalize_verdict("banana") is None


# --------------------------------------------------------------------------- #
# Trajectory accumulation
# --------------------------------------------------------------------------- #
def _traj(*texts) -> Trajectory:
    t = Trajectory()
    for x in texts:
        t.add(parse_turn(x))
    return t


def test_trajectory_derived_views():
    t = _traj(
        "OBSERVATION: x\nHYPOTHESIS: h\nACTION: INSPECT 5",
        "RECONCILIATION: CONFIRMED\nBELIEF_UPDATE: P(fake)=0.7\nACTION: INSPECT 6",
        "RECONCILIATION: REFUTED\nBELIEF_UPDATE: P(fake)=0.55\nACTION: VERDICT AI confidence=0.6",
    )
    assert t.num_inspects() == 2
    assert t.answered is True
    assert t.final_verdict == AI
    assert t.final_confidence == 0.6
    assert t.belief_series() == [0.7, 0.55]
    assert t.final_belief() == 0.55
    # reconciliations exclude the first turn
    assert t.reconciliations() == [CONFIRMED, REFUTED]


def test_belief_steps_seed_from_prior():
    t = _traj(
        "OBSERVATION: x\nACTION: INSPECT 5",
        "RECONCILIATION: CONFIRMED\nBELIEF_UPDATE: P(fake)=0.7\nACTION: VERDICT AI confidence=0.8",
    )
    steps = t.belief_steps()
    assert len(steps) == 1
    recon, prev, new = steps[0]
    assert recon == CONFIRMED and prev == t.prior == 0.5 and new == 0.7


def test_unanswered_trajectory():
    t = _traj("OBSERVATION: x\nACTION: INSPECT 5")
    assert t.answered is False
    assert t.final_verdict is None
