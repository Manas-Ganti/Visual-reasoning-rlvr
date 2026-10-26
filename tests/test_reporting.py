"""Reporting tests: the collapse detector, the per-generator breakdown, and the
pre-GRPO group-variance gate.

All three are pure functions over episode-result dicts, so they are cheap to
test and worth testing — each exists to make a specific silent failure loud, and
a reporting function that quietly returns the wrong shape restores the silence.
"""

import sys

from env.reward import RewardConfig
from eval.harness import class_breakdown, compute_metrics, per_generator

sys.path.insert(0, "tools")
from group_variance_probe import outcome_component, summarize, verdict_line  # noqa: E402


def ep(index, truth, pred, reward=0.0, answered=True):
    return {
        "index": index,
        "ground_truth": truth,
        "predicted": pred,
        "correct": pred == truth,
        "answered": answered,
        "episode_reward": reward,
        "inspects_used": 1,
        "confidence": 0.8,
    }


# --------------------------------------------------------------------------- #
# class_breakdown — the failure the face substrate hid
# --------------------------------------------------------------------------- #
def test_class_collapse_is_visible():
    """Answering REAL for everything: respectable accuracy, zero signal."""
    results = [ep(i, "AI", "REAL") for i in range(57)] + [ep(i, "REAL", "REAL") for i in range(43)]
    cb = class_breakdown(results)
    assert cb["AI"]["AI"] == 0          # zero recall on the positive class
    assert cb["AI"]["REAL"] == 57
    assert cb["predicted_AI_rate"] == 0.0
    # ... while the aggregate looks merely mediocre, which is the whole problem.
    assert compute_metrics(results)["accuracy"] == 0.43


def test_class_breakdown_counts_unanswered():
    results = [ep(0, "AI", None, answered=False), ep(1, "AI", "AI")]
    cb = class_breakdown(results)
    assert cb["AI"]["no_answer"] == 1
    assert cb["predicted_AI_rate"] == 1.0  # over ANSWERED episodes only


def test_class_breakdown_on_a_healthy_split():
    results = [ep(0, "AI", "AI"), ep(1, "AI", "REAL"), ep(2, "REAL", "REAL"), ep(3, "REAL", "AI")]
    cb = class_breakdown(results)
    assert cb["AI"] == {"n": 2, "AI": 1, "REAL": 1, "no_answer": 0}
    assert cb["predicted_AI_rate"] == 0.5


# --------------------------------------------------------------------------- #
# per_generator
# --------------------------------------------------------------------------- #
def test_per_generator_separates_strong_and_blind():
    records = [{"generator": "biggan"}] * 2 + [{"generator": "midjourney"}] * 2
    results = [ep(0, "AI", "AI"), ep(1, "AI", "AI"), ep(2, "AI", "REAL"), ep(3, "AI", "REAL")]
    rows = {r["generator"]: r for r in per_generator(results, records)}
    assert rows["biggan"]["recall"] == 1.0
    assert rows["midjourney"]["recall"] == 0.0


def test_per_generator_empty_without_the_field():
    records = [{}, {}]
    assert per_generator([ep(0, "AI", "AI"), ep(1, "REAL", "REAL")], records) == []


def test_per_generator_is_sorted():
    records = [{"generator": "wukong"}, {"generator": "adm"}]
    rows = per_generator([ep(0, "AI", "AI"), ep(1, "AI", "AI")], records)
    assert [r["generator"] for r in rows] == ["adm", "wukong"]


# --------------------------------------------------------------------------- #
# group variance — the pre-GRPO gate
# --------------------------------------------------------------------------- #
CFG = RewardConfig()


def _group(index, corrects, rewards=None):
    rewards = rewards or [0.0] * len(corrects)
    return [ep(index, "AI", "AI" if c else "REAL", reward=r) for c, r in zip(corrects, rewards)]


def test_all_wrong_groups_have_no_outcome_signal():
    """The face-substrate scenario: every rollout wrong, so the outcome term is
    constant and only the process terms move the advantage."""
    groups = {i: _group(i, [False] * 8, rewards=[0.1 * j for j in range(8)]) for i in range(10)}
    s = summarize(groups, CFG)
    assert s["all_wrong"] == 1.0
    assert s["usable_groups"] == 0.0
    assert s["mean_outcome_sd"] == 0.0     # dead
    assert s["mean_total_sd"] > 0.0        # but the gradient does NOT vanish
    assert "DO NOT LAUNCH" in verdict_line(s)
    assert "too hard" in verdict_line(s)


def test_all_correct_groups_are_also_dead():
    groups = {i: _group(i, [True] * 8) for i in range(10)}
    s = summarize(groups, CFG)
    assert s["all_correct"] == 1.0
    assert s["usable_groups"] == 0.0
    assert "too easy" in verdict_line(s)


def test_mixed_groups_are_usable():
    groups = {i: _group(i, [True, True, True, True, False, False, False, False]) for i in range(10)}
    s = summarize(groups, CFG)
    assert s["usable_groups"] == 1.0
    assert s["mean_pass_rate"] == 0.5
    assert s["mean_outcome_sd"] == CFG.w_correct  # maximal spread on a 50/50 split
    assert "HEALTHY" in verdict_line(s)


def test_thin_band_is_flagged_but_not_blocked():
    groups = {i: _group(i, [True] * 8) for i in range(8)}
    groups.update({i: _group(i, [True] * 6 + [False] * 2) for i in range(8, 10)})
    s = summarize(groups, CFG)
    assert s["usable_groups"] == 0.2
    assert "THIN" in verdict_line(s)


def test_outcome_component_is_symmetric():
    assert outcome_component(True, CFG) == CFG.w_correct
    assert outcome_component(False, CFG) == -CFG.w_correct


def test_summarize_counts_groups_and_rollouts():
    groups = {i: _group(i, [True, False]) for i in range(3)}
    s = summarize(groups, CFG)
    assert s["groups"] == 3 and s["rollouts"] == 6
