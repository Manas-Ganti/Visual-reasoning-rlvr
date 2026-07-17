"""Environment transition / budget / grid tests. Uses a synthetic manifest so it
needs no dataset download (CI-friendly)."""

import json
import os

import pytest
from PIL import Image

from data import degradation as degrade
from env import grid
from env.environment import InvestigationEnv


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def manifest(tmp_path):
    rows = []
    for i, label in enumerate((1, 0)):
        img = Image.new("RGB", (300, 300), (i * 10, 50, 90))
        p = tmp_path / f"img_{i}.png"
        img.save(p)
        rows.append({"id": f"s{i}", "file_name": str(p), "label": label, "split": "test"})
    mpath = tmp_path / "manifest.jsonl"
    mpath.write_text("\n".join(json.dumps(r) for r in rows))
    return str(mpath)


def _env(manifest, **kw):
    return InvestigationEnv(manifest_path=manifest, shuffle=False, **kw)


# --------------------------------------------------------------------------- #
# Grid geometry
# --------------------------------------------------------------------------- #
def test_num_cells():
    assert grid.num_cells(4) == 16


def test_cell_bbox_covers_edges():
    # cell 16 (bottom-right) of a 4x4 grid must reach the true image edge.
    assert grid.cell_bbox(300, 300, 16, 4) == (225, 225, 300, 300)
    assert grid.cell_bbox(300, 300, 1, 4) == (0, 0, 75, 75)


def test_cell_bbox_out_of_range():
    with pytest.raises(ValueError):
        grid.cell_bbox(300, 300, 17, 4)


def test_point_to_cell():
    assert grid.point_to_cell(10, 10, 300, 300, 4) == 1
    assert grid.point_to_cell(290, 290, 300, 300, 4) == 16
    assert grid.point_to_cell(150, 10, 300, 300, 4) == 3


def test_crop_and_overview_sizes():
    img = Image.new("RGB", (300, 300))
    crop = grid.crop_cell(img, 6, grid=4, upscale_to=336)
    assert max(crop.size) == 336
    ov = grid.make_overview(img, long_edge=140, restore_to=300)
    assert max(ov.size) == 300  # restored size, but detail destroyed


# --------------------------------------------------------------------------- #
# Degradation
# --------------------------------------------------------------------------- #
def test_degradation_levels_run():
    img = Image.new("RGB", (300, 300), (120, 120, 120))
    for level in degrade.LEVELS:
        out = degrade.apply(img, level)
        assert out.size == (300, 300)
    with pytest.raises(ValueError):
        degrade.apply(img, "bogus")


# --------------------------------------------------------------------------- #
# Environment transitions
# --------------------------------------------------------------------------- #
def test_reset_shows_overview_only(manifest):
    env = _env(manifest)
    obs, info = env.reset(options={"index": 0})
    assert len(obs["images"]) == 1
    assert obs["inspects_remaining"] == 4
    assert info["ground_truth"] == "AI"  # label 1


def test_inspect_reveals_and_costs_budget(manifest):
    env = _env(manifest)
    env.reset(options={"index": 0})
    obs, r, term, trunc, info = env.step("HYPOTHESIS: x\nACTION: INSPECT 6")
    assert not term and not trunc
    assert len(obs["images"]) == 2
    assert obs["inspects_remaining"] == 3
    assert info["inspected_cells"] == [6]
    assert r == 0.0  # reward credited only at episode end


def test_repeated_inspect_wastes_no_budget(manifest):
    env = _env(manifest)
    env.reset(options={"index": 0})
    env.step("ACTION: INSPECT 6")
    obs, r, term, trunc, info = env.step("ACTION: INSPECT 6")
    assert obs["inspects_remaining"] == 3  # unchanged
    assert len(obs["images"]) == 2         # no new reveal


def test_correct_verdict_positive_reward(manifest):
    env = _env(manifest)
    env.reset(options={"index": 0})  # AI
    obs, r, term, trunc, info = env.step("ACTION: VERDICT AI confidence=0.8")
    assert term
    assert info["correct"] is True
    assert r > 0


def test_wrong_confident_verdict_negative_reward(manifest):
    env = _env(manifest)
    env.reset(options={"index": 0})  # AI
    obs, r, term, trunc, info = env.step("ACTION: VERDICT REAL confidence=0.9")
    assert term
    assert info["correct"] is False
    assert r < 0
    assert info["reward_breakdown"]["confident_wrong"] < 0


def test_budget_exhaustion_then_truncation(manifest):
    env = _env(manifest, max_inspects=2)
    env.reset(options={"index": 0})
    term = trunc = False
    steps = 0
    # Never emit a verdict; env must truncate (no-answer) without hanging.
    while not (term or trunc) and steps < 20:
        obs, r, term, trunc, info = env.step(f"ACTION: INSPECT {steps % 16 + 1}")
        steps += 1
    assert trunc
    assert info["inspects_used"] <= 2  # budget respected
    assert info["reward_breakdown"]["no_answer"] < 0


def test_invalid_action_gives_feedback(manifest):
    env = _env(manifest)
    env.reset(options={"index": 0})
    obs, r, term, trunc, info = env.step("I am not sure.")
    assert info["action_type"] == "invalid"
    assert not term
    last = obs["messages"][-1]
    assert last["role"] == "user"


def test_get_trace_schema(manifest):
    env = _env(manifest)
    env.reset(options={"index": 0})
    env.step("HYPOTHESIS: iris\nACTION: INSPECT 6")
    env.step("RECONCILIATION: CONFIRMED\nBELIEF_UPDATE: P(fake)=0.8\nACTION: VERDICT AI confidence=0.7")
    trace = env.get_trace(global_step=3, phase="train")
    assert trace["answered"] and trace["prediction"] == "AI"
    assert trace["correct"] is True
    assert len(trace["turns"]) == 2
    assert set(["verdict_correct", "action_cost"]).issubset(trace["reward_breakdown"])
