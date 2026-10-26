"""Shortcut detection and split assignment for the HF streaming builder.

The shortcut warnings are the reason this builder exists. If real and fake
differ in shape or size, the label is predictable WITHOUT looking at the image,
that difference survives the environment's overview downsample untouched, and a
high pass rate would mean nothing. These tests pin the cases that matter.
"""

import pytest

from data.build_manifest_hf import assign_splits, shortcut_warnings, summarize_dims


def square(n, size=1024):
    return [(size, size)] * n


def landscape(n, w=500, h=375):
    return [(w, h)] * n


# --------------------------------------------------------------------------- #
# summarize_dims
# --------------------------------------------------------------------------- #
def test_summary_of_squares():
    s = summarize_dims(square(10))
    assert s["n"] == 10
    assert s["short_med"] == 1024
    assert s["aspect_med"] == 1.0
    assert s["square_frac"] == 1.0


def test_summary_of_landscapes():
    s = summarize_dims(landscape(10))
    assert s["short_med"] == 375
    assert s["aspect_med"] == pytest.approx(1.333, abs=1e-3)
    assert s["square_frac"] == 0.0


def test_summary_of_empty():
    assert summarize_dims([]) == {}


# --------------------------------------------------------------------------- #
# shortcut_warnings — the ones that would sink a run
# --------------------------------------------------------------------------- #
def test_midjourney_vs_imagenet_is_flagged():
    """The exact pairing that prompted this: 1024 squares vs ~500x375 photos."""
    w = shortcut_warnings(summarize_dims(landscape(50)), summarize_dims(square(50)))
    assert w, "a square/rectangle split must not pass silently"
    joined = " ".join(w)
    assert "aspect" in joined
    assert "squareness" in joined
    assert "disjoint" in joined             # trivially separable
    assert "resolution" in joined           # 375 vs 1024 is >2x


def test_matched_pair_is_clean():
    """Both classes 1024 square — e.g. celeb-a-hq vs an SDXL render of it."""
    assert shortcut_warnings(summarize_dims(square(50)), summarize_dims(square(50))) == []


def test_same_source_shapes_are_clean():
    """A diffusion reconstruction of the real image: identical geometry."""
    dims = landscape(30) + landscape(20, 640, 480)
    assert shortcut_warnings(summarize_dims(dims), summarize_dims(dims)) == []


def test_mild_aspect_difference_is_tolerated():
    real = summarize_dims(landscape(50, 500, 400))   # 1.25
    fake = summarize_dims(landscape(50, 512, 400))   # 1.28
    assert shortcut_warnings(real, fake) == []


def test_resolution_gap_alone_is_flagged():
    """Same shape, very different size — softer, but still worth a look."""
    real = summarize_dims(landscape(50, 400, 300))
    fake = summarize_dims(landscape(50, 1200, 900))
    w = shortcut_warnings(real, fake)
    assert len(w) == 1 and "resolution" in w[0]


def test_missing_class_produces_no_warnings():
    assert shortcut_warnings({}, summarize_dims(square(5))) == []


# --------------------------------------------------------------------------- #
# assign_splits — must match build_manifest_genimage's stratification
# --------------------------------------------------------------------------- #
def _rows(n_per_class, generator="flux"):
    return [{"generator": generator, "label": lab, "id": f"{lab}_{i}"}
            for lab in (0, 1) for i in range(n_per_class)]


def test_every_stratum_reaches_test():
    rows = _rows(100)
    assign_splits(rows, 0.1, 0.1, seed=0)
    for label in (0, 1):
        splits = {r["split"] for r in rows if r["label"] == label}
        assert splits == {"train", "val", "test"}


def test_split_proportions():
    rows = _rows(100)
    assign_splits(rows, 0.1, 0.1, seed=0)
    from collections import Counter
    c = Counter(r["split"] for r in rows)
    assert c["train"] == 160 and c["val"] == 20 and c["test"] == 20


def test_tiny_strata_stay_in_train():
    """Under 10 rows there is nothing to spare for held-out splits."""
    rows = _rows(4)
    assign_splits(rows, 0.1, 0.1, seed=0)
    assert all(r["split"] == "train" for r in rows)


def test_assignment_is_deterministic():
    a, b = _rows(50), _rows(50)
    assign_splits(a, 0.1, 0.1, seed=0)
    assign_splits(b, 0.1, 0.1, seed=0)
    assert [r["split"] for r in a] == [r["split"] for r in b]


def test_generators_are_stratified_independently():
    rows = _rows(50, "flux") + _rows(50, "sdxl")
    assign_splits(rows, 0.1, 0.1, seed=0)
    for gen in ("flux", "sdxl"):
        test_rows = [r for r in rows if r["generator"] == gen and r["split"] == "test"]
        assert len(test_rows) == 10


# --------------------------------------------------------------------------- #
# report_dimensions — resolution is only meaningful relative to the overview
# --------------------------------------------------------------------------- #
def _report(real_dims, fake_dims, min_edge, overview, capsys):
    from data.build_manifest_hf import add_pass_frac, report_dimensions

    report_dimensions(add_pass_frac(summarize_dims(real_dims), min_edge),
                      add_pass_frac(summarize_dims(fake_dims), min_edge),
                      min_edge, overview)
    return capsys.readouterr().out


def test_same_images_pass_or_fail_on_the_overview_setting(capsys):
    """GenImage SD1.4 (512) + ImageNet (~375): unusable at 140, fine at 64.
    There is no absolute pixel threshold — only the ratio matters."""
    real, fake = landscape(30, 500, 375), square(30, 512)

    at_140 = _report(real, fake, 384, 140, capsys)
    assert "too low" in at_140
    assert "--overview-long-edge" in at_140      # tells you what would work

    at_64 = _report(real, fake, 384, 64, capsys)
    assert "too low" not in at_64


def test_faces_fail_at_any_sane_overview(capsys):
    """300px with 75px cells: the ratio can be rescued, the absolute detail
    cannot. Blurring harder does raise the gain, which is exactly why the gain
    number alone is not the whole story."""
    out = _report(landscape(20, 300, 300), landscape(20, 300, 300), 256, 140, capsys)
    assert "too low" in out
    assert "cell 75px" in out


def test_gain_is_reported_per_class(capsys):
    out = _report(square(10, 1024), square(10, 1024), 512, 140, capsys)
    assert "7.3x" in out
    assert "too low" not in out
