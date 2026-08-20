"""Domain registry tests — the guardrail against a substrate switch leaving the
prompt behind.

The face artifact checklist survived the move to GenImage once already (it names
irises, teeth and earrings, which cannot be found on an ImageNet photograph of a
dog). These tests make that failure mode loud.
"""

import json
import pathlib

import pytest
from PIL import Image

from env import prompts
from env.environment import InvestigationEnv

FACE_ONLY_TERMS = ("iris", "irises", "earring", "teeth", "gum", "hairline", "skin")


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_every_dataset_maps_to_a_known_domain():
    for dataset, domain in prompts.DATASET_DOMAIN.items():
        assert domain in prompts.DOMAINS, f"{dataset} -> unknown domain {domain}"


def test_default_domain_is_known():
    assert prompts.DEFAULT_DOMAIN in prompts.DOMAINS


def test_genimage_resolves_to_image_domain():
    assert prompts.resolve_domain("genimage") == "image"


def test_faces_resolves_to_face_domain():
    assert prompts.resolve_domain("faces") == "face"


def test_unknown_dataset_falls_back_to_generic_domain():
    # The generic checklist is harmless on a new substrate; the face one is not.
    assert prompts.resolve_domain("some-new-substrate") == prompts.DEFAULT_DOMAIN


def test_resolve_domain_reads_env_var(monkeypatch):
    monkeypatch.setenv("VRR_DATASET", "faces")
    assert prompts.resolve_domain() == "face"
    monkeypatch.setenv("VRR_DATASET", "genimage")
    assert prompts.resolve_domain() == "image"


def test_env_var_default_mirrors_training_common():
    """prompts/ cannot import training/ (circular), so the default is duplicated.
    This is the assertion that keeps the two copies in step."""
    import re

    src = (pathlib.Path(__file__).parent.parent / "training" / "common.py").read_text()
    m = re.search(r'DATASET = \(os\.environ\.get\("VRR_DATASET"\) or "([^"]+)"\)', src)
    assert m, "could not find the DATASET default in training/common.py"
    assert prompts._DEFAULT_DATASET == m.group(1)


def test_unknown_domain_raises():
    with pytest.raises(KeyError):
        prompts.get_domain("not-a-domain")


# --------------------------------------------------------------------------- #
# Prompt content
# --------------------------------------------------------------------------- #
def test_image_domain_prompt_has_no_face_artifacts():
    text = prompts.system_prompt("image").lower()
    for term in FACE_ONLY_TERMS:
        assert term not in text, f"face-only term {term!r} leaked into the image prompt"


def test_image_domain_prompt_names_diffusion_artifacts():
    text = prompts.system_prompt("image").lower()
    for term in ("hands", "text", "geometry", "shadow"):
        assert term in text


def test_face_domain_prompt_still_has_face_artifacts():
    # The retired substrate stays reproducible.
    text = prompts.system_prompt("face").lower()
    assert "iris" in text and "earring" in text


def test_system_prompt_always_carries_the_format_spec():
    for domain in prompts.DOMAINS:
        text = prompts.system_prompt(domain)
        for field in ("OBSERVATION:", "REASONING:", "HYPOTHESIS:", "ACTION:",
                      "RECONCILIATION:", "BELIEF_UPDATE:"):
            assert field in text, f"{field} missing from {domain} prompt"
        assert "INSPECT <n>" in text and "VERDICT <AI|REAL>" in text


def test_hypothesis_example_matches_its_domain():
    assert "iris" in prompts.system_prompt("face")
    assert "iris" not in prompts.system_prompt("image")


def test_classify_prompt_is_domain_specific_and_one_word():
    assert "face" in prompts.classify_prompt("face")
    assert "face" not in prompts.classify_prompt("image")
    for domain in prompts.DOMAINS:
        assert "one word" in prompts.classify_prompt(domain).lower()


def test_no_module_level_prompt_constant():
    """A global resolved at import time is how the face checklist survived the
    substrate switch. It must stay gone."""
    assert not hasattr(prompts, "SYSTEM_PROMPT_FULL")


# --------------------------------------------------------------------------- #
# Environment wiring
# --------------------------------------------------------------------------- #
@pytest.fixture
def manifest(tmp_path):
    rows = []
    for i, label in enumerate((1, 0)):
        img = Image.new("RGB", (600, 600), (i * 10, 50, 90))
        p = tmp_path / f"img_{i}.png"
        img.save(p)
        rows.append({"id": f"s{i}", "file_name": str(p), "label": label, "split": "test"})
    mpath = tmp_path / "manifest.jsonl"
    mpath.write_text("\n".join(json.dumps(r) for r in rows))
    return str(mpath)


def test_env_uses_dataset_domain(manifest):
    env = InvestigationEnv(manifest_path=manifest, shuffle=False, dataset="genimage")
    assert env.domain == "image"
    obs, _ = env.reset(options={"index": 0})
    system = obs["messages"][0]["content"][0]["text"].lower()
    for term in FACE_ONLY_TERMS:
        assert term not in system


def test_env_explicit_domain_overrides_dataset(manifest):
    env = InvestigationEnv(manifest_path=manifest, shuffle=False,
                           dataset="genimage", domain="face")
    assert env.domain == "face"
    obs, _ = env.reset(options={"index": 0})
    assert "iris" in obs["messages"][0]["content"][0]["text"]
