"""vLLM must never silently score a partly-applied adapter."""

import json
import re
import struct

import pytest

from training import common, vllm_backend as vb

LANG_OLD = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
LANG_NEW = "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"
VISION = "base_model.model.model.visual.blocks.0.mlp.up_proj.lora_B.weight"


def _write(path, keys):
    header = json.dumps({"__metadata__": {}, **{
        k: {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]} for k in keys}}).encode()
    (path / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    return str(path)


def test_only_names_vllm_resolves_survive():
    assert vb.keys_vllm_drops([LANG_OLD, LANG_NEW, VISION]) == [LANG_NEW, VISION]


@pytest.mark.parametrize("bad", [LANG_NEW, VISION])
def test_partly_served_adapter_is_refused(tmp_path, bad):
    with pytest.raises(SystemExit, match="silently drop 1 of 2"):
        vb.require_vllm_can_serve(_write(tmp_path, [LANG_OLD, bad]))


def test_servable_adapter_passes(tmp_path):
    vb.require_vllm_can_serve(_write(tmp_path, [LANG_OLD]))


def test_unverifiable_adapter_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="Cannot verify"):
        vb.require_vllm_can_serve(str(tmp_path))


def test_lora_excludes_the_vision_tower_only():
    hit = lambda n: bool(re.fullmatch(common.VISION_MODULES, n))
    assert hit("model.visual.blocks.0.mlp.up_proj") and hit("visual.blocks.0.mlp.gate_proj")
    assert not hit("model.language_model.layers.0.mlp.up_proj")
