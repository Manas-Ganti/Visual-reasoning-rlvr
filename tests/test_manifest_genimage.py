"""Generator-name normalisation for the GenImage manifest builder.

The official release names its Stable Diffusion directories `sdv4` / `sdv5`
while mirrors ship `stable_diffusion_v_1_4`. Both must normalise to the same
canonical name, or `--generators sdv1.4` silently selects nothing and the
builder aborts on a layout that is perfectly fine.
"""

import pytest

from data.build_manifest_genimage import _generator_from_path, _label_from_path


@pytest.mark.parametrize("dirname,expected", [
    # Official release layout.
    ("imagenet_ai_0419_biggan", "biggan"),
    ("imagenet_ai_0419_sdv4", "sdv1.4"),
    ("imagenet_ai_0424_sdv5", "sdv1.5"),
    ("imagenet_ai_0424_wukong", "wukong"),
    ("imagenet_ai_0508_adm", "adm"),
    ("imagenet_glide", "glide"),
    ("imagenet_midjourney", "midjourney"),
    ("imagenet_vqdm", "vqdm"),
    # Mirror spellings.
    ("stable_diffusion_v_1_4", "sdv1.4"),
    ("stable_diffusion_v_1_5", "sdv1.5"),
    ("sdv1.4", "sdv1.4"),
    ("SDv1.5", "sdv1.5"),
])
def test_generator_names_normalise(dirname, expected):
    path = f"/data/GenImage/{dirname}/train/ai"
    assert _generator_from_path("/data/GenImage", path) == expected


def test_all_eight_generators_are_recognised():
    """No official-layout directory may fall through to the raw name."""
    official = [
        "imagenet_ai_0419_biggan", "imagenet_ai_0419_sdv4", "imagenet_ai_0424_sdv5",
        "imagenet_ai_0424_wukong", "imagenet_ai_0508_adm", "imagenet_glide",
        "imagenet_midjourney", "imagenet_vqdm",
    ]
    names = {_generator_from_path("/r", f"/r/{d}/train/ai") for d in official}
    assert len(names) == 8, f"collision or fallthrough: {sorted(names)}"
    assert not any(n.startswith("imagenet") for n in names), sorted(names)


def test_unknown_directory_falls_back_to_its_own_name():
    assert _generator_from_path("/r", "/r/some_new_model/train/ai") == "some_new_model"


@pytest.mark.parametrize("path,label", [
    ("/r/imagenet_midjourney/train/ai", 1),
    ("/r/imagenet_midjourney/train/nature", 0),
    ("/r/imagenet_glide/val/fake", 1),
    ("/r/imagenet_glide/val/real", 0),
    ("/r/imagenet_glide/val", None),
])
def test_class_folders(path, label):
    assert _label_from_path(path) == label
