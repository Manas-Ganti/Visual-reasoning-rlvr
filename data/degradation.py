"""Image degradation — one of the two honest difficulty axes (the other being
inspect-budget tightness).

The substrate is StyleGAN2-only, so there are no multi-generator tiers to lean
on. Instead we make the *same* images harder by degrading them: JPEG compression
and blur/downscale both erode the fine, high-frequency cues (hair strands, iris
edges, skin micro-texture) that betray a GAN face. The eval harness reports pass
rate per degradation level, and Stage-3 verification holds out an unseen level to
prove the policy generalizes to difficulty it never trained on.

Every transform is deterministic given ``(image, level)`` so a manifest row +
level fully reproduces the pixels the agent saw. Applied consistently to the
overview AND every inspect crop by the environment.
"""

from __future__ import annotations

import io

from PIL import Image, ImageFilter

# Ordered easy -> hard. Names are the manifest/CLI level identifiers.
LEVELS: tuple[str, ...] = ("clean", "jpeg", "blur_downscale")


def apply(image: Image.Image, level: str = "clean") -> Image.Image:
    """Return a degraded copy of ``image`` at difficulty ``level``.

    - ``clean``          : untouched (a defensive copy).
    - ``jpeg``           : re-encode at low JPEG quality, injecting block/ringing
                           artifacts that mask subtle generation tells.
    - ``blur_downscale`` : halve resolution then restore + light Gaussian blur,
                           destroying high-frequency detail.
    """
    if level == "clean":
        return image.copy()
    if level == "jpeg":
        return _jpeg(image, quality=30)
    if level == "blur_downscale":
        return _blur_downscale(image, scale=0.5, blur_radius=0.8)
    raise ValueError(f"unknown degradation level {level!r}; expected one of {LEVELS}")


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _blur_downscale(image: Image.Image, scale: float, blur_radius: float) -> Image.Image:
    w, h = image.size
    small = image.resize(
        (max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.BILINEAR
    )
    restored = small.resize((w, h), Image.Resampling.BICUBIC)
    return restored.filter(ImageFilter.GaussianBlur(radius=blur_radius))
