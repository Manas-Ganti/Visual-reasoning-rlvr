"""Grid geometry for the inspect action.

The image is divided into a ``grid × grid`` lattice of cells numbered row-major
starting at 1. With the default 4×4 grid over a 300×300 face image::

     1  2  3  4
     5  6  7  8
     9 10 11 12
    13 14 15 16

``inspect(cell)`` crops the *full-resolution* pixels of that cell and hands the
agent a high-res reveal — the only way to sharpen a region beyond the blurred
overview it starts with. Keeping this math in one small, pure module makes it
trivial to unit-test and to reuse from the environment, the evidence-slice
builder (which maps GradCAM peaks back to a cell), and the demo renderer.
"""

from __future__ import annotations

from typing import Optional

from PIL import Image


def num_cells(grid: int = 4) -> int:
    return grid * grid


def cell_rowcol(cell: int, grid: int = 4) -> tuple[int, int]:
    """(row, col), both 0-indexed, for a 1-indexed row-major ``cell``."""
    if not 1 <= cell <= grid * grid:
        raise ValueError(f"cell {cell} out of range for {grid}x{grid} grid")
    return divmod(cell - 1, grid)


def cell_bbox(width: int, height: int, cell: int, grid: int = 4):
    """Return the ``(left, upper, right, lower)`` pixel box for ``cell``.

    Boundaries are rounded (not floored) so the rightmost / bottommost cells
    reach the true image edge even when the dimension isn't divisible by
    ``grid`` — otherwise a strip of pixels would never be inspectable.
    """
    row, col = cell_rowcol(cell, grid)
    left = round(col * width / grid)
    right = round((col + 1) * width / grid)
    upper = round(row * height / grid)
    lower = round((row + 1) * height / grid)
    return left, upper, right, lower


def point_to_cell(x: int, y: int, width: int, height: int, grid: int = 4) -> int:
    """Map a pixel coordinate onto its 1-indexed cell. Used to turn a GradCAM
    saliency peak into the grid cell it lands in (evidence-slice builder)."""
    col = min(grid - 1, max(0, int(x * grid / max(width, 1))))
    row = min(grid - 1, max(0, int(y * grid / max(height, 1))))
    return row * grid + col + 1


def crop_cell(
    image: Image.Image,
    cell: int,
    grid: int = 4,
    upscale_to: Optional[int] = 336,
) -> Image.Image:
    """Crop ``cell`` from ``image`` and upscale it to a fixed reveal size.

    Upscaling the small native crop (a 4×4 cell of a 300px image is ~75px) back
    up to ``upscale_to`` on its long edge is what makes ``inspect`` a *zoom*: the
    VLM receives the same region at higher effective resolution, surfacing
    generation artifacts (warped hair strands, iris asymmetry, fused earrings)
    that are invisible in the blurred overview. Set ``upscale_to=None`` to keep
    native crop size.
    """
    box = cell_bbox(image.width, image.height, cell, grid)
    crop = image.crop(box)
    if upscale_to:
        w, h = crop.size
        longest = max(w, h)
        if 0 < longest != upscale_to:
            scale = upscale_to / longest
            crop = crop.resize(
                (max(1, round(w * scale)), max(1, round(h * scale))),
                Image.Resampling.LANCZOS,
            )
    return crop


def make_overview(
    image: Image.Image, long_edge: int = 140, restore_to: Optional[int] = None
) -> Image.Image:
    """Build the low-resolution overview shown at reset (partial observability).

    Downsamples to ``long_edge`` so fine artifacts are *not* resolvable from the
    overview alone — the agent must spend inspect budget to sharpen regions. If
    ``restore_to`` is given the thumbnail is scaled back up to that long edge
    (bicubic) so the VLM's image processor sees a consistent input size while the
    detail stays destroyed. This is what makes the correct answer unreachable
    without investigation.
    """
    w, h = image.size
    longest = max(w, h)
    scale = long_edge / longest if longest > long_edge else 1.0
    small = image.resize(
        (max(1, round(w * scale)), max(1, round(h * scale))),
        Image.Resampling.BILINEAR,
    )
    if restore_to:
        s = restore_to / max(small.size)
        small = small.resize(
            (max(1, round(small.width * s)), max(1, round(small.height * s))),
            Image.Resampling.BICUBIC,
        )
    return small
