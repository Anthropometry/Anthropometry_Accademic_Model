"""Silhouette preparation and input checks.

Decision (DISCUSSION D-011): crop each silhouette to the body, pad to a 2:1 frame without
stretching, then resize to IMG_H x IMG_W.
- Why crop: BodyM subjects stood 5.5-6.5 ft from the camera, and our users will stand
  wherever they stand. Cropping removes distance from the picture so the network learns body
  SHAPE, not how big the person happened to appear.
- Where real size comes from instead: the height input. That is why height is required and
  must be measured, not guessed — a 2% height error makes every measurement ~2% off.
- Why pad, not stretch: stretching a wide body into a narrow frame changes its proportions,
  which are exactly what the model must read.

Input checks: the model was trained on clean, single-person, full-body silhouettes. Anything
else (an empty mask, two people, a cut-off body) gives a confident but meaningless answer. We
refuse those inputs rather than return wrong numbers (docs/SECURITY.md, "input validation").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .config import (
    HEIGHT_LIMITS_CM,
    HEIGHT_TRAIN_RANGE_CM,
    WEIGHT_LIMITS_KG,
    WEIGHT_TRAIN_RANGE_KG,
)

# A full-body silhouette normally covers a few % to ~40% of the frame.
_MIN_AREA, _MAX_AREA = 0.01, 0.6
_MARGIN = 0.03           # 3% of body height around the crop, so edges are not clipped
_MAX_PIXELS = 40_000_000  # refuse absurdly large images (memory exhaustion via upload)


@dataclass
class CheckResult:
    ok: bool
    problems: list[str] = field(default_factory=list)   # input refused
    warnings: list[str] = field(default_factory=list)   # input accepted, result less certain


def load_mask(path: str | Path) -> np.ndarray:
    """Load a silhouette as a boolean array (True = body)."""
    with Image.open(path) as im:
        if im.width * im.height > _MAX_PIXELS:
            raise ValueError(f"image too large: {im.width}x{im.height}")
        return np.asarray(im.convert("L")) > 127


def check_silhouette(mask: np.ndarray) -> CheckResult:
    r = CheckResult(ok=True)
    area = mask.mean()
    if area < _MIN_AREA:
        r.problems.append(f"almost no body visible ({area:.1%} of the image)")
    elif area > _MAX_AREA:
        r.problems.append(f"body fills {area:.0%} of the image: too close, or not a silhouette")
    if mask.size and (mask[0].any() or mask[-1].any()):
        r.warnings.append("body touches the top or bottom edge: head or feet may be cut off")
    r.ok = not r.problems
    return r


def check_inputs(height_cm: float, weight_kg: float | None) -> CheckResult:
    r = CheckResult(ok=True)
    lo, hi = HEIGHT_LIMITS_CM
    if not lo <= height_cm <= hi:
        r.problems.append(f"height {height_cm} cm is outside {lo:g}-{hi:g} cm")
    elif not HEIGHT_TRAIN_RANGE_CM[0] <= height_cm <= HEIGHT_TRAIN_RANGE_CM[1]:
        r.warnings.append("height is outside the training range: treat results with caution")
    if weight_kg is not None:
        lo, hi = WEIGHT_LIMITS_KG
        if not lo <= weight_kg <= hi:
            r.problems.append(f"weight {weight_kg} kg is outside {lo:g}-{hi:g} kg")
        elif not WEIGHT_TRAIN_RANGE_KG[0] <= weight_kg <= WEIGHT_TRAIN_RANGE_KG[1]:
            r.warnings.append("weight is outside the training range: treat results with caution")
    r.ok = not r.problems
    return r


def crop_pad_resize(mask: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    """Body-centred, aspect-preserving, uint8 (0/255) array of shape (img_h, img_w)."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros((img_h, img_w), np.uint8)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = int(_MARGIN * (y1 - y0))
    y0, y1 = max(0, y0 - m), min(mask.shape[0], y1 + m)
    x0, x1 = max(0, x0 - m), min(mask.shape[1], x1 + m)
    crop = mask[y0:y1, x0:x1]

    h, w = crop.shape
    target = img_h / img_w
    if h / w > target:      # taller than the frame: pad left and right
        pad = int(round(h / target)) - w
        crop = np.pad(crop, ((0, 0), (pad // 2, pad - pad // 2)))
    else:                   # wider than the frame: pad top and bottom
        pad = int(round(w * target)) - h
        crop = np.pad(crop, ((pad // 2, pad - pad // 2), (0, 0)))

    im = Image.fromarray((crop * 255).astype(np.uint8))
    return np.asarray(im.resize((img_w, img_h), Image.BILINEAR))


def prepare_pair(front: str | Path, side: str | Path, img_h: int, img_w: int) -> np.ndarray:
    """Front + side silhouettes as one (2, img_h, img_w) uint8 array."""
    return np.stack([crop_pad_resize(load_mask(front), img_h, img_w),
                     crop_pad_resize(load_mask(side), img_h, img_w)])
