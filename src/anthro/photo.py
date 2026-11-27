"""S7a — from a phone photo to a BodyM-like silhouette, refusing photos we can't trust.

specs/s7a-photo-silhouette/spec.md (approved by Ryan, 25 Nov 2026).

What it does
------------
1. `load_photo`: reads JPEG/PNG, applies the phone's rotation (EXIF), refuses absurd sizes.
2. `clean`: keeps the largest body, fills small holes (not the gap between the legs), drops specks.
3. `check`: refuses, with what to fix: no person; more than one person; head or feet cut off; body too small
   in the frame; phone tilted beyond the limit (when the app recorded tilt); arms away from the body in the
   side photo.
4. `to_bodym`: crops head to heel and places the body on BodyM's full-resolution canvas (960 x 720, body
   about 832 pixels tall), white on black, crisp (0/255) edges; side photos are mirrored to face BodyM's way.

Why these decisions
-------------------
* BodyM scale and crisp edges: the geometry (S3b) and the hybrid model's edge feature were tuned on them.
  Challenge (27 Nov 2026, D-054): the first version kept soft edges, assuming BodyM's were soft. The diagnosis
  on real photos showed BodyM's edge-softness feature is ~0 for every person, while ours sat about 150,000
  standard deviations away; the network was fed an input it had never seen. The silhouette is now made crisp.
* Only small holes are filled (under 0.5% of the body): when feet touch, the gap between the legs is a
  closed hole, and filling it would turn two legs into one block.
* Side direction (D-044): BodyM's side silhouettes all face one way. Direction is read from the feet: in a
  side view the toes stick out in front of the ankle. When it can't be read, the photo is used as is and the
  result says so.
* Refuse, don't guess: every refusal names the problem and the fix, so the person can retake.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage  # installed with scikit-learn, already a dependency

CANVAS = (960, 720)          # BodyM full-resolution mask size (rows, columns)
BODY_PX = 832                # typical BodyM body height on that canvas
MIN_BODY_PX = 600            # below this the photo is too far away: each pixel is worth too much
MAX_PHOTO_PIXELS = 40_000_000
MIN_PHOTO_PIXELS = 300_000
SMALL_HOLE = 0.005           # holes below this share of the body area are filled
SECOND_PERSON = 0.15         # another blob this big (vs the largest) is a second person
EDGE_PX = 2                  # body this close to the top or bottom edge counts as cut off
MAX_TILT_DEG = 5.0


class PhotoError(ValueError):
    """The photo can't be measured; the message says what to fix."""


@dataclass
class Silhouette:
    mask: np.ndarray                         # uint8 0-255 on the BodyM canvas
    facing: str | None = None                # side photos: "left", "right" or None (unknown)
    mirrored: bool = False
    notes: list[str] = field(default_factory=list)


def load_photo(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        n = im.width * im.height
        if n > MAX_PHOTO_PIXELS:
            raise PhotoError(f"photo too large ({im.width}x{im.height})")
        if n < MIN_PHOTO_PIXELS:
            raise PhotoError(f"photo too small ({im.width}x{im.height}): use the phone camera's normal size")
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))


def clean(prob: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray, int]:
    """(body mask, soft mask, number of person-sized blobs). Soft mask = probability inside the body only."""
    binary = prob >= threshold
    labels, n = ndimage.label(binary)
    if n == 0:
        return binary, np.zeros_like(prob), 0
    sizes = ndimage.sum(binary, labels, index=np.arange(1, n + 1))
    biggest = int(np.argmax(sizes)) + 1
    people = int((sizes >= SECOND_PERSON * sizes.max()).sum())
    body = labels == biggest
    holes, nh = ndimage.label(~body)
    if nh:
        hole_sizes = ndimage.sum(~body, holes, index=np.arange(1, nh + 1))
        border = set(np.unique(np.concatenate([holes[0], holes[-1], holes[:, 0], holes[:, -1]])))
        for i, s in enumerate(hole_sizes, start=1):
            if i not in border and s < SMALL_HOLE * body.sum():
                body |= holes == i
    soft = np.where(body, np.maximum(prob, threshold), np.minimum(prob, threshold - 1e-3))
    # Soft edges only right at the boundary: anything not touching the body is background.
    near = ndimage.binary_dilation(body, iterations=2)
    soft = np.where(near, soft, 0.0)
    return body, soft.astype(np.float32), people


def check(body: np.ndarray, people: int, view: str, tilt: tuple[float, float] | None = None) -> list[str]:
    problems = []
    if people == 0 or not body.any():
        return ["no person found in the photo: stand in the middle of the frame, whole body visible"]
    if people > 1:
        problems.append("more than one person in the photo: only the person being measured should be visible")
    rows = np.flatnonzero(body.any(axis=1))
    top, bottom = int(rows[0]), int(rows[-1])
    if top <= EDGE_PX:
        problems.append("the head touches the top of the photo: step back or tilt the phone less")
    if bottom >= body.shape[0] - 1 - EDGE_PX:
        problems.append("the feet touch the bottom of the photo: the whole body, feet included, must show")
    if bottom - top + 1 < MIN_BODY_PX:
        problems.append(f"the body is only {bottom - top + 1} pixels tall in the photo "
                        f"(needs {MIN_BODY_PX}): come closer or use a higher-resolution photo")
    if tilt is not None:
        pitch, roll = tilt
        if max(abs(pitch), abs(roll)) > MAX_TILT_DEG:
            problems.append(f"the phone was tilted ({pitch:.0f} deg forward, {roll:.0f} deg sideways): "
                            f"hold it upright (within {MAX_TILT_DEG:g} deg)")
    if view == "side" and arms_away_in_side(body, top, bottom):
        problems.append("in the side photo the arms are away from the body: let them hang by your sides")
    return problems


def arms_away_in_side(body: np.ndarray, top: int, bottom: int) -> bool:
    """Side view: arms hanging by the sides overlap the torso; an arm held forward or back shows as a
    separate run of body pixels in most rows of the upper torso (30-50% of body height)."""
    h = bottom - top + 1
    band = body[top + int(0.30 * h): top + int(0.50 * h)]
    runs = [int(np.count_nonzero(np.diff(np.concatenate([[0], r.astype(int), [0]])) == 1)) for r in band]
    return bool(np.mean(np.array(runs) >= 2) > 0.6)


def facing(body: np.ndarray) -> str | None:
    """Which way a side silhouette faces, from the feet: toes stick out in front of the ankle."""
    rows = np.flatnonzero(body.any(axis=1))
    if rows.size < 50:
        return None
    top, bottom = rows[0], rows[-1]
    h = bottom - top + 1

    def centre(a, b):
        cols = np.nonzero(body[top + int(a * h): top + int(b * h)])[1]
        return float(cols.mean()) if cols.size else None

    ankle, foot = centre(0.90, 0.95), centre(0.97, 1.0)
    if ankle is None or foot is None:
        return None
    shift = foot - ankle
    width = max(1.0, float(body[top + int(0.97 * h): bottom + 1].any(axis=0).sum()))
    if abs(shift) < 0.05 * width:      # too small to call
        return None
    return "right" if shift > 0 else "left"


def to_bodym(soft: np.ndarray, body: np.ndarray) -> np.ndarray:
    """Crop head to heel (columns: the body's extent), scale to BODY_PX tall, centre on the BodyM canvas."""
    rows = np.flatnonzero(body.any(axis=1))
    cols = np.flatnonzero(body.any(axis=0))
    crop = soft[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1]
    scale = BODY_PX / crop.shape[0]
    new_w = max(1, int(round(crop.shape[1] * scale)))
    if new_w > CANVAS[1]:
        raise PhotoError("the body is too wide for its height: arms or legs are spread too far apart")
    im = Image.fromarray((crop * 255).astype(np.uint8)).resize((new_w, BODY_PX), Image.BILINEAR)
    canvas = np.zeros(CANVAS, np.uint8)
    y0, x0 = (CANVAS[0] - BODY_PX) // 2, (CANVAS[1] - new_w) // 2
    canvas[y0: y0 + BODY_PX, x0: x0 + new_w] = np.asarray(im)
    return np.where(canvas >= 128, 255, 0).astype(np.uint8)     # crisp, like BodyM's masks (D-054)


def silhouette(rgb: np.ndarray, seg, view: str, bodym_facing: str | None = None,
               tilt: tuple[float, float] | None = None) -> Silhouette:
    """Photo -> BodyM-like silhouette, or PhotoError listing every problem found."""
    body, soft, people = clean(seg(rgb))
    problems = check(body, people, view, tilt)
    if problems:
        raise PhotoError(f"{view} photo: " + "; ".join(problems))
    out = Silhouette(to_bodym(soft, body))
    if view == "side":
        out.facing = facing(body)
        if bodym_facing and out.facing and out.facing != bodym_facing:
            out.mask = out.mask[:, ::-1].copy()
            out.mirrored = True
        elif bodym_facing and out.facing is None:
            out.notes.append("side photo: direction could not be read from the feet; used as taken")
    return out


def bodym_facing(data_root: str | Path, n: int = 60) -> str | None:
    """Majority direction of BodyM's side silhouettes (training split, first n)."""
    from .preprocess import load_mask
    files = sorted((Path(data_root) / "train" / "mask_left").glob("*.png"))[:n]
    votes = [facing(load_mask(p)) for p in files]
    votes = [v for v in votes if v]
    if not votes:
        return None
    return max(set(votes), key=votes.count)


def overlay(rgb: np.ndarray, body: np.ndarray) -> np.ndarray:
    """The photo with the body outline in red, for the visual check (shown in Colab, never saved)."""
    edge = body & ~ndimage.binary_erosion(body, iterations=2)
    out = rgb.copy()
    out[edge] = (255, 0, 0)
    return out
