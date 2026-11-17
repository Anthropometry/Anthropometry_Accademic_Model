"""S3b — measure width and depth from silhouettes and turn them into circumferences.

specs/S3b-geometric-girth.md (approved 17 Nov 2026).

What it does
------------
1. `profile()` reads a FULL-RESOLUTION front and side mask and records, at 500 evenly spaced
   heights from the top of the head to the soles:
   - front torso width (cm),
   - side depth (cm) and how far the body extends to each side of the side-view midline,
   - how many separate body pieces the front row has (3 = left arm | torso | right arm).
2. `level_features()` finds the chest, waist and hip heights on that profile and returns width W,
   depth D, the ellipse perimeter C0 and quality flags at each.
3. `ramanujan()` is the ellipse perimeter; the learned correction k lives in geometry_fit.py.

Why each decision
-----------------
* Full resolution, never the 256 px training images: at 256 px one pixel is ~0.7 cm of body and a
  1 cm width error becomes ~1.7 cm of girth. The scale is height_cm / silhouette height in pixels,
  computed per mask, so front and side photos taken at different zoom still line up.
* Heights are FRACTIONS of the body (0 = top of head, 1 = soles), not pixel rows: people differ in
  height and photos in resolution, but "40% of the way down" means roughly the same place.
* Torso width = the run of body pixels that contains the torso's centre column. In A-pose the
  arms are separate runs and are excluded automatically. When an arm touches the body, the run
  swallows it; we cannot tell, so the row is flagged (`merge`) instead of trusted.
* Levels are found by rules inside height bands: waist = narrowest front width, hip = deepest side
  depth, chest = a fixed offset below the armpit (first row where both arms separate from the
  torso). BodyM's levels are fixed SMPL vertex loops with no published heights, so the band limits
  and the offset are TUNED on training people (geometry_fit.py), not assumed.
* Ellipse first, correction learned later: published "under 1 cm" ellipse results come from small
  or synthetic samples; no large study shows the formula alone beating a trained model (research
  brief, 17 Nov 2026). The ellipse is the explainable starting point, not the answer.
* Fullness (weight vs. silhouette volume) is a SOFT feature: body density is 1.02-1.07 kg/L
  depending on body fat, so weight gives volume only to about ±3%.

Challenges met and fixed
------------------------
* A per-row Python loop over ~9,000 photo pairs x ~1,000 rows was too slow; every per-row quantity
  is computed with whole-array numpy operations instead.
* Some rows have no body at the centre column (e.g. between the legs). Those rows get width 0 and
  are never chosen as a level; a level that would land on one is refused with GeometryError.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_ROWS = 500                 # heights sampled per body; 170 cm / 500 ≈ 3.4 mm apart
BODY_DENSITY_KG_PER_L = 1.03  # Siri-based whole-body density, mid-range (1.02-1.07)
LEVELS = ("chest", "waist", "hip")

# Starting search bands (fractions from the top). Tuned on training people in geometry_fit.py.
DEFAULT_PARAMS = {
    "armpit_band": [0.12, 0.40],   # where to look for the first row with both arms separated
    "chest_offset": 0.03,          # chest = armpit + this fraction of body height
    "chest_fallback": 0.27,        # used (and flagged) when no armpit gap is found
    "waist_band": [0.33, 0.47],
    "hip_band": [0.45, 0.60],
}


class GeometryError(ValueError):
    """The silhouette can't be measured; the caller should ask for a retake, not guess."""


@dataclass
class _Sil:
    mask: np.ndarray
    top: int
    bottom: int
    cm_per_px: float
    centre: int


def _prepare(mask: np.ndarray, height_cm: float) -> _Sil:
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size < 50:
        raise GeometryError("no body found in the silhouette (fewer than 50 body rows)")
    top, bottom = int(rows[0]), int(rows[-1])
    px_height = bottom - top + 1
    # Torso centre: average column of body pixels between 35% and 55% of the body height. Arms in
    # A-pose sit roughly symmetrically, so they barely move this average.
    a, b = top + int(0.35 * px_height), top + int(0.55 * px_height)
    cols = np.nonzero(mask[a:b])[1]
    if cols.size == 0:
        raise GeometryError("no torso found between 35% and 55% of body height")
    return _Sil(mask, top, bottom, height_cm / px_height, int(round(cols.mean())))


def _sample_rows(s: _Sil) -> np.ndarray:
    frac = (np.arange(N_ROWS) + 0.5) / N_ROWS
    return (s.top + frac * (s.bottom - s.top)).round().astype(int)


def _centre_run(m: np.ndarray, c: int) -> tuple[np.ndarray, np.ndarray]:
    """For each row, pixels from the centre column to the run's left and right edges.

    Vectorised: the first background pixel left of c is the first True in the reversed,
    inverted left half; same idea to the right. Rows where the centre pixel is background get 0.
    """
    left = ~m[:, : c + 1][:, ::-1]
    right = ~m[:, c:]
    lx = np.where(left.any(1), left.argmax(1), left.shape[1])   # pixels until background, leftwards
    rx = np.where(right.any(1), right.argmax(1), right.shape[1])
    on = m[:, c]
    return np.where(on, lx, 0), np.where(on, rx, 0)


def _n_runs(m: np.ndarray) -> np.ndarray:
    starts = m[:, 1:] & ~m[:, :-1]
    return starts.sum(1) + m[:, 0]


def ramanujan(a, b):
    """Ramanujan's ellipse perimeter from the half-width a and half-depth b (any units).

    Accurate to well under 0.01% for body-like shapes (D/W between 0.5 and 1); exact for a circle.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))


def edge_softness(raw: np.ndarray) -> float:
    """Share of pixels that are neither clearly body nor clearly background.

    A crisp mask scores ~0; a blurry or anti-aliased one scores higher. Recorded as a quality
    flag because blur moves the edge and edge error is multiplied ~2π in a girth.
    """
    raw = np.asarray(raw)
    if raw.dtype == bool:
        return 0.0
    grey = (raw > 25) & (raw < 230)
    body = raw >= 128
    return float(grey.sum() / max(body.sum(), 1))


def profile(front: np.ndarray, side: np.ndarray, height_cm: float) -> dict:
    """Row-by-row width and depth (cm) at N_ROWS fractions of body height.

    `front` and `side` are boolean masks (True = body) at their original resolution.
    """
    if not (100.0 <= float(height_cm) <= 230.0):
        raise GeometryError(f"height {height_cm} cm is outside 100-230 cm")
    f, s = _prepare(front, height_cm), _prepare(side, height_cm)
    fr, sr = _sample_rows(f), _sample_rows(s)

    fl, frr = _centre_run(f.mask[fr], f.centre)
    sl, srr = _centre_run(s.mask[sr], s.centre)
    w = np.where(fl > 0, fl + frr - 1, 0) * f.cm_per_px
    d = np.where(sl > 0, sl + srr - 1, 0) * s.cm_per_px

    # Fullness: weight-volume vs silhouette volume, using whole rows (arms included: it is a
    # rough whole-body figure on purpose, see module docstring).
    tot_w = f.mask[f.top:f.bottom + 1].sum(1) * f.cm_per_px
    tot_d = s.mask[s.top:s.bottom + 1].sum(1) * s.cm_per_px
    # resample the side rows onto the front rows so the two line up by body fraction
    tot_d = np.interp(np.linspace(0, 1, tot_w.size), np.linspace(0, 1, tot_d.size), tot_d)
    sil_volume_l = float((np.pi / 4 * tot_w * tot_d * f.cm_per_px).sum() / 1000)

    return {
        "w_cm": w.astype(np.float32),
        "d_cm": d.astype(np.float32),
        "d_left_cm": (sl * s.cm_per_px).astype(np.float32),    # side-view extent, one side of midline
        "d_right_cm": (srr * s.cm_per_px).astype(np.float32),  # ... and the other
        "n_runs": _n_runs(f.mask[fr]).astype(np.uint8),
        "mask_px_height": float(f.bottom - f.top + 1),
        "sil_volume_l": sil_volume_l,
    }


def fullness(sil_volume_l, weight_kg):
    """>1: heavier than the outline suggests (fuller cross-sections); <1: lighter."""
    return (np.asarray(weight_kg, float) / BODY_DENSITY_KG_PER_L) / np.asarray(sil_volume_l, float)


def _band(lo: float, hi: float) -> slice:
    return slice(int(lo * N_ROWS), int(np.ceil(hi * N_ROWS)))


def levels_batch(w, d, n_runs, params: dict | None = None) -> dict:
    """Chest, waist and hip row (0..N_ROWS-1) for MANY profiles at once (arrays of shape N x N_ROWS).

    Returns rows, `merge` flags (arm touching the torso, or no armpit gap found) and `valid`
    (False where a level has no body at the torso centre). Vectorised because level search in
    geometry_fit.py evaluates hundreds of band settings on ~6,000 training photo pairs.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    w, d, n_runs = np.atleast_2d(w), np.atleast_2d(d), np.atleast_2d(n_runs)
    idx = np.arange(w.shape[0])
    rows, merge = {}, {}

    ab = _band(*p["armpit_band"])
    sep = n_runs[:, ab] >= 3
    # 3 consecutive separated rows, so a stray pixel isn't taken for the armpit
    ok = sep[:, :-2] & sep[:, 1:-1] & sep[:, 2:]
    found = ok.any(1)
    chest = np.where(found, ab.start + ok.argmax(1) + int(round(p["chest_offset"] * N_ROWS)),
                     int(p["chest_fallback"] * N_ROWS))
    rows["chest"] = np.minimum(chest, N_ROWS - 1)
    merge["chest"] = ~found | (n_runs[idx, rows["chest"]] < 3)

    wb = _band(*p["waist_band"])
    ww = np.where(w[:, wb] > 0, w[:, wb], np.inf)
    rows["waist"] = wb.start + ww.argmin(1)

    hb = _band(*p["hip_band"])
    rows["hip"] = hb.start + d[:, hb].argmax(1)

    valid = np.ones(w.shape[0], bool)
    for lv in LEVELS:
        r = rows[lv]
        if lv != "chest":
            merge[lv] = n_runs[idx, r] < 3
        valid &= (w[idx, r] > 0) & (d[idx, r] > 0)
    return {"rows": rows, "merge": merge, "valid": valid}


def find_levels(w, d, n_runs, params: dict | None = None) -> dict:
    """Same as levels_batch for ONE profile; refuses instead of returning an invalid level."""
    b = levels_batch(w[None], d[None], n_runs[None], params)
    if not b["valid"][0]:
        bad = [lv for lv in LEVELS if w[b["rows"][lv][0]] <= 0 or d[b["rows"][lv][0]] <= 0]
        raise GeometryError(f"{', '.join(bad)}: no body at the torso centre at that height")
    return {"rows": {lv: int(b["rows"][lv][0]) for lv in LEVELS},
            "merge": {lv: bool(b["merge"][lv][0]) for lv in LEVELS}}


def level_features(prof: dict, params: dict | None = None) -> dict:
    """W, D, ellipse perimeter C0, D/W and flags at chest, waist and hip, for one profile."""
    lv = find_levels(prof["w_cm"], prof["d_cm"], prof["n_runs"], params)
    feats = {}
    for name, r in lv["rows"].items():
        W, D = float(prof["w_cm"][r]), float(prof["d_cm"][r])
        feats[name] = {
            "row": r,
            "W": W,
            "D": D,
            "C0": float(ramanujan(W / 2, D / 2)),
            "DW": D / W,
            "side_split": float(prof["d_left_cm"][r] / max(D, 1e-6)),  # front/back balance
            "merge": lv["merge"][name],
        }
    return feats
