"""S5b — rough outline estimates for under-bust, high hip and knee girth.

specs/s5b-geometry-rough/spec.md (approved by Ryan, 25 Nov 2026). Front length comes from ANSUR (D-033).

What it does
------------
From the same full-resolution silhouettes and profile as S3b (`geometry.profile`):
* **Under-bust:** front width and side depth a fixed distance below the chest level, ellipse perimeter,
  times the chest correction k (S3b) evaluated with this row's own shape (D/W) and the person's BMI etc.
* **High hip:** halfway in height between the waist and hip levels, ellipse, times the average of the
  waist and hip corrections at that row.
* **Knee:** at knee height, ONE leg: its width in the front view and the legs' depth in the side view
  (in a side view the two legs overlap, so the widest run is one leg's depth), ellipse, no correction.

Why these decisions
-------------------
* Every value is `rough` with no expected error (D-034): neither BodyM nor ANSUR has these measurements
  next to a photo, so no error can be measured. Ryan's tape data is the first check.
* The under-bust offset (4.5% of height below the chest level, about 7.5 cm at 165 cm) and the textbook
  knee height (28.5% of stature from the floor) are stated assumptions. ANSUR's knee-height ratio
  replaces the textbook one once `ansur-fit` stores it (`level_ratios` in the formula file).
* Corrections are borrowed from the nearest fitted level, not invented for new ones; the knee gets none.
* A level the outline can't measure (arm or legs touching, nothing at the centre) returns a reason, and
  the code stays `not_available` with that reason in the result: refuse, don't guess.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import geometry as geo

UNDERBUST_OFFSET = 0.045          # fraction of body height below the chest level (stated assumption)
KNEE_FROM_FLOOR_TEXTBOOK = 0.285  # knee height / stature, textbook segment proportion
MIN_LEG_CM = 5.0                  # a run narrower than this at knee height is not a leg


def ellipse_girth(width_cm: float, depth_cm: float) -> float:
    return float(geo.ramanujan(width_cm / 2.0, depth_cm / 2.0))


def _k(geo_fit: dict, level: str, dw: float, bmi: float, male: float, fullness: float, merge: float) -> float:
    co = geo_fit["correction"][level]
    x = np.nan_to_num(np.array([dw, bmi, male, fullness, merge], float))
    return float(co["intercept"] + x @ np.asarray(co["coef"], float))


def _runs(row: np.ndarray) -> list[tuple[int, int]]:
    """(start, length) of each run of body pixels in one image row."""
    padded = np.concatenate([[False], row.astype(bool), [False]])
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    return [(int(a), int(b - a)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def knee(front: np.ndarray, side: np.ndarray, height_cm: float, from_top: float) -> tuple[float | None, str]:
    """One leg's girth at `from_top` (fraction of body height from the head), or None and a reason."""
    f, s = geo._prepare(front, height_cm), geo._prepare(side, height_cm)
    fy = int(round(f.top + from_top * (f.bottom - f.top)))
    sy = int(round(s.top + from_top * (s.bottom - s.top)))
    legs = [n * f.cm_per_px for _, n in _runs(f.mask[fy]) if n * f.cm_per_px >= MIN_LEG_CM]
    if len(legs) < 2:
        return None, "legs touch (or only one leg visible) at knee height in the front photo"
    w = float(np.mean(sorted(legs)[-2:]))           # the two widest runs are the legs
    side_runs = [n * s.cm_per_px for _, n in _runs(s.mask[sy])]
    if not side_runs:
        return None, "no leg found at knee height in the side photo"
    d = float(max(side_runs))
    return ellipse_girth(w, d), "outline ellipse at knee height, one leg, no correction"


def summarise_people(df, root, geo_fit: dict, knee_from_floor: float):
    """S5b on BodyM people: per person (median over their photo pairs), the share each code could be measured
    for and the spread of values, overall and by sex and BMI band. A sanity check, not a score (D-034)."""
    import pandas as pd

    from .config import HEIGHT_COL, PHOTO_COL, SEX_COL, SUBJECT_COL, WEIGHT_COL
    from .evaluate import bmi_band
    from .preprocess import load_mask
    codes = ("underbust_girth", "high_hip_girth", "knee_girth")
    recs = []
    for row in df.itertuples(index=False):
        d = Path(root) / row.split
        pid = getattr(row, PHOTO_COL)
        person = {"height_cm": getattr(row, HEIGHT_COL), "weight_kg": getattr(row, WEIGHT_COL),
                  "sex": str(getattr(row, SEX_COL)).lower()}
        est = estimate(load_mask(d / "mask" / f"{pid}.png"), load_mask(d / "mask_left" / f"{pid}.png"),
                       person, geo_fit, knee_from_floor)
        recs.append({SUBJECT_COL: getattr(row, SUBJECT_COL), **{c: est[c][0] for c in codes}})
    per = pd.DataFrame(recs).groupby(SUBJECT_COL)[list(codes)].median()
    people = df.groupby(SUBJECT_COL).first().loc[per.index]
    spread = pd.DataFrame({c: {"measured_pct": round(per[c].notna().mean() * 100, 1),
                               "median_cm": round(per[c].median(), 1),
                               "p5_cm": round(per[c].quantile(0.05), 1),
                               "p95_cm": round(per[c].quantile(0.95), 1)}
                           for c in codes}).T
    groups = per.notna().groupby([people[SEX_COL].str.lower(), bmi_band(people)], observed=True)
    share = (groups.mean() * 100).round(1)
    share["people"] = groups.size()
    return spread, share


def estimate(front: np.ndarray, side: np.ndarray, person: dict, geo_fit: dict,
             knee_from_floor: float = KNEE_FROM_FLOOR_TEXTBOOK) -> dict[str, tuple[float | None, str]]:
    """{code: (value_cm or None, method or reason)} for underbust_girth, high_hip_girth, knee_girth."""
    h, wkg = person["height_cm"], person["weight_kg"]
    out: dict[str, tuple[float | None, str]] = {}
    try:
        p = geo.profile(front, side, h)
    except geo.GeometryError as e:
        reason = f"outline could not be measured ({e})"
        return {c: (None, reason) for c in ("underbust_girth", "high_hip_girth", "knee_girth")}
    lv = geo.levels_batch(p["w_cm"], p["d_cm"], p["n_runs"], geo_fit["params"])
    rows = {k: int(v[0]) for k, v in lv["rows"].items()}
    bmi, male = wkg / (h / 100) ** 2, float(person["sex"] == "male")
    full = float(geo.fullness(max(p["sil_volume_l"], 1e-6), wkg))

    def at(r):
        w, d = float(p["w_cm"][r]), float(p["d_cm"][r])
        return (w, d, float(p["n_runs"][r] < 3)) if w > 0 and d > 0 else None

    r = min(rows["chest"] + int(round(UNDERBUST_OFFSET * geo.N_ROWS)), geo.N_ROWS - 1)
    m = at(r)
    if m and lv["valid"][0]:
        w, d, merge = m
        k = _k(geo_fit, "chest", d / w, bmi, male, full, merge)
        out["underbust_girth"] = (k * ellipse_girth(w, d),
                                  "outline ellipse below the chest level x chest correction")
    else:
        out["underbust_girth"] = (None, "no torso outline below the chest level")

    r = (rows["waist"] + rows["hip"]) // 2
    m = at(r)
    if m and lv["valid"][0]:
        w, d, merge = m
        k = np.mean([_k(geo_fit, lvl, d / w, bmi, male, full, merge) for lvl in ("waist", "hip")])
        out["high_hip_girth"] = (float(k) * ellipse_girth(w, d),
                                 "outline ellipse halfway between waist and hip x mean correction")
    else:
        out["high_hip_girth"] = (None, "no torso outline between the waist and hip levels")

    try:
        out["knee_girth"] = knee(front, side, h, 1.0 - knee_from_floor)
    except geo.GeometryError as e:
        out["knee_girth"] = (None, f"outline could not be measured ({e})")
    return out
