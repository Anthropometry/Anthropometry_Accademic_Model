"""S3c — shape check: does a superellipse or a front/back term beat the oval correction?

specs/s3c-shape-check/spec.md (approved by Ryan, 19 Nov 2026, D-023). A diagnostic: no model is
trained and nothing is saved except a JSON of numbers.

What it does
------------
1. For chest, waist and hip, measures whether the oval's error k = true / oval follows the
   depth-to-width ratio (D/W) or the front/back balance of the side view (`side_split`).
2. Compares three ways of turning width W and depth D into a girth, by 5-fold cross-validation
   grouped by person, scored per person:
   A. current S3b correction:  C_hat = k_hat * C0,  k_hat linear in D/W, BMI, sex, fullness, merge
   B. A plus `side_split` as a sixth feature
   C. superellipse: |x/a|^n + |y/b|^n = 1. For each training person find the n whose perimeter equals
      the true girth, learn n linearly from the six features, predict perimeter(a, b, n_hat).

Why these decisions
-------------------
* The photos give the half-width a and half-depth b, never the boxiness n, so n has to be predicted
  from the same features as k. If C is no better than A, a multiplicative k already captures what a
  superellipse would add.
* A superellipse has no closed-form perimeter. It is computed numerically once, on a table of
  shape ratio (b/a) x n, and interpolated: exact enough (checked against Ramanujan at n = 2) and
  fast for ~5,500 photo pairs.
* Same people, folds and per-person scoring as `geometry-fit`, so the three numbers are comparable
  with E2 and with each other. Validation people stay excluded.

Challenges met and fixed
------------------------
* A true girth larger than the rectangle 2(W + D), or smaller than a pinched shape, has no n in
  range. Such n are clipped to 1.5-8 rather than dropped, so every person still counts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import geometry as geo
from . import geometry_fit as gfit
from .config import HEIGHT_COL, PHOTO_COL, SUBJECT_COL, TOLERANCE_CM, WEIGHT_COL
from .data import split_by_person
from .evaluate import bmi_band

N_MIN, N_MAX = 1.5, 8.0
_RATIOS = np.round(np.arange(0.20, 2.001, 0.01), 3)      # b / a; side depth over front width
_NS = np.round(np.arange(N_MIN, N_MAX + 1e-9, 0.05), 3)
_TABLE: np.ndarray | None = None                          # unit perimeters, shape (ratios, ns)


def superellipse_perimeter_unit(ratio: float, n: float, points: int = 2000) -> float:
    """Perimeter of |x|^n + |y/ratio|^n = 1 (half-width 1, half-depth `ratio`), numerically."""
    t = np.linspace(0.0, np.pi / 2, points)
    x = np.cos(t) ** (2.0 / n)
    y = ratio * np.sin(t) ** (2.0 / n)
    return float(4 * np.hypot(np.diff(x), np.diff(y)).sum())


def _table() -> np.ndarray:
    global _TABLE
    if _TABLE is None:
        _TABLE = np.array([[superellipse_perimeter_unit(r, n) for n in _NS] for r in _RATIOS])
    return _TABLE


def _unit_curve(ratio: float) -> np.ndarray:
    """Unit perimeter as a function of n, for one shape ratio (interpolated between table rows)."""
    tab = _table()
    r = float(np.clip(ratio, _RATIOS[0], _RATIOS[-1]))
    i = int(np.clip(np.searchsorted(_RATIOS, r) - 1, 0, len(_RATIOS) - 2))
    w = (r - _RATIOS[i]) / (_RATIOS[i + 1] - _RATIOS[i])
    return (1 - w) * tab[i] + w * tab[i + 1]


def superellipse_perimeter(a, b, n):
    """Perimeter for half-width a, half-depth b and boxiness n (arrays allowed)."""
    a, b, n = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float), np.asarray(n, float))
    out = np.full(a.shape, np.nan)
    for idx in np.ndindex(a.shape):
        if np.isfinite(a[idx]) and np.isfinite(b[idx]) and np.isfinite(n[idx]) and a[idx] > 0:
            nn = float(np.clip(n[idx], N_MIN, N_MAX))
            out[idx] = a[idx] * np.interp(nn, _NS, _unit_curve(b[idx] / a[idx]))
    return out


def fit_n(a, b, girth):
    """The n (clipped to 1.5-8) whose superellipse perimeter equals `girth`."""
    a, b, girth = (np.asarray(v, float) for v in (a, b, girth))
    out = np.full(a.shape, np.nan)
    for i in range(a.size):
        if np.isfinite(a[i]) and np.isfinite(b[i]) and np.isfinite(girth[i]) and a[i] > 0:
            curve = _unit_curve(b[i] / a[i])            # increases with n: boxier = longer
            out[i] = np.interp(girth[i] / a[i], curve, _NS)  # clips at both ends
    return out


def _quintile_table(t: pd.DataFrame, k: np.ndarray, col: str) -> pd.Series:
    v = t[col].to_numpy(float)
    ok = np.isfinite(v) & np.isfinite(k)
    # labels=False: with many equal values pandas merges groups, and fixed labels would no longer fit
    bins = pd.qcut(v[ok], 5, labels=False, duplicates="drop")
    names = ["lowest", "low", "middle", "high", "highest"]
    out = pd.Series(k[ok]).groupby(bins).mean().round(3)
    out.index = [f"{names[int(i)]} {col}" for i in out.index]
    return out


def run(train_split: pd.DataFrame, c: dict, fit_: dict, k: int = 5, verbose: bool = True) -> dict:
    if "seed" not in fit_ or "params" not in fit_:
        raise ValueError("not a geometry fit: run `anthro geometry-fit` first")
    order = {pid: i for i, pid in enumerate(c["photo_id"])}
    fit_df, _ = split_by_person(train_split, fit_["val_fraction"], fit_["seed"])
    if fit_df[SUBJECT_COL].nunique() != fit_.get("fit_people", fit_df[SUBJECT_COL].nunique()):
        raise ValueError("this geometry fit was made from different training people: rerun geometry-fit")
    sel = np.array([order[str(p)] for p in fit_df[PHOTO_COL]])
    cs = {key: (v[sel] if isinstance(v, np.ndarray) and v.shape[:1] == (len(c["ok"]),) else v)
          for key, v in c.items()}
    fit_df = fit_df.reset_index(drop=True)
    t = gfit.level_table(fit_df, cs, fit_["params"])
    folds = gfit._folds(fit_df, k)
    truth_all = fit_df.groupby(SUBJECT_COL)[list(geo.LEVELS)].first()
    base_X = np.column_stack([fit_df[HEIGHT_COL], fit_df[WEIGHT_COL], t["male"]])

    report: dict = {"fit_people": int(fit_df[SUBJECT_COL].nunique()), "levels": {}}
    for name in geo.LEVELS:
        W, D = t[f"{name}_W"].to_numpy(float), t[f"{name}_D"].to_numpy(float)
        C0 = t[f"{name}_C0"].to_numpy(float)
        y = fit_df[name].to_numpy(float)
        kk = y / C0
        X5 = gfit._corr_X(t, name)
        X6 = np.column_stack([X5, t[f"{name}_side_split"].to_numpy(float)])
        truth = truth_all[name]

        # 1. Does the oval's error follow the shape?
        corr = {col: round(float(pd.Series(kk).corr(t[f"{name}_{col}"])), 3) for col in ("DW", "side_split")}

        # 2. Three corrections, same folds
        pred_a = gfit._cv_linear(t, X5, kk, folds) * C0
        pred_b = gfit._cv_linear(t, X6, kk, folds) * C0
        n_true = fit_n(W / 2, D / 2, y)
        n_hat = gfit._cv_linear(t, X6, n_true, folds)
        pred_c = superellipse_perimeter(W / 2, D / 2, n_hat)
        pred_base = gfit._cv_linear(t, base_X, y, folds)
        scores = {
            "A oval x k (current)": gfit._score(gfit._per_person(t, pred_a) - truth),
            "B oval x k + front/back": gfit._score(gfit._per_person(t, pred_b) - truth),
            "C superellipse (n learned)": gfit._score(gfit._per_person(t, pred_c) - truth),
            "baseline h+w+sex": gfit._score(gfit._per_person(t, pred_base) - truth),
        }
        good = np.isfinite(n_true)
        report["levels"][name] = {
            "corr_k_with": corr,
            "k_by_DW_quintile": _quintile_table(t, kk, f"{name}_DW").to_dict(),
            "k_by_side_split_quintile": _quintile_table(t, kk, f"{name}_side_split").to_dict(),
            "n_true": {"median": round(float(np.nanmedian(n_true)), 2),
                       "p10": round(float(np.nanpercentile(n_true[good], 10)), 2),
                       "p90": round(float(np.nanpercentile(n_true[good], 90)), 2),
                       "at_limit_pct": round(float(np.mean((n_true[good] <= N_MIN) | (n_true[good] >= N_MAX))
                                                   * 100), 1)},
            "scores": scores,
        }
        # best variant by BMI band (fairness check from the spec)
        best = min(("A oval x k (current)", pred_a), ("B oval x k + front/back", pred_b),
                   ("C superellipse (n learned)", pred_c),
                   key=lambda v: gfit._person_mae(t, v[1], truth))
        err = (gfit._per_person(t, best[1]) - truth).abs().dropna()
        meta = fit_df.groupby(SUBJECT_COL).first().loc[err.index]
        band = bmi_band(meta).astype(str)
        report["levels"][name]["best"] = best[0]
        report["levels"][name]["best_by_bmi"] = (
            pd.DataFrame({"err": err, "bmi": band}).groupby("bmi")["err"]
            .agg(people="size", MAE_cm="mean",
                 pct=lambda e: (e <= TOLERANCE_CM).mean() * 100).round(2).to_dict(orient="index"))

    if verbose:
        _print(report)
    return report


def _print(report: dict) -> None:
    print(f"\nshape check on {report['fit_people']} training people (validation people excluded), "
          "5-fold CV grouped by person")
    for name, r in report["levels"].items():
        print(f"\n================ {name} ================")
        print(f"correlation of k (true / oval) with D/W: {r['corr_k_with']['DW']:+.3f}   "
              f"with front/back balance: {r['corr_k_with']['side_split']:+.3f}")
        print("mean k in five equal groups, from lowest to highest D/W:")
        print("  " + "  ".join(f"{v:.3f}" for v in r["k_by_DW_quintile"].values()))
        print("mean k in five equal groups, from lowest to highest front/back balance:")
        print("  " + "  ".join(f"{v:.3f}" for v in r["k_by_side_split_quintile"].values()))
        n = r["n_true"]
        print(f"boxiness n that matches each person (2 = oval): median {n['median']}, "
              f"10th-90th percentile {n['p10']}-{n['p90']}, at the 1.5/8 limits {n['at_limit_pct']}%")
        print(pd.DataFrame(r["scores"]).T.to_string())
        print(f"best variant: {r['best']}; by BMI band:")
        print(pd.DataFrame(r["best_by_bmi"]).T.to_string())
