"""Scoring: always per PERSON, against the ±1.5 cm target.

Three questions every result must answer:
1. How far off is the average person?           -> MAE (mean absolute error), cm
2. How many people get a usable measurement?    -> % within ±1.5 cm
3. Does one person get the same answer twice?   -> spread across their photos (consistency)

Why both 1 and 2: an average can hide a lot. On the baseline, calf averages 1.40 cm (under
target) yet only 64% of people are within ±1.5 cm. The supervisor's target is about people,
so the percentage is the headline figure; MAE is the research-standard companion.

Subgroup reporting (by sex and BMI band) is part of evaluation, not an extra: a model can look
fine on average and fail one group of bodies (docs/ACCESSIBILITY.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HEIGHT_COL, MEASUREMENTS, SEX_COL, SUBJECT_COL, TOLERANCE_CM, WEIGHT_COL

MIN_GROUP = 20


def per_person_predictions(df: pd.DataFrame, pred_cm: np.ndarray) -> pd.DataFrame:
    """Average each person's photo-pair predictions into one prediction per person."""
    p = pd.DataFrame(pred_cm, columns=list(MEASUREMENTS))
    p[SUBJECT_COL] = df[SUBJECT_COL].to_numpy()
    return p.groupby(SUBJECT_COL)[list(MEASUREMENTS)].mean()


def per_person_errors(df: pd.DataFrame, pred_cm: np.ndarray) -> pd.DataFrame:
    pred = per_person_predictions(df, pred_cm)
    truth = df.groupby(SUBJECT_COL)[list(MEASUREMENTS)].first().loc[pred.index]
    return (pred - truth).abs()


def summarise(err: pd.DataFrame, tol: float = TOLERANCE_CM) -> pd.DataFrame:
    return pd.DataFrame({
        "MAE_cm": err.mean(),
        "p90_cm": err.quantile(0.9),                  # the error 9 in 10 people stay under
        f"pct_within_{tol:g}cm": (err <= tol).mean() * 100,
    }).round(2)


def tolerance_view(err: pd.DataFrame, tols=(1.5, 2.0, 3.0)) -> pd.DataFrame:
    """% of people within each tolerance, per measurement. Exploration only: target stays ±1.5 cm (D-002)."""
    return pd.DataFrame({f"within_{t:g}cm_pct": (err <= t).mean() * 100 for t in tols}).round(1)


def consistency(df: pd.DataFrame, pred_cm: np.ndarray) -> pd.Series:
    """Average standard deviation (cm) of one person's predictions across their photo pairs.

    Only people with 2+ pairs count. Small values mean small changes in pose or framing
    don't change the answer — a reliability figure for the report.
    """
    p = pd.DataFrame(pred_cm, columns=list(MEASUREMENTS))
    p[SUBJECT_COL] = df[SUBJECT_COL].to_numpy()
    return p.groupby(SUBJECT_COL)[list(MEASUREMENTS)].std().mean().round(2)


def bmi_band(df: pd.DataFrame) -> pd.Series:
    bmi = df[WEIGHT_COL] / (df[HEIGHT_COL] / 100) ** 2
    return pd.cut(bmi, [0, 18.5, 25, 30, 35, np.inf],
                  labels=["<18.5", "18.5-25", "25-30", "30-35", "35+"], right=False)


def by_group(df: pd.DataFrame, err: pd.DataFrame, tol: float = TOLERANCE_CM) -> pd.DataFrame:
    """% of people within tolerance, averaged over measurements, per sex x BMI band."""
    meta = df.groupby(SUBJECT_COL).first().loc[err.index]
    groups = pd.DataFrame({"sex": meta[SEX_COL].str.lower(), "bmi": bmi_band(meta)}, index=err.index)
    within = (err <= tol).mean(axis=1) * 100
    out = within.groupby([groups["sex"], groups["bmi"]], observed=True).agg(["mean", "size"])
    out = out.rename(columns={"mean": f"pct_within_{tol:g}cm", "size": "people"}).round(1)
    # Fewer than 20 people: the percentage moves by 5+ points per person, so don't over-read it.
    out["note"] = np.where(out["people"] < MIN_GROUP, "few people: unreliable", "")
    return out


# ---------------------------------------------------------------- S3b additions

def bootstrap_ci(err: pd.DataFrame, n: int = 1000, seed: int = 0, tol: float = TOLERANCE_CM) -> pd.DataFrame:
    """95% confidence intervals for MAE and % within tolerance, resampling PEOPLE.

    Why: Test-A has 87 people, so one person moves the % column by ~1.1 points. An interval tells
    the reader whether a difference between two models is bigger than that luck.
    """
    rng = np.random.default_rng(seed)
    E = err.to_numpy(np.float64)
    idx = rng.integers(0, len(E), (n, len(E)))
    mae = E[idx].mean(1)
    pct = (E[idx] <= tol).mean(1) * 100
    lo, hi = np.percentile(mae, [2.5, 97.5], 0), np.percentile(pct, [2.5, 97.5], 0)
    cols = err.columns
    return pd.DataFrame({
        "MAE_95ci": [f"{lo[0][j]:.2f}-{lo[1][j]:.2f}" for j in range(len(cols))],
        f"pct_{tol:g}_95ci": [f"{hi[0][j]:.1f}-{hi[1][j]:.1f}" for j in range(len(cols))],
    }, index=cols)


RETAKE_LEVELS = ("chest", "waist", "hip")


def disagreement(df: pd.DataFrame, geo_table: pd.DataFrame, pred_cm: np.ndarray) -> pd.Series:
    """Per person: average over their photo pairs of the largest |geometry - network| gap (cm)
    across chest, waist and hip. A pair whose geometry could not be measured counts as infinite:
    if we can't cross-check it, the honest answer is "please retake"."""
    p = pd.DataFrame(pred_cm, columns=list(MEASUREMENTS))
    gaps = np.column_stack([(geo_table[f"{lv}_C_hat"].to_numpy() - p[lv].to_numpy()) for lv in RETAKE_LEVELS])
    worst = np.where(np.isfinite(gaps).all(1), np.abs(gaps).max(1), np.inf)
    return pd.Series(worst, index=df[SUBJECT_COL].to_numpy()).groupby(level=0).mean()


def retake_threshold(val_disagreement: pd.Series, share: float) -> float:
    """Gap (cm) above which the warning fires, chosen so it fires for `share` of validation people."""
    if not 0 < share < 1:
        raise ValueError("retake share must be between 0 and 1")
    return float(np.quantile(val_disagreement.to_numpy(), 1 - share))


def retake_report(err: pd.DataFrame, flagged: pd.Series, tol: float = TOLERANCE_CM) -> pd.DataFrame:
    rows = {}
    for name, mask in (("not flagged", ~flagged), ("flagged: retake", flagged)):
        e = err.loc[mask[mask].index.intersection(err.index)]
        r = {"people": len(e)}
        for lv in RETAKE_LEVELS:
            r[f"{lv}_MAE"] = round(float(e[lv].mean()), 2) if len(e) else np.nan
            r[f"{lv}_pct"] = round(float((e[lv] <= tol).mean() * 100), 1) if len(e) else np.nan
        r["all14_pct"] = round(float((e <= tol).to_numpy().mean() * 100), 1) if len(e) else np.nan
        rows[name] = r
    return pd.DataFrame(rows).T
