"""Step 2 — the height + weight + sex baseline.

Why a baseline at all: a photo model is only worth its cost if it beats what three numbers
already give. This straight-line model (measurement ≈ a·height + b·weight + c·male + d) is
that bar. On Test-A, per person, it already reaches an average error ≤ 1.5 cm on 9 of the
14 measurements; it fails on waist (3.31), chest (2.90), hip (2.02), leg length (1.81) and
thigh (1.68) — the measurements where body shape varies for the same height and weight
(docs/RESULTS.md).

Challenge met and fixed: the first version trained and scored on photo rows, so people with
many photos counted many times. It now uses one row per person (DISCUSSION D-007).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .config import HEIGHT_COL, MEASUREMENTS, SEX_COL, WEIGHT_COL
from .data import people
from .evaluate import summarise


def features(df: pd.DataFrame) -> np.ndarray:
    male = (df[SEX_COL].str.lower() == "male").astype(float)
    return np.column_stack([df[HEIGHT_COL], df[WEIGHT_COL], male])


def fit(train: pd.DataFrame) -> dict[str, LinearRegression]:
    tr = people(train)
    return {m: LinearRegression().fit(features(tr), tr[m]) for m in MEASUREMENTS}


def errors(models: dict[str, LinearRegression], test: pd.DataFrame) -> pd.DataFrame:
    """Absolute error in cm, one row per person, one column per measurement."""
    te = people(test)
    X = features(te)
    return pd.DataFrame({m: np.abs(models[m].predict(X) - te[m].to_numpy()) for m in MEASUREMENTS},
                        index=te["subject_id"])


def run(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    return summarise(errors(fit(train), test))
