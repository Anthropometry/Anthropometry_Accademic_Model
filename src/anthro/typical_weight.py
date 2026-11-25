"""S7-0 / D-027 — a typical weight for a person who doesn't know theirs.

Ryan's decision (24 Nov 2026): people without a weight are not refused. They get a typical weight
for their height and sex, the result carries a warning, and every value that used it is marked
`weight_assumed`.

How: a straight line, weight = a + b x height, fitted separately for women and men on the BodyM
training people. It is a population average, not this person: someone heavier or lighter than
typical will get girths that are too small or too large, which is why the warning is mandatory.
New checkpoints store the fit (plain numbers); for older ones (r1-r4) it is refitted from BodyM's
training table, which takes a second.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HEIGHT_COL, SEX_COL, SUBJECT_COL, WEIGHT_COL

WARNING = ("weight not given: a typical weight for this height and sex was used "
           "({:.1f} kg); girths may be off for anyone heavier or lighter than typical")


def fit(train_df: pd.DataFrame) -> dict:
    """{sex: {"intercept", "slope", "people"}} from one row per person."""
    people = train_df.drop_duplicates(SUBJECT_COL)
    out = {}
    for sex in ("female", "male"):
        p = people[people[SEX_COL].str.lower() == sex]
        if len(p) < 2:
            raise ValueError(f"need at least 2 {sex} people to fit a typical weight")
        slope, intercept = np.polyfit(p[HEIGHT_COL].to_numpy(float), p[WEIGHT_COL].to_numpy(float), 1)
        out[sex] = {"intercept": float(intercept), "slope": float(slope), "people": int(len(p))}
    return out


def typical(fit_: dict, sex: str, height_cm: float) -> float:
    f = fit_[sex]
    return float(f["intercept"] + f["slope"] * height_cm)
