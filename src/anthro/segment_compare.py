"""S7a — compare segmentation tools on Ryan's test photos (D-043), with rules fixed before looking.

specs/s7a-photo-silhouette/spec.md, "How the winner is chosen".

Input: a CSV with one row per photo pair: `person, front, side, height_cm, sex` and optionally `age_range`,
`weight_kg`, and tape values in columns named by app code (e.g. `waist_girth`). Two rows with the same
`person` are repeat photos of that person (for stability).

For each tool it reports:
1. refused pairs, with reasons (photo checks);
2. **height agreement:** share of pairs where the model's own height is 2 cm or more from the tape height
   (the D-029 warning): the one check that has a tape value for every pair;
3. **retake rate** (outline vs photo model, S3b);
4. **stability:** for people with repeats, the average spread (max - min) of each girth across their pairs;
5. **tape errors**, when tape columns exist: per code, average error and % within ±1.5 cm (per person).

Decision rule (fixed before looking): tape errors decide when there are tape values for at least 3 people;
otherwise fewer height flags, then smaller stability spread, then fewer retakes.
Only numbers are printed and saved; never photos, silhouettes or names (use initials or numbers for `person`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import catalog, predict
from .photo import PhotoError

REQUIRED = ("person", "front", "side", "height_cm", "sex")
STABILITY_CODES = ("bust_girth", "chest_girth", "waist_girth", "hip_girth", "thigh_girth")


def load_pairs(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"person": str})
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"pairs file is missing columns: {', '.join(sorted(missing))}")
    return df


def run_tool(pairs: pd.DataFrame, seg, ckpt, bodym_side, **kw) -> tuple[pd.DataFrame, list[dict]]:
    """One row per pair: person, refused reason or the predicted values, height flag, retake flag."""
    rows, docs = [], []
    codes = list(catalog.by_code())
    for r in pairs.itertuples(index=False):
        base = {"person": r.person}
        try:
            weight = getattr(r, "weight_kg", None)
            weight = None if weight is None or pd.isna(weight) else float(weight)
            age = getattr(r, "age_range", None)
            age = None if age is None or pd.isna(age) else str(age)
            doc = predict.predict_photos(r.front, r.side, float(r.height_cm), str(r.sex).lower(), ckpt, seg,
                                         bodym_side, weight_kg=weight, age_range=age, **kw)
        except (PhotoError, predict.PredictError, ValueError) as e:
            rows.append({**base, "refused": str(e)})
            continue
        docs.append(doc)
        vals = {c: doc["measurements"].get(c, {}).get("value_cm") for c in codes}
        rows.append({**base, "refused": None,
                     "height_flag": any("from the tape height" in w for w in doc["warnings"]),
                     "retake": doc["retake"]["flag"], **vals})
    return pd.DataFrame(rows), docs


def scores(result: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    ok = result[result["refused"].isna()]
    out = {"pairs": int(len(result)), "refused": int(result["refused"].notna().sum()),
           "refusal_reasons": result["refused"].dropna().tolist(),
           "height_flag_pct": round(float(ok["height_flag"].mean() * 100), 1) if len(ok) else None,
           "retake_pct": round(float(ok["retake"].mean() * 100), 1) if len(ok) else None}
    rep = ok.groupby("person").filter(lambda g: len(g) >= 2)
    spread = {}
    for c in STABILITY_CODES:
        if c in rep and rep[c].notna().any():
            s = rep.groupby("person")[c].agg(lambda v: float(np.nanmax(v) - np.nanmin(v)))
            spread[c] = round(float(s.mean()), 2)
    out["stability_spread_cm"] = spread
    out["stability_people"] = int(rep["person"].nunique()) if len(rep) else 0
    tape_codes = [c for c in catalog.by_code() if c in pairs.columns and c != "height"]
    errors = {}
    if tape_codes and len(ok):
        pred = ok.groupby("person")[tape_codes].mean()
        # Tape values that are not plain numbers (e.g. "42.0_tbr", a value the measurer marked for
        # re-checking) count as not measured rather than stopping the comparison.
        taped = pairs[["person"]].assign(**{c: pd.to_numeric(pairs[c], errors="coerce") for c in tape_codes})
        tape = taped.groupby("person")[tape_codes].mean().reindex(pred.index)
        for c in tape_codes:
            d = (pred[c] - tape[c]).dropna()
            e = d.abs()
            if len(e):
                # bias: + means the model reads larger than the tape. A large bias with every error of the
                # same sign points at a definition or clothing difference, not at random model error.
                errors[c] = {"mae_cm": round(float(e.mean()), 2), "bias_cm": round(float(d.mean()), 2),
                             "same_sign_pct": round(float(max((d > 0).mean(), (d < 0).mean()) * 100), 1),
                             "pct_within_1.5": round(float((e <= 1.5).mean() * 100), 1),
                             "people": int(len(e))}
    out["tape_errors"] = errors
    out["tape_people"] = max((v["people"] for v in errors.values()), default=0)
    return out


def decide(all_scores: dict[str, dict]) -> tuple[str, str]:
    """The fixed rule from the module docstring: (winner, why)."""
    tools = list(all_scores)
    if all(s["tape_people"] >= 3 for s in all_scores.values()):
        def mean_pct(s):
            return np.mean([v["pct_within_1.5"] for v in s["tape_errors"].values()])
        best = max(tools, key=lambda t: mean_pct(all_scores[t]))
        return best, "higher average % within ±1.5 cm against the tape"

    def key(t):
        s = all_scores[t]
        spread = np.mean(list(s["stability_spread_cm"].values())) if s["stability_spread_cm"] else np.inf
        return (s["height_flag_pct"] if s["height_flag_pct"] is not None else np.inf, spread,
                s["retake_pct"] if s["retake_pct"] is not None else np.inf, s["refused"])
    best = min(tools, key=key)
    return best, "fewer height flags, then smaller repeat spread, then fewer retakes (no tape for 3+ people)"
