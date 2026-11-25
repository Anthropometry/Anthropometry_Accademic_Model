"""S3b — cache geometry for every photo pair, tune the levels, learn the ellipse correction.

specs/s3b-geometric-girth/spec.md, experiments E0-E2 (approved 17 Nov 2026).

What it does
------------
* `build_cache()` measures every photo pair once (full-resolution masks) and saves the profiles as
  one .npz per split on Drive. About 9,000 pairs; a cache means this is paid once.
* `fit()` uses only the TRAINING part of the train split (the validation people that choose the
  network's best epoch are excluded, same seed and fraction as `anthro train`):
  1. per level, searches the height band (waist, hip) or armpit offset (chest) that makes the
     ellipse perimeter C0 best predict the true girth, by 5-fold cross-validation grouped by
     person;
  2. fits the correction k = true / C0 as a linear function of D/W, BMI, sex, fullness and the
     merge flag: C_hat = k_hat * C0. Linear so it stays explainable ("the ellipse under-reads
     your waist by 3%");
  3. reports E1 (ellipse alone), the k table by sex x BMI band, and E2 (C_hat vs the height +
     weight + sex baseline on the same folds);
  4. returns a plain-JSON fit: chosen params, coefficients, and the feature scaler the hybrid
     network uses.

Why these decisions
-------------------
* Grouped by person: a person's photo pairs on both sides of a fold would flatter the score (D-008).
* Validation people excluded from the fit: otherwise the geometric features would carry
  information about the people used to pick the network's best epoch, making the hybrid look
  better than it is.
* Scores per person (photo predictions averaged), as everywhere in this project (D-007).
* Test-A and Test-B are never touched here (POLICY: choose on validation, test once).

Challenges met and fixed
------------------------
* Object (string) arrays need pickle to load from .npz; ids are stored as fixed-width unicode
  arrays so the cache loads with allow_pickle=False.
* A level that cannot be measured (no body at the torso centre) is recorded as invalid and left
  out of fitting, rather than filled with a guess.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm

from . import geometry as geo
from .config import HEIGHT_COL, PHOTO_COL, SEX_COL, SUBJECT_COL, TOLERANCE_CM, WEIGHT_COL
from .data import split_by_person
from .evaluate import bmi_band
from .preprocess import load_mask

_PROFILE_KEYS = ("w_cm", "d_cm", "d_left_cm", "d_right_cm", "n_runs")
_SCALAR_KEYS = ("mask_px_height", "sil_volume_l", "edge_soft", "mask_h", "mask_w")
CORR_FEATURES = ("DW", "bmi", "male", "fullness", "merge")
# Per-level features handed to the hybrid network (D-019), then 3 whole-body ones + a validity flag.
LEVEL_FEATURES = ("W", "D", "C0", "C_hat", "DW", "side_split", "merge")
BODY_FEATURES = ("fullness", "edge_soft", "mask_px_height", "geo_ok")

# Search grids.
# 17 Nov, first real fit: all three levels chose a grid edge -> grids widened.
# 17 Nov, second fit: the hip kept choosing the narrowest band, i.e. a FIXED height matched BodyM
# better than "deepest point in a band". Both waist and hip now also try fixed heights (a band one
# sample row wide), from 0.30 to 0.64 of body height.
_FIXED = [[float(round(f, 2)), float(round(f + 0.002, 3))] for f in np.arange(0.30, 0.641, 0.02)]
SEARCH = {
    "chest": {"chest_offset": [float(round(x, 3)) for x in np.arange(0.0, 0.181, 0.02)]},
    "waist": {"waist_band": [[lo, hi] for lo in (0.30, 0.33, 0.36, 0.39, 0.42)
                             for hi in (0.46, 0.49, 0.52, 0.55) if hi - lo >= 0.06] + _FIXED},
    "hip": {"hip_band": [[lo, hi] for lo in (0.47, 0.50, 0.53, 0.56)
                         for hi in (0.53, 0.56, 0.59, 0.62, 0.65) if hi - lo >= 0.03] + _FIXED},
}


# ---------------------------------------------------------------- cache (E0)

def cache_path(cache_dir: str | Path, split: str, data_fp: str) -> Path:
    return Path(cache_dir) / f"{split}_geometry_{data_fp}.npz"


def build_cache(df: pd.DataFrame, root: str | Path, cache_dir: str | Path, split: str,
                data_fp: str) -> dict:
    path = cache_path(cache_dir, split, data_fp)
    if path.exists():
        c = load_cache(path)
        if len(c["photo_id"]) == len(df) and (c["photo_id"] == df[PHOTO_COL].astype(str).to_numpy()).all():
            return c
    n = len(df)
    out = {k: np.zeros((n, geo.N_ROWS), np.uint8 if k == "n_runs" else np.float32) for k in _PROFILE_KEYS}
    out.update({k: np.zeros(n, np.float32) for k in _SCALAR_KEYS})
    out["ok"] = np.zeros(n, bool)
    errors: dict[str, int] = {}
    for i, row in enumerate(tqdm(df.itertuples(index=False), total=n, desc=f"geometry {split}")):
        d = Path(root) / row.split
        pid = getattr(row, PHOTO_COL)
        try:
            m = measure_pair(d / "mask" / f"{pid}.png", d / "mask_left" / f"{pid}.png",
                             getattr(row, HEIGHT_COL))
        except geo.GeometryError as e:
            errors[str(e)] = errors.get(str(e), 0) + 1
            continue
        for k in (*_PROFILE_KEYS, *_SCALAR_KEYS):
            out[k][i] = m[k]
        out["ok"][i] = True
    out["photo_id"] = df[PHOTO_COL].astype(str).to_numpy().astype("U")
    out["subject_id"] = df[SUBJECT_COL].astype(str).to_numpy().astype("U")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    if errors:
        print(f"{split}: {n - int(out['ok'].sum())} of {n} pairs could not be measured:")
        for msg, k in sorted(errors.items(), key=lambda x: -x[1]):
            print(f"  {k:5d}  {msg}")
    return out


def measure_pair(front_path, side_path, height_cm: float) -> dict:
    """Width/depth profile and body features of one photo pair (raises geo.GeometryError).
    Shared by the cache builder and by single-person prediction (S7-0), so both measure alike."""
    raw_f = _load_raw(Path(front_path))
    front = load_mask(front_path)     # refuses oversized images
    side = load_mask(side_path)
    p = geo.profile(front, side, height_cm)
    m = {k: p[k] for k in _PROFILE_KEYS}
    m.update(mask_px_height=p["mask_px_height"], sil_volume_l=p["sil_volume_l"],
             edge_soft=geo.edge_softness(raw_f), mask_h=front.shape[0], mask_w=front.shape[1])
    return m


def single_cache(m: dict | None) -> dict:
    """A one-row cache (the layout `level_table` reads) from `measure_pair`; None = unmeasurable."""
    c = {k: np.zeros((1, geo.N_ROWS), np.uint8 if k == "n_runs" else np.float32) for k in _PROFILE_KEYS}
    c.update({k: np.zeros(1, np.float32) for k in _SCALAR_KEYS})
    c["ok"] = np.array([m is not None])
    if m is not None:
        for k in (*_PROFILE_KEYS, *_SCALAR_KEYS):
            c[k][0] = m[k]
    return c


def _load_raw(path: Path) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("L"))


def load_cache(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as z:     # never unpickle a downloaded file
        return {k: z[k] for k in z.files}


def diagnostics(c: dict, params: dict | None = None) -> pd.DataFrame:
    """E0: can full-resolution geometry work on this data?"""
    ok = c["ok"]
    lv = geo.levels_batch(c["w_cm"][ok], c["d_cm"][ok], c["n_runs"][ok], params)
    px = c["mask_px_height"][ok]
    rows = {
        "pairs": len(ok),
        "measured (%)": 100 * ok.mean(),
        "mask size (median HxW)": f"{int(np.median(c['mask_h'][ok]))}x{int(np.median(c['mask_w'][ok]))}",
        "body height px (median)": float(np.median(px)) if px.size else np.nan,
        "body height px (5th pct)": float(np.percentile(px, 5)) if px.size else np.nan,
        "mm per pixel (median, 170 cm)": 1700 / np.median(px) if px.size else np.nan,
        "levels valid (%)": 100 * lv["valid"].mean() if px.size else np.nan,
        "edge softness (median)": float(np.median(c["edge_soft"][ok])) if px.size else np.nan,
    }
    for name in geo.LEVELS:
        rows[f"{name}: arm touching / no gap (%)"] = 100 * lv["merge"][name].mean() if px.size else np.nan
    return pd.Series(rows, name="value").to_frame()


# ---------------------------------------------------------------- features

def _meta(df: pd.DataFrame) -> pd.DataFrame:
    bmi = df[WEIGHT_COL] / (df[HEIGHT_COL] / 100) ** 2
    return pd.DataFrame({"bmi": bmi.to_numpy(), "male": (df[SEX_COL].str.lower() == "male").to_numpy(float)})


def level_table(df: pd.DataFrame, c: dict, params: dict | None = None) -> pd.DataFrame:
    """One row per photo pair: W, D, C0, D/W, side split, merge per level, plus body features."""
    ok = c["ok"].copy()
    lv = geo.levels_batch(c["w_cm"], c["d_cm"], c["n_runs"], params)
    ok &= lv["valid"]
    idx = np.arange(len(ok))
    t = _meta(df)
    t["fullness"] = np.where(ok, geo.fullness(np.maximum(c["sil_volume_l"], 1e-6), df[WEIGHT_COL]), np.nan)
    t["edge_soft"] = c["edge_soft"]
    t["mask_px_height"] = c["mask_px_height"]
    t["geo_ok"] = ok.astype(float)
    for name in geo.LEVELS:
        r = lv["rows"][name]
        W, D = c["w_cm"][idx, r].astype(float), c["d_cm"][idx, r].astype(float)
        t[f"{name}_W"] = np.where(ok, W, np.nan)
        t[f"{name}_D"] = np.where(ok, D, np.nan)
        t[f"{name}_C0"] = np.where(ok, geo.ramanujan(W / 2, D / 2), np.nan)
        t[f"{name}_DW"] = np.where(ok, D / np.maximum(W, 1e-6), np.nan)
        t[f"{name}_side_split"] = np.where(ok, c["d_left_cm"][idx, r] / np.maximum(D, 1e-6), np.nan)
        t[f"{name}_merge"] = lv["merge"][name].astype(float)
    t[SUBJECT_COL] = df[SUBJECT_COL].to_numpy()
    return t


def _corr_X(t: pd.DataFrame, name: str) -> np.ndarray:
    return np.column_stack([t[f"{name}_DW"], t["bmi"], t["male"], t["fullness"], t[f"{name}_merge"]])


def apply_correction(t: pd.DataFrame, fit_: dict) -> pd.DataFrame:
    """Add C_hat = k_hat * C0 per level, using fitted coefficients (plain values)."""
    t = t.copy()
    for name in geo.LEVELS:
        co = fit_["correction"][name]
        X = np.nan_to_num(_corr_X(t, name), nan=0.0)
        k = co["intercept"] + X @ np.asarray(co["coef"])
        t[f"{name}_C_hat"] = k * t[f"{name}_C0"]
    return t


def feature_names() -> list[str]:
    return [f"{lv}_{f}" for lv in geo.LEVELS for f in LEVEL_FEATURES] + list(BODY_FEATURES)


def feature_matrix(t: pd.DataFrame, fit_: dict) -> np.ndarray:
    """Standardised features for the hybrid network. Unmeasurable values become the training
    average (0 after standardising); `geo_ok` and the merge flags tell the network so."""
    t = apply_correction(t, fit_)
    X = t[feature_names()].to_numpy(np.float64)
    mean, std = np.asarray(fit_["scaler"]["mean"]), np.asarray(fit_["scaler"]["std"])
    Z = (X - mean) / std
    return np.nan_to_num(Z, nan=0.0).astype(np.float32)


# ---------------------------------------------------------------- fitting (E1, E2)

def _per_person(t: pd.DataFrame, values: np.ndarray) -> pd.Series:
    return pd.Series(values, index=t[SUBJECT_COL].to_numpy()).groupby(level=0).mean()


def _score(err: pd.Series) -> dict:
    err = err.dropna().abs()
    return {"people": int(err.size), "MAE_cm": round(float(err.mean()), 2),
            f"pct_within_{TOLERANCE_CM:g}cm": round(float((err <= TOLERANCE_CM).mean() * 100), 1)}


def _folds(t: pd.DataFrame, k: int = 5):
    return list(GroupKFold(n_splits=k).split(t, groups=t[SUBJECT_COL]))


def _cv_linear(t: pd.DataFrame, X: np.ndarray, y: np.ndarray, folds) -> np.ndarray:
    """Out-of-fold predictions; rows with NaN are left NaN."""
    pred = np.full(len(y), np.nan)
    good = np.isfinite(X).all(1) & np.isfinite(y)
    for tr, te in folds:
        tr, te = tr[good[tr]], te[good[te]]
        if len(tr) < 5 or len(te) == 0:
            continue
        m = LinearRegression().fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def _person_mae(t: pd.DataFrame, pred: np.ndarray, truth: pd.Series) -> float:
    p = _per_person(t, pred).dropna()
    return float((p - truth.loc[p.index]).abs().mean()) if len(p) else np.inf


def _at_edge(options: list, best: int) -> bool:
    """True if the choice sits at the outer limit of what was searched, so the best may lie beyond.

    Scalar settings: the smallest or largest value. Bands: the lowest start or the highest end of
    all searched bands (the outer limits of the height range searched).
    """
    chosen = options[best]
    if not isinstance(chosen, list):
        return best in (0, len(options) - 1)
    return chosen[0] <= min(o[0] for o in options) or chosen[1] >= max(o[1] for o in options)


def fit(train_split: pd.DataFrame, c: dict, seed: int = 0, val_fraction: float = 0.1,
        k: int = 5, verbose: bool = True) -> dict:
    """Tune levels and fit the correction on the training part of the train split."""
    order = {pid: i for i, pid in enumerate(c["photo_id"])}
    fit_df, val_df = split_by_person(train_split, val_fraction, seed)
    sel = np.array([order[str(p)] for p in fit_df[PHOTO_COL]])
    cs = {key: (v[sel] if isinstance(v, np.ndarray) and v.shape[:1] == (len(c["ok"]),) else v)
          for key, v in c.items()}
    fit_df = fit_df.reset_index(drop=True)
    truth_all = fit_df.groupby(SUBJECT_COL)[list(geo.LEVELS)].first()
    folds = _folds(fit_df, k)

    params = dict(geo.DEFAULT_PARAMS)
    search_log = {}
    for name in geo.LEVELS:
        key, options = next(iter(SEARCH[name].items()))
        scores = []
        for opt in options:
            # Score each candidate level with the SAME model used afterwards (k from D/W, BMI,
            # sex, fullness, merge). Second real fit: a simpler score (straight line on C0)
            # chose levels that made the final corrected geometry worse.
            t = level_table(fit_df, cs, {**params, key: opt})
            C0 = t[f"{name}_C0"].to_numpy(float)
            k_pred = _cv_linear(t, _corr_X(t, name), fit_df[name].to_numpy(float) / C0, folds)
            scores.append(_person_mae(t, k_pred * C0, truth_all[name]))
        best = int(np.nanargmin(scores))
        params[key] = options[best]
        search_log[name] = {"param": key, "options": options, "cv_mae_cm": [round(s, 3) for s in scores],
                            "chosen": options[best], "at_edge": _at_edge(options, best)}

    t = level_table(fit_df, cs, params)
    correction, e1, e2, k_rows = {}, {}, {}, []
    base_X = np.column_stack([fit_df[HEIGHT_COL], fit_df[WEIGHT_COL], t["male"]])
    for name in geo.LEVELS:
        y = fit_df[name].to_numpy(float)
        C0 = t[f"{name}_C0"].to_numpy(float)
        truth = truth_all[name]
        # E1: the ellipse alone, no learning at all
        e1[name] = _score(_per_person(t, C0) - truth)
        ratio = y / C0
        X = _corr_X(t, name)
        k_pred = _cv_linear(t, X, ratio, folds)
        e2[name] = {"geometry C_hat": _score(_per_person(t, k_pred * C0) - truth),
                    "baseline h+w+sex": _score(_per_person(t, _cv_linear(t, base_X, y, folds)) - truth)}
        good = np.isfinite(X).all(1) & np.isfinite(ratio)
        m = LinearRegression().fit(X[good], ratio[good])
        correction[name] = {"features": list(CORR_FEATURES), "intercept": float(m.intercept_),
                            "coef": [float(v) for v in m.coef_]}
        groups = pd.DataFrame({"sex": fit_df[SEX_COL].str.lower(), "bmi": bmi_band(fit_df).astype(str),
                               "k": ratio})
        g = groups.groupby(["sex", "bmi"])["k"].agg(["mean", "std", "size"]).round(3)
        g["level"] = name
        k_rows.append(g.reset_index())

    fit_ = {"params": params, "correction": correction, "seed": seed, "val_fraction": val_fraction,
            "fit_people": int(fit_df[SUBJECT_COL].nunique()), "search": search_log,
            "e1_ellipse_only": e1, "e2_corrected_vs_baseline": e2}
    tc = apply_correction(t, fit_)
    X = tc[feature_names()].to_numpy(np.float64)
    fit_["scaler"] = {"mean": np.nan_to_num(np.nanmean(X, 0)).tolist(),
                      "std": (np.nan_to_num(np.nanstd(X, 0), nan=1.0) + 1e-6).tolist(),
                      "features": feature_names()}
    k_table = pd.concat(k_rows, ignore_index=True)

    if verbose:
        print(f"\nfitted on {fit_['fit_people']} training people (validation people excluded), "
              f"{k}-fold CV grouped by person")
        print("\nlevel search (CV MAE of the corrected geometry, cm):")
        for name, s in search_log.items():
            edge = "   <- WARNING: at the edge of the search grid" if s["at_edge"] else ""
            ch = s["chosen"]
            kind = " (fixed height)" if isinstance(ch, list) and ch[1] - ch[0] < 0.01 else ""
            print(f"  {name:5s} {s['param']} = {ch}{kind}   (best {min(s['cv_mae_cm']):.2f} cm){edge}")
        print("\n== E1: ellipse alone (C0 vs true girth, per person) ==")
        print(pd.DataFrame(e1).T.to_string())
        print("\n== k = true / ellipse, by sex and BMI band (k < 1: ellipse over-reads) ==")
        print(k_table.pivot_table(index=["sex", "bmi"], columns="level", values="mean").round(3).to_string())
        print("\n== E2: corrected geometry vs height + weight + sex baseline (CV, per person) ==")
        rows = {(lv, m): v for lv, d in e2.items() for m, v in d.items()}
        print(pd.DataFrame(rows).T.to_string())
    fit_["k_table"] = k_table.to_dict(orient="records")
    return fit_


def to_plain(obj):
    """JSON round-trip so a fit holds only plain Python values (safe in a weights_only checkpoint)."""
    import json
    return json.loads(json.dumps(obj, default=lambda v: v.item() if hasattr(v, "item") else str(v)))


def features_for(split_df: pd.DataFrame, rows_df: pd.DataFrame, root, cache_dir, split: str,
                 data_fp: str, fit_: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """Standardised hybrid features and the raw table (with C_hat) for `rows_df`, a subset of
    `split_df` (e.g. the training or validation people of the train split)."""
    c = build_cache(split_df, root, cache_dir, split, data_fp)
    order = {pid: i for i, pid in enumerate(c["photo_id"])}
    sel = np.array([order[str(p)] for p in rows_df[PHOTO_COL]])
    cs = {k: (v[sel] if isinstance(v, np.ndarray) and v.shape[:1] == (len(c["ok"]),) else v)
          for k, v in c.items()}
    t = level_table(rows_df.reset_index(drop=True), cs, fit_["params"])
    return feature_matrix(t, fit_), apply_correction(t, fit_)
