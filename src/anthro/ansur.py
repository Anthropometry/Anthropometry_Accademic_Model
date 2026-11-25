"""S5a — formulas learned on ANSUR II for the app measurements BodyM doesn't have.

specs/s5a-ansur/spec.md (approved by Ryan, 25 Nov 2026).

What it does
------------
* Loads the two public ANSUR II files (`ANSUR II MALE Public.csv`, `ANSUR II FEMALE Public.csv`):
  millimetres → cm, tenths of a kg → kg; refuses missing columns, unit mistakes, and a folder inside
  this repository (ANSUR stays on Drive, never committed: D-014).
* For each app code and sex, fits two straight-line formulas and keeps the one with the lower held-out
  error (5 folds, split by person):
    - basic: height, weight, BMI (what the person types);
    - extended: basic + the photo model's measurements that ANSUR has in a comparable form, with those
      inputs blurred by noise the size of the photo model's Test-B error (in real use they are photo
      estimates, not tape).
* Reports per code and sex: average error, 90th-percentile error and % within ±1.5 cm, for the sex-average
  baseline, basic and extended (blurred and not); by BMI band; and for ANSUR's Black participants separately
  (D-030). Also a check refitted on ANSUR's self-reported height and weight.
* Writes `ansur_formulas.json` (plain numbers only) and supplies `derived` estimates from it.

Why these decisions
-------------------
* Straight lines: one coefficient per input is explainable in the writeup and hard to overfit on 2,000-4,000
  people. Rejected: boosting or neural nets (opaque, little expected gain on smooth body-size relations).
* The blur: without it the extended formula looks far better than it can be, because ANSUR's own inputs are
  exact tape measurements. sigma = MAE x sqrt(pi/2) makes the blur's average size equal the Test-B MAE.
* ANSUR's waist is at the navel (omphalion), the app's at the narrowest point; lengths from the waist are
  marked `differs`. Lengths are differences of standing heights (vertical), not a tape along the body.
* Nothing about a single ANSUR person is printed or saved: only coefficients and summary errors.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from . import catalog
from .config import TOLERANCE_CM

FORMAT = "anthro-ansur-formulas/1"
FILES = {"male": "ANSUR II MALE Public.csv", "female": "ANSUR II FEMALE Public.csv"}
BLACK = 2                      # ANSUR's DODRace code for Black participants (self-reported)
FOLDS = 5

# App code -> (how to compute it from ANSUR columns, definition match, sexes). Values in mm until converted.
TARGETS: dict[str, tuple[tuple[str, ...], str]] = {
    "neck_girth": (("neckcircumferencebase",), "same"),
    "shoulder_length": (("shoulderlength",), "same"),
    "hip_girth": (("buttockcircumference",), "same"),
    "back_length": (("waistbacklength",), "differs"),
    "inseam": (("crotchheight", "-lateralmalleolusheight"), "presumed"),
    "outseam": (("waistheightomphalion", "-lateralmalleolusheight"), "differs"),
    "waist_to_hip": (("waistheightomphalion", "-buttockheight"), "differs"),
    "waist_to_knee": (("waistheightomphalion", "-kneeheightmidpatella"), "differs"),
    "shoulder_to_knee": (("acromialheight", "-kneeheightmidpatella"), "differs"),   # D-031
}
# Photo-model output (BodyM name) -> ANSUR column measuring the same thing on tape (extended formulas).
PHOTO_INPUTS = {
    "chest": "chestcircumference", "waist": "waistcircumference", "shoulder-breadth": "biacromialbreadth",
    "thigh": "thighcircumference", "calf": "calfcircumference", "wrist": "wristcircumference",
    "hip": "buttockcircumference",
}
# The big girths for the mix test (spec: "Mix test"): BodyM name -> ANSUR column.
MIX_GIRTHS = {"chest": "chestcircumference", "waist": "waistcircumference"}
BASIC = ("height_cm", "weight_kg", "bmi")
OTHER = ("subjectid", "gender", "dodrace", "stature", "weightkg", "heightin", "weightlbs")
# Loose plausible medians (cm) per column: catches a file in inches or cm instead of mm.
PLAUSIBLE_MEDIAN_CM = (2.0, 200.0)


class AnsurError(ValueError):
    """The ANSUR files or a formula file can't be used; the message says why."""


def needed_columns() -> set[str]:
    cols = {c.lstrip("-") for parts, _ in TARGETS.values() for c in parts}
    return cols | set(PHOTO_INPUTS.values()) | set(OTHER)


def fingerprint(folder: str | Path) -> str:
    h = hashlib.sha256()
    for name in sorted(FILES.values()):
        h.update((Path(folder) / name).read_bytes())
    return h.hexdigest()[:16]


def load(folder: str | Path) -> pd.DataFrame:
    """Both files as one table in cm and kg, one row per person, lowercase column names."""
    folder = Path(folder).resolve()
    repo = Path(__file__).resolve().parents[2]
    if folder == repo or repo in folder.parents:
        raise AnsurError("ANSUR files must stay outside the repository (keep them on Drive)")
    frames = []
    for sex, name in FILES.items():
        p = folder / name
        if not p.exists():
            raise AnsurError(f"{name} not found in {folder.name}")
        try:
            df = pd.read_csv(p)
        except UnicodeDecodeError:            # the public female file is not UTF-8
            df = pd.read_csv(p, encoding="latin-1")
        df.columns = [c.strip().lower() for c in df.columns]
        missing = needed_columns() - set(df.columns)
        if missing:
            raise AnsurError(f"{name} is missing columns: {', '.join(sorted(missing))}")
        df = df[sorted(needed_columns())].copy()
        df["sex"] = sex
        mm = sorted(needed_columns() - set(OTHER)) + ["stature"]
        df[mm] = df[mm].astype(float) / 10.0                   # mm -> cm
        med = df[mm].median()                                   # checked per file: one bad file can't hide
        bad = med[(med < PLAUSIBLE_MEDIAN_CM[0]) | (med > PLAUSIBLE_MEDIAN_CM[1])]
        if len(bad):
            raise AnsurError(f"{name}: implausible values (wrong units?): {', '.join(bad.index)}")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["height_cm"] = df["stature"]
    df["weight_kg"] = df["weightkg"].astype(float) / 10.0     # tenths of a kg -> kg
    df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2
    df["stated_height_cm"] = df["heightin"].astype(float) * 2.54
    df["stated_weight_kg"] = df["weightlbs"].astype(float) * 0.45359237
    if df["subjectid"].duplicated().any():
        raise AnsurError("a subject appears twice across the two files")
    for code, (parts, _) in TARGETS.items():
        df[code] = sum((-df[c[1:]] if c.startswith("-") else df[c]) for c in parts)
    return df.reset_index(drop=True)


def folds(df: pd.DataFrame, k: int = FOLDS, seed: int = 0) -> np.ndarray:
    """Fold number per row, assigned by person (one row per person in ANSUR, but kept explicit)."""
    ids = df["subjectid"].drop_duplicates().sample(frac=1.0, random_state=seed).to_numpy()
    fold_of = {sid: i % k for i, sid in enumerate(ids)}
    return df["subjectid"].map(fold_of).to_numpy()


def sigma_from_mae(mae: float) -> float:
    return float(mae) * float(np.sqrt(np.pi / 2))


def _X(df: pd.DataFrame, inputs: list[str], noise: dict[str, float] | None, rng) -> np.ndarray:
    cols = []
    for name in inputs:
        if name in BASIC:
            cols.append(df[name].to_numpy(float))
        else:
            v = df[PHOTO_INPUTS[name]].to_numpy(float)
            if noise is not None:
                v = v + rng.normal(0.0, sigma_from_mae(noise.get(name, 0.0)), len(v))
            cols.append(v)
    return np.column_stack(cols)


def _scores(err: np.ndarray, tol: float = TOLERANCE_CM) -> dict:
    err = np.abs(err)
    return {"mae_cm": round(float(err.mean()), 2), "p90_cm": round(float(np.quantile(err, 0.9)), 2),
            "pct_within": round(float((err <= tol).mean() * 100), 1), "people": int(len(err))}


def cv_errors(df: pd.DataFrame, target: str, inputs: list[str], noise=None, seed: int = 0) -> np.ndarray:
    """Held-out signed errors (cm), one per person: each fold is predicted by a fit on the others."""
    f = folds(df, seed=seed)
    rng = np.random.default_rng(seed)
    X, y = _X(df, inputs, noise, rng), df[target].to_numpy(float)
    X_clean = _X(df, inputs, None, rng)       # the formula is fitted on tape values, applied to blurred ones
    out = np.zeros(len(df))
    for k in np.unique(f):
        tr, te = f != k, f == k
        m = LinearRegression().fit(X_clean[tr], y[tr])
        out[te] = m.predict(X[te]) - y[te]
    return out


def extended_inputs(code: str) -> list[str]:
    own = TARGETS[code][0][0]
    return [*BASIC, *[n for n, col in PHOTO_INPUTS.items() if col != own]]


def fit(df: pd.DataFrame, noise: dict[str, float], catalogue_version: str, data_fp: str,
        noise_basis: str) -> tuple[dict, pd.DataFrame]:
    """Formulas (plain numbers) and a score table for printing."""
    formulas, rows = {}, []
    for code in TARGETS:
        formulas[code] = {}
        for sex in catalog.SEXES:
            if sex not in catalog.by_code()[code]["sexes"]:
                continue
            d = df[df["sex"] == sex].reset_index(drop=True)
            y = d[code].to_numpy(float)
            fo = folds(d)
            base_err = np.zeros(len(d))
            for k in np.unique(fo):
                base_err[fo == k] = y[fo != k].mean() - y[fo == k]
            basic_err = cv_errors(d, code, list(BASIC))
            ext_in = extended_inputs(code)
            ext_err = cv_errors(d, code, ext_in, noise)
            ext_clean = cv_errors(d, code, ext_in, None)
            sb, se = _scores(basic_err), _scores(ext_err)
            kind, inputs, err = (("extended", ext_in, ext_err) if se["mae_cm"] < sb["mae_cm"]
                                 else ("basic", list(BASIC), basic_err))
            m = LinearRegression().fit(_X(d, inputs, None, None), y)
            formulas[code][sex] = {
                "kind": kind, "inputs": inputs, "intercept": float(m.intercept_),
                "coef": [float(c) for c in m.coef_], "mae_cm": _scores(err)["mae_cm"],
                "pct_within": _scores(err)["pct_within"],
                "range_cm": [float(np.quantile(y, 0.01)), float(np.quantile(y, 0.99))],
                "definition_match": TARGETS[code][1],
            }
            for name, e in (("sex average", base_err), ("basic", basic_err), ("extended, blurred", ext_err),
                            ("extended, tape inputs", ext_clean)):
                rows.append({"code": code, "sex": sex, "formula": name, **_scores(e),
                             "chosen": name.startswith(kind) and name != "extended, tape inputs"})
    doc = {"format": FORMAT, "catalogue": catalogue_version, "ansur_fingerprint": data_fp,
           "noise_basis": noise_basis, "noise_mae_cm": {k: float(v) for k, v in noise.items()},
           "girth_basic": girth_basic(df), "formulas": formulas, "mix": {}}
    return doc, pd.DataFrame(rows)


def girth_basic(df: pd.DataFrame) -> dict:
    """Basic formulas (height, weight, BMI) for ANSUR chest and waist, per sex: the mix test's ANSUR side."""
    out = {}
    for name, col in MIX_GIRTHS.items():
        out[name] = {}
        for sex in catalog.SEXES:
            d = df[df["sex"] == sex]
            m = LinearRegression().fit(d[list(BASIC)].to_numpy(float), d[col].to_numpy(float))
            out[name][sex] = {"intercept": float(m.intercept_), "coef": [float(c) for c in m.coef_]}
    return out


def subgroup_table(df: pd.DataFrame, doc: dict, noise: dict[str, float]) -> pd.DataFrame:
    """Chosen formula's held-out error by BMI band and for Black participants vs everyone else (D-030)."""
    rows = []
    bands = pd.cut(df["bmi"], [0, 25, 30, 35, np.inf], labels=["<25", "25-30", "30-35", "35+"], right=False)
    for code, by_sex in doc["formulas"].items():
        for sex, f in by_sex.items():
            mask = (df["sex"] == sex).to_numpy()
            d = df[mask].reset_index(drop=True)
            err = np.abs(cv_errors(d, code, f["inputs"], noise if f["kind"] == "extended" else None))
            groups = {"all": np.ones(len(d), bool),
                      "Black participants": (d["dodrace"] == BLACK).to_numpy(),
                      "all other participants": (d["dodrace"] != BLACK).to_numpy()}
            b_mask = bands[mask].to_numpy()
            groups.update({f"BMI {b}": (b_mask == b) for b in ["<25", "25-30", "30-35", "35+"]})
            for g, sel in groups.items():
                if sel.sum() >= 20:            # small groups give noise, not evidence
                    rows.append({"code": code, "sex": sex, "group": g, **_scores(err[sel])})
    return pd.DataFrame(rows)


def self_report_check(df: pd.DataFrame) -> pd.DataFrame:
    """Basic formula fitted on measured height/weight, applied with self-reported ones (cost of stating)."""
    rows = []
    stated = df.assign(height_cm=df["stated_height_cm"], weight_kg=df["stated_weight_kg"])
    stated["bmi"] = stated["weight_kg"] / (stated["height_cm"] / 100) ** 2
    ok = stated[["height_cm", "weight_kg"]].notna().all(axis=1).to_numpy()
    for code in TARGETS:
        for sex in catalog.SEXES:
            if sex not in catalog.by_code()[code]["sexes"]:
                continue
            sel = ok & (df["sex"] == sex).to_numpy()
            d, s = df[sel].reset_index(drop=True), stated[sel].reset_index(drop=True)
            f = folds(d)
            y = d[code].to_numpy(float)
            e_meas, e_stat = np.zeros(len(d)), np.zeros(len(d))
            for k in np.unique(f):
                m = LinearRegression().fit(d.loc[f != k, list(BASIC)].to_numpy(float), y[f != k])
                e_meas[f == k] = m.predict(d.loc[f == k, list(BASIC)].to_numpy(float)) - y[f == k]
                e_stat[f == k] = m.predict(s.loc[f == k, list(BASIC)].to_numpy(float)) - y[f == k]
            rows.append({"code": code, "sex": sex, "measured_mae_cm": _scores(e_meas)["mae_cm"],
                         "stated_mae_cm": _scores(e_stat)["mae_cm"], "people": int(sel.sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- using a formula file

def load_formulas(path: str | Path) -> dict:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("format") != FORMAT:
        raise AnsurError(f"not an ANSUR formula file ({doc.get('format')!r}, expected {FORMAT!r})")
    if doc.get("catalogue") != catalog.version():
        raise AnsurError(f"formulas were made for catalogue {doc.get('catalogue')}, this code uses "
                         f"{catalog.version()}: rerun `anthro ansur-fit`")
    return doc


def _inputs(person: dict, photo_cm: dict[str, float]) -> dict[str, float]:
    h, w = person["height_cm"], person["weight_kg"]
    return {"height_cm": h, "weight_kg": w, "bmi": w / (h / 100) ** 2, **photo_cm}


def apply(f: dict, person: dict, photo_cm: dict[str, float]) -> float:
    vals = _inputs(person, photo_cm)
    return float(f["intercept"] + sum(c * vals[n] for c, n in zip(f["coef"], f["inputs"], strict=True)))


def ansur_girth(doc: dict, name: str, person: dict) -> float:
    g = doc["girth_basic"][name][person["sex"]]
    return apply({**g, "inputs": list(BASIC)}, person, {})


# ---------------------------------------------------------------- mix test (chest/bust and waist)

def mix_table(df: pd.DataFrame, pred_cm: np.ndarray, doc: dict) -> pd.DataFrame:
    """One row per BodyM person: truth, photo-model prediction and ANSUR basic estimate for each mix girth."""
    from .config import HEIGHT_COL, MEASUREMENTS, SEX_COL, SUBJECT_COL, WEIGHT_COL
    from .evaluate import per_person_predictions
    pred = per_person_predictions(df, pred_cm)
    people = df.groupby(SUBJECT_COL)[[*MEASUREMENTS, HEIGHT_COL, WEIGHT_COL, SEX_COL]].first().loc[pred.index]
    t = pd.DataFrame(index=pred.index)
    t["male"] = (people[SEX_COL].str.lower() == "male").astype(float)
    for name in MIX_GIRTHS:
        t[f"truth_{name}"] = people[name]
        t[f"photo_{name}"] = pred[name]
        t[f"ansur_{name}"] = [ansur_girth(doc, name, {"sex": s.lower(), "height_cm": h, "weight_kg": w})
                              for s, h, w in zip(people[SEX_COL], people[HEIGHT_COL], people[WEIGHT_COL],
                                                 strict=True)]
    return t


def _mix_X(t: pd.DataFrame, name: str) -> np.ndarray:
    return t[[f"photo_{name}", f"ansur_{name}", "male"]].to_numpy(float)


def mix_decide(val: pd.DataFrame, seed: int = 0) -> dict:
    """Per girth, on validation people: held-out mix errors (5 folds by person) vs the photo model alone.

    Kept only if the mix is better on BOTH average error and % within ±1.5 cm (spec, step 4)."""
    out = {}
    ids = val.index.to_numpy()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    k = min(FOLDS, len(ids))
    fold = np.empty(len(ids), int)
    fold[order] = np.arange(len(ids)) % k
    for name in MIX_GIRTHS:
        X, y = _mix_X(val, name), val[f"truth_{name}"].to_numpy(float)
        mix_err = np.zeros(len(y))
        for f in range(k):
            m = LinearRegression().fit(X[fold != f], y[fold != f])
            mix_err[fold == f] = m.predict(X[fold == f]) - y[fold == f]
        photo = _scores(val[f"photo_{name}"].to_numpy(float) - y)
        mix = _scores(mix_err)
        keep = mix["mae_cm"] < photo["mae_cm"] and mix["pct_within"] > photo["pct_within"]
        m = LinearRegression().fit(X, y)
        out[name] = {"kept": bool(keep), "intercept": float(m.intercept_),
                     "coef": [float(c) for c in m.coef_],
                     "inputs": ["photo", "ansur", "male"], "validation": {"photo_model": photo, "mix": mix}}
    return out


def mix_apply(m: dict, photo_cm: float, ansur_cm: float, male: bool) -> float:
    a, b, c = m["coef"]
    return float(m["intercept"] + a * photo_cm + b * ansur_cm + c * float(male))


def mix_score(test: pd.DataFrame, mix: dict) -> dict:
    """Test-B scores for kept mixes, next to the photo model alone (scored once, after the decision)."""
    out = {}
    for name, m in mix.items():
        if not m["kept"]:
            continue
        y = test[f"truth_{name}"].to_numpy(float)
        pred = np.array([mix_apply(m, p, a, bool(s)) for p, a, s in
                         zip(test[f"photo_{name}"], test[f"ansur_{name}"], test["male"], strict=True)])
        out[name] = {"photo_model": _scores(test[f"photo_{name}"].to_numpy(float) - y),
                     "mix": _scores(pred - y)}
    return out
