"""S7-0 — measure one adult from a front and a side silhouette: the demo's engine.

specs/s7-0-foundation/spec.md (approved by Ryan, 24 Nov 2026). Command:
    anthro predict-silhouettes --front F --side S --height H [--weight W] --sex female
                               --age-range 25-34 --ckpt runs/r4/best.pt [--out result.json]

What it does
------------
1. Checks the person: adults only (catalog.check_person), height and weight in range.
2. Fills a missing weight with a typical weight for height and sex (D-027), with a warning, and
   marks every photo-model value `weight_assumed`.
3. Runs the trained photo model on the silhouettes (plus the geometry features if the checkpoint
   is a hybrid, as r4 is), giving BodyM's 14 numbers.
4. Routes them to the app's codes (suppliers.route) and writes an `anthro-output/1` document.

Why these decisions
-------------------
* Input is a silhouette pair, like BodyM's, because turning a phone photo into a silhouette needs
  a segmentation model: a new dependency, so its own spec (S7a) with Ryan's approval first.
* Retake warning uses the same rule as evaluation (largest gap between geometry and network over
  chest, waist, hip), with the threshold recorded in the evaluation report, never re-chosen here.
* Error figures come from the checkpoint folder's `eval_report.json`, Test-B section (D-028). If
  there is none, errors are left empty (null) rather than guessed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import catalog, checkpoint, output, suppliers, typical_weight
from . import geometry as geo
from . import geometry_fit as gfit
from .config import HEIGHT_COL, MEASUREMENTS, SEX_COL, SUBJECT_COL, WEIGHT_COL
from .evaluate import RETAKE_LEVELS
from .preprocess import check_inputs, check_silhouette, crop_pad_resize, load_mask

FIXED_WARNINGS = (
    "everyday clothing has not been tested: BodyM photos are of tight clothing",
    "not yet validated on Ghanaian bodies: trained on BodyM (US adults)",
)
BMI_WARNING = "BMI 35 or more: the model is less accurate for this group (docs/RESULTS.md)"


class PredictError(ValueError):
    """The person or the photos can't be measured; the message says why."""


def _person(sex, age_range, height_cm, weight_kg, height_kind, clothing, tw_fit,
            weight_kind="stated") -> tuple[dict, list[str]]:
    try:
        catalog.check_person(sex, age_range)
    except catalog.CatalogError as e:
        raise PredictError(str(e)) from e
    checks = check_inputs(height_cm, weight_kg)
    if not checks.ok:
        raise PredictError("; ".join(checks.problems))
    warnings = list(checks.warnings)
    if weight_kg is None:
        weight_kg = typical_weight.typical(tw_fit, sex, height_cm)
        weight_kind = "assumed"
        warnings.insert(0, typical_weight.WARNING.format(weight_kg))
    if weight_kg / (height_cm / 100) ** 2 >= 35:
        warnings.append(BMI_WARNING)
    person = {"sex": sex, "age_range": age_range, "height_cm": float(height_cm), "height_kind": height_kind,
              "weight_kg": round(float(weight_kg), 1), "weight_kind": weight_kind, "clothing": clothing}
    return person, warnings


def _tw_fit(ck: dict, data_root: str | None) -> dict:
    fit_ = (ck.get("extra") or {}).get("typical_weight")
    if fit_:
        return fit_
    if data_root is None:
        raise PredictError("this checkpoint has no typical-weight fit (trained before S7-0): "
                           "give --weight, or --data BODYM so it can be fitted from the training table")
    from .data import load_split
    return typical_weight.fit(load_split(data_root, "train"))


def run_model(net, cfg, xs, ys, ck, front_path, side_path, person):
    """BodyM's 14 values for one pair, retake reasons, and the geometry table (None if not hybrid)."""
    front, side = load_mask(front_path), load_mask(side_path)
    for name, m in (("front", front), ("side", side)):
        c = check_silhouette(m)
        if not c.ok:
            raise PredictError(f"{name} silhouette: " + "; ".join(c.problems))
    img = np.stack([crop_pad_resize(front, cfg.img_h, cfg.img_w),
                    crop_pad_resize(side, cfg.img_h, cfg.img_w)]).astype(np.float32) / 255.0
    hw = xs.encode(np.array([[person["height_cm"], person["weight_kg"]]], np.float64))[0]
    num = np.array([hw[0], hw[1], float(person["sex"] == "male")], np.float32)
    h, w = img.shape[1:]
    x = torch.cat([torch.from_numpy(img), torch.from_numpy(num)[:, None, None].expand(3, h, w)], 0)[None]

    reasons: list[str] = []
    geo_fit, g, table = ck.get("geo_fit"), torch.zeros(1, 0), None
    if geo_fit is not None:
        try:
            m = gfit.measure_pair(front_path, side_path, person["height_cm"])
        except geo.GeometryError as e:
            m = None
            reasons.append(f"the outline could not be measured ({e}): retake the photos")
        df = pd.DataFrame({HEIGHT_COL: [person["height_cm"]], WEIGHT_COL: [person["weight_kg"]],
                           SEX_COL: [person["sex"]], SUBJECT_COL: ["person"]})
        t = gfit.level_table(df, gfit.single_cache(m), geo_fit["params"])
        g = torch.from_numpy(gfit.feature_matrix(t, geo_fit))
        table = gfit.apply_correction(t, geo_fit)
        if m is not None and not bool(t["geo_ok"].iloc[0]):
            reasons.append("chest, waist or hip level could not be found on the outline: retake the photos")
    from .model import forward
    with torch.no_grad():
        z = forward(net, x, g).cpu().numpy()
    values = dict(zip(MEASUREMENTS, ys.decode(z)[0].tolist(), strict=True))
    return values, reasons, table


def retake_check(values: dict, table, threshold: float | None) -> list[str]:
    if table is None or threshold is None or not bool(table["geo_ok"].iloc[0]):
        return []
    gaps = {lv: abs(float(table[f"{lv}_C_hat"].iloc[0]) - values[lv]) for lv in RETAKE_LEVELS}
    worst = max(gaps, key=gaps.get)
    if gaps[worst] > threshold:
        return [f"outline geometry and photo model disagree on {worst} by {gaps[worst]:.1f} cm "
                f"(limit {threshold:.1f} cm): retake the photos"]
    return []


def load_report(ckpt: str | Path, report_path: str | Path | None) -> dict | None:
    p = Path(report_path) if report_path else Path(ckpt).parent / "eval_report.json"
    return json.loads(p.read_text()) if p.exists() else None


def predict(front, side, height_cm, sex, ckpt, weight_kg=None, age_range=None, height_kind="measured",
            clothing=None, data_root=None, report_path=None, routing_path=None,
            weight_kind="stated") -> dict:
    net, cfg, xs, ys, ck = checkpoint.load(ckpt, "cpu")
    person, warnings = _person(sex, age_range, height_cm, weight_kg, height_kind, clothing,
                               _tw_fit(ck, data_root) if weight_kg is None else None, weight_kind)
    values, reasons, table = run_model(net, cfg, xs, ys, ck, front, side, person)

    report = load_report(ckpt, report_path)
    errors, basis, threshold = {}, None, None
    if report is None:
        warnings.append("no evaluation report found next to the checkpoint: expected errors are not shown")
    else:
        errors, basis = suppliers.errors_from_report(report, "testB")
        threshold = (report["testB"].get("retake") or {}).get("threshold_cm")
    reasons += retake_check(values, table, threshold)

    ctx = suppliers.Context(person, values, ["weight_assumed"] if person["weight_kind"] == "assumed" else [],
                            errors, basis)
    estimates = suppliers.route(ctx, [suppliers.Input(), suppliers.PhotoModel()],
                                suppliers.load_routing(routing_path))
    model_version = (f"{Path(ckpt).parent.name} (anthro {ck.get('package_version')}, "
                     f"data {ck.get('data_fingerprint')})")
    return output.build(person, estimates, reasons, [*warnings, *FIXED_WARNINGS], model_version)


def to_text(doc: dict) -> str:
    """The result as a readable table, for the notebook and the class demo."""
    p = doc["person"]
    lines = [f"{p['sex']}, {p.get('age_range') or 'age not given'}, "
             f"height {p['height_cm']:g} cm ({p['height_kind']}), "
             f"weight {p['weight_kg']:g} kg ({p['weight_kind']})",
             f"retake: {'YES' if doc['retake']['flag'] else 'no'}"]
    lines += [f"  - {r}" for r in doc["retake"]["reasons"]]
    lines.append("")
    labels = catalog.by_code()
    lines.append(f"{'measurement':26s} {'cm':>7s}  {'quality':14s} {'source':12s} {'± error':>8s}")
    for code, m in doc["measurements"].items():
        v = "—" if m["value_cm"] is None else f"{m['value_cm']:.1f}"
        e = "—" if m["expected_error_cm"] is None else f"{m['expected_error_cm']:.2f}"
        q = m["quality"] + (" *" if m["definition_match"] in ("presumed", "differs") else "")
        lines.append(f"{labels[code]['label'][:26]:26s} {v:>7s}  {q:14s} {(m['source'] or '—'):12s} {e:>8s}")
    lines.append("\n* definition presumed or known to differ from the app's (BodyM virtual tape)")
    lines.append("warnings:")
    lines += [f"  - {w}" for w in doc["warnings"]]
    return "\n".join(lines)
