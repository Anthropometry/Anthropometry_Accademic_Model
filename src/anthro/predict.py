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

from . import catalog, checkpoint, geometry_rough, output, suppliers, typical_weight
from . import geometry as geo
from . import geometry_fit as gfit
from .config import HEIGHT_AGREEMENT_CM, HEIGHT_COL, MEASUREMENTS, SEX_COL, SUBJECT_COL, WEIGHT_COL
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


def rough_estimates(front, side, person, geo_fit, formulas) -> dict:
    """S5b outline estimates; empty when the checkpoint has no geometry fit (non-hybrid models)."""
    if geo_fit is None:
        return {}
    ratio = ((formulas or {}).get("level_ratios") or {}).get("knee_from_floor",
                                                             geometry_rough.KNEE_FROM_FLOOR_TEXTBOOK)
    return geometry_rough.estimate(load_mask(front), load_mask(side), person, geo_fit, ratio)


def sanity_checks(estimates: dict, sex: str, rough: dict) -> tuple[dict, list[str]]:
    """S5b: rough values outside the app's adult range are withheld; out-of-order ones are flagged."""
    out, notes = dict(estimates), []
    labels = catalog.by_code()
    for code, (value, why) in rough.items():
        if value is None and code in catalog.codes_for(sex):
            notes.append(f"{labels[code]['label'].lower()} not available: {why}")
    for code, e in estimates.items():
        if e.quality != "rough":
            continue
        lo, hi = labels[code]["adult_range_cm"]
        if not lo <= e.value_cm <= hi:
            out[code] = output.NOT_AVAILABLE
            notes.append(f"{labels[code]['label'].lower()} withheld: the outline gave {e.value_cm:.1f} cm, "
                         f"outside the adult range {lo}-{hi} cm")

    def v(code):
        e = out.get(code)
        return e.value_cm if e is not None and e.value_cm is not None else None

    checks = [("underbust_girth", lambda u: v("bust_girth") is None or u < v("bust_girth"),
               "under-bust came out larger than bust"),
              ("high_hip_girth", lambda x: (v("waist_girth") is None or x > v("waist_girth"))
               and (v("hip_girth") is None or x < v("hip_girth")),
               "high hip came out outside the waist-to-hip range")]
    for code, ok, rule in checks:
        e = out.get(code)
        if e is not None and e.quality == "rough" and not ok(e.value_cm):
            e.flags = [*e.flags, "order_check_failed"]
            # Some bodies really have this shape (e.g. a belly larger than the hips), so this is a prompt
            # to check, never a reason to withhold the value.
            notes.append(f"{labels[code]['label'].lower()}: {rule}. This can be a real body shape (for "
                         "example a belly larger than the hips) or an outline error: check it with a tape")
    return out, notes


def height_warnings(person: dict, model_height_cm: float) -> list[str]:
    """D-029: the model's height estimate should agree with the given height within 2 cm."""
    gap = model_height_cm - person["height_cm"]
    if abs(gap) < HEIGHT_AGREEMENT_CM:
        return []
    if person["height_kind"] == "measured":
        return [f"the photo model's height ({model_height_cm:.1f} cm) is {abs(gap):.1f} cm from the tape "
                "height: check the typed height and that the photos show the whole body, camera level"]
    return [f"the photo model's height ({model_height_cm:.1f} cm) was more than {HEIGHT_AGREEMENT_CM:g} cm "
            f"from the stated height; reported as {person['height_cm'] + (2 if gap > 0 else -2):.1f} cm "
            "(held within 2 cm)"]


PACKAGED_ANSUR = "data/ansur_formulas.json"


def load_ansur(path):
    """S5a formulas: the file given, else the packaged one (committed after Ryan's run), else none."""
    from importlib import resources

    from . import ansur
    if path:
        return ansur.load_formulas(path)
    packaged = resources.files("anthro").joinpath(PACKAGED_ANSUR)
    return ansur.load_formulas(str(packaged)) if packaged.is_file() else None


def load_report(ckpt: str | Path, report_path: str | Path | None) -> dict | None:
    p = Path(report_path) if report_path else Path(ckpt).parent / "eval_report.json"
    return json.loads(p.read_text()) if p.exists() else None


def predict(front, side, height_cm, sex, ckpt, weight_kg=None, age_range=None, height_kind="measured",
            clothing=None, data_root=None, report_path=None, routing_path=None,
            weight_kind="stated", ansur_path=None, use_ansur=True) -> dict:
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
    warnings += height_warnings(person, values["height"])

    ctx = suppliers.Context(person, values, ["weight_assumed"] if person["weight_kind"] == "assumed" else [],
                            errors, basis)
    chain: list = [suppliers.Input(), suppliers.PhotoModel()]
    formulas = load_ansur(ansur_path) if use_ansur else None
    if formulas is not None:
        chain += [suppliers.Ansur(formulas), suppliers.Mix(formulas)]
    rough = rough_estimates(front, side, person, ck.get("geo_fit"), formulas)
    chain.append(suppliers.Geometry(rough))
    estimates = suppliers.route(ctx, chain, suppliers.load_routing(routing_path))
    estimates, notes = sanity_checks(estimates, person["sex"], rough)
    warnings += notes
    if any("fitted_on_ansur_black_participants" in e.flags for e in estimates.values()):
        # D-032. The text is shown to users, so it carries no internal references.
        warnings.append("inseam and hip use formulas fitted on the Black participants of a US Army survey "
                        "(the closest group available for Ghanaian users); replaced once Ghanaian tape "
                        "data exists")
    model_version = (f"{Path(ckpt).parent.name} (anthro {ck.get('package_version')}, "
                     f"data {ck.get('data_fingerprint')})")
    return output.build(person, estimates, reasons, [*warnings, *FIXED_WARNINGS], model_version)


def predict_photos(front_photo, side_photo, height_cm, sex, ckpt, seg, bodym_side: str | None,
                   save_dir=None, front_tilt=None, side_tilt=None, **kw) -> dict:
    """S7a: two phone photos -> BodyM-like silhouettes -> the same result as `predict` (plus `inputs`).

    Silhouettes go to `save_dir` when given (for checking; Drive, never the repository), otherwise to a
    temporary folder that is deleted afterwards."""
    import tempfile

    from PIL import Image

    from . import photo
    sil_f = photo.silhouette(photo.load_photo(front_photo), seg, "front", tilt=front_tilt)
    sil_s = photo.silhouette(photo.load_photo(side_photo), seg, "side", bodym_side, tilt=side_tilt)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(save_dir) if save_dir else Path(tmp)
        d.mkdir(parents=True, exist_ok=True)
        fp, sp = d / "front_silhouette.png", d / "side_silhouette.png"
        Image.fromarray(sil_f.mask).save(fp)
        Image.fromarray(sil_s.mask).save(sp)
        doc = predict(fp, sp, height_cm, sex, ckpt, **kw)
    doc["inputs"] = {"kind": "photos", "segmenter": seg.name, "segmenter_version": seg.version(),
                     "side_facing": sil_s.facing, "side_mirrored": sil_s.mirrored}
    doc["warnings"] += sil_f.notes + sil_s.notes
    return doc


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
