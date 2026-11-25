"""S7-0 — suppliers and routing: which source fills which measurement.

specs/s7-0-foundation/spec.md (approved by Ryan, 24 Nov 2026).

What it does
------------
* A *supplier* is anything that can give values for some app codes: the photo model today; ANSUR
  formulas (S5a), outline geometry (S5b), tape calibration (S6) and our own model (S8) later.
  Each one takes the same `Context` (the person and what earlier steps computed) and returns
  `{code: Estimate}` for the codes it can fill.
* `route()` walks the routing table (`data/routing_default.json` or a file given by the user):
  for each code in the person's set, the first listed supplier that is installed and gives a value
  wins; a code nobody fills is `not_available`.

Why these decisions
-------------------
* Swapping a source (e.g. waist from the photo model to our own model) becomes a one-line change
  in the routing file, testable side by side, with no code change. Rejected: one large function
  with special cases per measurement.
* Suppliers that aren't built yet are simply absent, so the routing file can already describe the
  final plan while today's output stays honest (those codes show `not_available`).
* Errors shown to users come from a recorded evaluation report only (Test-B: D-028), never typed
  in by hand, so a number can't drift from what was measured.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Protocol

from . import catalog
from .config import HEIGHT_AGREEMENT_CM
from .output import NOT_AVAILABLE, Estimate


@dataclass
class Context:
    person: dict                                 # sex, age_range, height_cm, weight_kg, weight_kind, ...
    bodym_cm: dict[str, float] = field(default_factory=dict)   # photo model output, BodyM names
    flags: list[str] = field(default_factory=list)            # added to every value (weight_assumed)
    errors: dict[str, float] = field(default_factory=dict)    # BodyM name -> Test-B per-person MAE
    error_basis: str | None = None


class Supplier(Protocol):
    name: str

    def estimate(self, ctx: Context) -> dict[str, Estimate]: ...


class PhotoModel:
    """The trained photo model's 14 BodyM outputs, translated to app codes (catalog.BODYM_TO_CODE)."""

    name = "photo_model"

    def estimate(self, ctx: Context) -> dict[str, Estimate]:
        out = {}
        mapped = catalog.bodym_to_codes(ctx.person["sex"], ctx.bodym_cm)
        for code, (value, match, bodym_name) in mapped.items():
            err = ctx.errors.get(bodym_name)
            out[code] = Estimate(value, self.name, "measured", match, err,
                                 ctx.error_basis if err is not None else None, list(ctx.flags))
        if "height" in out and ctx.person.get("height_kind") == "stated":
            out["height"] = _within_stated(out["height"], ctx.person["height_cm"])
        return out


def _within_stated(est: Estimate, stated_cm: float) -> Estimate:
    """D-029: an estimate of a stated height stays within HEIGHT_AGREEMENT_CM of what was said."""
    lo, hi = stated_cm - HEIGHT_AGREEMENT_CM, stated_cm + HEIGHT_AGREEMENT_CM
    if lo <= est.value_cm <= hi:
        return est
    est.value_cm = min(max(est.value_cm, lo), hi)
    est.flags = [*est.flags, "held_within_2cm_of_stated"]
    return est


class Input:
    """Height as given, when it was measured with a tape (D-029, approved by Ryan 25 Nov 2026): a
    measured height is more accurate than any estimate of it. A stated (self-reported) height falls
    through to the photo model, held within 2 cm of what the person said."""

    name = "input"

    def estimate(self, ctx: Context) -> dict[str, Estimate]:
        if ctx.person.get("height_kind") != "measured":
            return {}
        return {"height": Estimate(ctx.person["height_cm"], self.name, "measured", "same")}


class Ansur:
    """S5a: formulas learned on ANSUR II (anthro.ansur) for the codes BodyM lacks; quality `derived`."""

    name = "ansur"
    BASIS = "ANSUR II held-out (5 folds by person), photo-model-sized input noise"

    def __init__(self, formulas: dict):
        self.doc = formulas

    def estimate(self, ctx: Context) -> dict[str, Estimate]:
        from . import ansur
        out = {}
        for code, by_sex in self.doc["formulas"].items():
            f = by_sex.get(ctx.person["sex"])
            if f is None:
                continue
            if any(n not in ansur.BASIC and n not in ctx.bodym_cm for n in f["inputs"]):
                continue                     # an extended formula needs photo-model values we don't have
            value = ansur.apply(f, ctx.person, ctx.bodym_cm)
            flags = list(ctx.flags)
            lo, hi = f["range_cm"]
            if not lo <= value <= hi:
                flags.append("outside_formula_range")
            out[code] = Estimate(value, self.name, "derived", f["definition_match"], f["mae_cm"],
                                 self.BASIS, flags)
        return out


class Mix:
    """S5a mix test: photo model and ANSUR basic estimate combined for chest/bust and waist, only where the
    mix beat the photo model alone on validation people. Error shown: its Test-B score (D-028)."""

    name = "mix"

    def __init__(self, formulas: dict):
        self.doc = formulas

    def estimate(self, ctx: Context) -> dict[str, Estimate]:
        from . import ansur
        out = {}
        male = ctx.person["sex"] == "male"
        for name, m in self.doc.get("mix", {}).items():
            if not m.get("kept") or name not in ctx.bodym_cm:
                continue
            rule = catalog.BODYM_TO_CODE[name]
            code, match = rule.get(ctx.person["sex"], rule.get("*"))
            a_cm = ansur.ansur_girth(self.doc, name, ctx.person)
            value = ansur.mix_apply(m, ctx.bodym_cm[name], a_cm, male)
            tb = (m.get("testB") or {}).get("mix")
            err = tb["mae_cm"] if tb else None
            basis = "BodyM testB, per-person average error (photo model + ANSUR mix)" if tb else None
            out[code] = Estimate(value, self.name, "measured", match, err, basis, list(ctx.flags))
        return out


def load_routing(path: str | Path | None = None) -> dict[str, list[str]]:
    text = (Path(path).read_text(encoding="utf-8") if path else
            resources.files("anthro").joinpath("data/routing_default.json").read_text(encoding="utf-8"))
    table = {k: v for k, v in json.loads(text).items() if not k.startswith("_")}
    unknown = set(table) - set(catalog.by_code())
    if unknown:
        raise ValueError(f"routing names codes not in the catalogue: {', '.join(sorted(unknown))}")
    missing = set(catalog.by_code()) - set(table)
    if missing:
        raise ValueError(f"routing has no entry for: {', '.join(sorted(missing))}")
    return table


def route(ctx: Context, suppliers: list[Supplier], routing: dict[str, list[str]]) -> dict[str, Estimate]:
    """One Estimate per code in the person's set; each supplier is asked once."""
    results = {s.name: s.estimate(ctx) for s in suppliers}
    chosen = {}
    for code in catalog.codes_for(ctx.person["sex"], ctx.person.get("age_range")):
        chosen[code] = next((results[n][code] for n in routing[code]
                             if n in results and code in results[n]), NOT_AVAILABLE)
    return chosen


def errors_from_report(report: dict, split: str = "testB") -> tuple[dict[str, float], str]:
    """Per-measurement per-person MAE (cm) from an `anthro evaluate` report (D-028: Test-B)."""
    if split not in report:
        raise ValueError(f"evaluation report has no {split} section: run `anthro evaluate --test {split}`")
    rows = report[split]["table"]
    errors = {r["index"]: float(r["photo_MAE_cm"]) for r in rows}
    return errors, f"BodyM {split}, per-person average error"
