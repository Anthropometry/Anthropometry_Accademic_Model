"""S7-0 — the result format `anthro-output/1`: one JSON per person, every measurement in their set.

specs/s7-0-foundation/spec.md (approved by Ryan, 24 Nov 2026). The JSON Schema that documents the
format is `data/anthro-output-1.schema.json`; `validate()` checks the same rules by hand, so no new
dependency is needed.

Why these decisions
-------------------
* Every value says where it came from (`source`), how far to trust it (`quality`: measured /
  derived / rough / not_available), whether its definition matches the app's (`definition_match`),
  and its expected error, taken only from a recorded evaluation (`expected_error_cm`, Test-B: D-028),
  otherwise null. A tailor must never mistake a rough estimate for a measured one.
* Only the person's own set appears (women 24, men 18); a code nobody supplies is `not_available`
  rather than missing, so readers see the gap.
* Versioned (`schema`): a change to the format is `anthro-output/2`, so nothing reading it breaks
  silently.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from . import catalog

SCHEMA = "anthro-output/1"
QUALITIES = ("measured", "derived", "rough", "not_available")
MATCHES = ("same", "presumed", "differs")
WEIGHT_KINDS = ("scale", "stated", "assumed")
HEIGHT_KINDS = ("measured", "stated")


@dataclass
class Estimate:
    value_cm: float | None
    source: str | None
    quality: str
    definition_match: str | None = None
    expected_error_cm: float | None = None
    error_basis: str | None = None
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        err = self.expected_error_cm
        return {"value_cm": None if self.value_cm is None else round(float(self.value_cm), 1),
                "source": self.source, "quality": self.quality,
                "definition_match": self.definition_match,
                "expected_error_cm": None if err is None else round(float(err), 2),
                "error_basis": self.error_basis, "flags": list(self.flags)}


NOT_AVAILABLE = Estimate(None, None, "not_available")


def build(person: dict, estimates: dict[str, Estimate], retake_reasons: list[str], warnings: list[str],
          model_version: str, now: dt.datetime | None = None) -> dict:
    codes = catalog.codes_for(person["sex"], person.get("age_range"))
    doc = {
        "schema": SCHEMA,
        "catalogue": catalog.version(),
        "model_version": model_version,
        "created_at": (now or dt.datetime.now(dt.timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "person": dict(person),     # a copy: later edits to the caller's dict must not change the result
        "retake": {"flag": bool(retake_reasons), "reasons": list(retake_reasons)},
        "warnings": list(dict.fromkeys(warnings)),      # keep order, drop repeats
        "measurements": {c: estimates.get(c, NOT_AVAILABLE).to_dict() for c in codes},
    }
    problems = validate(doc)
    if problems:
        raise ValueError("built an invalid anthro-output/1 document: " + "; ".join(problems))
    return doc


def validate(doc: dict) -> list[str]:
    """Problems with an `anthro-output/1` document (empty list = valid)."""
    p: list[str] = []
    if doc.get("schema") != SCHEMA:
        return [f"schema is {doc.get('schema')!r}, not {SCHEMA!r}"]
    for key in ("catalogue", "model_version", "created_at", "person", "retake", "warnings", "measurements"):
        if key not in doc:
            p.append(f"missing {key}")
    if p:
        return p
    person = doc["person"]
    try:
        expected = catalog.codes_for(person.get("sex"), person.get("age_range"))
    except catalog.CatalogError as e:
        return [str(e)]
    if not isinstance(person.get("height_cm"), (int, float)):
        p.append("person.height_cm must be a number")
    if person.get("height_kind") not in HEIGHT_KINDS:
        p.append(f"person.height_kind must be one of {HEIGHT_KINDS}")
    if person.get("weight_kind") not in WEIGHT_KINDS:
        p.append(f"person.weight_kind must be one of {WEIGHT_KINDS}")
    if not isinstance(doc["retake"].get("flag"), bool) or not isinstance(doc["retake"].get("reasons"), list):
        p.append("retake must have a boolean flag and a list of reasons")
    if set(doc["measurements"]) != set(expected):
        p.append("measurements must be exactly the person's set "
                 f"({len(expected)} codes for {person.get('sex')})")
    for code, m in doc["measurements"].items():
        if m.get("quality") not in QUALITIES:
            p.append(f"{code}: quality {m.get('quality')!r}")
        if m.get("quality") == "not_available":
            if m.get("value_cm") is not None:
                p.append(f"{code}: not_available must have no value")
        elif not isinstance(m.get("value_cm"), (int, float)) or not m.get("source"):
            p.append(f"{code}: needs a number and a source")
        if m.get("definition_match") not in (*MATCHES, None):
            p.append(f"{code}: definition_match {m.get('definition_match')!r}")
        if m.get("expected_error_cm") is not None and not m.get("error_basis"):
            p.append(f"{code}: an expected error needs its basis")
    return p
