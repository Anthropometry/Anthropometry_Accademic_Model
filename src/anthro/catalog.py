"""S7-0 — the measurement catalogue: the app's 25 codes are the backbone of this repository.

specs/s7-0-foundation/spec.md (approved by Ryan, 24 Nov 2026).

What it does
------------
* Loads `data/measurements.json`, a stamped copy of the AnthropometryDC app's catalogue
  (`scripts/sync_catalog.py` makes it), and answers "which measurements does this person get?":
  women 24, men 18, adults only.
* Translates BodyM's 14 names into the app's codes, each with how well the definitions match.

Why these decisions
-------------------
* The app's codes (`waist_girth`, `inseam`, …) are used everywhere: in training, prediction and
  results. When the app's export arrives, its columns already match; BodyM is the one translated.
* `definition_match` is carried with every BodyM value (same / presumed / differs), because BodyM's
  virtual-tape paths were never published (research, 24 Nov 2026): the output must not pretend a
  BodyM hip is a tailor's hip.
* Adults only: BodyM has no children, so the model must not measure anyone under 18 (Ryan, 24 Nov).
* BodyM's ankle, forearm, leg-length and shoulder-to-crotch are not reported (the app doesn't collect
  them) but stay available internally as inputs for later ANSUR formulas (S5a). Leg-length is not
  treated as the app's inseam until its definition is checked.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

ADULT_RANGES = ("18-24", "25-34", "35-44", "45-54", "55-64", "65+")
CHILD_RANGES = ("8-9", "10-12", "13-14", "15-17")
SEXES = ("female", "male")

# BodyM name -> (app code, definition match). Chest depends on sex.
BODYM_TO_CODE: dict[str, dict[str, tuple[str, str]]] = {
    "height": {"*": ("height", "same")},
    "waist": {"*": ("waist_girth", "same")},           # narrowest point, as the app (level search, S3b)
    "chest": {"male": ("chest_girth", "presumed"), "female": ("bust_girth", "presumed")},
    "hip": {"*": ("hip_girth", "differs")},             # BodyM: fixed height; app: fullest part
    "shoulder-breadth": {"*": ("shoulder_width", "differs")},
    "arm-length": {"*": ("arm_length", "presumed")},
    "bicep": {"*": ("upper_arm_girth", "presumed")},
    "wrist": {"*": ("wrist_girth", "presumed")},
    "thigh": {"*": ("thigh_girth", "presumed")},
    "calf": {"*": ("calf_girth", "presumed")},
}
BODYM_INTERNAL_ONLY = ("ankle", "forearm", "leg-length", "shoulder-to-crotch")


class CatalogError(ValueError):
    """A person or a code the catalogue can't serve; the message says why."""


@lru_cache(maxsize=1)
def catalogue() -> dict:
    text = resources.files("anthro").joinpath("data/measurements.json").read_text(encoding="utf-8")
    return json.loads(text)


def version() -> str:
    return catalogue()["catalogue_version"]


def by_code() -> dict[str, dict]:
    return {m["code"]: m for m in catalogue()["measurements"]}


def check_person(sex: str, age_range: str | None) -> None:
    if sex not in SEXES:
        raise CatalogError(f"sex must be 'female' or 'male', not {sex!r}")
    if age_range is None:
        return
    if age_range in CHILD_RANGES:
        raise CatalogError(f"age range {age_range}: this model is for adults only (18+); "
                           "it was trained on adults and has not been tested on children")
    if age_range not in ADULT_RANGES:
        raise CatalogError(f"unknown age range {age_range!r}; expected one of {', '.join(ADULT_RANGES)}")


def codes_for(sex: str, age_range: str | None = None) -> list[str]:
    """The person's measurement set, in the app's order: 24 for women, 18 for men."""
    check_person(sex, age_range)
    return [m["code"] for m in catalogue()["measurements"] if sex in m["sexes"]]


def bodym_to_codes(sex: str, values_cm: dict[str, float]) -> dict[str, tuple[float, str, str]]:
    """BodyM predictions -> {app code: (value, definition match, BodyM name)}, for this sex."""
    out = {}
    for name, value in values_cm.items():
        rule = BODYM_TO_CODE.get(name)
        if rule is None:
            continue                       # internal-only BodyM measurement
        code, match = rule.get(sex, rule.get("*", (None, None)))
        if code is not None:
            out[code] = (float(value), match, name)
    return out
