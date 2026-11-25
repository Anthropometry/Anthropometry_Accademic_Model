"""S7-0 — read an AnthropometryDC export folder into one table, in catalogue codes.

specs/s7-0-foundation/spec.md section 5 (approved by Ryan, 24 Nov 2026). The format is the app's
F8 export (`anthro-export-YYYY-MM-DD-HH-MM/`: README.txt, participants.csv, measurements.csv,
photos.csv, images/<id>/{front,side,back}.jpg, metadata/<id>.json).

What it returns (`Export`)
--------------------------
* `people`: one row per participant: sex, age range, clothing, stated and scale weight kept
  separately, stated height, plus one column per catalogue code (value in cm; empty where skipped
  or not in the person's set), `<code>__readings` (both readings, for measurer repeatability) and
  `<code>__skip_reason`.
* `photos`: one row per photo with its path, phone tilt and head-to-heel pixels.
* `format`, `warnings`, and counts of rows left out (practice, under 18).

Checks (refusals raise `ExportError`)
-------------------------------------
* Format stamp: none yet in the app. Unstamped → format 0 with a warning; `anthro-export/1` →
  accepted; anything else → refused, so a future format change can't be misread silently.
* Missing columns → refused, naming them.
* Practice rows → refused unless `include_practice=True` (then kept and marked).
* Codes not in the catalogue → refused (the copies have drifted: run scripts/sync_catalog.py).
* Under-18 rows are left out with a count: this model is for adults only (Ryan, 24 Nov).

Policy: this only reads a folder Ryan points it at. It never touches the app's database or
storage, and it is tested only on fake folders the tests generate.
"""

from __future__ import annotations

import json
import warnings as _w
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import catalog

KNOWN_FORMATS = ("anthro-export/1",)
PARTICIPANT_COLS = {"export_id", "practice", "sex", "age_range", "clothing", "stated_height_cm",
                    "stated_weight_kg", "weight_kg", "weight_source"}
MEASUREMENT_COLS = {"export_id", "code", "value_cm", "readings_cm", "skipped", "skip_reason"}
PHOTO_COLS = {"export_id", "slot", "file", "level_pitch_deg", "level_roll_deg", "head_top_y", "heel_y"}


class ExportError(ValueError):
    """The folder can't be read as an AnthropometryDC export; the message says why."""


@dataclass
class Export:
    format: str
    people: pd.DataFrame
    photos: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    left_out: dict[str, int] = field(default_factory=dict)


def _truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def _read(folder: Path, name: str, need: set[str]) -> pd.DataFrame:
    p = folder / name
    if not p.exists():
        raise ExportError(f"{name} is missing from {folder.name}")
    df = pd.read_csv(p, dtype={"export_id": str})
    missing = need - set(df.columns)
    if missing:
        raise ExportError(f"{name} is missing columns: {', '.join(sorted(missing))}")
    return df


def detect_format(folder: Path) -> str:
    """The format stamp from metadata/*.json or README.txt; '0' if there is none."""
    stamps = set()
    for meta in sorted((folder / "metadata").glob("*.json")):
        fmt = json.loads(meta.read_text(encoding="utf-8")).get("format")
        if fmt:
            stamps.add(fmt)
    readme = folder / "README.txt"
    if readme.exists():
        for line in readme.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("format:"):
                stamps.add(line.split(":", 1)[1].strip())
    if len(stamps) > 1:
        raise ExportError(f"the export carries more than one format stamp: {', '.join(sorted(stamps))}")
    return stamps.pop() if stamps else "0"


def load(folder: str | Path, include_practice: bool = False) -> Export:
    folder = Path(folder)
    if not folder.is_dir():
        raise ExportError(f"{folder} is not a folder")
    fmt = detect_format(folder)
    notes = []
    if fmt == "0":
        notes.append("export has no format stamp: read as format 0 (the F8 layout of Nov 2026)")
    elif fmt not in KNOWN_FORMATS:
        raise ExportError(f"unknown export format {fmt!r}; this loader reads {', '.join(KNOWN_FORMATS)}")

    parts = _read(folder, "participants.csv", PARTICIPANT_COLS)
    meas = _read(folder, "measurements.csv", MEASUREMENT_COLS)
    photos = _read(folder, "photos.csv", PHOTO_COLS)

    unknown = set(meas["code"]) - set(catalog.by_code())
    if unknown:
        raise ExportError(f"measurement codes not in the catalogue: {', '.join(sorted(unknown))} "
                          "(update with scripts/sync_catalog.py)")

    left_out = {}
    practice = _truthy(parts["practice"])
    if practice.any() and not include_practice:
        raise ExportError(f"{int(practice.sum())} practice row(s) in the export: practice data is not "
                          "research data; use --include-practice to keep them, marked")
    parts = parts.assign(practice=practice)
    child = parts["age_range"].isin(catalog.CHILD_RANGES)
    if "age_group" in parts:
        child |= parts["age_group"].astype(str).str.lower().eq("child")
    if child.any():
        left_out["under_18"] = int(child.sum())
        notes.append(f"{int(child.sum())} participant(s) under 18 left out: this model is for adults only")
    parts = parts[~child]

    people = parts.set_index("export_id")
    meas = meas[meas["export_id"].isin(people.index)]
    skipped = _truthy(meas["skipped"])
    values = meas[~skipped].pivot_table(index="export_id", columns="code", values="value_cm", aggfunc="first")
    readings = meas.pivot_table(index="export_id", columns="code", values="readings_cm", aggfunc="first")
    reasons = meas[skipped].pivot_table(index="export_id", columns="code", values="skip_reason",
                                        aggfunc="first")
    for code in catalog.by_code():
        people[code] = values[code] if code in values else pd.NA
        people[f"{code}__readings"] = (readings[code].map(_parse_readings) if code in readings else pd.NA)
        people[f"{code}__skip_reason"] = reasons[code] if code in reasons else pd.NA
    # A value for a code outside the person's set (e.g. bust girth on a man) is a data-entry error.
    for sex in catalog.SEXES:
        outside = set(catalog.by_code()) - set(catalog.codes_for(sex))
        rows = people["sex"] == sex
        for code in outside:
            bad = rows & people[code].notna()
            if bad.any():
                notes.append(f"{int(bad.sum())} {sex} participant(s) have {code}, "
                             "which is not in their set: ignored")
                people.loc[bad, code] = pd.NA

    photos = photos[photos["export_id"].isin(people.index)].copy()
    photos["path"] = [str(folder / f) for f in photos["file"]]      # the app writes images/<id>/<slot>.jpg
    for n in notes:
        _w.warn(n, stacklevel=2)
    return Export(fmt, people.reset_index(), photos.reset_index(drop=True), notes, left_out)


def _parse_readings(v) -> list[float] | None:
    if pd.isna(v) or str(v).strip() == "":
        return None
    return [float(x) for x in str(v).split("|")]
