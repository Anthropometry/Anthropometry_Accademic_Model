"""Load BodyM and shape it into tables the rest of the code can trust.

BodyM layout (confirmed by running notebooks/01_explore_bodym.ipynb on the real data, 15 Nov 2026):

    <root>/<split>/hwg_metadata.csv          subject_id, gender, height_cm, weight_kg   (inputs)
    <root>/<split>/measurements.csv          subject_id + 14 measurements in cm          (answers)
    <root>/<split>/subject_to_photo_map.csv  subject_id, photo_id                        (links)
    <root>/<split>/mask/<photo_id>.png       front silhouette
    <root>/<split>/mask_left/<photo_id>.png  side silhouette

    split  people  photo pairs
    train   2018     6134      (~3 pairs per person)
    testA     87     1684      (~19 pairs per person)
    testB    400     1160      (~3 pairs per person)

Challenges met and fixed
------------------------
* Rows are PHOTOS, not people. Joining the three tables gives one row per photo pair, so a
  person with 19 pairs appears 19 times. Scoring on those rows let a few Test-A people
  dominate the result. Fix: `people()` gives one row per person, and every score in
  evaluate.py averages a person's photos first (DISCUSSION D-007).
* Two different "heights". `height_cm` is the entered height (an INPUT); `height` is
  head-to-heel measured on the 3D scan (an ANSWER). They differ by ~1 cm on average. Using
  the scan height as an input would leak the answer, so it is never used as a feature.
* Leakage between train and validation. Splitting photos at random would put one person's
  photos on both sides and inflate the score. Fix: `split_by_person()` (DISCUSSION D-008).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MEASUREMENTS, PHOTO_COL, SUBJECT_COL

SPLITS = ("train", "testA", "testB")
_META_COLS = {SUBJECT_COL, "gender", "height_cm", "weight_kg"}


class BodyMFormatError(ValueError):
    """The files on disk don't match the layout this code was written against."""


def _read(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise BodyMFormatError(f"missing file: {path}")
    df = pd.read_csv(path)
    missing = required - set(df.columns)
    if missing:
        raise BodyMFormatError(f"{path.name} lacks columns {sorted(missing)}")
    return df


def load_split(root: str | Path, split: str) -> pd.DataFrame:
    """One row per photo pair, with that person's inputs and 14 measurements.

    Validates as it loads, because a silent mismatch (a renamed column, a duplicated
    subject) would otherwise surface much later as a puzzling accuracy number.
    """
    d = Path(root) / split
    meta = _read(d / "hwg_metadata.csv", _META_COLS)
    meas = _read(d / "measurements.csv", {SUBJECT_COL, *MEASUREMENTS})
    pmap = _read(d / "subject_to_photo_map.csv", {SUBJECT_COL, PHOTO_COL})

    for name, df in (("hwg_metadata", meta), ("measurements", meas)):
        dupes = df[SUBJECT_COL].duplicated().sum()
        if dupes:
            raise BodyMFormatError(f"{split}/{name}.csv has {dupes} repeated subject_id rows")

    df = pmap.merge(meta, on=SUBJECT_COL, how="inner").merge(meas, on=SUBJECT_COL, how="inner")
    if len(df) != len(pmap):
        lost = len(pmap) - len(df)
        raise BodyMFormatError(f"{split}: {lost} photos have no matching person in the tables")
    if df[[*MEASUREMENTS, "height_cm", "weight_kg"]].isna().any().any():
        raise BodyMFormatError(f"{split}: missing values in inputs or measurements")

    df["split"] = split
    return df.reset_index(drop=True)


def load_all(root: str | Path) -> dict[str, pd.DataFrame]:
    return {s: load_split(root, s) for s in SPLITS}


def people(df: pd.DataFrame) -> pd.DataFrame:
    """One row per person. Safe because inputs and answers are identical across a person's photos."""
    return df.drop_duplicates(SUBJECT_COL).reset_index(drop=True)


def split_by_person(df: pd.DataFrame, val_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold back whole people (never single photos) for validation."""
    rng = np.random.default_rng(seed)
    ids = np.sort(df[SUBJECT_COL].unique())  # sorted first, so the split is the same on every machine
    n_val = max(1, int(round(len(ids) * val_fraction)))
    val_ids = set(rng.choice(ids, size=n_val, replace=False))
    is_val = df[SUBJECT_COL].isin(val_ids)
    train, val = df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)
    assert not set(train[SUBJECT_COL]) & set(val[SUBJECT_COL]), "person in both train and val"
    return train, val


def fingerprint(root: str | Path) -> str:
    """Short hash of the nine BodyM tables.

    Stored in every checkpoint: if BodyM is ever re-downloaded and has changed, or a file is
    corrupted, results can no longer be silently compared with older ones (see docs/SECURITY.md,
    "data integrity").
    """
    h = hashlib.sha256()
    for split in SPLITS:
        for name in ("hwg_metadata.csv", "measurements.csv", "subject_to_photo_map.csv"):
            p = Path(root) / split / name
            if p.exists():
                h.update(p.read_bytes())
    return h.hexdigest()[:16]
