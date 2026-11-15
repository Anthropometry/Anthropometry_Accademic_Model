import pandas as pd
import pytest

from anthro.data import BodyMFormatError, fingerprint, load_all, load_split, people, split_by_person


def test_rows_are_photo_pairs_and_people_dedupes(bodym):
    df = load_split(bodym, "testA")
    assert len(df) == 6 * 3            # 6 people x 3 pairs
    assert len(people(df)) == 6        # the per-person view used for every score (D-007)


def test_split_by_person_never_shares_a_person(bodym):
    train = load_all(bodym)["train"]
    tr, va = split_by_person(train, 0.25, seed=0)
    assert not set(tr.subject_id) & set(va.subject_id)
    assert set(tr.subject_id) | set(va.subject_id) == set(train.subject_id)


def test_split_is_reproducible(bodym):
    train = load_split(bodym, "train")
    a = split_by_person(train, 0.25, seed=1)[1].subject_id.unique()
    b = split_by_person(train, 0.25, seed=1)[1].subject_id.unique()
    assert sorted(a) == sorted(b)


def test_missing_column_is_refused(bodym):
    p = bodym / "testB" / "measurements.csv"
    pd.read_csv(p).drop(columns=["waist"]).to_csv(p, index=False)
    with pytest.raises(BodyMFormatError, match="waist"):
        load_split(bodym, "testB")


def test_duplicate_subject_is_refused(bodym):
    p = bodym / "testB" / "hwg_metadata.csv"
    df = pd.read_csv(p)
    pd.concat([df, df.iloc[:1]]).to_csv(p, index=False)
    with pytest.raises(BodyMFormatError, match="repeated"):
        load_split(bodym, "testB")


def test_fingerprint_changes_when_data_changes(bodym):
    before = fingerprint(bodym)
    p = bodym / "train" / "measurements.csv"
    p.write_text(p.read_text().replace("train-0", "train-X", 1))
    assert fingerprint(bodym) != before
