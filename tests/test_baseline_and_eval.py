import numpy as np

from anthro import baseline
from anthro.config import MEASUREMENTS
from anthro.data import load_all, people
from anthro.evaluate import by_group, consistency, per_person_errors, summarise


def test_baseline_scores_one_row_per_person(bodym):
    s = load_all(bodym)
    err = baseline.errors(baseline.fit(s["train"]), s["testA"])
    assert len(err) == len(people(s["testA"])) == 6
    assert list(err.columns) == list(MEASUREMENTS)


def test_per_person_averaging_counts_each_person_once(bodym):
    df = load_all(bodym)["testA"]                       # 3 pairs per person
    truth = df[list(MEASUREMENTS)].to_numpy()
    err = per_person_errors(df, truth + 1.0)            # every prediction 1 cm too high
    assert len(err) == 6
    assert np.allclose(err.to_numpy(), 1.0)


def test_summary_columns_and_tolerance(bodym):
    df = load_all(bodym)["testA"]
    err = per_person_errors(df, df[list(MEASUREMENTS)].to_numpy() + 2.0)   # all 2 cm off
    s = summarise(err)
    assert (s["pct_within_1.5cm"] == 0).all()
    assert np.allclose(s["MAE_cm"], 2.0)


def test_consistency_is_zero_for_identical_predictions(bodym):
    df = load_all(bodym)["testA"]
    assert (consistency(df, df[list(MEASUREMENTS)].to_numpy()) == 0).all()


def test_group_report_covers_every_person(bodym):
    df = load_all(bodym)["testA"]
    err = per_person_errors(df, df[list(MEASUREMENTS)].to_numpy())
    assert by_group(df, err)["people"].sum() == 6
