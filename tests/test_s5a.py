"""S5a on fake data: ANSUR loading, formulas, honest held-out errors, mix test, and the ANSUR supplier.

Fake ANSUR files with the real column names; numbers are invented, so only the rules are tested here.
"""

import json

import numpy as np
import pandas as pd
import pytest
from conftest import make_fake_ansur, make_hybrid_run

from anthro import ansur, catalog, output, predict, suppliers
from anthro.cli import main

NOISE = {"chest": 2.4, "waist": 2.4, "shoulder-breadth": 0.7, "thigh": 1.3, "calf": 1.0, "wrist": 0.5,
         "hip": 2.0}


@pytest.fixture(scope="module")
def fake_ansur(tmp_path_factory):
    return make_fake_ansur(tmp_path_factory.mktemp("ansur") / "ansur2")


@pytest.fixture(scope="module")
def table(fake_ansur):
    return ansur.load(fake_ansur)


@pytest.fixture(scope="module")
def formulas(table):
    doc, scores = ansur.fit(table, NOISE, catalog.version(), "fp", "test")
    return doc, scores


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return make_hybrid_run(tmp_path_factory.mktemp("s5a"))


# ---------------------------------------------------------------- loading

def test_loads_both_files_in_cm_and_kg(table):
    assert set(table["sex"]) == {"female", "male"} and len(table) == 500
    assert 150 < table["height_cm"].median() < 190 and 50 < table["weight_kg"].median() < 100
    assert 30 < table["neck_girth"].median() < 50 and table["inseam"].min() > 0
    assert table["stated_height_cm"].notna().all()


def test_missing_column_is_refused(tmp_path):
    f = make_fake_ansur(tmp_path / "a")
    p = f / ansur.FILES["male"]
    pd.read_csv(p).drop(columns=["crotchheight"]).to_csv(p, index=False)
    with pytest.raises(ansur.AnsurError, match="crotchheight"):
        ansur.load(f)


def test_missing_file_and_wrong_units_are_refused(tmp_path):
    f = make_fake_ansur(tmp_path / "a")
    p = f / ansur.FILES["female"]
    df = pd.read_csv(p)
    df["neckcircumferencebase"] = df["neckcircumferencebase"] * 1000        # as if in micrometres
    df.to_csv(p, index=False)
    with pytest.raises(ansur.AnsurError, match="units"):
        ansur.load(f)
    p.unlink()
    with pytest.raises(ansur.AnsurError, match="not found"):
        ansur.load(f)


def test_a_folder_inside_the_repository_is_refused():
    with pytest.raises(ansur.AnsurError, match="outside the repository"):
        ansur.load(ansur.Path(__file__).resolve().parents[1] / "data")


def test_usual_download_names_are_accepted_and_a_miss_lists_the_folder(tmp_path):
    import zipfile
    f = make_fake_ansur(tmp_path / "a")
    male = f / ansur.FILES["male"]
    male.rename(f / "ANSUR_II_MALE_Public.csv")
    female = f / ansur.FILES["female"]
    with zipfile.ZipFile(f / "ANSUR II FEMALE Public.zip", "w") as z:
        z.write(female, female.name)
    female.unlink()
    assert len(ansur.load(f)) == 500
    (f / "ANSUR_II_MALE_Public.csv").rename(f / "men.csv")
    with pytest.raises(ansur.AnsurError, match="men.csv"):
        ansur.load(f)


def test_duplicate_subject_is_refused(tmp_path):
    f = make_fake_ansur(tmp_path / "a")
    p = f / ansur.FILES["male"]
    df = pd.read_csv(p)
    df.loc[1, "subjectid"] = df.loc[0, "subjectid"]
    df.to_csv(p, index=False)
    with pytest.raises(ansur.AnsurError, match="twice"):
        ansur.load(f)


def test_latin1_file_is_read(tmp_path):
    f = make_fake_ansur(tmp_path / "a")
    p = f / ansur.FILES["female"]
    df = pd.read_csv(p)
    df["note"] = "Bogotá"
    df.to_csv(p, index=False, encoding="latin-1")
    assert len(ansur.load(f)) == 500


# ---------------------------------------------------------------- formulas and scores

def test_folds_are_by_person(table):
    f = ansur.folds(table)
    assert table.assign(f=f).groupby("subjectid")["f"].nunique().max() == 1
    assert set(f) == set(range(ansur.FOLDS))


def test_noise_has_the_test_b_size():
    rng = np.random.default_rng(0)
    assert abs(np.abs(rng.normal(0, ansur.sigma_from_mae(2.4), 200_000)).mean() - 2.4) < 0.02


def test_fit_covers_the_right_codes_per_sex_and_is_plain_json(formulas):
    doc, scores = formulas
    assert json.loads(json.dumps(doc)) == doc and doc["format"] == ansur.FORMAT
    women = {c for c, s in doc["formulas"].items() if "female" in s}
    men = {c for c, s in doc["formulas"].items() if "male" in s}
    assert len(women) == 9 and len(men) == 7
    assert doc["formulas"]["shoulder_to_knee"]["female"]["definition_match"] == "differs"   # D-031
    assert {"sex average", "basic", "extended, blurred", "extended, tape inputs"} <= set(scores["formula"])
    assert scores.groupby(["code", "sex"])["chosen"].sum().eq(1).all()


def test_blur_makes_the_extended_formula_honest(table):
    clean = np.abs(ansur.cv_errors(table[table.sex == "male"].reset_index(drop=True), "hip_girth",
                                   ansur.extended_inputs("neck_girth"), None)).mean()
    blurred = np.abs(ansur.cv_errors(table[table.sex == "male"].reset_index(drop=True), "hip_girth",
                                     ansur.extended_inputs("neck_girth"), {k: 5.0 for k in NOISE})).mean()
    assert blurred > clean


def test_hip_formula_never_uses_the_photo_models_hip():
    assert "hip" not in ansur.extended_inputs("hip_girth") and "hip" in ansur.extended_inputs("neck_girth")


def test_subgroups_include_black_participants_and_bmi_bands(table, formulas):
    t = ansur.subgroup_table(table, formulas[0], NOISE)
    assert {"all", "Black participants", "all other participants"} <= set(t["group"])
    assert t["group"].str.startswith("BMI").any() and (t["people"] >= 20).all()


def test_self_report_check_runs(table):
    t = ansur.self_report_check(table)
    assert len(t) == 16 and (t["stated_mae_cm"] > 0).all()


def test_formula_file_for_another_catalogue_is_refused(formulas, tmp_path):
    p = tmp_path / "f.json"
    p.write_text(json.dumps({**formulas[0], "catalogue": "anthropometrydc-measurements@old"}))
    with pytest.raises(ansur.AnsurError, match="rerun"):
        ansur.load_formulas(p)
    p.write_text(json.dumps({"format": "something/9"}))
    with pytest.raises(ansur.AnsurError, match="not an ANSUR formula file"):
        ansur.load_formulas(p)


# ---------------------------------------------------------------- mix

def _mix_frame(n=60, seed=0, ansur_helps=True):
    rng = np.random.default_rng(seed)
    truth = rng.normal(90, 10, n)
    t = pd.DataFrame({"male": rng.integers(0, 2, n).astype(float)})
    for name in ansur.MIX_GIRTHS:
        t[f"truth_{name}"] = truth
        t[f"photo_{name}"] = truth + rng.normal(0, 3, n)
        t[f"ansur_{name}"] = truth + rng.normal(0, 3 if ansur_helps else 60, n)
    return t


def test_mix_of_two_independent_estimates_is_kept():
    m = ansur.mix_decide(_mix_frame(n=300))
    assert all(v["kept"] for v in m.values())
    assert m["waist"]["validation"]["mix"]["mae_cm"] < m["waist"]["validation"]["photo_model"]["mae_cm"]


def test_mix_that_loses_is_not_kept_and_not_routed(formulas):
    m = ansur.mix_decide(_mix_frame(n=40, ansur_helps=False, seed=3))
    m = {k: {**v, "kept": False} for k, v in m.items()}          # whatever the fold noise did
    doc = {**formulas[0], "mix": m}
    ctx = suppliers.Context({"sex": "male", "height_cm": 175, "weight_kg": 70}, {"chest": 100, "waist": 85})
    assert suppliers.Mix(doc).estimate(ctx) == {}


def test_mix_score_on_test_people():
    m = ansur.mix_decide(_mix_frame(n=300))
    s = ansur.mix_score(_mix_frame(n=100, seed=5), m)
    assert set(s) == set(ansur.MIX_GIRTHS) and s["chest"]["mix"]["people"] == 100


def test_cli_fit_and_mix_end_to_end(run, fake_ansur, tmp_path, capsys):
    out = tmp_path / "ansur_formulas.json"
    main(["ansur-fit", "--ansur", str(fake_ansur), "--report", str(run["ckpt"].parent / "eval_report.json"),
          "--out", str(out)])
    text = capsys.readouterr().out
    assert "Black participants" in text and "self-reported" in text and "chosen" in text
    main(["ansur-mix", "--data", str(run["data"]), "--ckpt", str(run["ckpt"]), "--formulas", str(out),
          "--cache", str(run["cache"]), "--geo-cache", str(run["geo"])])
    text = capsys.readouterr().out
    assert "mix test on" in text and ("KEPT" in text or "not kept" in text)
    doc = ansur.load_formulas(out)
    assert set(doc["mix"]) == set(ansur.MIX_GIRTHS)
    for m in doc["mix"].values():
        assert ("testB" in m) == m["kept"]                        # Test-B scored only after a keep decision


# ---------------------------------------------------------------- supplier and prediction

def test_woman_gets_9_derived_codes_and_man_7(run, formulas, tmp_path):
    p = tmp_path / "f.json"
    p.write_text(json.dumps(formulas[0]))
    w = predict.predict(run["front"], run["side"], 165, "female", run["ckpt"], weight_kg=60, ansur_path=p)
    m = predict.predict(run["front"], run["side"], 178, "male", run["ckpt"], weight_kg=80, ansur_path=p)
    for doc, n in ((w, 9), (m, 7)):
        assert output.validate(doc) == []
        derived = [c for c, v in doc["measurements"].items() if v["quality"] == "derived"]
        assert len(derived) == n
        assert all(doc["measurements"][c]["expected_error_cm"] is not None for c in derived)
    assert w["measurements"]["hip_girth"]["source"] == "ansur"
    assert w["measurements"]["hip_girth"]["definition_match"] == "same"
    assert w["measurements"]["bust_point_distance"]["quality"] == "not_available"


def test_derived_values_carry_weight_assumed(formulas):
    person = {"sex": "female", "height_cm": 165.0, "weight_kg": 60.0, "height_kind": "measured"}
    photo = {"chest": 90, "waist": 75, "shoulder-breadth": 36, "thigh": 55, "calf": 36, "wrist": 15,
             "hip": 98}
    est = suppliers.Ansur(formulas[0]).estimate(suppliers.Context(person, photo, ["weight_assumed"]))
    assert est and all("weight_assumed" in e.flags for e in est.values())


def test_extended_formula_needs_photo_values(formulas):
    doc = json.loads(json.dumps(formulas[0]))
    for by_sex in doc["formulas"].values():
        for f in by_sex.values():
            f.update(kind="extended", inputs=["height_cm", "chest"], coef=[0.1, 0.3])
    person = {"sex": "male", "height_cm": 175.0, "weight_kg": 70.0}
    assert suppliers.Ansur(doc).estimate(suppliers.Context(person, {})) == {}


def test_value_outside_the_formula_range_is_flagged(formulas):
    doc = json.loads(json.dumps(formulas[0]))
    doc["formulas"]["neck_girth"]["male"]["range_cm"] = [1.0, 2.0]
    person = {"sex": "male", "height_cm": 175.0, "weight_kg": 70.0}
    photo = {"chest": 100, "waist": 85, "shoulder-breadth": 40, "thigh": 58, "calf": 38, "wrist": 17,
             "hip": 100}
    est = suppliers.Ansur(doc).estimate(suppliers.Context(person, photo))
    assert "outside_formula_range" in est["neck_girth"].flags


def test_kept_mix_fills_waist_with_its_test_b_error(formulas):
    m = ansur.mix_decide(_mix_frame(n=300))
    for v in m.values():
        v["testB"] = {"mix": {"mae_cm": 1.9}}
    doc = {**formulas[0], "mix": m}
    ctx = suppliers.Context({"sex": "female", "height_cm": 165, "weight_kg": 60}, {"chest": 90, "waist": 75})
    est = suppliers.Mix(doc).estimate(ctx)
    assert set(est) == {"bust_girth", "waist_girth"} and est["waist_girth"].expected_error_cm == 1.9
    assert est["waist_girth"].source == "mix"


# ---------------------------------------------------------------- F1, F2, tolerance view

def _bodym_people(ratio, n=50):
    rng = np.random.default_rng(1)
    h = rng.normal(170, 9, n)
    return pd.DataFrame({"subject_id": [f"p{i}" for i in range(n)], "height": h,
                         "leg-length": h * ratio + rng.normal(0, 1, n)})


def test_leg_check_passes_only_when_the_ratios_match(table):
    ansur_ratio = float((table["crotchheight"] / table["stature"]).median())
    assert ansur.leg_check(table, _bodym_people(ansur_ratio))["passed"]
    failed = ansur.leg_check(table, _bodym_people(ansur_ratio + 0.05))
    assert not failed["passed"] and failed["difference"] > ansur.LEG_RATIO_TOLERANCE


def test_leg_formulas_are_tried_only_for_leg_codes_and_win_when_informative(table):
    noise = {**NOISE, "leg-length": 0.01}            # an almost exact leg length must help the leg codes
    doc, scores = ansur.fit(table, noise, catalog.version(), "fp", "test", leg_ok=True)
    tried = set(scores.loc[scores.formula == "extended + leg, blurred", "code"])
    assert tried == set(ansur.LEG_CODES)
    assert doc["formulas"]["inseam"]["male"]["kind"] == "extended + leg"
    assert "leg-length" in doc["formulas"]["inseam"]["male"]["inputs"]
    _, without = ansur.fit(table, NOISE, catalog.version(), "fp", "test", leg_ok=False)
    assert "extended + leg, blurred" not in set(without["formula"])


def test_leg_formula_uses_the_photo_models_leg_length(table):
    doc, _ = ansur.fit(table, {**NOISE, "leg-length": 0.01}, catalog.version(), "fp", "test", leg_ok=True)
    person = {"sex": "male", "height_cm": 176.0, "weight_kg": 85.0, "height_kind": "measured"}
    photo = {"chest": 100, "waist": 88, "shoulder-breadth": 40, "thigh": 58, "calf": 38, "wrist": 17,
             "hip": 100, "leg-length": 82.0}
    short = suppliers.Ansur(doc).estimate(suppliers.Context(person, photo))["inseam"].value_cm
    long_ = suppliers.Ansur(doc).estimate(suppliers.Context(person, {**photo, "leg-length": 90.0}))
    assert long_["inseam"].value_cm > short + 5


def test_black_fitted_alternative_is_stored_but_off_until_ryan_decides(table, formulas):
    doc = json.loads(json.dumps(formulas[0]))
    alt, t = ansur.black_fitted(table, doc, NOISE)
    assert set(alt) == set(ansur.BLACK_FIT_CODES) and set(alt["inseam"]) == {"female", "male"}
    assert {"general, on Black participants", "Black-fitted, on Black participants"} <= set(t["formula"])
    doc["black_fitted"] = alt
    person = {"sex": "male", "height_cm": 176.0, "weight_kg": 85.0}
    photo = {"chest": 100, "waist": 88, "shoulder-breadth": 40, "thigh": 58, "calf": 38, "wrist": 17,
             "hip": 100}
    general = suppliers.Ansur(doc).estimate(suppliers.Context(person, photo))["inseam"].value_cm
    for f in alt["inseam"].values():
        f.update(kept=True, intercept=f["intercept"] + 10)          # make the switch visible
    assert suppliers.Ansur(doc).estimate(suppliers.Context(person, photo))["inseam"].value_cm == general
    doc["prefer_black_fitted"] = True
    assert suppliers.Ansur(doc).estimate(suppliers.Context(person, photo))["inseam"].value_cm > general + 5


def test_cli_fit_runs_the_leg_check_and_f2(run, fake_ansur, tmp_path, capsys):
    out = tmp_path / "f.json"
    main(["ansur-fit", "--ansur", str(fake_ansur), "--report", str(run["ckpt"].parent / "eval_report.json"),
          "--out", str(out), "--bodym", str(run["data"])])
    text = capsys.readouterr().out
    assert "F1 leg check" in text and "F2" in text and "pct_within_3" in text
    doc = ansur.load_formulas(out)
    assert doc["leg_check"] is not None and doc["prefer_black_fitted"] is False


def test_evaluate_reports_the_tolerance_view(run):
    report = json.loads((run["ckpt"].parent / "eval_report.json").read_text())
    rows = report["testB"]["tolerance_view"]
    assert {"within_1.5cm_pct", "within_2cm_pct", "within_3cm_pct"} <= set(rows[0])
    assert all(r["within_1.5cm_pct"] <= r["within_2cm_pct"] <= r["within_3cm_pct"] for r in rows)
