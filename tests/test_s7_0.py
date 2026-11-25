"""S7-0 on fake data: catalogue, output format, routing, typical weight, prediction, export loader.

Proves the rules in specs/s7-0-foundation/spec.md hold (allow and refuse cases). Accuracy is not
tested here: the model is trained for one epoch on fake rectangles.
"""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import make_hybrid_run

from anthro import catalog, checkpoint, export_loader, output, predict, suppliers, typical_weight
from anthro.cli import main
from anthro.config import MEASUREMENTS
from anthro.data import load_split
from anthro.output import Estimate

FEMALE_ONLY = {"bust_girth", "underbust_girth", "bust_point_distance", "shoulder_to_bust_point",
               "high_hip_girth", "waist_to_knee", "shoulder_to_knee"}


# ---------------------------------------------------------------- one trained hybrid run, shared

@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return make_hybrid_run(tmp_path_factory.mktemp("s7"))


# ---------------------------------------------------------------- catalogue

def test_catalogue_matches_the_app_rules():
    assert len(catalog.catalogue()["measurements"]) == 25
    women, men = catalog.codes_for("female"), catalog.codes_for("male")
    assert len(women) == 24 and len(men) == 18
    assert set(women) - set(men) == FEMALE_ONLY and set(men) - set(women) == {"chest_girth"}
    assert catalog.version().startswith("anthropometrydc-measurements@")
    assert all(m["adult_range_cm"][0] < m["adult_range_cm"][1] for m in catalog.catalogue()["measurements"])


@pytest.mark.parametrize("age", catalog.CHILD_RANGES)
def test_children_are_refused(age):
    with pytest.raises(catalog.CatalogError, match="adults only"):
        catalog.codes_for("female", age)


@pytest.mark.parametrize("sex,age", [("other", None), ("female", "30-40")])
def test_unknown_sex_or_age_range_is_refused(sex, age):
    with pytest.raises(catalog.CatalogError):
        catalog.codes_for(sex, age)


def test_bodym_chest_becomes_bust_for_women_and_chest_for_men():
    vals = {m: 50.0 for m in MEASUREMENTS}
    women, men = catalog.bodym_to_codes("female", vals), catalog.bodym_to_codes("male", vals)
    assert "bust_girth" in women and "chest_girth" not in women
    assert "chest_girth" in men and "bust_girth" not in men
    assert women["hip_girth"][1] == "differs" and women["waist_girth"][1] == "same"
    assert not set(catalog.BODYM_INTERNAL_ONLY) & {v[2] for v in women.values()}


# ---------------------------------------------------------------- output format

PERSON = {"sex": "male", "age_range": "25-34", "height_cm": 175.0, "height_kind": "measured",
          "weight_kg": 70.0, "weight_kind": "stated", "clothing": None}


def test_output_has_exactly_the_persons_set_and_validates():
    doc = output.build(PERSON, {"waist_girth": Estimate(80.0, "photo_model", "measured", "same", 2.4, "x")},
                       [], ["w"], "test")
    assert len(doc["measurements"]) == 18 and output.validate(doc) == []
    assert doc["measurements"]["neck_girth"]["quality"] == "not_available"
    assert json.loads(json.dumps(doc)) == doc


@pytest.mark.parametrize("breaker,msg", [
    (lambda d: d.update(schema="anthro-output/2"), "schema"),
    (lambda d: d["measurements"].pop("waist_girth"), "exactly the person's set"),
    (lambda d: d["measurements"]["neck_girth"].update(value_cm=3.0), "not_available must have no value"),
    (lambda d: d["measurements"]["waist_girth"].update(error_basis=None), "needs its basis"),
    (lambda d: d["measurements"]["waist_girth"].update(quality="guess"), "quality"),
    (lambda d: d["measurements"]["waist_girth"].update(definition_match="maybe"), "definition_match"),
    (lambda d: d["measurements"]["waist_girth"].update(source=None), "needs a number and a source"),
    (lambda d: d["person"].update(weight_kind="guessed"), "weight_kind"),
    (lambda d: d["person"].update(height_kind=None, height_cm="tall"), "height"),
    (lambda d: d["retake"].update(flag="no"), "retake"),
    (lambda d: d["person"].update(age_range="15-17"), "adults only"),
    (lambda d: d.pop("warnings"), "missing warnings"),
])
def test_invalid_output_is_caught(breaker, msg):
    doc = output.build(PERSON, {"waist_girth": Estimate(80.0, "photo_model", "measured", "same", 2.4, "x")},
                       [], [], "test")
    breaker(doc)
    assert any(msg in p for p in output.validate(doc))


def test_building_an_invalid_document_is_refused():
    with pytest.raises(ValueError, match="invalid"):
        output.build(PERSON, {"waist_girth": Estimate(None, "photo_model", "measured")}, [], [], "t")


# ---------------------------------------------------------------- routing

class FakeOwnModel:
    name = "own_model"

    def estimate(self, ctx):
        return {"waist_girth": Estimate(81.0, self.name, "measured", "same")}


def test_routing_file_decides_the_supplier_without_code_change(tmp_path):
    ctx = suppliers.Context(PERSON, {m: 50.0 for m in MEASUREMENTS})
    both = [suppliers.PhotoModel(), FakeOwnModel()]
    default = suppliers.route(ctx, both, suppliers.load_routing())
    assert default["waist_girth"].source == "own_model"          # own model listed first
    table = suppliers.load_routing()
    table["waist_girth"] = ["photo_model", "own_model"]
    p = tmp_path / "routing.json"
    p.write_text(json.dumps(table))
    assert suppliers.route(ctx, both, suppliers.load_routing(p))["waist_girth"].source == "photo_model"
    # Today (photo model only): ANSUR and geometry codes are honestly not available; hip falls back.
    today = suppliers.route(ctx, [suppliers.PhotoModel()], suppliers.load_routing())
    assert today["neck_girth"].quality == "not_available"
    assert today["hip_girth"].source == "photo_model" and today["hip_girth"].definition_match == "differs"


@pytest.mark.parametrize("change,msg", [({"belly_girth": ["photo_model"]}, "not in the catalogue"),
                                        ({"waist_girth": None}, "no entry")])
def test_bad_routing_files_are_refused(tmp_path, change, msg):
    table = suppliers.load_routing()
    table.update(change)
    table = {k: v for k, v in table.items() if v is not None}
    p = tmp_path / "r.json"
    p.write_text(json.dumps(table))
    with pytest.raises(ValueError, match=msg):
        suppliers.load_routing(p)


def test_errors_come_from_the_test_b_report_only():
    report = {"testB": {"table": [{"index": "waist", "photo_MAE_cm": 2.4}]}}
    errors, basis = suppliers.errors_from_report(report)
    assert errors == {"waist": 2.4} and "testB" in basis
    with pytest.raises(ValueError, match="no testB"):
        suppliers.errors_from_report({"testA": {}})


# ---------------------------------------------------------------- typical weight (D-027)

def test_typical_weight_rises_with_height_per_sex(bodym):
    fit_ = typical_weight.fit(load_split(bodym, "train"))
    assert set(fit_) == {"female", "male"} and json.loads(json.dumps(fit_)) == fit_
    for sex in fit_:
        assert np.isfinite(typical_weight.typical(fit_, sex, 170))


def test_typical_weight_needs_both_sexes():
    df = pd.DataFrame({"subject_id": ["a", "b"], "gender": ["male", "male"],
                       "height_cm": [170, 180], "weight_kg": [70, 80]})
    with pytest.raises(ValueError, match="female"):
        typical_weight.fit(df)


# ---------------------------------------------------------------- prediction

def test_woman_gets_24_measurements_with_test_b_errors(run):
    doc = predict.predict(run["front"], run["side"], 165, "female", run["ckpt"], weight_kg=60,
                          age_range="25-34", use_ansur=False)
    assert output.validate(doc) == [] and len(doc["measurements"]) == 24
    m = doc["measurements"]
    measured = [c for c, v in m.items() if v["quality"] == "measured"]
    assert len(measured) == 10 and "bust_girth" in measured and "chest_girth" not in m
    assert m["waist_girth"]["expected_error_cm"] is not None and "testB" in m["waist_girth"]["error_basis"]
    assert m["bust_point_distance"]["quality"] == "not_available"
    assert doc["person"]["weight_kind"] == "stated" and not m["waist_girth"]["flags"]
    assert m["height"]["value_cm"] == 165 and m["height"]["source"] == "input"     # tape height kept
    assert any("tape height" in w for w in doc["warnings"])    # the 1-epoch fake model is far off
    assert any("Ghanaian" in w for w in doc["warnings"])


def test_missing_weight_gets_a_typical_weight_and_a_warning(run, tmp_path, capsys):
    out = tmp_path / "man.json"
    main(["predict-silhouettes", "--front", str(run["front"]), "--side", str(run["side"]),
          "--height", "178", "--sex", "male", "--age-range", "35-44", "--ckpt", str(run["ckpt"]),
          "--out", str(out), "--no-ansur"])
    text = capsys.readouterr().out
    assert "assumed" in text and "Chest" in text
    doc = json.loads(out.read_text())
    assert output.validate(doc) == [] and len(doc["measurements"]) == 18
    assert doc["person"]["weight_kind"] == "assumed"
    assert doc["measurements"]["waist_girth"]["flags"] == ["weight_assumed"]
    assert doc["warnings"][0].startswith("weight not given")


def test_cli_refuses_a_child_and_a_bad_height(run):
    base = ["predict-silhouettes", "--front", str(run["front"]), "--side", str(run["side"]),
            "--sex", "female", "--ckpt", str(run["ckpt"]), "--weight", "50"]
    with pytest.raises(SystemExit, match="adults only"):
        main([*base, "--height", "150", "--age-range", "13-14"])
    with pytest.raises(SystemExit, match="outside"):
        main([*base, "--height", "300"])


def test_stated_height_is_estimated_by_the_model_within_2cm(run):
    doc = predict.predict(run["front"], run["side"], 165, "female", run["ckpt"], weight_kg=60,
                          height_kind="stated")
    h = doc["measurements"]["height"]
    assert h["source"] == "photo_model" and 163 <= h["value_cm"] <= 167


@pytest.mark.parametrize("model_cm,expected", [(178.0, 172.0), (165.0, 168.0), (169.0, 169.0)])
def test_stated_height_is_held_within_2cm(model_cm, expected):
    person = {**PERSON, "height_cm": 170.0, "height_kind": "stated"}
    est = suppliers.PhotoModel().estimate(suppliers.Context(person, {"height": model_cm}))["height"]
    assert est.value_cm == expected
    assert ("held_within_2cm_of_stated" in est.flags) == (model_cm != expected)


def test_height_gap_of_2cm_or_more_is_warned():
    measured = {**PERSON, "height_cm": 170.0}
    assert predict.height_warnings(measured, 171.9) == []
    assert "tape height" in predict.height_warnings(measured, 172.5)[0]
    stated = {**measured, "height_kind": "stated"}
    assert "172.0" in predict.height_warnings(stated, 175.0)[0]
    assert "168.0" in predict.height_warnings(stated, 160.0)[0]


def test_high_bmi_is_warned(run):
    doc = predict.predict(run["front"], run["side"], 160, "female", run["ckpt"], weight_kg=100)
    assert predict.BMI_WARNING in doc["warnings"]


def test_no_report_means_no_errors_shown(run, tmp_path):
    doc = predict.predict(run["front"], run["side"], 170, "male", run["ckpt"], weight_kg=70,
                          report_path=tmp_path / "none.json", use_ansur=False)
    assert all(v["expected_error_cm"] is None for v in doc["measurements"].values())
    assert any("no evaluation report" in w for w in doc["warnings"])


def test_old_checkpoint_needs_weight_or_data(run, tmp_path):
    ck = torch.load(run["ckpt"], weights_only=True)
    ck["extra"].pop("typical_weight")                       # what r1-r4 look like
    old = tmp_path / "old" / "best.pt"
    old.parent.mkdir()
    torch.save(ck, old)
    with pytest.raises(predict.PredictError, match="typical-weight"):
        predict.predict(run["front"], run["side"], 170, "male", old)
    doc = predict.predict(run["front"], run["side"], 170, "male", old, data_root=run["data"])
    assert doc["person"]["weight_kind"] == "assumed"


def test_empty_silhouette_is_refused(run, tmp_path):
    from PIL import Image
    blank = tmp_path / "blank.png"
    Image.fromarray(np.zeros((320, 240), np.uint8)).save(blank)
    with pytest.raises(predict.PredictError, match="front silhouette"):
        predict.predict(blank, run["side"], 170, "male", run["ckpt"], weight_kg=70)


def test_retake_fires_when_geometry_and_network_disagree():
    table = pd.DataFrame({"geo_ok": [1.0], "chest_C_hat": [100.0], "waist_C_hat": [80.0],
                          "hip_C_hat": [95.0]})
    values = {"chest": 100.0, "waist": 70.0, "hip": 95.0}
    reasons = predict.retake_check(values, table, threshold=7.8)
    assert len(reasons) == 1 and "waist" in reasons[0]
    assert predict.retake_check(values, table, threshold=12) == []
    assert predict.retake_check(values, None, threshold=7.8) == []


def test_loaded_checkpoint_still_safe(run):
    *_rest, ck = checkpoint.load(run["ckpt"])        # weights_only=True inside
    assert set(ck["extra"]["typical_weight"]) == {"female", "male"}


# ---------------------------------------------------------------- export loader

P_COLS = ["export_id", "practice", "sex", "age_group", "age_range", "clothing", "consent_version",
          "intimate_agreed", "weight_agreed", "stated_height_typed", "stated_height_unit", "stated_height_cm",
          "stated_weight_kg", "weight_kg", "weight_source", "collector", "phone_model", "created_date",
          "measured_date", "photos_date"]


def make_fake_export(folder, stamp=None, practice=False, extra_rows=()):
    """A fake F8 export with the app's real column layout (src/lib/exportData.ts) and invented values."""
    (folder / "metadata").mkdir(parents=True)
    rows = [["P001", practice, "female", "adult", "25-34", "tight", "v1", True, True, "165", "cm", 165,
             60, 61.2, "scale", "C01", "phone", "d", "d", "d"],
            ["P002", False, "male", "adult", "35-44", "tight", "v1", False, True, "5 10", "ftin", 177.8,
             None, None, "told", "C01", "phone", "d", "d", "d"],
            *extra_rows]
    pd.DataFrame(rows, columns=P_COLS).to_csv(folder / "participants.csv", index=False)
    m = [["P001", "waist_girth", 72.5, "72.4|72.6", "cm", False, None, False, None, None],
         ["P001", "bust_point_distance", None, "", "cm", True, "declined", False, None, None],
         ["P002", "waist_girth", 85.0, "85|85", "cm", False, None, False, None, None],
         ["P002", "bust_girth", 99.0, "99|99", "cm", False, None, False, None, None]]
    pd.DataFrame(m, columns=["export_id", "code", "value_cm", "readings_cm", "unit_typed", "skipped",
                             "skip_reason", "unusual_confirmed", "height_method", "hair_adds_height"]
                 ).to_csv(folder / "measurements.csv", index=False)
    f = [[i, s, f"images/{i}/{s}.jpg", 768, 1024, 1, "t", 1, 0.5, -0.2, 40, 990, 1, 0, True, 2]
         for i in ("P001", "P002") for s in ("front", "side")]
    pd.DataFrame(f, columns=["export_id", "slot", "file", "width", "height", "bytes", "taken_at", "zoom",
                             "level_pitch_deg", "level_roll_deg", "head_top_y", "heel_y", "people_detected",
                             "faces_detected", "a4_confirmed", "blur_version"]).to_csv(folder / "photos.csv",
                                                                                        index=False)
    readme = "AnthropometryDC export\n" + (f"format: {stamp}\n" if stamp else "")
    (folder / "README.txt").write_text(readme)
    for i in ("P001", "P002"):
        (folder / "metadata" / f"{i}.json").write_text(json.dumps({"participant": {"export_id": i}}))
    return folder


def test_fake_export_loads_with_readings_and_both_weights(tmp_path):
    with pytest.warns(UserWarning):
        ex = export_loader.load(make_fake_export(tmp_path / "x"))
    assert ex.format == "0" and len(ex.people) == 2 and len(ex.photos) == 4
    p = ex.people.set_index("export_id")
    assert p.loc["P001", "waist_girth"] == 72.5 and p.loc["P001", "waist_girth__readings"] == [72.4, 72.6]
    assert p.loc["P001", "stated_weight_kg"] == 60 and p.loc["P001", "weight_kg"] == 61.2
    assert p.loc["P001", "bust_point_distance__skip_reason"] == "declined"
    assert pd.isna(p.loc["P001", "bust_point_distance"])
    assert pd.isna(p.loc["P002", "bust_girth"])                 # not in a man's set: ignored
    assert any("not in their set" in w for w in ex.warnings)
    assert ex.photos["path"].iloc[0].endswith("images/P001/front.jpg")


def test_stamped_export_is_accepted(tmp_path):
    with pytest.warns(UserWarning):          # still warns about the bust value on a man
        ex = export_loader.load(make_fake_export(tmp_path / "x", stamp="anthro-export/1"))
    assert ex.format == "anthro-export/1"


def test_unknown_format_is_refused(tmp_path):
    with pytest.raises(export_loader.ExportError, match="unknown export format"):
        export_loader.load(make_fake_export(tmp_path / "x", stamp="anthro-export/9"))


def test_two_different_stamps_are_refused(tmp_path):
    f = make_fake_export(tmp_path / "x", stamp="anthro-export/1")
    (f / "metadata" / "P001.json").write_text(json.dumps({"format": "anthro-export/2"}))
    with pytest.raises(export_loader.ExportError, match="more than one"):
        export_loader.load(f)


def test_practice_rows_are_refused_unless_asked(tmp_path):
    f = make_fake_export(tmp_path / "x", practice=True)
    with pytest.raises(export_loader.ExportError, match="practice"):
        export_loader.load(f)
    with pytest.warns(UserWarning):
        ex = export_loader.load(f, include_practice=True)
    assert ex.people.set_index("export_id").loc["P001", "practice"]


def test_missing_column_or_file_is_refused(tmp_path):
    f = make_fake_export(tmp_path / "x")
    pd.read_csv(f / "photos.csv").drop(columns=["heel_y"]).to_csv(f / "photos.csv", index=False)
    with pytest.raises(export_loader.ExportError, match="heel_y"):
        export_loader.load(f)
    (f / "photos.csv").unlink()
    with pytest.raises(export_loader.ExportError, match="missing"):
        export_loader.load(f)
    with pytest.raises(export_loader.ExportError, match="not a folder"):
        export_loader.load(tmp_path / "nowhere")


def test_unknown_code_is_refused(tmp_path):
    f = make_fake_export(tmp_path / "x")
    m = pd.read_csv(f / "measurements.csv")
    m.loc[0, "code"] = "belly_girth"
    m.to_csv(f / "measurements.csv", index=False)
    with pytest.raises(export_loader.ExportError, match="belly_girth"):
        export_loader.load(f)


def test_children_are_left_out_with_a_count(tmp_path):
    child = ["P003", False, "female", "child", "13-14", "tight", "v1", False, False, "150", "cm", 150,
             None, None, "told", "C01", "phone", "d", "d", "d"]
    with pytest.warns(UserWarning, match="under 18"):
        ex = export_loader.load(make_fake_export(tmp_path / "x", extra_rows=[child]))
    assert ex.left_out == {"under_18": 1} and "P003" not in set(ex.people["export_id"])


def test_cli_read_export_prints_counts_only(tmp_path, capsys):
    f = make_fake_export(tmp_path / "x")
    with pytest.warns(UserWarning):
        main(["read-export", "--folder", str(f)])
    out = capsys.readouterr().out
    assert "2 participants" in out and "waist_girth" in out and "P001" not in out
    with pytest.raises(SystemExit, match="refused"):
        main(["read-export", "--folder", str(tmp_path / "nowhere")])


def test_packaged_ansur_formulas_are_used_by_default(run):
    doc = predict.predict(run["front"], run["side"], 165, "female", run["ckpt"], weight_kg=60)
    assert output.validate(doc) == []
    assert sum(v["quality"] == "derived" for v in doc["measurements"].values()) == 10   # 9 + front length


def test_packaged_black_fitted_inseam_is_flagged_and_warned(run):
    doc = predict.predict(run["front"], run["side"], 165, "female", run["ckpt"], weight_kg=60)
    assert "fitted_on_ansur_black_participants" in doc["measurements"]["inseam"]["flags"]
    assert "fitted_on_ansur_black_participants" not in doc["measurements"]["neck_girth"]["flags"]
    assert any("D-032" in w for w in doc["warnings"])
