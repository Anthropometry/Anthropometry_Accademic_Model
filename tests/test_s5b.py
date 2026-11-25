"""S5b on fake data: outline maths checked by hand, refusals, sanity checks, ANSUR front length, commands."""

import json

import numpy as np
import pytest
from conftest import make_fake_ansur, make_hybrid_run

from anthro import ansur, catalog, geometry_rough, output, predict, suppliers
from anthro.cli import main
from anthro.output import Estimate


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return make_hybrid_run(tmp_path_factory.mktemp("s5b"))


def _figure(h=1000, legs_gap=True):
    """A front and side mask, 1000 px tall: torso 200 px wide, two legs of 60 px with a gap (or merged)."""
    front, side = np.zeros((1100, 600), bool), np.zeros((1100, 400), bool)
    top = 50
    front[top:top + 550, 200:400] = True                    # head + torso
    side[top:top + 550, 150:290] = True
    if legs_gap:
        front[top + 550:top + h, 220:280] = True            # left leg 60 px
        front[top + 550:top + h, 320:380] = True            # right leg 60 px
    else:
        front[top + 550:top + h, 220:380] = True            # legs touching
    side[top + 550:top + h, 180:250] = True                 # legs overlap in the side view: 70 px
    return front, side


def test_ellipse_matches_the_hand_computed_ramanujan_value():
    # W=30, D=20: a=15, b=10, pi*(3*25 - sqrt(55*45)) = 79.33 cm
    assert geometry_rough.ellipse_girth(30, 20) == pytest.approx(79.33, abs=0.01)


def test_runs_are_found():
    assert geometry_rough._runs(np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1], bool)) == [(1, 2), (4, 1), (7, 3)]


def test_knee_is_measured_on_one_leg():
    front, side = _figure()
    value, why = geometry_rough.knee(front, side, 170.0, 0.7)
    cm = 170.0 / 1000                                       # body is 1000 rows tall (rows 50-1049)
    assert value == pytest.approx(geometry_rough.ellipse_girth(60 * cm, 70 * cm), rel=1e-6)
    assert "one leg" in why


def test_touching_legs_at_the_knee_are_refused():
    front, side = _figure(legs_gap=False)
    value, why = geometry_rough.knee(front, side, 170.0, 0.7)
    assert value is None and "legs touch" in why


def test_geometry_supplier_gives_rough_values_without_error():
    rough = {"knee_girth": (36.0, "x"), "underbust_girth": (None, "no outline")}
    est = suppliers.Geometry(rough).estimate(suppliers.Context({"sex": "female"}, {}, ["weight_assumed"]))
    assert set(est) == {"knee_girth"}
    e = est["knee_girth"]
    assert e.quality == "rough" and e.expected_error_cm is None and "outline_knee_one_leg" in e.flags
    assert "weight_assumed" in e.flags


def test_value_outside_the_adult_range_is_withheld_with_a_reason():
    est = {"knee_girth": Estimate(90.0, "geometry", "rough", "presumed")}
    out, notes = predict.sanity_checks(est, "male", {"knee_girth": (90.0, "x")})
    assert out["knee_girth"].quality == "not_available" and any("outside the adult range" in n for n in notes)


def test_out_of_order_values_are_flagged():
    est = {"underbust_girth": Estimate(95.0, "geometry", "rough", "presumed"),
           "bust_girth": Estimate(90.0, "photo_model", "measured", "presumed"),
           "high_hip_girth": Estimate(70.0, "geometry", "rough", "presumed"),
           "waist_girth": Estimate(75.0, "photo_model", "measured", "same"),
           "hip_girth": Estimate(98.0, "ansur", "derived", "same")}
    out, notes = predict.sanity_checks(est, "female", {})
    assert "order_check_failed" in out["underbust_girth"].flags
    assert "order_check_failed" in out["high_hip_girth"].flags and len(notes) == 2


def test_unmeasurable_rough_code_is_explained():
    _, notes = predict.sanity_checks({}, "female", {"knee_girth": (None, "legs touch at knee height")})
    assert any("knee girth not available: legs touch" in n for n in notes)


def test_estimate_on_a_real_pipeline_pair(run):
    from anthro import checkpoint
    from anthro.preprocess import load_mask
    *_rest, ck = checkpoint.load(run["ckpt"])
    person = {"height_cm": 165.0, "weight_kg": 60.0, "sex": "female"}
    est = geometry_rough.estimate(load_mask(run["front"]), load_mask(run["side"]), person, ck["geo_fit"])
    assert set(est) == {"underbust_girth", "high_hip_girth", "knee_girth"}
    assert all((v is None) == bool(why) or why for v, why in est.values())


def test_front_length_is_routed_to_ansur():
    from anthro import suppliers as s
    order = s.load_routing()["front_length"]
    assert order.index("ansur") < order.index("geometry")


def test_packaged_formulas_give_front_length_and_ansur_knee_height():
    from importlib import resources
    doc = json.loads(resources.files("anthro").joinpath("data/ansur_formulas.json").read_text())
    assert "front_length" in doc["formulas"] and 0.2 < doc["level_ratios"]["knee_from_floor"] < 0.35


def test_full_prediction_stays_valid_and_rough_rows_have_no_error(run):
    doc = predict.predict(run["front"], run["side"], 165, "female", run["ckpt"], weight_kg=60)
    assert output.validate(doc) == []
    assert doc["measurements"]["front_length"]["source"] == "ansur"
    rough = [m for m in doc["measurements"].values() if m["quality"] == "rough"]
    assert all(m["expected_error_cm"] is None for m in rough)
    assert doc["measurements"]["bust_point_distance"]["quality"] == "not_available"     # D-025


def test_bad_outline_gives_reasons_for_every_rough_code():
    blank = np.zeros((300, 200), bool)
    est = geometry_rough.estimate(blank, blank, {"height_cm": 170, "weight_kg": 70, "sex": "male"}, {})
    assert all(v is None and "could not be measured" in why for v, why in est.values())


# ---------------------------------------------------------------- ANSUR: front length and knee ratio

def test_front_length_is_fitted_only_when_ansur_has_the_column(tmp_path):
    with_front = ansur.load(make_fake_ansur(tmp_path / "a", front=True))
    without = ansur.load(make_fake_ansur(tmp_path / "b"))
    assert "front_length" in ansur.present_targets(with_front)
    assert "front_length" not in ansur.present_targets(without)
    doc, _ = ansur.fit(with_front, {}, catalog.version(), "fp", "t")
    f = doc["formulas"]["front_length"]
    assert set(f) == {"female", "male"} and f["male"]["definition_match"] == "differs"      # D-033
    assert 0.2 < doc["level_ratios"]["knee_from_floor"] < 0.35


def test_candidate_columns_lists_names_counts_and_medians_only(tmp_path):
    t = ansur.candidate_columns(make_fake_ansur(tmp_path / "a", front=True))
    assert {"column", "sex", "people", "median_cm"} == set(t.columns)
    assert "waistfrontlengthsitting" in set(t["column"]) and "kneeheightmidpatella" in set(t["column"])


def test_cli_ansur_columns_and_geometry_rough(run, tmp_path, capsys):
    main(["ansur-columns", "--ansur", str(make_fake_ansur(tmp_path / "a"))])
    assert "kneeheightmidpatella" in capsys.readouterr().out
    main(["geometry-rough", "--data", str(run["data"]), "--ckpt", str(run["ckpt"])])
    out = capsys.readouterr().out
    assert "measured_pct" in out and "No errors" in out and "Order against BodyM" in out


def test_geometry_rough_refuses_a_non_hybrid_checkpoint(run, tmp_path):
    import torch
    ck = torch.load(run["ckpt"], weights_only=True)
    ck["geo_fit"] = None
    p = tmp_path / "plain" / "best.pt"
    p.parent.mkdir()
    torch.save(ck, p)
    with pytest.raises(SystemExit, match="no geometry fit"):
        main(["geometry-rough", "--data", str(run["data"]), "--ckpt", str(p)])


def test_no_geometry_fit_means_no_rough_values(run):
    from importlib import resources
    doc = json.loads(resources.files("anthro").joinpath("data/ansur_formulas.json").read_text())
    person = {"height_cm": 170, "weight_kg": 70, "sex": "male"}
    assert predict.rough_estimates(run["front"], run["side"], person, None, doc) == {}    # no geometry fit


def test_knee_height_comes_from_ansur_when_stored_else_the_textbook(run, monkeypatch):
    from anthro import checkpoint
    *_rest, ck = checkpoint.load(run["ckpt"])
    seen = []
    monkeypatch.setattr(geometry_rough, "estimate", lambda f, s, p, g, ratio: seen.append(ratio) or {})
    person = {"height_cm": 170, "weight_kg": 70, "sex": "male"}
    predict.rough_estimates(run["front"], run["side"], person, ck["geo_fit"], {})
    predict.rough_estimates(run["front"], run["side"], person, ck["geo_fit"],
                            {"level_ratios": {"knee_from_floor": 0.27}})
    assert seen == [geometry_rough.KNEE_FROM_FLOOR_TEXTBOOK, 0.27]
