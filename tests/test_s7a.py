"""S7a on synthetic photos: cleaning, refusals, side direction, BodyM-like silhouettes, prediction,
comparison.

A fake segmenter reads the drawn person (pure red) so no model is downloaded in CI. The real tools are
tested for their wiring only (with their networks replaced).
"""

import json

import numpy as np
import pandas as pd
import pytest
from conftest import make_hybrid_run
from PIL import Image

from anthro import output, photo, predict, segment, segment_compare
from anthro.cli import main


class FakeSeg:
    name = "fake"

    def version(self):
        return "fake 1"

    def __call__(self, rgb):
        return ((rgb[..., 0] > 200) & (rgb[..., 1] < 60)).astype(np.float32)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return make_hybrid_run(tmp_path_factory.mktemp("s7a"))


def person(h=1500, w=1100, top=80, body=1300, feet="right", legs_gap=True, second=False, arms_out=False):
    """A drawn person (red) on a grey photo: head, torso, legs; for side views a foot pointing one way."""
    img = np.full((h, w, 3), 120, np.uint8)
    m = np.zeros((h, w), bool)
    cx = w // 2
    head = int(0.13 * body)
    m[top: top + head, cx - 60: cx + 60] = True
    torso_end = top + int(0.52 * body)
    m[top + head: torso_end, cx - 170: cx + 170] = True
    foot_top = top + int(0.97 * body)
    if legs_gap:
        m[torso_end: top + body, cx - 150: cx - 30] = True
        m[torso_end: top + body, cx + 30: cx + 150] = True
    else:
        m[torso_end: top + body, cx - 150: cx + 150] = True
    if feet in ("right", "left"):
        d = 1 if feet == "right" else -1
        x0, x1 = sorted((cx, cx + d * 260))
        m[foot_top: top + body, x0: x1] = True
    if arms_out:
        r0, r1 = top + int(0.30 * body), top + int(0.52 * body)
        m[r0 - 40: r0, cx + 170: cx + 290] = True              # shoulder joint
        m[r0:r1, cx + 230: cx + 290] = True                    # an arm held away, a gap from the torso
    if second:
        m[top: top + body, 20:220] = True                      # another person at the edge
    img[m] = (240, 20, 20)
    return img


def save(img, path):
    Image.fromarray(img).save(path)
    return path


# ---------------------------------------------------------------- cleaning and checks

def test_clean_keeps_the_body_drops_specks_and_fills_small_holes_only():
    prob = np.zeros((400, 300), np.float32)
    prob[20:380, 100:200] = 1
    prob[100:104, 140:144] = 0                                 # a small hole: filled
    prob[5:8, 5:8] = 1                                         # a speck: dropped
    body, soft, people = photo.clean(prob)
    assert people == 1 and body[101, 141] and not body[6, 6]
    assert soft.max() <= 1 and soft[body].min() >= 0.5 and soft[~body].max() < 0.5


def test_gap_between_legs_is_kept_when_feet_touch():
    prob = np.zeros((600, 400), np.float32)
    prob[20:300, 100:300] = 1                                  # torso
    prob[300:560, 100:180] = 1
    prob[300:560, 220:300] = 1                                 # legs with a gap
    prob[560:580, 100:300] = 1                                 # feet touching: the gap is now a closed hole
    body, _, _ = photo.clean(prob)
    assert not body[400, 200]                                  # big hole: not filled


def test_a_good_photo_passes_and_each_problem_is_refused():
    body, _, people = photo.clean(FakeSeg()(person()))
    assert photo.check(body, people, "front") == []
    cases = {
        "no person": (np.zeros((1500, 1100), bool), 0, "front", None),
        "more than one person": (*photo.clean(FakeSeg()(person(second=True)))[::2], "front", None),
        "head touches the top": (*photo.clean(FakeSeg()(person(top=0)))[::2], "front", None),
        "feet touch the bottom": (*photo.clean(FakeSeg()(person(top=200)))[::2], "front", None),
        "pixels tall": (*photo.clean(FakeSeg()(person(body=500)))[::2], "front", None),
        "tilted": (body, people, "front", (9.0, 1.0)),
        "arms are away": (*photo.clean(FakeSeg()(person(arms_out=True, legs_gap=False)))[::2], "side", None),
    }
    for want, (b, n, view, tilt) in cases.items():
        problems = photo.check(b, n, view, tilt)
        assert any(want in p for p in problems), (want, problems)


def test_side_direction_is_read_from_the_feet():
    for way in ("left", "right"):
        body, _, _ = photo.clean(FakeSeg()(person(feet=way, legs_gap=False)))
        assert photo.facing(body) == way
    body, _, _ = photo.clean(FakeSeg()(person(feet=None, legs_gap=False)))
    assert photo.facing(body) is None


def test_silhouette_is_bodym_sized_and_side_is_mirrored_to_bodym():
    s = photo.silhouette(person(feet="left", legs_gap=False), FakeSeg(), "side", bodym_facing="right")
    assert s.mask.shape == photo.CANVAS and s.facing == "left" and s.mirrored
    rows = np.flatnonzero((s.mask > 127).any(axis=1))
    assert abs((rows[-1] - rows[0] + 1) - photo.BODY_PX) <= 2
    body = s.mask > 127
    assert photo.facing(body) == "right"
    same = photo.silhouette(person(feet="right", legs_gap=False), FakeSeg(), "side", bodym_facing="right")
    assert not same.mirrored


def test_unknown_side_direction_is_used_as_taken_with_a_note():
    s = photo.silhouette(person(feet=None, legs_gap=False), FakeSeg(), "side", bodym_facing="right")
    assert not s.mirrored and s.notes


def test_refusal_lists_every_problem():
    with pytest.raises(photo.PhotoError, match="front photo: .*more than one person.*head touches"):
        photo.silhouette(person(top=0, second=True), FakeSeg(), "front")


def test_too_wide_body_is_refused():
    prob = np.zeros((900, 1000), np.float32)
    prob[100:150, 0:1000] = 1
    body, soft, _ = photo.clean(prob)
    with pytest.raises(photo.PhotoError, match="too wide"):
        photo.to_bodym(soft, body)


def test_load_photo_applies_rotation_and_refuses_bad_sizes(tmp_path):
    img = Image.fromarray(person()[:, :, :])
    exif = Image.Exif()
    exif[0x0112] = 6                                           # "rotate 90": how phones store portrait photos
    img.save(tmp_path / "rot.jpg", exif=exif)
    assert photo.load_photo(tmp_path / "rot.jpg").shape[:2] == (1100, 1500)
    Image.new("RGB", (300, 200)).save(tmp_path / "small.png")
    with pytest.raises(photo.PhotoError, match="too small"):
        photo.load_photo(tmp_path / "small.png")


def test_bodym_facing_votes_over_training_masks(tmp_path):
    d = tmp_path / "train" / "mask_left"
    d.mkdir(parents=True)
    for i in range(3):
        body, _, _ = photo.clean(FakeSeg()(person(feet="left", legs_gap=False)))
        Image.fromarray(body.astype(np.uint8) * 255).save(d / f"p{i}.png")
    assert photo.bodym_facing(tmp_path) == "left"
    assert photo.bodym_facing(tmp_path / "nothing") is None


def test_overlay_draws_the_outline():
    img = person()
    body, _, _ = photo.clean(FakeSeg()(img))
    out = photo.overlay(img, body)
    assert (out == (255, 0, 0)).all(axis=-1).sum() > 0 and out.shape == img.shape


# ---------------------------------------------------------------- prediction from photos

def _photos_from_run(run, tmp_path):
    """Photos drawn from the fake BodyM masks of the trained run (so the model sees familiar shapes)."""
    from anthro.preprocess import load_mask
    out = []
    for p in (run["front"], run["side"]):
        m = np.kron(load_mask(p), np.ones((4, 4), bool))       # 320x240 -> 1280x960
        img = np.full((m.shape[0] + 200, m.shape[1] + 200, 3), 120, np.uint8)
        img[100:-100, 100:-100][m] = (240, 20, 20)
        out.append(save(img, tmp_path / f"{p.parent.name}.png"))
    return out


def test_predict_photos_gives_the_same_result_as_its_silhouettes(run, tmp_path):
    front, side = _photos_from_run(run, tmp_path)
    sil = tmp_path / "sil"
    doc = predict.predict_photos(front, side, 165, "female", run["ckpt"], FakeSeg(), None, save_dir=sil,
                                 weight_kg=60)
    assert output.validate(doc) == [] and len(doc["measurements"]) == 24
    assert doc["inputs"]["kind"] == "photos" and doc["inputs"]["segmenter"] == "fake"
    again = predict.predict(sil / "front_silhouette.png", sil / "side_silhouette.png", 165, "female",
                            run["ckpt"], weight_kg=60)
    assert again["measurements"] == doc["measurements"]


def test_nothing_is_written_without_save_silhouettes(run, tmp_path):
    front, side = _photos_from_run(run, tmp_path)
    before = set(tmp_path.rglob("*"))
    predict.predict_photos(front, side, 165, "female", run["ckpt"], FakeSeg(), None, weight_kg=60)
    assert set(tmp_path.rglob("*")) == before


def test_cli_predict_photos_and_side_direction(run, tmp_path, capsys, monkeypatch):
    monkeypatch.setitem(segment.TOOLS, "torchvision", FakeSeg)
    front, side = _photos_from_run(run, tmp_path)
    out = tmp_path / "r.json"
    main(["predict-photos", "--front", str(front), "--side", str(side), "--height", "165", "--sex", "female",
          "--weight", "60", "--ckpt", str(run["ckpt"]), "--bodym-side", "right", "--out", str(out)])
    assert "segmentation: fake 1" in capsys.readouterr().out
    assert json.loads(out.read_text())["inputs"]["kind"] == "photos"
    main(["side-direction", "--data", str(run["data"])])
    assert "BodyM side silhouettes face" in capsys.readouterr().out
    bad = save(np.full((900, 700, 3), 120, np.uint8), tmp_path / "empty.png")
    with pytest.raises(SystemExit, match="refused: front photo: no person"):
        main(["predict-photos", "--front", str(bad), "--side", str(side), "--height", "165", "--sex",
              "female", "--ckpt", str(run["ckpt"]), "--weight", "60"])


# ---------------------------------------------------------------- comparison

def _pairs(run, tmp_path, tape=True):
    front, side = _photos_from_run(run, tmp_path)
    bad = save(np.full((900, 700, 3), 120, np.uint8), tmp_path / "empty.png")
    rows = []
    for p in ("A", "B", "C"):
        for _ in range(2):
            rows.append({"person": p, "front": front, "side": side, "height_cm": 165, "sex": "female",
                         "weight_kg": 60, "age_range": "25-34", **({"waist_girth": 75.0} if tape else {})})
    rows.append({"person": "D", "front": bad, "side": side, "height_cm": 170, "sex": "male", "weight_kg": 70,
                 "age_range": "25-34", **({"waist_girth": 85.0} if tape else {})})
    p = tmp_path / "pairs.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_comparison_scores_refusals_heights_stability_and_tape(run, tmp_path):
    pairs = segment_compare.load_pairs(_pairs(run, tmp_path))
    result, docs = segment_compare.run_tool(pairs, FakeSeg(), run["ckpt"], None)
    s = segment_compare.scores(result, pairs)
    assert s["pairs"] == 7 and s["refused"] == 1 and "no person" in s["refusal_reasons"][0]
    assert s["stability_people"] == 3 and s["stability_spread_cm"]["waist_girth"] == 0.0
    assert s["tape_errors"]["waist_girth"]["people"] == 3 and s["tape_people"] == 3
    assert 0 <= s["height_flag_pct"] <= 100 and len(docs) == 6


def test_decision_rule_uses_tape_first_then_height_flags():
    base = {"pairs": 6, "refused": 0, "retake_pct": 10.0, "stability_spread_cm": {"waist_girth": 1.0}}
    a = {**base, "height_flag_pct": 50.0, "tape_people": 3, "tape_errors": {"w": {"pct_within_1.5": 70.0}}}
    b = {**base, "height_flag_pct": 0.0, "tape_people": 3, "tape_errors": {"w": {"pct_within_1.5": 40.0}}}
    assert segment_compare.decide({"a": a, "b": b})[0] == "a"                 # tape decides
    a2, b2 = {**a, "tape_people": 1}, {**b, "tape_people": 1}
    assert segment_compare.decide({"a": a2, "b": b2})[0] == "b"               # then height flags


def test_pairs_file_needs_its_columns(tmp_path):
    p = tmp_path / "p.csv"
    pd.DataFrame({"person": ["A"], "front": ["f"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="height_cm"):
        segment_compare.load_pairs(p)


def test_cli_compare_segmenters(run, tmp_path, capsys, monkeypatch):
    monkeypatch.setitem(segment.TOOLS, "torchvision", FakeSeg)
    monkeypatch.setitem(segment.TOOLS, "rembg", FakeSeg)
    out = tmp_path / "c.json"
    main(["compare-segmenters", "--pairs", str(_pairs(run, tmp_path, tape=False)), "--ckpt", str(run["ckpt"]),
          "--bodym-side", "right", "--out", str(out)])
    text = capsys.readouterr().out
    assert "winner by the fixed rule" in text and "refused:" in text
    assert "_decision" in json.loads(out.read_text())


def test_compare_skips_a_tool_that_is_not_installed(run, tmp_path, capsys, monkeypatch):
    def missing():
        raise ImportError("rembg is not installed")
    monkeypatch.setitem(segment.TOOLS, "torchvision", FakeSeg)
    monkeypatch.setitem(segment.TOOLS, "rembg", missing)
    pairs = _pairs(run, tmp_path, tape=False)
    main(["compare-segmenters", "--pairs", str(pairs), "--ckpt", str(run["ckpt"])])
    assert "rembg: skipped" in capsys.readouterr().out


# ---------------------------------------------------------------- the real tools' wiring

def test_unknown_tool_is_refused():
    with pytest.raises(ValueError, match="unknown segmentation tool"):
        segment.get("magic")


def test_torchvision_tool_returns_the_person_probability(monkeypatch):
    import torch
    import torchvision.models.segmentation as tvseg

    class Net(torch.nn.Module):
        def forward(self, x):
            out = torch.zeros(x.shape[0], 21, x.shape[2], x.shape[3])
            out[:, 15, :, : x.shape[3] // 2] = 10.0             # "person" on the left half
            return {"out": out}

    monkeypatch.setattr(tvseg, "deeplabv3_resnet101", lambda weights=None: Net())
    seg = segment.TorchvisionSegmenter(device="cpu")
    prob = seg(np.zeros((64, 48, 3), np.uint8))
    assert prob.shape == (64, 48) and prob[:, :20].mean() > 0.9 and prob[:, 30:].mean() < 0.1
    assert "torchvision" in seg.version()


def test_rembg_tool_explains_how_to_install_when_missing(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "rembg":
            raise ImportError("no module")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"pip install -e \"\.\[photos\]\""):
        segment.RembgSegmenter()


def test_rembg_tool_wiring(monkeypatch):
    import sys
    import types
    fake = types.ModuleType("rembg")
    fake.__version__ = "0"
    fake.new_session = lambda model: object()
    fake.remove = lambda im, session=None, only_mask=True: Image.new("L", im.size, 255)
    monkeypatch.setitem(sys.modules, "rembg", fake)
    seg = segment.RembgSegmenter()
    assert seg(np.zeros((10, 8, 3), np.uint8)).min() == 1.0 and "u2net_human_seg" in seg.version()


# ---------------------------------------------------------------- impossible values (first real photos)

def test_model_value_outside_adult_range_is_withheld():
    from anthro import output, predict
    est = {"chest_girth": output.Estimate(-91308.5, "photo_model", "measured"),
           "waist_girth": output.Estimate(80.0, "photo_model", "measured"),
           "height": output.Estimate(174.0, "input", "measured")}
    out, notes = predict.sanity_checks(est, "male", {})
    assert out["chest_girth"].value_cm is None and out["chest_girth"].quality == "not_available"
    assert out["waist_girth"].value_cm == 80.0 and out["height"].value_cm == 174.0
    assert any("chest girth withheld: the model gave" in n for n in notes)


def test_infinite_outline_feature_is_clipped_and_reported():
    import numpy as np
    import pandas as pd

    from anthro import geometry_fit as gfit
    names = gfit.feature_names()
    fit_ = {"scaler": {"mean": [0.0] * len(names), "std": [1.0] * len(names)}}
    t = pd.DataFrame([{n: 1.0 for n in names}])
    t[names[0]] = np.inf
    orig = gfit.apply_correction
    gfit.apply_correction = lambda t_, f_: t_
    try:
        Z = gfit.feature_matrix(t, fit_)
        assert np.isfinite(Z).all() and np.abs(Z).max() <= gfit.FEATURE_LIMIT
        assert not gfit.features_in_range(t, fit_)
        t[names[0]] = 1.0
        assert gfit.features_in_range(t, fit_)
    finally:
        gfit.apply_correction = orig


def test_tape_values_that_are_not_numbers_count_as_missing():
    import pandas as pd
    from anthro import segment_compare
    result = pd.DataFrame({"person": ["A", "B"], "refused": [None, None], "waist_girth": [80.0, 90.0],
                           "height_flag": [False, False], "retake": [False, False]})
    pairs = pd.DataFrame({"person": ["A", "B"], "waist_girth": ["81.0", "42.0_tbr"]})
    s = segment_compare.scores(result, pairs)
    assert s["tape_errors"]["waist_girth"]["people"] == 1 and s["tape_errors"]["waist_girth"]["mae_cm"] == 1.0
