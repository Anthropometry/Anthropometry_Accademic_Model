"""S3b end to end on fake BodyM: geometry cache, fit, cm loss, hybrid training, retake, CIs.

Proves the pipeline runs and the safety rules hold; accuracy needs real BodyM on a GPU.
"""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from anthro import checkpoint
from anthro import geometry_fit as gfit
from anthro import train as train_mod
from anthro.cli import main
from anthro.config import TrainConfig
from anthro.data import fingerprint, load_all
from anthro.evaluate import bootstrap_ci, retake_threshold
from anthro.preprocess import crop_pad_resize

CFG = dict(img_h=64, img_w=32, epochs=1, batch_size=8, pretrained=False, num_workers=0, val_fraction=0.25)


# ---------------------------------------------------------------- A1, A2

def test_cm_loss_equals_mean_error_in_cm():
    std = torch.tensor([10.0, 1.0])
    pred, target = torch.tensor([[0.2, 0.5]]), torch.tensor([[0.0, 0.0]])
    # standardised errors 0.2 and 0.5 -> 2 cm and 0.5 cm -> mean 1.25 cm
    assert train_mod.loss_fn(pred, target, "cm", std).item() == pytest.approx(1.25)
    assert train_mod.loss_fn(pred, target, "std", std).item() == pytest.approx(0.35)   # r1 unchanged


def test_unknown_loss_is_refused():
    with pytest.raises(ValueError, match="loss"):
        TrainConfig(loss="mse")


def test_512_by_256_keeps_shape_without_stretching():
    mask = np.zeros((1200, 800), bool)
    mask[100:1100, 350:450] = True                    # 1000 tall, 100 wide body
    out = crop_pad_resize(mask, 512, 256)
    assert out.shape == (512, 256)
    rows, cols = np.nonzero(out > 127)
    ratio = (np.ptp(rows) + 1) / (np.ptp(cols) + 1)
    assert ratio == pytest.approx(10, rel=0.1)          # still 10:1, not squashed


# ---------------------------------------------------------------- B2, B3

@pytest.fixture
def geo_fit(bodym, tmp_path):
    splits = load_all(bodym)
    fp = fingerprint(bodym)
    c = gfit.build_cache(splits["train"], bodym, tmp_path / "geo", "train", fp)
    return gfit.fit(splits["train"], c, seed=0, val_fraction=0.25, k=3, verbose=False), tmp_path / "geo"


def test_geometry_cache_loads_without_pickle(bodym, tmp_path):
    splits = load_all(bodym)
    fp = fingerprint(bodym)
    gfit.build_cache(splits["testA"], bodym, tmp_path / "geo", "testA", fp)
    path = gfit.cache_path(tmp_path / "geo", "testA", fp)
    with np.load(path, allow_pickle=False) as z:
        assert z["ok"].all() and z["w_cm"].shape[1] == 500
    again = gfit.build_cache(splits["testA"], bodym, tmp_path / "geo", "testA", fp)   # reused
    assert len(again["photo_id"]) == len(splits["testA"])


def test_cv_folds_never_share_a_person(bodym):
    df = load_all(bodym)["train"].reset_index(drop=True)
    for tr, te in gfit._folds(df, 3):
        assert not set(df.subject_id.iloc[tr]) & set(df.subject_id.iloc[te])


def test_fit_is_plain_json_and_excludes_validation_people(geo_fit):
    fit_, _ = geo_fit
    plain = gfit.to_plain(fit_)
    assert json.loads(json.dumps(plain)) == plain
    assert fit_["fit_people"] == 18                     # 24 people minus 25% validation
    for name in ("chest", "waist", "hip"):
        assert fit_["search"][name]["chosen"] in fit_["search"][name]["options"]
    assert len(fit_["scaler"]["mean"]) == len(gfit.feature_names())


def test_feature_matrix_is_finite_and_right_width(bodym, geo_fit):
    fit_, cache = geo_fit
    splits = load_all(bodym)
    Z, tab = gfit.features_for(splits["train"], splits["train"], bodym, cache, "train",
                               fingerprint(bodym), fit_)
    assert Z.shape == (len(splits["train"]), len(gfit.feature_names()))
    assert np.isfinite(Z).all() and "waist_C_hat" in tab


# ---------------------------------------------------------------- B4, B5, C

def test_hybrid_trains_saves_and_reloads_safely(bodym, tmp_path, geo_fit):
    fit_, cache = geo_fit
    cfg = TrainConfig(loss="cm", **CFG)
    path = train_mod.run(str(bodym), str(tmp_path / "hy"), cfg, device="cpu",
                         geo_fit=gfit.to_plain(fit_), geo_cache=str(cache))
    net, cfg2, *_rest, ck = checkpoint.load(path)        # weights_only=True inside
    assert cfg2.n_geo == len(gfit.feature_names()) and cfg2.loss == "cm"
    assert ck["geo_fit"]["params"] == gfit.to_plain(fit_)["params"]


def test_fit_from_a_different_person_split_is_refused(bodym, tmp_path, geo_fit):
    fit_, cache = geo_fit
    cfg = TrainConfig(**{**CFG, "val_fraction": 0.5})
    with pytest.raises(ValueError, match="different person split"):
        train_mod.run(str(bodym), str(tmp_path / "bad"), cfg, device="cpu", geo_fit=fit_,
                      geo_cache=str(cache))


def test_cli_geometry_fit_train_and_evaluate_with_retake(bodym, tmp_path, capsys):
    geo_dir, fit_json, run = tmp_path / "g", tmp_path / "fit.json", tmp_path / "r"
    main(["geometry", "--data", str(bodym), "--out", str(geo_dir)])
    main(["geometry-fit", "--data", str(bodym), "--cache", str(geo_dir), "--out", str(fit_json),
          "--val-fraction", "0.25"])
    out = capsys.readouterr().out
    assert "E1: ellipse alone" in out and "E2: corrected geometry" in out
    main(["train", "--data", str(bodym), "--out", str(run), "--epochs", "1", "--batch-size", "8",
          "--img-h", "64", "--img-w", "32", "--workers", "0", "--no-pretrained", "--loss", "cm",
          "--val-fraction", "0.25",
          "--geo-fit", str(fit_json), "--geo-cache", str(geo_dir), "--cache", str(tmp_path / "c")])
    main(["evaluate", "--data", str(bodym), "--ckpt", str(run / "best.pt"), "--test", "testA",
          "--geo-cache", str(geo_dir), "--cache", str(tmp_path / "c"), "--bootstrap", "50"])
    out = capsys.readouterr().out
    assert "girths" in out and "lengths" in out and "retake warning" in out
    report = json.loads((run / "eval_report.json").read_text())
    assert "retake" in report["testA"]


def test_retake_threshold_fires_for_the_chosen_share():
    gaps = pd.Series(np.arange(100, dtype=float))
    thr = retake_threshold(gaps, 0.10)
    assert (gaps > thr).mean() == pytest.approx(0.10, abs=0.01)
    with pytest.raises(ValueError):
        retake_threshold(gaps, 1.5)


def test_bootstrap_intervals_contain_the_estimate():
    err = pd.DataFrame({"waist": np.linspace(0, 3, 50), "wrist": np.linspace(0, 1, 50)})
    ci = bootstrap_ci(err, n=200)
    lo, hi = map(float, ci.loc["waist", "MAE_95ci"].split("-"))
    assert lo <= err["waist"].mean() <= hi


def test_edge_of_search_grid_is_detected():
    assert gfit._at_edge([0.0, 0.02, 0.04], 2) and not gfit._at_edge([0.0, 0.02, 0.04], 1)
    bands = [[0.3, 0.46], [0.33, 0.49], [0.36, 0.52]]
    assert gfit._at_edge(bands, 0) and gfit._at_edge(bands, 2) and not gfit._at_edge(bands, 1)
