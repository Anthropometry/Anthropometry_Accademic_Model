"""S3c shape check: superellipse maths, the inverse, the CLI end to end, and refusals."""

import json

import numpy as np
import pytest

from anthro import geometry as geo
from anthro import geometry_shape as gs
from anthro.cli import main


def test_superellipse_with_n2_is_the_oval():
    for a, b in ((16.0, 12.0), (10.0, 10.0), (15.0, 6.0)):
        assert gs.superellipse_perimeter(a, b, 2.0) == pytest.approx(geo.ramanujan(a, b), rel=1e-3)


def test_boxier_shapes_are_longer_and_approach_the_rectangle():
    a, b = 16.0, 12.0
    p = [float(gs.superellipse_perimeter(a, b, n)) for n in (2.0, 2.5, 3.0, 5.0, 8.0)]
    assert p == sorted(p)
    assert p[-1] < 4 * (a + b)                     # never longer than the rectangle 2(W + D)
    assert p[-1] > 0.9 * 4 * (a + b)               # and within 10% of it at n = 8


def test_fit_n_recovers_the_boxiness_that_made_the_perimeter():
    a, b = np.array([16.0, 14.0, 18.0]), np.array([12.0, 13.0, 9.0])
    n = np.array([2.0, 2.6, 3.4])
    girth = gs.superellipse_perimeter(a, b, n)
    assert gs.fit_n(a, b, girth) == pytest.approx(n, abs=0.02)


def test_fit_n_clips_impossible_girths():
    assert gs.fit_n(np.array([16.0]), np.array([12.0]), np.array([500.0]))[0] == gs.N_MAX


def test_cli_shape_check_runs_and_writes_report(bodym, tmp_path, capsys):
    geo_dir, fit_json = tmp_path / "g", tmp_path / "fit.json"
    main(["geometry", "--data", str(bodym), "--out", str(geo_dir), "--split", "train"])
    main(["geometry-fit", "--data", str(bodym), "--cache", str(geo_dir), "--out", str(fit_json),
          "--val-fraction", "0.25"])
    main(["geometry-shape", "--data", str(bodym), "--cache", str(geo_dir), "--fit", str(fit_json)])
    out = capsys.readouterr().out
    assert "superellipse" in out and "oval x k" in out
    report = json.loads((tmp_path / "geo_shape_report.json").read_text())
    assert set(report["levels"]) == {"chest", "waist", "hip"}


def test_fit_from_other_data_is_refused(bodym, tmp_path):
    fit_json = tmp_path / "fit.json"
    fit_json.write_text(json.dumps({"seed": 0, "val_fraction": 0.25, "params": {},
                                    "data_fingerprint": "not-this-data"}))
    with pytest.raises(SystemExit, match="different data"):
        main(["geometry-shape", "--data", str(bodym), "--cache", str(tmp_path / "g"), "--fit", str(fit_json)])


def test_something_that_is_not_a_fit_is_refused(bodym):
    from anthro.data import load_all
    with pytest.raises(ValueError, match="not a geometry fit"):
        gs.run(load_all(bodym)["train"], {}, {"hello": 1})
