"""S7c: the server folder and the web API, with a fake segmenter and the tiny trained run (no downloads)."""

import json
import logging
import os

import numpy as np
import pytest
import torch
from conftest import make_hybrid_run
from fastapi.testclient import TestClient
from test_s7a import FakeSeg, _photos_from_run, save

from anthro import output, predict, serve, serving_files

KEY = "test-key-123"


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return make_hybrid_run(tmp_path_factory.mktemp("s7c"))


@pytest.fixture(scope="module")
def model_dir(run, tmp_path_factory):
    out = tmp_path_factory.mktemp("serving")
    serving_files.prepare(run["ckpt"], run["data"], out)
    return out


@pytest.fixture(scope="module")
def photos(run, tmp_path_factory):
    return _photos_from_run(run, tmp_path_factory.mktemp("photos"))


@pytest.fixture()
def client(model_dir):
    return TestClient(serve.create_app(model_dir, seg=FakeSeg(), api_key=KEY))


def _form(photos, **extra):
    f, s = photos
    files = {"front": ("front.png", f.read_bytes(), "image/png"),
             "side": ("side.png", s.read_bytes(), "image/png")}
    data = {"height_cm": "165", "sex": "female", "age_range": "25-34", "weight_kg": "60", **extra}
    return files, data


# ---------------------------------------------------------------- the server folder

def test_prepare_serving_makes_a_self_contained_folder(model_dir, run):
    ck = torch.load(model_dir / "model.pt", weights_only=True)
    assert set(ck["extra"]["typical_weight"]) == {"female", "male"} and ck["extra"]["model_name"] == "r"
    assert "bodym_side" in ck["extra"]
    m = json.loads((model_dir / "MANIFEST.json").read_text())
    assert set(m["files"]) == {"model.pt", "eval_report.json"} and "non-commercial" in m["licence"]
    assert len(m["files"]["model.pt"]["sha256"]) == 64


def test_prepare_serving_needs_the_test_b_report(run, tmp_path):
    ck = tmp_path / "x" / "best.pt"
    ck.parent.mkdir()
    ck.write_bytes(run["ckpt"].read_bytes())
    with pytest.raises(ValueError, match="testB"):
        serving_files.prepare(ck, run["data"], tmp_path / "out")


def test_cli_prepare_serving(run, tmp_path, capsys):
    from anthro.cli import main
    main(["prepare-serving", "--ckpt", str(run["ckpt"]), "--data", str(run["data"]),
          "--out-dir", str(tmp_path / "s")])
    assert "server folder ready" in capsys.readouterr().out
    bare = tmp_path / "bare" / "best.pt"                     # a checkpoint with no Test-B report beside it
    bare.parent.mkdir()
    bare.write_bytes(run["ckpt"].read_bytes())
    with pytest.raises(SystemExit, match="refused"):
        main(["prepare-serving", "--ckpt", str(bare), "--data", str(run["data"]),
              "--out-dir", str(tmp_path / "t")])


# ---------------------------------------------------------------- endpoints

def test_health_and_catalogue(client):
    h = client.get("/v1/health").json()
    assert h["status"] == "ok" and h["schema"] == "anthro-output/1" and "non-commercial" in h["licence"]
    assert client.get("/v1/catalogue").status_code == 401
    c = client.get("/v1/catalogue", headers={"X-API-Key": KEY}).json()
    assert len(c["measurements"]) == 25 and {"code", "label", "sexes"} <= set(c["measurements"][0])


def test_measure_returns_the_same_result_as_the_command(client, photos, model_dir):
    files, data = _form(photos)
    r = client.post("/v1/measure", files=files, data=data, headers={"X-API-Key": KEY})
    assert r.status_code == 200
    doc = r.json()
    assert output.validate(doc) == [] and len(doc["measurements"]) == 24
    assert doc["model_version"].startswith("r (")
    direct = predict.predict_photos(photos[0], photos[1], 165.0, "female", model_dir / "model.pt", FakeSeg(),
                                    None, weight_kg=60.0, age_range="25-34")
    assert doc["measurements"] == direct["measurements"]


def test_missing_weight_is_filled_from_the_model_folder(client, photos):
    files, data = _form(photos)
    data.pop("weight_kg")
    doc = client.post("/v1/measure", files=files, data=data, headers={"X-API-Key": KEY}).json()
    assert doc["person"]["weight_kind"] == "assumed"


def test_refusals_come_back_as_422_with_each_reason(client, photos, tmp_path):
    bad = save(np.full((900, 700, 3), 120, np.uint8), tmp_path / "empty.png")
    files, data = _form((bad, photos[1]))
    r = client.post("/v1/measure", files=files, data=data, headers={"X-API-Key": KEY})
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "refused" and body["reasons"][0].startswith("front photo: no person")
    files, data = _form(photos, age_range="13-14")
    r = client.post("/v1/measure", files=files, data=data, headers={"X-API-Key": KEY})
    assert r.status_code == 422 and "adults only" in r.json()["reasons"][0]
    files, data = _form(photos, height_kind="guessed")
    assert client.post("/v1/measure", files=files, data=data, headers={"X-API-Key": KEY}).status_code == 422


def test_key_is_required_and_checked(client, photos):
    files, data = _form(photos)
    assert client.post("/v1/measure", files=files, data=data).status_code == 401
    wrong = client.post("/v1/measure", files=files, data=data, headers={"X-API-Key": "wrong"})
    assert wrong.status_code == 401


def test_photo_too_large_is_413_and_empty_is_422(model_dir, photos):
    c = TestClient(serve.create_app(model_dir, seg=FakeSeg(), api_key=KEY, max_mb=0.001))
    files, data = _form(photos)
    assert c.post("/v1/measure", files=files, data=data, headers={"X-API-Key": KEY}).status_code == 413
    files["side"] = ("side.png", b"", "image/png")
    c2 = TestClient(serve.create_app(model_dir, seg=FakeSeg(), api_key=KEY))
    assert c2.post("/v1/measure", files=files, data=data, headers={"X-API-Key": KEY}).status_code == 422


def test_not_ready_reports_why(tmp_path, photos):
    c = TestClient(serve.create_app(tmp_path, seg=FakeSeg(), api_key=KEY))
    r = c.get("/v1/health")
    assert r.status_code == 503 and "model.pt" in r.json()["reason"]
    files, data = _form(photos)
    assert c.post("/v1/measure", files=files, data=data, headers={"X-API-Key": KEY}).status_code == 503


def test_a_model_without_typical_weight_is_not_served(run, tmp_path):
    ck = torch.load(run["ckpt"], weights_only=True)
    ck["extra"].pop("typical_weight")
    torch.save(ck, tmp_path / "model.pt")
    r = TestClient(serve.create_app(tmp_path, seg=FakeSeg(), api_key=KEY)).get("/v1/health")
    assert r.status_code == 503 and "prepare-serving" in r.json()["reason"]


def test_no_key_is_refused_unless_explicitly_allowed(model_dir, photos):
    with pytest.raises(ValueError, match="ANTHRO_API_KEY"):
        serve.create_app(model_dir, seg=FakeSeg())
    c = TestClient(serve.create_app(model_dir, seg=FakeSeg(), allow_no_key=True))
    files, data = _form(photos)
    assert c.post("/v1/measure", files=files, data=data).status_code == 200


def test_cors_only_for_allowed_origins(model_dir):
    c = TestClient(serve.create_app(model_dir, seg=FakeSeg(), api_key=KEY, origins=("https://app.example",)))
    ok = c.get("/v1/health", headers={"Origin": "https://app.example"})
    other = c.get("/v1/health", headers={"Origin": "https://evil.example"})
    assert ok.headers.get("access-control-allow-origin") == "https://app.example"
    assert "access-control-allow-origin" not in other.headers


def test_logs_hold_no_values_and_nothing_is_kept(client, photos, caplog, tmp_path):
    import tempfile
    places = [p for p in ("/dev/shm", tempfile.gettempdir()) if os.path.isdir(p)]  # noqa: S108 (read-only look)
    before = {p: set(os.listdir(p)) for p in places}
    files, data = _form(photos)
    with caplog.at_level(logging.INFO, logger="anthro.serve"):
        doc = client.post("/v1/measure", files=files, data=data, headers={"X-API-Key": KEY}).json()
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "status=ok" in text
    assert str(doc["measurements"]["waist_girth"]["value_cm"]) not in text and "165" not in text
    for p in places:
        assert set(os.listdir(p)) <= before[p]                   # no leftovers


def test_app_from_environment(model_dir, monkeypatch):
    monkeypatch.setenv("ANTHRO_MODEL_DIR", str(model_dir))
    monkeypatch.setenv("ANTHRO_API_KEY", KEY)
    monkeypatch.setenv("ANTHRO_ALLOWED_ORIGINS", "https://a.example, https://b.example")
    from anthro import segment
    monkeypatch.setitem(segment.TOOLS, "torchvision", FakeSeg)
    c = TestClient(serve.create_app_from_env())
    assert c.get("/v1/health").json()["segmenter"] == "fake"


def test_reasons_are_split_per_photo_problem():
    assert serve._reasons("front photo: a; b") == ["front photo: a", "front photo: b"]
    assert serve._reasons("age range 13-14: adults only") == ["age range 13-14: adults only"]
