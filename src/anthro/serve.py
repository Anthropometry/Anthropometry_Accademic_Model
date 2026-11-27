"""S7c — the web API: two photos and a person's details in, the `anthro-output/1` result out.

specs/s7c-serve/spec.md (approved by Ryan, 26 Nov 2026; packages D-045; tool D-046).

Run it:  uvicorn anthro.serve:create_app_from_env --factory --host 0.0.0.0 --port 8080
(the Docker image does this). Settings come from environment variables:

| Variable | Meaning | Default |
| --- | --- | --- |
| `ANTHRO_MODEL_DIR` | folder made by `anthro prepare-serving` (model.pt, eval_report.json) | `/models` |
| `ANTHRO_API_KEY` | the key every request must send as `X-API-Key` | required |
| `ANTHRO_ALLOW_NO_KEY` | `1` to run without a key (laptop practice only) | off |
| `ANTHRO_TOOL` | segmentation tool: `torchvision` or `rembg` | `torchvision` (D-046) |
| `ANTHRO_ALLOWED_ORIGINS` | comma-separated web addresses allowed to call it from a browser | none |
| `ANTHRO_MAX_MB` | largest photo accepted, in MB | 15 |

Endpoints
---------
* `GET /v1/health` — 200 when ready (model name, schema, catalogue, tool, licence); 503 + reason if not.
* `GET /v1/catalogue` — the measurement list for screens (code, label, sexes, adult range, type).
* `POST /v1/measure` — multipart form: `front`, `side` (photo files), `height_cm`, `sex`, and optionally
  `height_kind`, `age_range`, `weight_kg`, `weight_kind`, `clothing`, `front_pitch`, `front_roll`,
  `side_pitch`, `side_roll`. 200 → the result; 422 → refused (`{"error": "refused", "reasons": [...]}`);
  401 → wrong or missing key; 413 → photo too large; 503 → not ready.

Why these decisions
-------------------
* The server adds no measuring logic: `/v1/measure` calls `predict.predict_photos` (S7a), so the API and the
  command line can never disagree.
* One request at a time per process (a lock): the model and the segmentation tool are large, and two at once
  would double the memory; a busy server queues rather than crashing.
* Nothing kept: photos are read from memory; the silhouettes the model reads live in a temporary in-memory
  folder deleted with the answer. Logs hold only the status and the time taken.
* Key check in constant time (`hmac.compare_digest`), so the key can't be guessed by timing.
* Non-commercial: the model is trained on BodyM (CC BY-NC 4.0); `/v1/health` says so.
"""

from __future__ import annotations

import hmac
import logging
import os
import threading
import time
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import catalog, output, predict, segment
from .photo import PhotoError

log = logging.getLogger("anthro.serve")
LICENCE = "trained on BodyM (CC BY-NC 4.0): non-commercial use only (school project and demos)"


class _State:
    def __init__(self):
        self.model_path: Path | None = None
        self.seg = None
        self.bodym_side: str | None = None
        self.model_name: str | None = None
        self.error: str | None = None
        self.lock = threading.Lock()


def create_app(model_dir: str | Path, seg=None, api_key: str | None = None, allow_no_key: bool = False,
               origins: tuple[str, ...] = (), max_mb: float = 15.0, tool: str = "torchvision") -> FastAPI:
    if not api_key and not allow_no_key:
        raise ValueError("ANTHRO_API_KEY is not set (set ANTHRO_ALLOW_NO_KEY=1 only for laptop practice)")
    st = _State()
    try:
        st.model_path = Path(model_dir) / "model.pt"
        *_rest, ck = predict.load_model(st.model_path)
        extra = ck.get("extra") or {}
        if not extra.get("typical_weight"):
            raise ValueError("model.pt has no typical-weight line: make it with `anthro prepare-serving`")
        st.bodym_side, st.model_name = extra.get("bodym_side"), extra.get("model_name")
        st.seg = seg if seg is not None else segment.get(tool)
    except Exception as e:  # noqa: BLE001 — the reason is reported by /v1/health instead of crashing
        st.error = f"{type(e).__name__}: {e}"
        log.error("not ready: %s", st.error)

    app = FastAPI(title="Anthropometry model API", version="1",
                  description="Two photos and a person's details in; the anthro-output/1 result out. "
                              + LICENCE)
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(origins), allow_methods=["GET", "POST"],
                           allow_headers=["X-API-Key", "Content-Type"])
    max_bytes = int(max_mb * 1024 * 1024)

    def check_key(x_api_key: str | None = Header(default=None)):
        if allow_no_key and not api_key:
            return
        if not x_api_key or not hmac.compare_digest(x_api_key.encode(), api_key.encode()):
            raise HTTPException(status_code=401, detail="missing or wrong X-API-Key")

    def ready():
        if st.error:
            raise HTTPException(status_code=503, detail=f"model not ready: {st.error}")

    @app.get("/v1/health")
    def health():
        if st.error:
            return JSONResponse(status_code=503, content={"status": "not ready", "reason": st.error})
        return {"status": "ok", "schema": output.SCHEMA, "catalogue": catalog.version(),
                "model_name": st.model_name, "segmenter": st.seg.name, "licence": LICENCE}

    @app.get("/v1/catalogue", dependencies=[Depends(check_key)])
    def catalogue():
        return {"catalogue": catalog.version(),
                "measurements": [{k: m[k] for k in ("code", "label", "type", "sexes", "adult_range_cm")}
                                 for m in catalog.catalogue()["measurements"]]}

    def _read(f: UploadFile, name: str) -> bytes:
        data = f.file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(status_code=413, detail=f"{name} photo is larger than {max_mb:g} MB")
        if not data:
            raise HTTPException(status_code=422, detail=f"{name} photo is empty")
        return data

    @app.post("/v1/measure", dependencies=[Depends(check_key)])
    def measure(front: UploadFile = File(...), side: UploadFile = File(...),  # noqa: B008 (FastAPI form)
                height_cm: float = Form(...), sex: str = Form(...),
                height_kind: str = Form("measured"), age_range: str | None = Form(None),
                weight_kg: float | None = Form(None), weight_kind: str = Form("stated"),
                clothing: str | None = Form(None),
                front_pitch: float | None = Form(None), front_roll: float | None = Form(None),
                side_pitch: float | None = Form(None), side_roll: float | None = Form(None)):
        ready()
        t0 = time.perf_counter()
        fb, sb = _read(front, "front"), _read(side, "side")
        tilt = lambda p, r: None if p is None else (p, r or 0.0)  # noqa: E731
        if height_kind not in ("measured", "stated") or weight_kind not in ("scale", "stated"):
            raise HTTPException(status_code=422,
                                detail="height_kind: measured|stated; weight_kind: scale|stated")
        try:
            with st.lock:
                doc = predict.predict_photos(fb, sb, height_cm, sex, st.model_path, st.seg, st.bodym_side,
                                             front_tilt=tilt(front_pitch, front_roll),
                                             side_tilt=tilt(side_pitch, side_roll), weight_kg=weight_kg,
                                             age_range=age_range, height_kind=height_kind, clothing=clothing,
                                             weight_kind=weight_kind)
        except (PhotoError, predict.PredictError, catalog.CatalogError, ValueError) as e:
            log.info("measure status=refused ms=%d", (time.perf_counter() - t0) * 1000)
            return JSONResponse(status_code=422, content={"error": "refused", "reasons": _reasons(str(e))})
        log.info("measure status=ok ms=%d", (time.perf_counter() - t0) * 1000)
        return doc

    return app


def _reasons(message: str) -> list[str]:
    """'front photo: a; b' -> ['front photo: a', 'front photo: b'] so each reason can be shown on its own."""
    head, sep, rest = message.partition(": ")
    if sep and head.endswith("photo"):
        return [f"{head}: {r}" for r in rest.split("; ")]
    return [message]


def create_app_from_env() -> FastAPI:
    logging.basicConfig(level=logging.INFO)
    env = os.environ.get
    origins = tuple(o.strip() for o in env("ANTHRO_ALLOWED_ORIGINS", "").split(",") if o.strip())
    return create_app(env("ANTHRO_MODEL_DIR", "/models"), api_key=env("ANTHRO_API_KEY") or None,
                      allow_no_key=env("ANTHRO_ALLOW_NO_KEY") == "1", origins=origins,
                      max_mb=float(env("ANTHRO_MAX_MB", "15")), tool=env("ANTHRO_TOOL", "torchvision"))
