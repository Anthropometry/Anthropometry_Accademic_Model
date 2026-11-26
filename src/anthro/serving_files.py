"""S7c — prepare the model folder a server needs, from a trained run (run once, in Colab).

specs/s7c-serve/spec.md (approved by Ryan, 26 Nov 2026).

What it does
------------
`anthro prepare-serving --ckpt runs/r4/best.pt --data bodym --out-dir DRIVE/serving` writes:
* `model.pt`: the checkpoint plus the three things a server cannot work out without BodyM: the typical-weight
  line (D-027, for people with no weight), the direction BodyM's side silhouettes face (D-044) and the
  model's name (e.g. "r4");
* `eval_report.json`: the run's Test-B report, which the per-measurement errors come from (D-028);
* `MANIFEST.json`: sizes and SHA-256 fingerprints of both, and what they are.

Why
---
* A server has no BodyM (licence, size); everything it needs travels in one small folder.
* The checkpoint stays plain tensors and values, loaded with `weights_only=True` (D-012).
* The folder goes on Drive or the server, never into GitHub (D-014).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch

from . import photo, typical_weight
from .data import load_split


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(ckpt: str | Path, data_root: str | Path, out_dir: str | Path) -> dict:
    ckpt, out = Path(ckpt), Path(out_dir)
    report = ckpt.parent / "eval_report.json"
    if not report.exists() or "testB" not in json.loads(report.read_text()):
        raise ValueError(f"{report} with a testB section is needed (run `anthro evaluate --test testB`)")
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    extra = dict(ck.get("extra") or {})
    if not extra.get("typical_weight"):
        extra["typical_weight"] = typical_weight.fit(load_split(data_root, "train"))
    extra["bodym_side"] = photo.bodym_facing(data_root)
    extra["model_name"] = ckpt.parent.name
    ck["extra"] = extra
    out.mkdir(parents=True, exist_ok=True)
    torch.save(ck, out / "model.pt")
    shutil.copyfile(report, out / "eval_report.json")
    manifest = {
        "model_name": extra["model_name"], "bodym_side": extra["bodym_side"],
        "licence": "trained on BodyM (CC BY-NC 4.0): non-commercial use only",
        "files": {n: {"bytes": (out / n).stat().st_size, "sha256": _sha256(out / n)}
                  for n in ("model.pt", "eval_report.json")},
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
