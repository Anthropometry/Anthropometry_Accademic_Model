"""Save and load trained models safely.

Security decision (DISCUSSION D-012): checkpoints hold only tensors and plain Python values
(numbers, strings, lists, dicts), and are always loaded with `torch.load(weights_only=True)`.

Why it matters: a PyTorch checkpoint is a pickle file, and loading an arbitrary pickle can run
arbitrary code on your machine. `weights_only=True` refuses anything that isn't data. Our first
notebook version stored numpy arrays and loaded with `weights_only=False`; that is fixed here by
storing the standardisation statistics as plain lists.

Each checkpoint also records the training config, the data fingerprint and the package version,
so every result can be traced to how it was produced (docs/SECURITY.md, "traceability").
"""

from __future__ import annotations

from pathlib import Path

import torch

from . import __version__
from .config import MEASUREMENTS, TrainConfig
from .dataset import Scaler
from .model import build


def save(path: str | Path, net: torch.nn.Module, cfg: TrainConfig, x_scaler: Scaler,
         y_scaler: Scaler, data_fp: str, extra: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": net.state_dict(),
        "config": cfg.to_dict(),
        "x_scaler": x_scaler.to_dict(),
        "y_scaler": y_scaler.to_dict(),
        "measurements": list(MEASUREMENTS),
        "data_fingerprint": data_fp,
        "package_version": __version__,
        "extra": extra or {},
    }, path)


def load(path: str | Path, device: str = "cpu"):
    ck = torch.load(path, map_location=device, weights_only=True)
    if ck["measurements"] != list(MEASUREMENTS):
        raise ValueError("checkpoint was trained on a different measurement list")
    cfg = TrainConfig(**ck["config"])
    net = build(pretrained=False)          # weights come from the checkpoint, not the internet
    net.load_state_dict(ck["state_dict"])
    net.to(device).eval()
    return net, cfg, Scaler(**ck["x_scaler"]), Scaler(**ck["y_scaler"]), ck
