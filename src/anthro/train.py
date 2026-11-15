"""Step 3 — train the BMnet-style model and choose the best epoch on held-back people.

Training decisions
------------------
* Loss: L1 (mean absolute error) on standardised measurements. It matches how we report
  results (average error in cm) and is less thrown by a few unusual bodies than squared error.
* Optimiser: AdamW with a one-cycle learning-rate schedule — a standard, robust recipe for
  fine-tuning a pre-trained network, with few settings to tune.
* Model selection: after each epoch we score the validation PEOPLE (per person, averaged over
  their photos) and keep the best epoch. Test-A and Test-B are never looked at during
  training, so their scores stay an honest estimate.
* Mixed precision on GPU: roughly twice as fast on a Colab T4, same accuracy.
* Reproducibility: fixed seeds and a person-level split that is the same on every machine.
  Results can still vary slightly between GPU runs; report the seed with every result.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from . import checkpoint, model
from .config import HEIGHT_COL, MEASUREMENTS, WEIGHT_COL, TrainConfig
from .data import fingerprint, load_all, split_by_person
from .dataset import Scaler, SilhouetteSet, cache_images
from .evaluate import per_person_errors


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict(net: torch.nn.Module, loader: DataLoader, y_scaler: Scaler, device: str) -> np.ndarray:
    """Predictions in cm, one row per photo pair, in loader order."""
    net.eval()
    out = []
    with torch.no_grad():
        for xb, _ in loader:
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                out.append(net(xb.to(device)).float().cpu().numpy())
    return y_scaler.decode(np.concatenate(out))


def run(data_root: str, out_dir: str, cfg: TrainConfig, cache_dir: str | None = None,
        device: str | None = None) -> Path:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_dir or str(out / "cache")

    splits = load_all(data_root)
    data_fp = fingerprint(data_root)
    train_df, val_df = split_by_person(splits["train"], cfg.val_fraction, cfg.seed)
    print(f"device {device} | data {data_fp} | train {train_df.subject_id.nunique()} people "
          f"/ {len(train_df)} pairs | val {val_df.subject_id.nunique()} people / {len(val_df)} pairs")

    # Statistics from the training part only (see dataset.py).
    x_scaler = Scaler.fit(train_df[[HEIGHT_COL, WEIGHT_COL]].to_numpy(np.float64))
    y_scaler = Scaler.fit(train_df[list(MEASUREMENTS)].to_numpy(np.float64))

    # The cache name includes the split seed and fraction, so a different split can't reuse it.
    tag = f"s{cfg.seed}v{cfg.val_fraction:g}"
    imgs_tr = cache_images(train_df, data_root, cache_dir, f"train_{tag}", cfg.img_h, cfg.img_w, data_fp)
    imgs_va = cache_images(val_df, data_root, cache_dir, f"val_{tag}", cfg.img_h, cfg.img_w, data_fp)

    g = torch.Generator().manual_seed(cfg.seed)
    tr_loader = DataLoader(SilhouetteSet(imgs_tr, train_df, x_scaler, y_scaler, augment=cfg.augment),
                           batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                           num_workers=cfg.num_workers, generator=g)
    va_loader = DataLoader(SilhouetteSet(imgs_va, val_df, x_scaler, y_scaler),
                           batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    if len(tr_loader) == 0:
        raise ValueError("training set smaller than one batch: lower batch_size")

    net = model.build(cfg.pretrained).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * len(tr_loader))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    ckpt_path = out / "best.pt"
    best, log = float("inf"), []
    for epoch in range(1, cfg.epochs + 1):
        t0, net_loss = time.time(), 0.0
        net.train()
        for xb, yb in tqdm(tr_loader, desc=f"epoch {epoch}/{cfg.epochs}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                loss = F.l1_loss(net(xb), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            net_loss += loss.item()

        val_err = per_person_errors(val_df, predict(net, va_loader, y_scaler, device))
        val_mae = float(val_err.to_numpy().mean())
        row = {"epoch": epoch, "train_loss": round(net_loss / len(tr_loader), 4),
               "val_mae_cm": round(val_mae, 3), "seconds": round(time.time() - t0, 1)}
        saved = val_mae < best
        if saved:
            best = val_mae
            checkpoint.save(ckpt_path, net, cfg, x_scaler, y_scaler, data_fp, extra={"epoch": epoch})
        log.append(row)
        print(f"epoch {epoch:2d} | train loss {row['train_loss']:.3f} | val MAE {val_mae:.2f} cm"
              f" | {row['seconds']}s{'  <- saved' if saved else ''}")

    (out / "train_log.json").write_text(json.dumps({"config": cfg.to_dict(), "data_fingerprint": data_fp,
                                                    "epochs": log}, indent=2))
    return ckpt_path
