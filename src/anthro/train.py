"""Step 3 — train the BMnet-style model and choose the best epoch on held-back people.

Training decisions
------------------
* Loss: L1 (mean absolute error), less thrown by a few unusual bodies than squared error.
  - `loss="std"` (run r1): on standardised measurements. Challenge found after r1: one standard
    unit of waist is ~10 cm, of wrist ~1 cm, so a 10 cm waist error cost the same as a 1 cm wrist
    error and the model under-worked chest and waist.
  - `loss="cm"` (S3b, D-017): each measurement's error is multiplied back by its training spread,
    so the loss is the average error in cm, the same thing we score.
* S3b hybrid (D-019): with a geometry fit, the geometric features of each photo pair go to the
  network too. The fit must come from the same person split (seed, fraction): otherwise the
  validation people would have helped fit the geometry and the best epoch would look too good.
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

from . import checkpoint, model, typical_weight
from . import geometry_fit as gfit
from .config import HEIGHT_COL, MEASUREMENTS, WEIGHT_COL, TrainConfig
from .data import fingerprint, load_all, split_by_person
from .dataset import Scaler, SilhouetteSet, cache_images
from .evaluate import per_person_errors, summarise


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict(net: torch.nn.Module, loader: DataLoader, y_scaler: Scaler, device: str) -> np.ndarray:
    """Predictions in cm, one row per photo pair, in loader order."""
    net.eval()
    out = []
    with torch.no_grad():
        for xb, gb, _ in loader:
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                out.append(model.forward(net, xb.to(device), gb.to(device)).float().cpu().numpy())
    return y_scaler.decode(np.concatenate(out))


def loss_fn(pred: torch.Tensor, target: torch.Tensor, kind: str, cm_w: torch.Tensor) -> torch.Tensor:
    """L1 loss; "cm" multiplies each measurement's standardised error by its spread in cm."""
    if kind == "cm":
        return ((pred.float() - target.float()).abs() * cm_w).mean()
    return F.l1_loss(pred, target)            # r1 behaviour, unchanged


def run(data_root: str, out_dir: str, cfg: TrainConfig, cache_dir: str | None = None,
        device: str | None = None, geo_fit: dict | None = None, geo_cache: str | None = None) -> Path:
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

    tw_fit = typical_weight.fit(train_df)     # S7-0 / D-027: stored so prediction can fill a missing weight

    # The cache name includes the split seed and fraction, so a different split can't reuse it.
    tag = f"s{cfg.seed}v{cfg.val_fraction:g}"
    imgs_tr = cache_images(train_df, data_root, cache_dir, f"train_{tag}", cfg.img_h, cfg.img_w, data_fp)
    imgs_va = cache_images(val_df, data_root, cache_dir, f"val_{tag}", cfg.img_h, cfg.img_w, data_fp)

    geo_tr = geo_va = None
    if geo_fit is not None:
        if (geo_fit["seed"], geo_fit["val_fraction"]) != (cfg.seed, cfg.val_fraction):
            raise ValueError("geometry fit used a different person split: rerun `anthro geometry-fit` "
                             f"with --seed {cfg.seed} --val-fraction {cfg.val_fraction:g}")
        gc = geo_cache or cache_dir
        geo_tr, _ = gfit.features_for(splits["train"], train_df, data_root, gc, "train", data_fp, geo_fit)
        geo_va, _ = gfit.features_for(splits["train"], val_df, data_root, gc, "train", data_fp, geo_fit)
        cfg.n_geo = geo_tr.shape[1]
        print(f"hybrid: {cfg.n_geo} geometric features per photo pair")

    g = torch.Generator().manual_seed(cfg.seed)
    tr_loader = DataLoader(SilhouetteSet(imgs_tr, train_df, x_scaler, y_scaler, augment=cfg.augment,
                                         geo=geo_tr),
                           batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                           num_workers=cfg.num_workers, generator=g)
    va_loader = DataLoader(SilhouetteSet(imgs_va, val_df, x_scaler, y_scaler, geo=geo_va),
                           batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    if len(tr_loader) == 0:
        raise ValueError("training set smaller than one batch: lower batch_size")

    net = model.build(cfg.pretrained, cfg.n_geo).to(device)
    # cm loss: per-measurement weights = training spread in cm (constant, not learned)
    cm_w = torch.tensor(y_scaler.std, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * len(tr_loader))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    ckpt_path = out / "best.pt"
    best, log = float("inf"), []
    for epoch in range(1, cfg.epochs + 1):
        t0, net_loss = time.time(), 0.0
        net.train()
        for xb, gb, yb in tqdm(tr_loader, desc=f"epoch {epoch}/{cfg.epochs}", leave=False):
            xb, gb, yb = xb.to(device), gb.to(device), yb.to(device)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                loss = loss_fn(model.forward(net, xb, gb), yb, cfg.loss, cm_w)
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
            best_table = summarise(val_err)
            checkpoint.save(ckpt_path, net, cfg, x_scaler, y_scaler, data_fp,
                            extra={"epoch": epoch, "typical_weight": tw_fit},
                            geo_fit=gfit.to_plain(geo_fit) if geo_fit is not None else None)
        log.append(row)
        print(f"epoch {epoch:2d} | train loss {row['train_loss']:.3f} | val MAE {val_mae:.2f} cm"
              f" | {row['seconds']}s{'  <- saved' if saved else ''}")

    # Per-measurement validation scores at the best epoch: what S3b compares runs on, so the
    # choice between runs never needs Test-A or Test-B.
    print("\nvalidation people at the best epoch (per person):")
    print(best_table.to_string())
    (out / "train_log.json").write_text(json.dumps({
        "config": cfg.to_dict(), "data_fingerprint": data_fp, "epochs": log,
        "best_val": best_table.reset_index().to_dict(orient="records")}, indent=2, default=float))
    return ckpt_path
