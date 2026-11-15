"""Turn tables + silhouettes into tensors the network can learn from.

Design decisions
----------------
* Cache prepared images once. Reading ~12,000 PNGs every epoch is slow, and Colab
  disconnects often. Each split's prepared images are saved as one .npy file (on Google
  Drive in Colab), keyed by image size and data fingerprint, so a stale cache is never reused.
* Standardise numbers using TRAINING statistics only. Networks learn best on values near 0,
  so heights, weights and measurements become "distance from the training average". Test
  statistics are never used: that would quietly leak test information into training.
* 5 input channels: front, side, then height, weight and sex each spread over a whole
  channel. This follows BMnet, which concatenated height and weight as constant-valued
  channels so an image network can use them from its first layer.
* Light augmentation (±3% shift and scale) teaches the model that small framing changes do
  not change the body. No flipping or rotation: a mirrored side view is a different pose,
  and rotation would undo the level-camera assumption.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .config import HEIGHT_COL, MEASUREMENTS, PHOTO_COL, SEX_COL, WEIGHT_COL
from .preprocess import prepare_pair


def cache_images(df: pd.DataFrame, root: str | Path, cache_dir: str | Path, name: str,
                 img_h: int, img_w: int, data_fp: str) -> np.ndarray:
    path = Path(cache_dir) / f"{name}_{img_h}x{img_w}_{data_fp}.npy"
    if path.exists():
        arr = np.load(path, allow_pickle=False)   # never unpickle: a .npy can carry code otherwise
        if arr.shape == (len(df), 2, img_h, img_w):
            return arr
    arr = np.zeros((len(df), 2, img_h, img_w), np.uint8)
    for i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df), desc=f"prepare {name}")):
        d = Path(root) / row.split
        pid = getattr(row, PHOTO_COL)
        arr[i] = prepare_pair(d / "mask" / f"{pid}.png", d / "mask_left" / f"{pid}.png", img_h, img_w)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return arr


class Scaler:
    """Mean/std standardiser fitted on training data; stored in the checkpoint as plain lists."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, np.float64)
        self.std = np.asarray(std, np.float64)

    @classmethod
    def fit(cls, values: np.ndarray) -> Scaler:
        return cls(values.mean(0), values.std(0) + 1e-8)

    def encode(self, v):
        return (np.asarray(v) - self.mean) / self.std

    def decode(self, z):
        return np.asarray(z) * self.std + self.mean

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def numeric_inputs(df: pd.DataFrame, x_scaler: Scaler) -> np.ndarray:
    hw = x_scaler.encode(df[[HEIGHT_COL, WEIGHT_COL]].to_numpy(np.float64))
    male = (df[SEX_COL].str.lower() == "male").to_numpy(np.float64)[:, None]
    return np.hstack([hw, male]).astype(np.float32)


class SilhouetteSet(Dataset):
    def __init__(self, images: np.ndarray, df: pd.DataFrame, x_scaler: Scaler, y_scaler: Scaler,
                 augment: bool = False):
        assert len(images) == len(df), "images and table rows are out of step"
        self.images = images
        self.num = numeric_inputs(df, x_scaler)
        self.y = y_scaler.encode(df[list(MEASUREMENTS)].to_numpy(np.float64)).astype(np.float32)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i):
        im = torch.from_numpy(self.images[i]).float() / 255.0   # (2, H, W)
        if self.augment:
            im = _jitter(im)
        _, h, w = im.shape
        num = torch.from_numpy(self.num[i])[:, None, None].expand(3, h, w)
        return torch.cat([im, num], 0), torch.from_numpy(self.y[i])


def _jitter(im: torch.Tensor, amount: float = 0.03) -> torch.Tensor:
    s = 1 + (torch.rand(1).item() - 0.5) * 2 * amount
    tx, ty = ((torch.rand(2) - 0.5) * 2 * amount).tolist()
    theta = torch.tensor([[s, 0.0, tx], [0.0, s, ty]])[None]
    grid = F.affine_grid(theta, (1, *im.shape), align_corners=False)
    return F.grid_sample(im[None], grid, align_corners=False)[0]
