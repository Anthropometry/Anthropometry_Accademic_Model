"""A tiny fake BodyM with the real file layout, so tests run anywhere (including CI) without
downloading BodyM — and without ever putting BodyM itself in the repository (licence, D-014)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from anthro.config import MEASUREMENTS


def make_fake_bodym(root, people=(24, 6, 6), pairs=(2, 3, 1), seed=0):
    rng = np.random.default_rng(seed)
    for split, n, k in zip(("train", "testA", "testB"), people, pairs, strict=True):
        d = root / split
        (d / "mask").mkdir(parents=True)
        (d / "mask_left").mkdir()
        ids = [f"{split}-{i}" for i in range(n)]
        h = rng.normal(170, 9, n)
        w = rng.normal(72, 12, n)
        pd.DataFrame({"subject_id": ids, "gender": rng.choice(["male", "female"], n),
                      "height_cm": h, "weight_kg": w}).to_csv(d / "hwg_metadata.csv", index=False)
        meas = {m: 0.3 * h + 0.5 * w + rng.normal(0, 2, n) for m in MEASUREMENTS}
        pd.DataFrame({"subject_id": ids, **meas}).to_csv(d / "measurements.csv", index=False)
        rows = []
        for i, sid in enumerate(ids):
            for j in range(k):
                pid = f"{sid}-p{j}"
                rows.append((sid, pid))
                for folder, width in (("mask", w[i]), ("mask_left", w[i] * 0.7)):
                    im = np.zeros((320, 240), np.uint8)
                    top, bh, bw = 20, int(h[i] * 1.5), int(width)
                    cx = 120 + int(rng.integers(-10, 10))
                    im[top:top + bh, cx - bw // 2:cx + bw // 2] = 255
                    Image.fromarray(im).save(d / folder / f"{pid}.png")
        pmap = pd.DataFrame(rows, columns=["subject_id", "photo_id"])
        pmap.to_csv(d / "subject_to_photo_map.csv", index=False)
    return root


@pytest.fixture
def bodym(tmp_path):
    return make_fake_bodym(tmp_path / "bodym")
