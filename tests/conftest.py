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


def make_hybrid_run(tmp):
    """Fake BodyM -> geometry -> fit -> hybrid training -> Test-B evaluation, as on Colab (S7-0, S5a)."""
    from anthro.cli import main
    data = make_fake_bodym(tmp / "bodym")
    geo_dir, fit_json, out = tmp / "g", tmp / "fit.json", tmp / "r"
    main(["geometry", "--data", str(data), "--out", str(geo_dir)])
    main(["geometry-fit", "--data", str(data), "--cache", str(geo_dir), "--out", str(fit_json),
          "--val-fraction", "0.25"])
    main(["train", "--data", str(data), "--out", str(out), "--epochs", "1", "--batch-size", "8",
          "--img-h", "64", "--img-w", "32", "--workers", "0", "--no-pretrained", "--loss", "cm",
          "--val-fraction", "0.25", "--geo-fit", str(fit_json), "--geo-cache", str(geo_dir),
          "--cache", str(tmp / "c")])
    main(["evaluate", "--data", str(data), "--ckpt", str(out / "best.pt"), "--test", "testB",
          "--geo-cache", str(geo_dir), "--cache", str(tmp / "c")])
    pid = "testB-0-p0"
    return {"data": data, "ckpt": out / "best.pt", "tmp": tmp, "geo": geo_dir, "cache": tmp / "c",
            "front": data / "testB" / "mask" / f"{pid}.png",
            "side": data / "testB" / "mask_left" / f"{pid}.png"}


def make_fake_ansur(folder, n=(300, 200), seed=0, front=False):
    """Two fake ANSUR II files with the real column names (mixed case as in the public files), mm and
    tenths of a kg. Values are invented: plausible sizes driven by height and weight, plus noise."""
    from anthro import ansur
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    start = 10000
    for (sex, name), k in zip(ansur.FILES.items(), n, strict=True):
        h = rng.normal(1760 if sex == "male" else 1630, 70, k)            # mm
        w = rng.normal(850 if sex == "male" else 680, 120, k)              # tenths of a kg
        cols = {c: 0.0 for c in ansur.needed_columns()}
        size = (w / 10) / 75

        def g(base, size=size, k=k):
            return base * (0.6 + 0.4 * size) + rng.normal(0, base * 0.02, k)

        def L(frac, h=h, k=k):  # noqa: N802
            return frac * h + rng.normal(0, 8, k)
        cols.update({
            "neckcircumferencebase": g(400), "shoulderlength": L(0.087), "buttockcircumference": g(1000),
            "waistbacklength": L(0.27), "crotchheight": L(0.47), "lateralmalleolusheight": L(0.04),
            "waistheightomphalion": L(0.6), "buttockheight": L(0.5), "kneeheightmidpatella": L(0.28),
            "acromialheight": L(0.82), "chestcircumference": g(1000), "waistcircumference": g(900),
            "biacromialbreadth": L(0.23), "thighcircumference": g(600), "calfcircumference": g(380),
            "wristcircumference": g(170),
        })
        if front:                                    # S5b: ANSUR's seated front length (optional column)
            cols["waistfrontlengthsitting"] = L(0.25)
        cols.update({"subjectid": np.arange(start, start + k), "gender": sex.capitalize(),
                     "dodrace": rng.choice([1, 2, 3], k, p=[0.6, 0.3, 0.1]), "stature": h, "weightkg": w,
                     "heightin": h / 25.4 + rng.normal(0, 0.5, k),
                     "weightlbs": w / 10 / 0.4536 + rng.normal(0, 3, k)})
        start += k
        df = pd.DataFrame(cols)
        if sex == "female":
            df = df.rename(columns={"subjectid": "SubjectId", "dodrace": "DODRace", "gender": "Gender"})
        df.to_csv(folder / name, index=False)
    return folder
