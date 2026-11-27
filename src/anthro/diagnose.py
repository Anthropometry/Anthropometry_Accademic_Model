"""Why do phone-photo outlines fall outside what the model saw in training? (S7a diagnosis)

Compares the outline measurements (the hybrid network's geometry features) of one phone-photo pair with
those of BodyM Test-B people of the same sex and similar height, in standard deviations from the training
average. A feature far out for the phone photo but normal for BodyM shows what differs: pose, clothing,
framing (body pixel height) or how the outline is drawn (edge softness).

Nothing is saved: the silhouettes go to a temporary folder and the figure is only shown.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import geometry as geo
from . import geometry_fit as gfit
from . import photo
from .config import HEIGHT_COL, PHOTO_COL, SEX_COL, SUBJECT_COL, WEIGHT_COL


def feature_z(front_path, side_path, height_cm: float, weight_kg: float, sex: str,
              geo_fit: dict) -> pd.Series:
    """Each geometry feature of one silhouette pair, in standard deviations from the training average."""
    m = gfit.measure_pair(front_path, side_path, height_cm)
    df = pd.DataFrame({HEIGHT_COL: [height_cm], WEIGHT_COL: [weight_kg], SEX_COL: [sex], SUBJECT_COL: ["p"]})
    t = gfit.apply_correction(gfit.level_table(df, gfit.single_cache(m), geo_fit["params"]), geo_fit)
    x = t[gfit.feature_names()].to_numpy(np.float64)[0]
    z = (x - np.asarray(geo_fit["scaler"]["mean"])) / np.asarray(geo_fit["scaler"]["std"])
    return pd.Series(z, index=gfit.feature_names())


def bodym_neighbours(data_root, sex: str, height_cm: float, n: int = 3) -> pd.DataFrame:
    """n Test-B photo pairs of the same sex, nearest in height."""
    from .data import load_split
    df = load_split(data_root, "testB")
    df = df[df["gender"].str.lower() == sex.lower()].copy()
    df["gap"] = (df["height_cm"] - height_cm).abs()
    return df.nsmallest(n, "gap")


def compare_with_bodym(front_photo, side_photo, height_cm: float, weight_kg: float, sex: str, ckpt,
                       data_root, tool: str = "torchvision", n: int = 3, show: bool = True) -> pd.DataFrame:
    """Table of the features furthest from training for the phone pair, next to the same features for n
    BodyM people; optionally shows the silhouettes side by side."""
    from . import checkpoint, segment
    ck = checkpoint.load(ckpt, "cpu")[4]          # weights_only loading (D-012)
    geo_fit = ck.get("geo_fit")
    if geo_fit is None:
        raise ValueError("this checkpoint has no geometry features (not a hybrid model)")
    seg = segment.get(tool)
    side_dir = photo.bodym_facing(data_root)
    sf = photo.silhouette(photo.load_photo(front_photo), seg, "front")
    ss = photo.silhouette(photo.load_photo(side_photo), seg, "side", side_dir)
    root = Path(data_root) / "testB"
    nb = bodym_neighbours(data_root, sex, height_cm, n)
    cols = {}
    with tempfile.TemporaryDirectory() as tmp:
        from PIL import Image
        fp, sp = Path(tmp) / "f.png", Path(tmp) / "s.png"
        Image.fromarray(sf.mask).save(fp)
        Image.fromarray(ss.mask).save(sp)
        cols["phone"] = feature_z(fp, sp, height_cm, weight_kg, sex, geo_fit)
        pics = [("phone front", sf.mask), ("phone side", ss.mask)]
        for i, r in enumerate(nb.itertuples(index=False), 1):
            pid = getattr(r, PHOTO_COL)
            f, s = root / "mask" / f"{pid}.png", root / "mask_left" / f"{pid}.png"
            try:
                cols[f"bodym{i}"] = feature_z(f, s, r.height_cm, r.weight_kg, sex, geo_fit)
            except geo.GeometryError as e:
                print(f"bodym{i}: outline not measurable ({e})")
                continue
            if i == 1:
                pics += [("BodyM front", np.asarray(Image.open(f))),
                         ("BodyM side", np.asarray(Image.open(s)))]
    table = pd.DataFrame(cols)
    table = table.reindex(table["phone"].abs().sort_values(ascending=False).index).round(1)
    if show:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, len(pics), figsize=(3 * len(pics), 4))
        for a, (title, im) in zip(ax, pics, strict=True):
            a.imshow(im, cmap="gray")
            a.set_title(title)
            a.axis("off")
        plt.show()
    return table
