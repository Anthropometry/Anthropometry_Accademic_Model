"""Command line: the same three commands work on a laptop, in Colab and in CI.

    anthro baseline --data bodym --test testA
    anthro train    --data bodym --out runs/r1 [--epochs 25] [--cache DIR]
    anthro evaluate --data bodym --ckpt runs/r1/best.pt [--test testA testB]

Why a CLI rather than logic in notebooks: notebooks are hard to review in a pull request and
easy to run out of order. Notebooks here only call these commands, so what the supervisor
reads in the repo is exactly what ran.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from . import baseline, checkpoint
from . import train as train_mod
from .config import TrainConfig
from .data import fingerprint, load_all, people
from .dataset import SilhouetteSet, cache_images
from .evaluate import by_group, consistency, per_person_errors, summarise


def _baseline(a) -> None:
    splits = load_all(a.data)
    for t in a.test:
        print(f"\n== height + weight + sex baseline on {t}: {len(people(splits[t]))} people ==")
        print(baseline.run(splits["train"], splits[t]).to_string())


def _train(a) -> None:
    cfg = TrainConfig(epochs=a.epochs, batch_size=a.batch_size, img_h=a.img_h, img_w=a.img_w,
                      pretrained=not a.no_pretrained, seed=a.seed, num_workers=a.workers)
    path = train_mod.run(a.data, a.out, cfg, cache_dir=a.cache)
    print(f"\nbest model: {path}")


def _evaluate(a) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, cfg, xs, ys, ck = checkpoint.load(a.ckpt, device)
    if ck["data_fingerprint"] != fingerprint(a.data):
        print("WARNING: data differs from the data this model was trained on")
    splits = load_all(a.data)
    report = {}
    for t in a.test:
        df = splits[t]
        imgs = cache_images(df, a.data, a.cache or Path(a.ckpt).parent / "cache", t,
                            cfg.img_h, cfg.img_w, ck["data_fingerprint"])
        loader = DataLoader(SilhouetteSet(imgs, df, xs, ys), batch_size=cfg.batch_size)
        pred = train_mod.predict(net, loader, ys, device)
        err = per_person_errors(df, pred)
        photo = summarise(err)
        base = baseline.run(splits["train"], df)
        table = base.add_prefix("baseline_").join(photo.add_prefix("photo_"))
        print(f"\n== {t}: {len(err)} people (per person, averaged over their photos) ==")
        print(table.to_string())
        spread = consistency(df, pred)
        if spread.isna().all():
            print("\nconsistency: not measurable — nobody in this split has 2+ photo pairs")
        else:
            print("\nconsistency: average spread (cm) of one person's predictions across photos")
            print(spread.to_string())
        print("\n% within tolerance by sex and BMI band")
        print(by_group(df, err).to_string())
        report[t] = table.reset_index().to_dict(orient="records")
    out = Path(a.ckpt).parent / "eval_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nsaved {out}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="anthro")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("baseline", help="height + weight + sex linear baseline")
    b.add_argument("--data", required=True)
    b.add_argument("--test", nargs="+", default=["testA", "testB"])
    b.set_defaults(fn=_baseline)

    t = sub.add_parser("train", help="train the BMnet-style model")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--cache", default=None, help="where prepared images are cached (e.g. on Drive)")
    t.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    t.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    t.add_argument("--img-h", type=int, default=TrainConfig.img_h)
    t.add_argument("--img-w", type=int, default=TrainConfig.img_w)
    t.add_argument("--seed", type=int, default=TrainConfig.seed)
    t.add_argument("--workers", type=int, default=TrainConfig.num_workers)
    t.add_argument("--no-pretrained", action="store_true", help="for tests and offline runs")
    t.set_defaults(fn=_train)

    e = sub.add_parser("evaluate", help="score a trained model per person against the baseline")
    e.add_argument("--data", required=True)
    e.add_argument("--ckpt", required=True)
    e.add_argument("--cache", default=None)
    e.add_argument("--test", nargs="+", default=["testA", "testB"])
    e.set_defaults(fn=_evaluate)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
