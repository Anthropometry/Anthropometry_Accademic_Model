"""Command line: the same three commands work on a laptop, in Colab and in CI.

    anthro baseline --data bodym --test testA
    anthro train    --data bodym --out runs/r1 [--epochs 25] [--cache DIR]
    anthro evaluate --data bodym --ckpt runs/r1/best.pt [--test testA testB]
    anthro geometry     --data bodym --out GEO_CACHE                       (S3b, E0)
    anthro geometry-fit --data bodym --cache GEO_CACHE --out geo_fit.json  (S3b, E1-E2)
    anthro train ... --loss cm [--geo-fit geo_fit.json --geo-cache GEO_CACHE]
    anthro evaluate ... [--geo-cache GEO_CACHE] [--bootstrap 1000] [--retake-share 0.10]
    anthro geometry-shape --data bodym --cache GEO_CACHE --fit geo_fit.json     (S3c shape check)
    anthro predict-silhouettes --front F --side S --height H [--weight W] --sex female
                               --age-range 25-34 --ckpt runs/r4/best.pt [--out result.json]  (S7-0)
    anthro read-export --folder anthro-export-YYYY-MM-DD-HH-MM [--include-practice]     (S7-0)
    anthro ansur-fit --ansur DRIVE/ansur2 --report runs/r4/eval_report.json --out ansur_formulas.json (S5a)
    anthro ansur-mix --data bodym --ckpt runs/r4/best.pt --formulas ansur_formulas.json             (S5a)

Why a CLI rather than logic in notebooks: notebooks are hard to review in a pull request and
easy to run out of order. Notebooks here only call these commands, so what the supervisor
reads in the repo is exactly what ran.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import baseline, catalog, checkpoint, geometry_shape
from . import geometry_fit as gfit
from . import train as train_mod
from .config import GIRTHS, LENGTHS, TrainConfig
from .data import fingerprint, load_all, people, split_by_person
from .dataset import SilhouetteSet, cache_images
from .evaluate import (
    bootstrap_ci,
    by_group,
    consistency,
    disagreement,
    per_person_errors,
    retake_report,
    retake_threshold,
    summarise,
)


def _baseline(a) -> None:
    splits = load_all(a.data)
    for t in a.test:
        print(f"\n== height + weight + sex baseline on {t}: {len(people(splits[t]))} people ==")
        print(baseline.run(splits["train"], splits[t]).to_string())


def _train(a) -> None:
    cfg = TrainConfig(epochs=a.epochs, batch_size=a.batch_size, img_h=a.img_h, img_w=a.img_w,
                      pretrained=not a.no_pretrained, seed=a.seed, num_workers=a.workers, loss=a.loss,
                      val_fraction=a.val_fraction)
    geo_fit = json.loads(Path(a.geo_fit).read_text()) if a.geo_fit else None
    path = train_mod.run(a.data, a.out, cfg, cache_dir=a.cache, geo_fit=geo_fit, geo_cache=a.geo_cache)
    print(f"\nbest model: {path}")


def _geometry(a) -> None:
    splits = load_all(a.data)
    fp = fingerprint(a.data)
    for t in a.split:
        c = gfit.build_cache(splits[t], a.data, a.out, t, fp)
        print(f"\n== E0 geometry diagnostics: {t} ==")
        print(gfit.diagnostics(c).to_string())


def _geometry_fit(a) -> None:
    splits = load_all(a.data)
    fp = fingerprint(a.data)
    c = gfit.build_cache(splits["train"], a.data, a.cache, "train", fp)
    fit_ = gfit.fit(splits["train"], c, seed=a.seed, val_fraction=a.val_fraction)
    fit_["data_fingerprint"] = fp
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(gfit.to_plain(fit_), indent=2))
    print(f"\nsaved {a.out}")


def _geometry_shape(a) -> None:
    splits = load_all(a.data)
    fp = fingerprint(a.data)
    fit_ = json.loads(Path(a.fit).read_text())
    if fit_.get("data_fingerprint") not in (None, fp):
        raise SystemExit("this geometry fit was made on different data: rerun `anthro geometry-fit`")
    c = gfit.build_cache(splits["train"], a.data, a.cache, "train", fp)
    report = geometry_shape.run(splits["train"], c, fit_)
    out = Path(a.fit).with_name("geo_shape_report.json")
    out.write_text(json.dumps(gfit.to_plain(report), indent=2))
    print(f"\nsaved {out}")


def _print_split(table) -> None:
    print("girths (wrap around the body):")
    print(table.loc[[m for m in GIRTHS if m in table.index]].to_string())
    print("lengths (run along the body):")
    print(table.loc[[m for m in LENGTHS if m in table.index]].to_string())


def _evaluate(a) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, cfg, xs, ys, ck = checkpoint.load(a.ckpt, device)
    fp = ck["data_fingerprint"]
    if fp != fingerprint(a.data):
        print("WARNING: data differs from the data this model was trained on")
    splits = load_all(a.data)
    cache = a.cache or Path(a.ckpt).parent / "cache"
    geo_fit = ck.get("geo_fit")
    geo_cache = a.geo_cache or cache

    def predict(df, split, name, rows_of=None):
        imgs = cache_images(df, a.data, cache, name, cfg.img_h, cfg.img_w, fp)
        geo, gtab = None, None
        if geo_fit is not None:
            geo, gtab = gfit.features_for(rows_of if rows_of is not None else df, df, a.data,
                                          geo_cache, split, fp, geo_fit)
        loader = DataLoader(SilhouetteSet(imgs, df, xs, ys, geo=geo), batch_size=cfg.batch_size)
        return train_mod.predict(net, loader, ys, device), gtab

    threshold = None
    if geo_fit is not None and a.retake_share > 0:
        _, val_df0 = split_by_person(splits["train"], cfg.val_fraction, cfg.seed)
        tag0 = f"s{cfg.seed}v{cfg.val_fraction:g}"
        vpred, vtab = predict(val_df0, "train", f"val_{tag0}", rows_of=splits["train"])
        threshold = retake_threshold(disagreement(val_df0, vtab, vpred), a.retake_share)
        print(f"retake warning: fires when geometry and network differ by more than {threshold:.2f} cm "
              f"(chosen on validation people to fire for {a.retake_share:.0%})")

    report = {}
    fit_df, val_df = split_by_person(splits["train"], cfg.val_fraction, cfg.seed)
    tag = f"s{cfg.seed}v{cfg.val_fraction:g}"
    for t in a.test:
        if t == "val":
            # The validation people this model chose its best epoch on: the fair place to compare
            # runs (S3b), since Test-A/B stay untouched until the final choice. The baseline is
            # refitted without them so it doesn't see their answers either.
            df, base_train = val_df, fit_df
            pred, gtab = predict(df, "train", f"val_{tag}", rows_of=splits["train"])
        else:
            df, base_train = splits[t], splits["train"]
            pred, gtab = predict(df, t, t)
        err = per_person_errors(df, pred)
        photo = summarise(err)
        base = baseline.run(base_train, df)
        table = base.add_prefix("baseline_").join(photo.add_prefix("photo_"))
        if a.bootstrap:
            table = table.join(bootstrap_ci(err, a.bootstrap).add_prefix("photo_"))
        print(f"\n== {t}: {len(err)} people (per person, averaged over their photos) ==")
        _print_split(table)
        spread = consistency(df, pred)
        if spread.isna().all():
            print("\nconsistency: not measurable — nobody in this split has 2+ photo pairs")
        else:
            print("\nconsistency: average spread (cm) of one person's predictions across photos")
            print(spread.to_string())
        print("\n% within tolerance by sex and BMI band")
        print(by_group(df, err).to_string())
        entry = {"table": table.reset_index().to_dict(orient="records")}
        if threshold is not None:
            dis = disagreement(df, gtab, pred).loc[err.index]
            flagged = dis > threshold
            rr = retake_report(err, flagged)
            print(f"\nretake warning on {t}: {flagged.mean():.0%} of people flagged")
            print(rr.to_string())
            print("\n% within tolerance by sex and BMI band, people NOT flagged")
            print(by_group(df, err.loc[~flagged]).to_string())
            entry["retake"] = {"threshold_cm": threshold, "flagged_share": float(flagged.mean()),
                               "report": rr.reset_index().to_dict(orient="records")}
        report[t] = entry
    out = Path(a.ckpt).parent / ("eval_report_val.json" if a.test == ["val"] else "eval_report.json")
    out.write_text(json.dumps(gfit.to_plain(report), indent=2))
    print(f"\nsaved {out}")


def _predict_silhouettes(a) -> None:
    from . import predict
    try:
        doc = predict.predict(a.front, a.side, a.height, a.sex, a.ckpt, weight_kg=a.weight,
                              age_range=a.age_range, height_kind=a.height_kind, clothing=a.clothing,
                              data_root=a.data, report_path=a.report, routing_path=a.routing,
                              weight_kind=a.weight_kind, ansur_path=a.ansur)
    except (predict.PredictError, ValueError) as e:
        raise SystemExit(f"refused: {e}") from e
    print(predict.to_text(doc))
    if a.out:
        Path(a.out).write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
        print(f"\nsaved {a.out}")


def _ansur_fit(a) -> None:
    from . import ansur, suppliers
    df = ansur.load(a.ansur)
    report = json.loads(Path(a.report).read_text())
    errors, basis = suppliers.errors_from_report(report, "testB")
    noise = {n: errors[n] for n in ansur.PHOTO_INPUTS if n in errors}
    doc, scores = ansur.fit(df, noise, catalog.version(), ansur.fingerprint(a.ansur), basis)
    print(f"ANSUR II: {int((df.sex == 'female').sum())} women, {int((df.sex == 'male').sum())} men")
    blur = ", ".join(f"{k} {v:.2f}" for k, v in noise.items())
    print(f"photo-model blur (Test-B per-person MAE, cm): {blur}")
    print("\n== held-out error per formula (5 folds by person); 'chosen' = used in the output ==")
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(scores.to_string(index=False))
        print("\n== chosen formula by group: BMI bands, and Black participants separately (D-030) ==")
        print(ansur.subgroup_table(df, doc, noise).to_string(index=False))
        print("\n== cost of a stated height and weight (basic formula, ANSUR's self-reported values) ==")
        print(ansur.self_report_check(df).to_string(index=False))
    Path(a.out).write_text(json.dumps(doc, indent=2) + "\n")
    print(f"\nsaved {a.out} (coefficients and summary errors only)")


def _split_predictions(a, bundle, df, split, name, rows_of=None):
    """Photo-model predictions for a split, exactly as `anthro evaluate` makes them."""
    net, cfg, xs, ys, ck = bundle
    fp, geo_fit = ck["data_fingerprint"], ck.get("geo_fit")
    cache = a.cache or Path(a.ckpt).parent / "cache"
    imgs = cache_images(df, a.data, cache, name, cfg.img_h, cfg.img_w, fp)
    geo = None
    if geo_fit is not None:
        geo, _ = gfit.features_for(rows_of if rows_of is not None else df, df, a.data, a.geo_cache or cache,
                                   split, fp, geo_fit)
    loader = DataLoader(SilhouetteSet(imgs, df, xs, ys, geo=geo), batch_size=cfg.batch_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return train_mod.predict(net.to(device), loader, ys, device)


def _ansur_mix(a) -> None:
    from . import ansur
    doc = ansur.load_formulas(a.formulas)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = checkpoint.load(a.ckpt, device)
    cfg = bundle[1]
    splits = load_all(a.data)
    _, val_df = split_by_person(splits["train"], cfg.val_fraction, cfg.seed)
    tag = f"s{cfg.seed}v{cfg.val_fraction:g}"
    val = ansur.mix_table(val_df, _split_predictions(a, bundle, val_df, "train", f"val_{tag}",
                                                     rows_of=splits["train"]), doc)
    mix = ansur.mix_decide(val)
    print(f"== mix test on {len(val)} validation people (5 folds by person) ==")
    for name, m in mix.items():
        v = m["validation"]
        print(f"{name:6s} photo model: MAE {v['photo_model']['mae_cm']:.2f} cm, "
              f"{v['photo_model']['pct_within']:.1f}% "
              f"within ±1.5 | mix: MAE {v['mix']['mae_cm']:.2f} cm, {v['mix']['pct_within']:.1f}% "
              f"-> {'KEPT' if m['kept'] else 'not kept (photo model stays)'}")
    if any(m["kept"] for m in mix.values()):
        tb = splits["testB"]
        test = ansur.mix_table(tb, _split_predictions(a, bundle, tb, "testB", "testB"), doc)
        scored = ansur.mix_score(test, mix)
        print(f"\n== kept mixes, scored once on Test-B ({len(test)} people) ==")
        for name, s in scored.items():
            mix[name]["testB"] = s
            print(f"{name:6s} photo model: MAE {s['photo_model']['mae_cm']:.2f} cm, "
                  f"{s['photo_model']['pct_within']:.1f}% | mix: MAE {s['mix']['mae_cm']:.2f} cm, "
                  f"{s['mix']['pct_within']:.1f}%")
    doc["mix"] = mix
    Path(a.formulas).write_text(json.dumps(doc, indent=2) + "\n")
    print(f"\nsaved the mix decision into {a.formulas}")


def _read_export(a) -> None:
    from . import export_loader
    try:
        ex = export_loader.load(a.folder, include_practice=a.include_practice)
    except export_loader.ExportError as e:
        raise SystemExit(f"refused: {e}") from e
    codes = [c for c in catalog.by_code() if c in ex.people.columns]
    print(f"format {ex.format} | {len(ex.people)} participants | {len(ex.photos)} photos")
    for n in ex.warnings:
        print(f"note: {n}")
    # Counts only: no values or ids are printed, so logs and saved notebook outputs stay anonymous.
    filled = ex.people[codes].notna().sum()
    print("measurements filled (participants):")
    print(filled.to_string())


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
    t.add_argument("--val-fraction", type=float, default=TrainConfig.val_fraction)
    t.add_argument("--workers", type=int, default=TrainConfig.num_workers)
    t.add_argument("--no-pretrained", action="store_true", help="for tests and offline runs")
    t.add_argument("--loss", choices=["std", "cm"], default=TrainConfig.loss,
                   help="std = run r1; cm = error in cm (S3b, D-017)")
    t.add_argument("--geo-fit", default=None, help="geometry fit JSON from `anthro geometry-fit` (hybrid)")
    t.add_argument("--geo-cache", default=None, help="where geometry caches live (default: --cache)")
    t.set_defaults(fn=_train)

    g = sub.add_parser("geometry", help="S3b E0: measure width and depth on every photo pair")
    g.add_argument("--data", required=True)
    g.add_argument("--out", required=True, help="cache folder (e.g. on Drive)")
    g.add_argument("--split", nargs="+", default=["train", "testA", "testB"])
    g.set_defaults(fn=_geometry)

    gf = sub.add_parser("geometry-fit", help="S3b E1-E2: tune levels and fit the ellipse correction")
    gf.add_argument("--data", required=True)
    gf.add_argument("--cache", required=True)
    gf.add_argument("--out", required=True, help="where to write the fit JSON")
    gf.add_argument("--seed", type=int, default=TrainConfig.seed)
    gf.add_argument("--val-fraction", type=float, default=TrainConfig.val_fraction)
    gf.set_defaults(fn=_geometry_fit)

    gs = sub.add_parser("geometry-shape", help="S3c: does a superellipse or front/back term beat the oval?")
    gs.add_argument("--data", required=True)
    gs.add_argument("--cache", required=True, help="geometry cache folder from `anthro geometry`")
    gs.add_argument("--fit", required=True, help="geo_fit.json from `anthro geometry-fit`")
    gs.set_defaults(fn=_geometry_shape)

    e = sub.add_parser("evaluate", help="score a trained model per person against the baseline")
    e.add_argument("--data", required=True)
    e.add_argument("--ckpt", required=True)
    e.add_argument("--cache", default=None)
    e.add_argument("--test", nargs="+", default=["testA", "testB"],
                   help="testA, testB, or val (the model's own validation people, for comparing runs)")
    e.add_argument("--geo-cache", default=None)
    e.add_argument("--bootstrap", type=int, default=0, help="95%% confidence intervals from N resamples")
    e.add_argument("--retake-share", type=float, default=0.10,
                   help="hybrid only: share of validation people the retake warning fires for (D-020)")
    e.set_defaults(fn=_evaluate)

    pr = sub.add_parser("predict-silhouettes",
                        help="S7-0: one adult's full measurement set from two silhouettes")
    pr.add_argument("--front", required=True, help="front silhouette (white on black), as BodyM mask")
    pr.add_argument("--side", required=True, help="side silhouette, like BodyM's mask_left")
    pr.add_argument("--height", type=float, required=True, help="cm")
    pr.add_argument("--height-kind", choices=("measured", "stated"), default="measured")
    pr.add_argument("--weight", type=float, default=None, help="kg; left out: typical weight (D-027)")
    pr.add_argument("--weight-kind", choices=("scale", "stated"), default="stated")
    pr.add_argument("--sex", choices=("female", "male"), required=True)
    pr.add_argument("--age-range", default=None, help="18-24 ... 65+ (adults only)")
    pr.add_argument("--clothing", default=None)
    pr.add_argument("--ckpt", required=True)
    pr.add_argument("--report", default=None, help="default: eval_report.json beside the checkpoint")
    pr.add_argument("--routing", default=None, help="default: packaged routing_default.json")
    pr.add_argument("--data", default=None, help="BodyM: only for a typical weight with r1-r4")
    pr.add_argument("--out", default=None, help="save the anthro-output/1 JSON here")
    pr.add_argument("--ansur", default=None, help="S5a formula file (default: the packaged one, if any)")
    pr.set_defaults(fn=_predict_silhouettes)

    af = sub.add_parser("ansur-fit", help="S5a: fit ANSUR II formulas for the codes BodyM lacks")
    af.add_argument("--ansur", required=True, help="folder with the two ANSUR II public CSV files (on Drive)")
    af.add_argument("--report", required=True, help="evaluation report with a testB section (blur size)")
    af.add_argument("--out", required=True)
    af.set_defaults(fn=_ansur_fit)

    am = sub.add_parser("ansur-mix", help="S5a: does mixing ANSUR into chest/waist beat the photo model?")
    am.add_argument("--data", required=True)
    am.add_argument("--ckpt", required=True)
    am.add_argument("--formulas", required=True, help="ansur_formulas.json from ansur-fit (updated in place)")
    am.add_argument("--cache", default=None)
    am.add_argument("--geo-cache", default=None)
    am.set_defaults(fn=_ansur_mix)

    rx = sub.add_parser("read-export", help="S7-0: check and summarise an AnthropometryDC export folder")
    rx.add_argument("--folder", required=True)
    rx.add_argument("--include-practice", action="store_true")
    rx.set_defaults(fn=_read_export)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
