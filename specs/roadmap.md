# Roadmap

Seven steps in order, one pull request per step. A step is ticked only after Ryan has run it on the
real data and its numbers are in [RESULTS.md](../docs/RESULTS.md).

```
S1 Load BodyM → S2 Baseline → S3 Photo model → S4 Compare → S5 SMPL-X extras
→ Gate: ethics approval + AnthropometryDC export → S6 Ghanaian calibration → S7 Demo
```

## S1 Load and understand BodyM — done, 15 Nov 2026

- [x] Download to Drive once, restore per session (D-010)
- [x] Layout confirmed on real data: 3 splits, 3 tables each, front `mask/` + side `mask_left/` per photo pair
- [x] 2,018 / 87 / 400 people in train / Test-A / Test-B (2,505 total, as published)

## S2 Height + weight baseline — done, 15 Nov 2026

- [x] Scored per person (D-007): Test-A results in RESULTS.md
- [x] Test-B baseline added to RESULTS.md (`anthro baseline --data bodym --test testB`)

## S3 Photo model — first run done, 15 Nov 2026 (run r1)

- [x] `anthro train` / `anthro evaluate` built and tested on fake data
- [x] Ryan approved specs/s3-photo-model/spec.md (15 Nov 2026)
- [x] First full run on a Colab T4; per-person Test-A / Test-B tables in RESULTS.md
- [x] Epoch log added to RESULTS.md (best: epoch 25, val MAE 1.02 cm)
- [ ] Seeds 1 and 2 to measure run-to-run variation (about 7 min each)

## S3b Geometric girth + training fixes — done 17 Nov 2026 (final: r4; Test-A up, Test-B unchanged)

- [x] Ryan approved specs/s3b-geometric-girth/spec.md (D-017 to D-020), 17 Nov 2026
- [x] A1 loss in cm · A2 512×256 images (built, tested on fake data)
- [x] Geometry: width, depth, ellipse, levels, correction, fullness, flags (built)
- [x] E0–E2 on real BodyM (three fits; levels: chest at armpit, waist narrowest 36–49%, hip fixed 52%)
- [x] Hybrid model · retake flag · bootstrap confidence intervals · girths and lengths split (built)
- [x] E3a (r2), E3b (r3), E4 (r4) compared on validation; r4 chosen
- [x] Final Test-A / Test-B scored once on r4 (RESULTS.md)
- [ ] Proposed next: capture-variation augmentation + repeat seeds (needs a spec)
- Execution prompt: specs/s3b-geometric-girth/execute-prompt.md

## S3c Shape check (superellipse) — done 19 Nov 2026

- [x] Spec approved (specs/s3c-shape-check/spec.md, D-023)
- [x] `anthro geometry-shape`: k vs D/W and front/back; oval × k vs + front/back vs superellipse
- [x] Real run on BodyM (19 Nov): superellipse tested, no gain; oval × k stays (RESULTS.md)

## S4 Compare photo model vs baseline

- [x] Per measurement: MAE and % within ±1.5 cm, baseline vs photo, Test-A and Test-B
- [x] By sex and BMI band
- [ ] Lengths and girths reported separately
- [x] Consistency across a person's photos

## S7-0 Foundation for the full measurement set and our own data — built 25 Nov 2026, waiting for a Colab run

- [x] Spec approved (specs/s7-0-foundation/spec.md)
- [x] Catalogue from the app's codes; output format anthro-output/1; suppliers + routing; export loader
- [x] `anthro predict-silhouettes` and `anthro read-export`; notebooks/04_s7_predict.ipynb
- [ ] Ryan's Colab run of notebook 04 with r4

## Next after S7-0

- S7a predict pipeline: photo → silhouette → measurements (needs a segmentation model: new dependency) — built 25 Nov 2026, waiting for Ryan's test photos (notebook 07)
- S5a ANSUR II supplier (D-024; licence checked 25 Nov 2026) — done 25 Nov 2026 (formulas packaged; mix not kept; F1 failed its check; F2 Black-fitted inseam and hip in use, D-032)
- S5b rough estimates for under-bust, front length, high hip, knee (ANSUR first, then outline geometry) — built 25 Nov 2026, waiting for Ryan's Colab run (notebook 06)
- S8 train on the app's data (missing values allowed, both weights, clothing)

## S5 SMPL-X extras — set aside (D-026, 24 Nov 2026)

- [ ] Fit SMPL-X shape to the predicted (or corrected) measurements + height + weight
- [ ] Virtual tape for neck, armhole, back length, rise; inconsistency flag when no plausible body fits
- [ ] Ryan registers for SMPL-X; model files stay on his Drive (D-014)

## Gate — before S6

- [ ] Ethics approval for the Ghanaian sample
- [ ] AnthropometryDC collects photos + tape for 40+ adults in fitted clothing; Ryan exports a de-identified file
- [ ] Map AnthropometryDC's ISO 8559-1 measurement codes to BodyM's 14 (definitions differ: tape vs scan)

## S6 Ghanaian calibration

- [ ] Per-measurement offset learned with leave-one-out cross-validation
- [ ] Before/after per person; % within ±1.5 cm; two-measurer disagreement as the floor

- S7c serve the model as a web API (free host), laptop and phone options weighed — built 26 Nov 2026 (API, Docker, laptop route); host Oracle Always Free (D-048), deploy guide written; waiting for Ryan's notebook 08 run and server setup
- S9 offline on the phone (later): convert and shrink the models, phone-sized segmentation

## S7 Demo

- [ ] Upload front + side photo, enter height (weight optional), get measurements with "confirm with tape" flags
- [ ] Accessibility checks in docs/ACCESSIBILITY.md pass
