# Results

Only numbers from Ryan's real runs go here, each with its date, data fingerprint and commit.

## S3c — Shape check: superellipse vs oval correction (19 Nov 2026, commit 34dec41)

1,816 training people (validation excluded), 5-fold CV grouped by person, per person.

### Does the oval's error k follow the body's shape?

| Level | corr(k, D/W) | k by D/W, lowest → highest fifth | corr(k, front/back) | k by front/back, lowest → highest fifth | Boxiness n matching each person: median (10th–90th pct) | n at the 1.5/8 limits |
| --- | ---: | --- | ---: | --- | --- | ---: |
| chest | +0.25 | 1.135 · 1.146 · 1.146 · 1.137 · 1.116 | −0.13 | 1.160 · 1.139 · 1.133 · 1.125 · 1.123 | 5.0 (2.5–8.0) | 23.3% |
| waist | +0.08 | 1.116 · 1.110 · 1.104 · 1.091 · 1.045 | −0.23 | 1.116 · 1.102 · 1.097 · 1.078 · 1.072 | 3.6 (2.2–5.8) | 5.4% |
| hip | −0.30 | 1.099 · 1.085 · 1.077 · 1.070 · 1.037 | +0.14 | 1.060 · 1.074 · 1.075 · 1.077 · 1.081 | 3.1 (2.1–4.9) | 3.6% |

### Three corrections compared (CV MAE cm / % within ±1.5 cm)

| Level | A oval × k (current) | B + front/back | C superellipse (n learned) | Baseline h+w+sex |
| --- | --- | --- | --- | --- |
| chest | **3.37** / 28.6 | 3.38 / 27.9 | 3.41 / 28.4 | 3.03 / 32.3 |
| waist | **2.67** / 37.6 | 2.67 / 37.7 | 2.73 / 38.8 | 3.51 / 28.4 |
| hip | 2.84 / 36.7 | **2.76** / 37.3 | 2.90 / 34.8 | 2.23 / 42.7 |

Reading:
- **Real cross-sections are boxier than ovals:** the boxiness that matches each person is about
  3 (hip), 3.6 (waist) and 5 (chest), against 2 for an oval. The chest often reaches the limit
  (23%): its outline (hollows, shoulder blades) is not well described by any superellipse.
- **k does follow shape, weakly** (correlations 0.08–0.30), but the current correction already
  uses D/W, so modelling the shape explicitly adds nothing: no variant beats A by the 0.1 cm the
  spec required (best: hip with the front/back term, −0.08 cm).
- **The boxiness varies from person to person, and the features cannot predict it**, which is
  why the superellipse (C) is no better than a single learned factor.
- Larger bodies are measured worst by every variant (BMI 35+: chest 5.1 cm, waist 4.1 cm).
- **Decision (per spec): superellipse tested, no gain.** The current oval × k stays.

## S3b — r4 retrained with the same settings, test scored a second time (19 Nov 2026)

Notebook 03 was rerun, which retrained r4 with identical settings (seed 0) and overwrote the first
r4. GPU training is not exactly repeatable, so this is a second draw of the same recipe. Recorded
as a measure of run-to-run variation, **not** as a second candidate (no choice is made from it).

| % within ±1.5 cm | r1 | r4 first run (17 Nov) | r4 second run (19 Nov) |
| --- | ---: | ---: | ---: |
| Test-A waist | 63.2 | 70.1 | 58.6 |
| Test-A hip | 72.4 | 63.2 | 73.6 |
| Test-A chest | 56.3 | 57.5 | 52.9 |
| Test-A mean of 14 | 83.5 | 84.0 | 83.3 |
| Test-A mean MAE (cm) | 0.88 | 0.84 | 0.89 |
| Test-B mean of 14 | 74.5 | 74.2 | 73.0 |
| Test-B mean MAE (cm) | 1.22 | 1.22 | 1.27 |
| Retake: flagged A / B | — | 8% / 32% | 9% / 33% |
| Retake: all-14 % not flagged vs flagged, Test-B | — | 76.6 vs 69.1 | 74.4 vs 70.0 |
| Test-B women BMI 35+ (23 people) | 52.8 | 53.7 | 53.4 |

Reading:
- **Same recipe, two runs: waist 70 vs 59, hip 63 vs 74 on Test-A.** Single-measurement changes of
  ~10 points between runs are chance, so the earlier "waist 63 → 70" gain is **not** an
  improvement over r1. On average, r4 performs like r1 on both test sets.
- **What repeats:** the retake warning (fires ~9% A, ~33% B; flagged people measured worse), the
  BMI 35+ gap, and the overall level (~83% A, ~73–74% B).
- Consequence: report averages over several runs (seeds) before claiming any gain.

## S3b — Final test, scored once: run r4 on Test-A and Test-B (17 Nov 2026)

Run r4 (cm loss, 512×256 images, 25 geometric features), chosen on validation before this run.
95% confidence intervals from 1,000 resamples of people.

### % of people within ±1.5 cm (headline) and MAE

| Measurement | Test-A r1 | **Test-A r4** (95% CI) | r4 MAE A | Test-B r1 | **Test-B r4** (95% CI) | r4 MAE B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wrist | 100.0 | 100.0 (100–100) | 0.43 | 97.3 | 97.3 (95.5–98.8) | 0.52 |
| ankle | 94.3 | 97.7 (94.3–100) | 0.59 | 89.0 | 90.5 (87.8–93.2) | 0.72 |
| forearm | 96.6 | 97.7 (93.1–100) | 0.47 | 90.0 | 91.0 (87.8–93.8) | 0.67 |
| shoulder-breadth | 98.9 | 97.7 (94.3–100) | 0.48 | 91.8 | 91.8 (89.0–94.2) | 0.69 |
| bicep | 94.3 | 94.3 (88.5–98.9) | 0.67 | 82.0 | 82.0 (78.2–85.8) | 0.90 |
| arm-length | 89.7 | 94.3 (88.5–98.9) | 0.64 | 88.0 | 86.0 (82.8–89.2) | 0.83 |
| calf | 88.5 | 85.1 (77.0–92.0) | 0.73 | 79.5 | 80.0 (76.0–84.0) | 0.98 |
| height (scan) | 83.9 | 83.9 (75.9–90.8) | 0.94 | 72.5 | 70.3 (65.8–74.8) | 1.46 |
| shoulder-to-crotch | 81.6 | 82.8 (74.7–89.7) | 0.85 | 79.0 | 75.5 (71.5–79.5) | 1.11 |
| thigh | 72.4 | 78.2 (69.0–86.2) | 0.98 | 70.0 | 68.5 (63.7–73.0) | 1.23 |
| leg-length | 77.0 | 73.6 (64.4–82.8) | 1.10 | 67.0 | 70.5 (66.0–74.8) | 1.36 |
| waist | 63.2 | **70.1** (59.8–79.3) | 1.26 | 41.3 | 42.5 (38.0–47.2) | 2.34 |
| hip | 72.4 | 63.2 (52.9–72.4) | 1.17 | 53.8 | 50.8 (45.5–55.2) | 2.00 |
| chest | 56.3 | 57.5 (47.1–67.8) | **1.42** | 41.3 | 42.0 (37.5–46.8) | 2.32 |
| **mean of 14** | 83.5 | **84.0** | **0.84** | 74.5 | 74.2 | 1.22 |

r1 MAE for comparison: Test-A mean 0.88 (chest 1.67, waist 1.33, hip 1.12); Test-B mean 1.22. Published
BMnet, Test-A mean 0.92 (scoring protocol may differ).

### Consistency (spread of one person's predictions across photos, cm), r1 → r4

| | chest | waist | hip |
| --- | --- | --- | --- |
| Test-A | 1.25 → 1.21 | 1.34 → 1.33 | 0.94 → 1.03 |
| Test-B | 1.54 → **1.82** | 1.89 → **2.11** | 1.19 → **1.72** |

### Retake warning (E5): threshold 7.80 cm, set on validation people to fire for 10%

| | Flagged | Not flagged: all-14 % | Flagged: all-14 % | Not flagged: waist % | Flagged: waist % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Test-A | 8% (7 people) | 84.6 | 76.5 | 70.0 | 71.4 |
| Test-B | **32%** (128 people) | 76.6 | 69.1 | 47.4 | 32.0 |

### Sex × BMI (Test-B, all 14 pooled)

Women BMI 35+: 53.7% (23 people); men BMI 35+: 57.1% (8); BMI 18.5–25: 76–78%. Unchanged since r1.

### Reading

- **Test-A improved; Test-B did not.** Validation (same capture sessions as training) rose
  79.8 → 82.1% and Test-A 83.5 → 84.0% (waist 63 → 70%, chest MAE 1.67 → 1.42 cm), but Test-B is
  unchanged (74.5 → 74.2%, MAE 1.22 both). With 400 people the CIs are tight: a real null result.
- **Hip got worse on both** (A 72 → 63%, B 54 → 51%). Likely cause: the geometry places the hip at a
  fixed 52% of body height, which moves with camera tilt and framing.
- **r4 is less stable on Test-B photos** (spread up 0.2–0.5 cm). Sharper images and explicit widths read
  the outline in more detail, and so react more to capture differences. The research brief described
  Test-B as captured with more varied camera orientations and lighting, and predicted geometric
  features would be noisiest there.
- **The retake warning works as a photo-quality detector.** It fires 3× its target rate on Test-B and
  the people it flags are measured worse (69 vs 77% overall; waist 32 vs 47%).
- **Implication for the system:** under controlled capture (Test-A-like), the model reaches 84% of
  people within ±1.5 cm overall and 70% on the waist. The app's capture protocol (level phone, fixed
  distance, A-pose, fitted clothing) plus the retake warning is as important as the model.
- **Next lever (proposed, not run):** train with simulated capture variation (small tilts, shifts,
  perspective) so the model ignores it; repeat seeds to measure run-to-run variation.
- **Fairness gap persists:** BMI 35+ about 54–57% vs 76–78% at BMI 18.5–25 on Test-B.

## S3b — Geometry on real BodyM (E0–E2, 17 Nov 2026, commit 2e72aa9)

### E0: can full-resolution geometry work on BodyM? Yes.

| | train | Test-A | Test-B |
| --- | ---: | ---: | ---: |
| Photo pairs measured | 6,134 (100%) | 1,684 (100%) | 1,160 (100%) |
| Mask size | 960×720 | 960×720 | 960×720 |
| Body height in pixels (median / 5th pct) | 832 / 772 | 836 / 792 | 847 / 804 |
| mm per pixel (170 cm person) | 2.0 | 2.0 | 2.0 |
| All three levels measurable | 97.7% | 94.0% | 97.1% |
| Arm touching at chest / waist / hip | 1.3 / 0.1 / 2.3% | 1.4 / 0.2 / 6.8% | 1.2 / 0.0 / 2.3% |
| Edge softness (median) | 0 (hard binary masks) | 0 | 0 |

Reading: 2 mm per pixel meets the ≤ 2 mm the research brief recommends (the network's 256 px images
are 7 mm). Arms rarely touch the body. Masks are hard 0/255, so there is no sub-pixel edge detail.

### E1 + E2, first fit (search grid too narrow; rerun pending)

Fitted on 1,816 training people (validation people excluded), 5-fold CV grouped by person.

**The ellipse under-reads every girth (k = true ÷ ellipse > 1):**

| Level | k, slim (BMI 18.5–25) | k, heavy (BMI 35+) | Ellipse alone MAE | Ellipse alone % ±1.5 |
| --- | ---: | ---: | ---: | ---: |
| chest | 1.24 | 1.04–1.06 | 16.27 cm | 1.1% |
| waist | 1.08–1.11 | 1.08–1.12 | 7.81 cm | 5.4% |
| hip | 1.05 | 1.06–1.07 | 6.00 cm | 10.7% |

**Corrected geometry (k fitted) vs height + weight + sex baseline (CV, per person):**

| Level | Geometry MAE | Geometry % ±1.5 | Baseline MAE | Baseline % ±1.5 |
| --- | ---: | ---: | ---: | ---: |
| chest | 4.22 | 23.0 | 3.03 | 32.3 |
| waist | **2.71** | **38.4** | 3.51 | 28.4 |
| hip | 3.36 | 31.7 | 2.23 | 42.7 |

Reading:
- Torsos are boxier than ellipses: the true girth is 5–11% larger at waist and hip, as the
  superellipse discussion predicted.
- The chest is far from an ellipse in slim people (24% under-read), less so in heavy people. The
  chest profile has hollows (between and under the breasts, the spine) and a slim chest is flatter.
- Geometry alone beats the baseline **only for the waist**. For chest and hip it is worse. This
  matches the research: geometry is a feature for the network, not a replacement for it.
- **Flaw found:** all three chosen levels were at the edge of the search grid, so the best levels
  were outside the grid. The grid has been widened and `geometry-fit` now warns on an edge choice;
  E1/E2 must be rerun before E4.
- Second fit (wider grid): chest offset 0.08 (now inside the grid), waist band [0.39, 0.49], hip
  [0.50, 0.53], still at an edge. E2 MAE: chest 4.21, waist 3.10, hip 3.04 cm. **Second flaw:**
  levels were scored with a straight line on C0, not the correction used afterwards, so waist got
  worse after correction (2.71 → 3.10 cm) though the ellipse alone improved (7.81 → 6.14 cm). Fixed:
  levels are now scored with the full correction, and waist and hip also try fixed heights.

### E1 + E2, third fit (used for E4), 17 Nov 2026, commit 9f8e977

Chosen levels: **chest at the armpit** (offset 0.0; the edge is physical: above the armpit the arms
join the torso in the front photo), **waist = narrowest point between 36% and 49%** of body height,
**hip = fixed height at 52%** (not the deepest point: BodyM's hip is a fixed SMPL loop).

| Level | k range (true ÷ ellipse) | Ellipse alone MAE | Corrected MAE | Corrected % ±1.5 | Baseline MAE | Baseline % ±1.5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| chest | 1.09–1.16 | 11.54 | 3.37 | 28.6 | 3.03 | 32.3 |
| waist | 1.05–1.12 | 7.80 | **2.67** | **37.6** | 3.51 | 28.4 |
| hip | 1.05–1.13 | 7.24 | 2.84 | 36.7 | 2.23 | 42.7 |

Reading:
- With the chest at the armpit, k no longer depends on body size (1.24 slim vs 1.04 heavy before),
  so the earlier chest level was wrong for slim bodies.
- Geometry alone beats height + weight + sex on the waist, nearly ties on the chest, loses on the
  hip. Its value is tested in E4 (does it help the network?).
- For S6: BodyM's hip is at a fixed height, not the fullest point. A Ghanaian tailor's hip may be
  elsewhere; calibration must check the offset.

### E3a: loss in cm (run r2) vs run r1, on the same 202 validation people (17 Nov 2026)

Commit 95b0649 (`anthro evaluate --test val`). One person = 0.5 points on the % column.

| Measurement | r1 MAE | r2 MAE | r1 % ±1.5 | r2 % ±1.5 |
| --- | ---: | ---: | ---: | ---: |
| chest | 1.86 | 1.81 | 52.5 | 50.5 |
| waist | 2.08 | **1.93** | 48.5 | **51.5** |
| hip | 1.56 | 1.54 | 58.9 | 60.4 |
| thigh | 1.17 | 1.19 | 73.3 | 71.8 |
| other girths and lengths | | | within ±1 point | |
| overall val MAE (best epoch) | 1.02 | 1.01 | | |

Reading: a small, real gain on the waist (−0.15 cm, 6 people), none clearly on chest or hip, nothing
lost elsewhere; chest/waist predictions slightly more consistent across photos (chest 0.81 → 0.71 cm).
The expected large gain did not appear: the network shares almost all of its work across the 14
outputs, so reweighting the loss mostly redistributes effort in the last layer. The limit on chest
and waist is information in the image, not the training goal. **Decision: keep the cm loss** for
r3 and r4.

### E3b: loss in cm + 512×256 images (run r3), same 202 validation people (17 Nov 2026)

About 42 s per epoch on a T4 (12 s at 256×128); batch 64 fits. Best epoch 25, val MAE 0.95 cm.

| Measurement | r1 % ±1.5 | r2 % ±1.5 | r3 % ±1.5 | r3 MAE |
| --- | ---: | ---: | ---: | ---: |
| chest | 52.5 | 50.5 | **57.4** | 1.69 |
| waist | 48.5 | 51.5 | 52.5 | 1.98 |
| hip | 58.9 | 60.4 | 60.4 | **1.43** (p90 3.21 → 2.70) |
| thigh | 73.3 | 71.8 | **78.2** | 1.09 |
| leg-length | 78.7 | 76.7 | **82.2** | 0.94 |
| height | 84.2 | 82.2 | 86.6 | 0.79 |
| shoulder-to-crotch | 78.2 | 81.7 | 84.7 | 0.85 |
| **mean of 14** | 79.8 | 79.9 | **82.3** | |

Reading: image detail was a real limit. Chest (+7 points, about 14 people), thigh and leg length
improved well beyond the 0.5-point-per-person noise; nothing got worse. The waist barely moved:
sharper outlines don't show the shape between the front and side views. That is the target for the
geometry (E4). Women with BMI 35+: about 55% (8 people, unreliable alone, but a persistent gap).
One seed per run: 1–2 point differences may be chance. **Decision: 512×256 for r4.**

### E4: hybrid = r3 + geometric features (run r4), same 202 validation people (17 Nov 2026)

25 geometric features per photo pair (third geometry fit). About 41 s per epoch; best epoch 25,
val MAE 0.94 cm.

| Measurement | r3 % ±1.5 | r4 % ±1.5 | r3 MAE | r4 MAE |
| --- | ---: | ---: | ---: | ---: |
| chest | 57.4 | 55.4 | 1.69 | **1.65** |
| waist | 52.5 | **54.0** | 1.98 | **1.87** |
| hip | 60.4 | **65.3** | 1.43 | **1.41** (p90 2.70 → 2.97) |
| thigh | 78.2 | 81.2 | 1.09 | 0.99 |
| calf | 88.1 | 83.7 | 0.81 | 0.82 |
| leg-length | 82.2 | 78.2 | 0.94 | 0.93 |
| **mean of 14** | 82.3 | 82.1 | 0.95 | 0.94 |

Reading: a small gain where the geometry was aimed (hip +5 points; waist −0.11 cm, the first
waist movement since the cm loss; chest MAE −0.04 cm), none overall. Calf and leg length, which the
geometry does not touch, moved −4 points: a measure of run-to-run chance with one seed. The network
already sees the outline at 512 px, so explicit width and depth add only a little independent
information.

**Choice for the final test (rule fixed beforehand: validation chest, waist, hip): r4**, narrowly:
lower MAE on all three, higher combined % (174.8 vs 170.3), equal overall, and it carries the
retake warning. Seeds were not repeated; whether r4's edge over r3 is real is untested.

## S3/S4 — Photo model vs baseline, run r1

Run 15 Nov 2026, Google Colab T4, commit `5f3543e`, data fingerprint `d9d4893ed2e13c7b`,
run folder `MyDrive/anthro/runs/r1`. Default settings from D-015, seed 0.
Every number is per person, averaging that person's photo pairs (D-007).

### Headline: % of people within ±1.5 cm (D-016)

| Measurement | Test-A baseline | Test-A photo | Test-B baseline | Test-B photo |
| --- | ---: | ---: | ---: | ---: |
| wrist | 98.85 | **100.00** | 93.75 | **97.25** |
| shoulder-breadth | 91.95 | **98.85** | 87.75 | **91.75** |
| forearm | 93.10 | **96.55** | 83.50 | **90.00** |
| ankle | 89.66 | **94.25** | 84.50 | **89.00** |
| bicep | 73.56 | **94.25** | 65.50 | **82.00** |
| arm-length | 67.82 | **89.66** | 71.00 | **88.00** |
| calf | 64.37 | **88.51** | 59.50 | **79.50** |
| height (scan) | 80.46 | **83.91** | 69.75 | **72.50** |
| shoulder-to-crotch | 70.11 | **81.61** | 70.00 | **79.00** |
| leg-length | 47.13 | **77.01** | 56.50 | **67.00** |
| hip | 45.98 | **72.41** | 42.25 | **53.75** |
| thigh | 55.17 | **72.41** | 51.25 | **70.00** |
| waist | 22.99 | **63.22** | 26.25 | **41.25** |
| chest | 35.63 | **56.32** | 28.00 | **41.25** |
| **mean of 14** | 66.9 | **83.5** | 63.5 | **74.5** |

### Companion: MAE (cm) and 90th-percentile error (cm)

| Measurement | A base MAE | A photo MAE | A photo p90 | B base MAE | B photo MAE | B photo p90 | BMnet paper (A) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ankle | 0.74 | 0.64 | 1.26 | 0.84 | 0.74 | 1.56 | 0.56 |
| arm-length | 1.12 | 0.66 | 1.52 | 1.19 | 0.83 | 1.67 | 0.71 |
| bicep | 1.15 | 0.69 | 1.28 | 1.37 | 0.91 | 1.88 | 0.64 |
| calf | 1.40 | 0.74 | 1.51 | 1.49 | 1.00 | 2.10 | 0.79 |
| chest | 2.90 | 1.67 | 3.54 | 3.28 | 2.35 | 4.84 | 1.88 |
| forearm | 0.76 | 0.47 | 0.97 | 0.89 | 0.67 | 1.47 | 0.52 |
| height (scan) | 1.08 | 0.97 | 1.81 | 1.58 | 1.45 | 2.83 | 0.89 |
| hip | 2.02 | 1.12 | 2.41 | 2.45 | 1.89 | 4.06 | 1.13 |
| leg-length | 1.81 | 1.16 | 2.63 | 1.76 | 1.39 | 2.83 | 1.12 |
| shoulder-breadth | 0.74 | 0.52 | 1.01 | 0.82 | 0.70 | 1.42 | 0.61 |
| shoulder-to-crotch | 1.18 | 0.92 | 1.91 | 1.23 | 1.08 | 2.21 | 0.89 |
| thigh | 1.68 | 1.09 | 1.99 | 1.85 | 1.30 | 2.98 | 1.12 |
| waist | 3.31 | 1.33 | 2.70 | 3.56 | 2.30 | 4.59 | 1.58 |
| wrist | 0.56 | 0.41 | 0.91 | 0.59 | 0.51 | 1.04 | 0.43 |
| **mean** | 1.46 | **0.88** | | 1.64 | **1.22** | | 0.92 |

Photo MAE is 39% lower than the baseline on Test-A and 25% lower on Test-B.

### Consistency: average spread (cm) of one person's predictions across their photos

| | ankle | arm | bicep | calf | chest | forearm | height | hip | leg | shoulder | sh-crotch | thigh | waist | wrist |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Test-A | 0.30 | 0.58 | 0.55 | 0.54 | 1.25 | 0.35 | 0.53 | 0.94 | 0.71 | 0.38 | 0.51 | 0.66 | 1.34 | 0.24 |
| Test-B | 0.33 | 0.52 | 0.66 | 0.63 | 1.54 | 0.43 | 0.65 | 1.19 | 0.69 | 0.39 | 0.55 | 0.80 | 1.89 | 0.33 |

### % within ±1.5 cm by sex and BMI band (all 14 measurements pooled)

Groups under 20 people are marked unreliable by the code; shown here in brackets.

| Sex | BMI | Test-A % (people) | Test-B % (people) |
| --- | --- | --- | --- |
| female | <18.5 | [88.1 (3)] | [76.3 (16)] |
| female | 18.5–25 | 87.2 (28) | 78.3 (148) |
| female | 25–30 | [87.1 (5)] | 75.0 (40) |
| female | 30–35 | [71.4 (3)] | [70.1 (16)] |
| female | 35+ | [71.4 (2)] | **52.8 (23)** |
| male | <18.5 | — | [78.6 (2)] |
| male | 18.5–25 | 81.8 (33) | 77.2 (71) |
| male | 25–30 | [83.3 (12)] | 73.7 (53) |
| male | 30–35 | [64.3 (1)] | 72.4 (23) |
| male | 35+ | — | [52.7 (8)] |

### Reading

- **Photos help every measurement on both test sets.** The biggest gains are where the baseline
  failed: waist 23→63% and hip 46→72% of people within tolerance on Test-A.
- **Matches the published BMnet on Test-A** (mean MAE 0.88 vs 0.92 cm). Not a like-for-like win:
  we average each person's ~19 photo pairs, which cancels some photo-to-photo noise; the paper may
  score per photo. Claim "on par", not "better".
- **Not yet at the ±1.5 cm target for girths.** Chest and waist: only 56–63% of people within
  tolerance on Test-A, 41% on Test-B. Their photo-to-photo spread alone (1.3–1.9 cm) is near the
  tolerance, so one photo pair will be noisier than these averaged figures.
- **Height barely improves** (it is an input). The remaining ~1–1.5 cm is the gap between the
  height a person gives and the scan's height; it sets a floor under every length.
- **Test-B is harder for both methods** (about 9 points lower on the mean %), so Test-B is the
  more honest guide to new people.
- **Fairness gap at BMI 35+:** about 53% within tolerance versus 77–78% at BMI 18.5–25 on Test-B
  (23 women, 8 men). Larger bodies are the group most likely to be failed; see ACCESSIBILITY.md.
  This must be reported to the supervisor and checked again on Ghanaian participants.

### Training log (r1)

Train 1,816 people / 5,533 pairs; validation 202 held-out training people / 601 pairs. About
12 s per epoch on a T4, so the whole run (25 epochs plus image caching) took about 7 minutes.

| Epoch | 1 | 3 | 5 | 7 | 10 | 13 | 16 | 19 | 22 | 25 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train loss (standardised) | 0.593 | 0.296 | 0.259 | 0.235 | 0.196 | 0.171 | 0.142 | 0.122 | 0.102 | 0.091 |
| Val MAE (cm, per person) | 2.08 | 1.50 | 1.26 | 1.23 | 1.22 | 1.12 | 1.06 | 1.05 | 1.02 | 1.02 |

Best model: epoch 25, validation MAE 1.02 cm (epochs 22–25 equal to two decimals).

- **No harmful overfitting:** validation error never rose; it flattened at 1.02 cm from epoch 22.
- **The flattening is mostly the schedule:** one-cycle drives the learning rate to almost zero in
  the last epochs, so the model stops changing. Epochs 16→25 gained only 0.04 cm, so simply
  training longer is unlikely to fix chest and waist.
- **Training loss keeps falling while validation stalls**: the model is starting to memorise the
  training people. Gains now need more varied training signal (stronger augmentation, other
  seeds), not more of the same.
- Validation (1.02) sits between Test-A (0.88) and Test-B (1.22): validation people have about
  3 photo pairs each, Test-A about 19, so Test-A averages away more noise.
- One run, one seed. How much the numbers move from luck alone is not yet known (next: seeds 1, 2).

## S2 — Height + weight + sex baseline, Test-A, per person

Run 15 Nov 2026, Google Colab, 2,018 training people, 87 Test-A people (one row each).

| Measurement | MAE (cm) | % within ±1.5 cm | Meets MAE ≤ 1.5? |
| --- | ---: | ---: | --- |
| wrist | 0.56 | 98.85 | yes |
| ankle | 0.74 | 89.66 | yes |
| shoulder-breadth | 0.74 | 91.95 | yes |
| forearm | 0.76 | 93.10 | yes |
| height (scan) | 1.08 | 80.46 | yes |
| arm-length | 1.12 | 67.82 | yes |
| bicep | 1.15 | 73.56 | yes |
| shoulder-to-crotch | 1.18 | 70.11 | yes |
| calf | 1.40 | 64.37 | yes |
| thigh | 1.68 | 55.17 | no |
| leg-length | 1.81 | 47.13 | no |
| hip | 2.02 | 45.98 | no |
| chest | 2.90 | 35.63 | no |
| waist | 3.31 | 22.99 | no |

Reading:
- 9 of 14 meet the MAE target from three numbers alone; the 5 that fail are where body shape
  varies at the same height and weight. These are what photos must improve.
- "MAE ≤ 1.5" is not "most people within ±1.5": calf meets the MAE target but only 64% of people
  are within tolerance. The % column is the headline (D-002).
- Even a properly measured height disagrees with scan height by about 1 cm on average.
- Test-A is 87 people, so one person ≈ 1.1 points on the % column; Test-B (400 people) is steadier.

Test-B baseline (400 people) is in the S3/S4 tables above.

Superseded: an earlier run scored per photo row (6,134 / 1,684 rows); numbers moved by at most
~0.3 cm. Kept out of this table (D-007).

## Reference — BMnet (paper, Test-A, cm)

Not our run: published by the BodyM authors (Ruiz et al., 3DV 2022), photos + height + weight.
Ankle 0.56 · arm 0.71 · bicep 0.64 · calf 0.79 · chest 1.88 · forearm 0.52 · head-to-heel 0.89 ·
hip 1.13 · leg 1.12 · shoulder 0.61 · shoulder-to-crotch 0.89 · thigh 1.12 · waist 1.58 · wrist 0.43.
Their scoring protocol may differ slightly from our per-person averaging.
