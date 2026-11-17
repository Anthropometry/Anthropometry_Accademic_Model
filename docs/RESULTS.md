# Results

Only numbers from Ryan's real runs go here, each with its date, data fingerprint and commit.

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
  Third fit pending.

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
