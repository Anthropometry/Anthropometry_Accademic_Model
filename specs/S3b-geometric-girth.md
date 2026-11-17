# S3b: Geometric girth step (width × depth → circumference), plus two training fixes

**Status:** approved by Ryan, 17 Nov 2026 (D-017 to D-020 as proposed; retake share 10%) — code built and tested on fake data, 17 Nov; real runs next (notebooks/03_s3b_geometry.ipynb)
**Branch:** `s3b-geometric-girth` (stacked on `s1-s3-model-pipeline`, PR #1)

## Goal (one sentence)

Make chest, waist and hip more accurate and explainable by measuring the body's width (front photo)
and depth (side photo) in cm, turning them into a circumference with a learned ellipse correction, and
giving those numbers to the photo model. Two training fixes come first.

## Why (from run r1 and the research brief, 17 Nov 2026)

- r1, Test-B, % of people within ±1.5 cm: chest 41%, waist 41%, hip 54%. The other measurements are 67–97%.
- **Cause 1, wrong training goal:** L1 loss on *standardised* targets. One unit of waist is about
  10 cm, one unit of wrist about 1 cm, so the model treats waist errors as cheap.
- **Cause 2, coarse images:** at 256 px tall, 1 px ≈ 0.7 cm of body. A 1 cm width error moves an
  ellipse perimeter about 1.7 cm.
- **Cause 3, shape:** the same width and depth give 88.4 cm (ellipse) or 90.7 cm (slightly squarer
  superellipse, n = 2.3). Two photos fix W and D, not the shape. The shape must be learned.
- **Evidence:** front + side ellipse methods report 0.6–0.9 cm mean error, but only on small,
  female-only or synthetic samples (Montazerian & Fol Leymarie 2022, 55 women; Zhang et al. 2024,
  120 virtual bodies). No large study shows geometry alone beating a trained network. The supported
  design is **geometry first, then learn the leftover error** (Zhang et al. 2024; Wang et al. 2020,
  *Symmetry* 12:1997, front/back half-ellipses split by D/W).
- **Realistic target:** cut chest/waist MAE by 0.2–0.5 cm on Test-B (about 45–55% within
  ±1.5 cm). 70%+ is plausible only on Test-A-style controlled photos. Human tape measurers' own
  allowable error is 12–14 mm for chest and waist (ANSUR 1988); ISO 20685-1 allows 9 mm.

## Inputs and outputs

| In | Out |
| --- | --- |
| Full-resolution front and side masks, height cm, weight kg, sex | Per level (chest, waist, hip): W cm, D cm, ellipse perimeter C₀, corrected perimeter Ĉ, quality flags |
| The above + the S3 inputs | 14 measurements (hybrid model) + a "retake photo" flag |

## Method and why

### Part A: two training fixes (run before Part B, each alone)

| ID | Change | Why | Rejected |
| --- | --- | --- | --- |
| A1 | Loss in cm: L1 on standardised outputs × each measurement's training std (`--loss cm`) | Matches how we score (cm, ±1.5 cm for all) | Keeping standardised loss (under-weights girths); squared error (outliers dominate) |
| A2 | Images 512×256 (`--img-h 512 --img-w 256`); batch 32 if memory runs out | Halves cm per pixel; T4 time ≈ 4×, about 25 min | 1024 px (memory; small extra gain) |

Run A1 then A1+A2. Keep whichever is better on **validation**. That becomes the base for Part B.

### Part B: the geometric step (`src/anthro/geometry.py`)

1. **Measure at full resolution.** Use the original masks, never the 256 px cache. Scale:
   cm per pixel = height_cm ÷ silhouette height in pixels.
   - E0 prints the mask sizes and silhouette heights in pixels. If the body is under ~600 px tall,
     record that 1 px ≈ 3 mm caps accuracy.
   - For every row, record:
     - **front torso width**: the run of body pixels containing the torso centre, with arms excluded;
     - **side depth**: total extent, plus the front extent and back extent measured from the torso midline.
2. **Find the levels.** BodyM's chest, waist and hip are fixed SMPL vertex loops, not published
   heights. So the levels are found on our own data. Search within height bands (fractions of
   silhouette height from the top), choosing per level:
   - **hip**: maximum side depth in the lower-torso band;
   - **waist**: minimum front width in the mid-torso band;
   - **chest**: a fixed offset below the armpit, where the armpit is the first row from the top
     where a background gap appears between arm and torso.

   Band limits and the chest offset are chosen by 5-fold **person-grouped** cross-validation on
   train only. The choice criterion: whichever makes C₀ best predict that person's true girth.
3. **Ellipse (Ramanujan):** a = W/2, b = D/2,
   C₀ = π[3(a + b) − √((3a + b)(a + 3b))].
4. **Learned correction:** Ĉ = k · C₀, where k = f(D/W, BMI, sex, fullness).
   - f is plain linear regression per level, so the result stays explainable: "the ellipse
     under-reads your waist by 3%".
   - Before fitting, plot k = C_true / C₀ by level, sex and BMI band (E1).
   - Add asymmetric front/back half-ellipses (Wang et al. 2020) **only** if E1 shows k's spread
     follows D/W or the front/back split.
5. **Fullness index (weight → volume), a soft feature only:**
   fullness = (weight ÷ 1.03 kg/L) ÷ Σ over rows (π/4 · W · D · row height).
   - Whole-body density is about 1.02–1.07 kg/L depending on body fat (Siri), so volume is only
     known to about ±3%, about ±1.5% in girth.
   - It is one number for the whole body, so it enters f as a feature, never as a hard constraint.
6. **Quality flags:**
   - `arm_merge`: no gap between arm and torso at the chest row;
   - `edge_soft`: anti-aliased or blurry edge;
   - `mask_px_height`: silhouette height in pixels.

   When `arm_merge` is set, chest W is unreliable and Ĉ_chest is not used.
7. **Hybrid:** append the per-level W, D, C₀, Ĉ and flags (standardised on train) to the ResNet's
   pooled features before the final layer. The model learns when to trust geometry.
   - This changes the approved S3 head (D-015), so it needs Ryan's approval (D-019).
   - Considered and rejected: feeding geometric features as constant image channels, the current
     route for height and weight, because it is wasteful for 15+ numbers.
8. **Retake flag:** |Ĉ − model prediction| > threshold, per level.
   - Pick the threshold on validation so it fires for a chosen share of people (Ryan decides,
     default 10%).
   - Report how much the error drops among people who are not flagged.

### Experiments, in order (each one decided on validation, never on test)

| E | What | Answers |
| --- | --- | --- |
| E0 | Mask sizes, silhouette px heights, level-finding success rate, `arm_merge` rate | Is full-resolution geometry possible on BodyM? |
| E1 | C₀ alone vs truth, and k by level × sex × BMI | How wrong is "the body is an ellipse"? (supervisor figure) |
| E2 | Ĉ = k · C₀ (linear f) vs S2 baseline vs r1 model | Does explainable geometry alone beat height + weight? |
| E3 | A1, then A1 + A2 | Fixes from causes 1 and 2 |
| E4 | Hybrid (best of E3 + geometric features) | Does geometry add to the network? |
| E5 | Retake flag on E4 | How much error the flag removes, and for whom |

Final: score **only the chosen configuration** on Test-A and Test-B, once. Report bootstrap 95%
confidence intervals (resampling people), because Test-A has only 87 people.

## Acceptance criteria (testable)

- [ ] `anthro geometry --data bodym --out DIR` writes per-photo W, D, C₀ and flags to a cache on
      Drive (never the repo). It runs on all ~9,000 photo pairs in under 20 min on Colab CPU.
- [ ] E0–E2 are reported in RESULTS.md, with the ellipse-only error per level and k by sex × BMI band.
- [ ] `anthro train --loss cm` and `--geo-features` work, and r1 settings are unchanged when they're off.
- [ ] Hybrid MAE on validation ≤ the best E3 model's on chest, waist and hip, or the result is
      reported honestly as "no gain".
- [ ] Final Test-A/Test-B tables report MAE, % within ±1.5 cm, sex × BMI bands and 95% CIs, with
      the baseline and r1 alongside.
- [ ] Checkpoints still load with `weights_only=True`. Geometric scalers are stored as plain lists.

## How it is scored

Per person, averaging photo pairs. Choices are made by 5-fold person-grouped CV, or on the 10%
validation people for network runs. The baseline and r1 are always shown alongside. Lengths and
girths are reported separately. % within ±1.5 cm is the headline (D-016).

## Security, licence and privacy checks

- Geometry caches and full-resolution masks stay on Drive. `.gitignore` already blocks `*.npy`
  and `*.csv`. The cache is a `.npz`, which is added to `.gitignore`.
- Full-resolution masks go through `load_mask` (refuses images over 40 MP) and `check_silhouette`.
  Geometry refuses masks with no torso run or levels outside the bands, rather than guessing.
- No new dependencies: numpy, pandas and scikit-learn are already installed.
- BodyM is CC BY-NC 4.0: school use with attribution only. Any commercial use of a model trained
  on it needs Amazon's permission (D-014).

## Accessibility and fairness checks

- Report k and errors by sex × BMI band. Expect the ellipse to break down for large abdomens: a CT
  study found waist ellipse agreement degrades above about 110 cm (Ciudin et al. 2014). Check
  whether geometry narrows or widens r1's BMI 35+ gap (53% vs 77%).
- BodyM is about 14% Black/African American. No West African cross-section data was found, so k
  learned on BodyM is **not assumed** to fit Ghanaian bodies. S6 recalibrates it on Ghanaian
  participants.
- People flagged for retake are told why ("arms touching your sides", "move further from the
  camera"), not just refused.

## Tests to write

| Test | Allow or refuse |
| --- | --- |
| Ramanujan perimeter: circle gives 2πr; known ellipse within 0.01% of numeric integration | allow |
| Synthetic box/ellipse masks: W, D recovered within 1 px; cm scale from height | allow |
| Torso with arms touching: `arm_merge` set, chest not used | allow (flagged) |
| Empty mask, no torso run, or level outside band | refuse with a clear message |
| `--loss cm` equals per-measurement std-weighted L1 on a toy batch | allow |
| `--geo-features` checkpoint saves and reloads with `weights_only=True` | allow |
| Retake threshold reproduces the chosen firing share on validation | allow |

## Ryan's answers (17 Nov 2026)

All four approved as proposed; retake flag share 10%.

## Questions as asked

1. **D-017:** approve loss in cm (A1)?
2. **D-018:** approve 512×256 images (A2), with about 4× training time?
3. **D-019:** approve changing the approved S3 head to accept geometric features (hybrid)?
4. **D-020:** what share of people may the retake flag fire for? (default 10%)
