# S3c: Shape check: does a superellipse or a front/back term beat the oval correction?

**Status:** approved by Ryan, 19 Nov 2026 (D-023)
**Branch:** `s3c-shape-check` (stacked on `repo-structure`, PR #3)

## Goal (one sentence)

Turn "a superellipse would not help" from a reasoned guess into a measured result, on BodyM training
people only, without retraining the network.

## Why

The S3b spec said: add a richer cross-section shape (superellipse, or separate front/back halves)
**only if** the oval's error k = true ÷ oval follows the depth-to-width ratio (D/W) or the front/back
balance. That check was never run: k was only reported by sex × BMI. The front/back balance
(`side_split`) is measured and given to the network, but not to the correction.

## Inputs and outputs

| In | Out |
| --- | --- |
| The geometry cache (`anthro geometry`) and the chosen levels (`geo_fit.json`) | Printed tables + `geo_shape_report.json` beside the fit |
| BodyM training people, validation people excluded (same split as `geometry-fit`) | No model, no checkpoint: a diagnostic only |

## Method and why

1. **Does k follow D/W or the front/back balance?** Per level, the correlation of k with D/W and with
   `side_split`, and the mean k in five equal-sized groups (quintiles) of each. A flat table = the
   shape term would not help; a slope = it might.
2. **Three corrections compared by 5-fold cross-validation grouped by person** (same folds, per-person
   scoring, as everywhere else):
   - **A, current:** Ĉ = k̂ × C₀, k̂ linear in D/W, BMI, sex, fullness, arm-merge (S3b).
   - **B, + front/back:** the same, with `side_split` added as a sixth feature.
   - **C, superellipse:** each slice is |x/a|ⁿ + |y/b|ⁿ = 1 (n = 2 oval, larger = boxier). For each
     training person, find the n whose perimeter equals the true girth; learn n̂ linearly from the
     six features; predict Ĉ = superellipse perimeter(a, b, n̂).
3. **Why C fits n per person and then learns it:** the photos give a and b only; n is what they can't
   see, so it must be predicted from the same features. If C ≈ A, the superellipse adds nothing a
   multiplicative k does not already capture.
4. **Perimeter of a superellipse:** no closed formula. Computed numerically (polyline with 2,000
   points per quarter) on a table of shape ratio b/a × n, then interpolated. Checked against
   Ramanujan at n = 2 and against the rectangle 2(W + D) at large n.

Rejected: retraining the network with superellipse features (costs a GPU run before we know whether
the geometry improves at all); fitting n per person without cross-validation (would flatter C).

## Acceptance criteria (testable)

- [ ] `anthro geometry-shape --data D --cache GEO --fit FIT` prints the k-vs-shape tables and the
      A/B/C comparison per level, and saves `geo_shape_report.json`.
- [ ] Uses only training people, validation people excluded, folds grouped by person.
- [ ] Superellipse perimeter matches Ramanujan within 0.1% at n = 2 and approaches 2(W + D) as n grows.
- [ ] Runs on Colab CPU in a few minutes.

## How it is scored

Per person (photos averaged), 5-fold CV grouped by person; MAE and % within ±1.5 cm per level;
baseline (height + weight + sex) shown alongside.

## Security, licence and privacy checks

- Data never committed: reads the Drive cache; writes one JSON of numbers next to the fit.
- Safe loading: cache loaded with `allow_pickle=False`; fit read as JSON.
- Inputs validated: refuses a fit made from a different person split; superellipse n limited to 1.5–8.

## Accessibility and fairness checks

- Results also by BMI band for the best variant, because the CT study found the oval degrades for
  large abdomens.

## Tests to write

| Test | Allow or refuse |
| --- | --- |
| Superellipse perimeter: n = 2 ≈ Ramanujan; n = 8 close to rectangle; increases with n | allow |
| Inverse: the n recovered from a perimeter matches the n that made it | allow |
| CLI runs end to end on fake BodyM and writes the report | allow |
| Fit made on different data, or not a fit at all | refuse |

## Decision afterwards

If B or C lowers CV MAE by ≥ 0.1 cm on a level, propose (to Ryan) using it in the hybrid features;
otherwise record "superellipse tested, no gain" in RESULTS.md and the tutorial.
