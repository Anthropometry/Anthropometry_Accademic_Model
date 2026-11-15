# Results

Only numbers from Ryan's real runs go here, each with its date, data fingerprint and commit.

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

Superseded: an earlier run scored per photo row (6,134 / 1,684 rows); numbers moved by at most
~0.3 cm. Kept out of this table (D-007).

## Reference — BMnet (paper, Test-A, cm)

Not our run: published by the BodyM authors (Ruiz et al., 3DV 2022), photos + height + weight.
Ankle 0.56 · arm 0.71 · bicep 0.64 · calf 0.79 · chest 1.88 · forearm 0.52 · head-to-heel 0.89 ·
hip 1.13 · leg 1.12 · shoulder 0.61 · shoulder-to-crotch 0.89 · thigh 1.12 · waist 1.58 · wrist 0.43.
Their scoring protocol may differ slightly from our per-person averaging.
