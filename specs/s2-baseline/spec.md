# S2: Height + weight + sex baseline

**Status:** verified on Test-A, 15 Nov 2026 (Test-B run pending)

## Goal

Measure how well three numbers alone predict each measurement: the bar every photo model must beat.

## Method and why

- Linear regression per measurement on height, weight, male (0/1). Simple, transparent, no tuning.
- Trained on one row per training person; scored on one row per test person (D-007).

## Acceptance criteria

- [x] Per-person scores, with the people count printed as a check (2,018 / 87)
- [x] MAE, 90th-percentile error and % within ±1.5 cm per measurement
- [ ] Test-B results added to docs/RESULTS.md

## Tests

`tests/test_baseline_and_eval.py`: one row per person; per-person averaging counts each person once;
tolerance maths; consistency; group report covers every person.
