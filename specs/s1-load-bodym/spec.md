# S1: Load and understand BodyM

**Status:** verified, 15 Nov 2026 (Ryan ran notebooks/01_explore_bodym.ipynb on the real data)

## Goal

Load BodyM into tables the rest of the project can trust, and record its real layout.

## Outcome

- Layout and split sizes recorded in docs/DATA.md and in `src/anthro/data.py`.
- `load_split` refuses missing files, missing columns, repeated people, unmatched photos and missing values.
- `fingerprint` hashes the tables so every checkpoint records which data trained it.

## Tests

`tests/test_data.py`: per-photo rows vs per-person view; person-level split never shares a person;
split reproducible; missing column refused; duplicate subject refused; fingerprint changes with data.
