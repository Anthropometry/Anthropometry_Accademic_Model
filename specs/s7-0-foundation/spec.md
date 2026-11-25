# S7-0: Foundation for the full measurement set and our own data

**Status:** approved by Ryan, 24 Nov 2026 (missing weight: typical weight with a warning, D-027; errors from Test-B, D-028)
**Branch:** `s7-0-foundation` (stacked on `s3c-shape-check`, PR #4)

## Goal (one sentence)

Make the app's own measurement list the backbone of the model repository, with one versioned output
format and swappable suppliers. Today's sources (BodyM photo model, later ANSUR and geometry) fill the
full adult set now, and a model trained on the app's data can replace them later without rewriting the
code.

## Why

- The class demo must output the **full adult set**: 24 measurements for women, 18 for men, adults only.
- The code today is built around BodyM's 14 names and the number 14. The app's export (AnthropometryDC
  F8, branch `f8-admin-export`) uses 25 different codes, optional measurements, two weights, clothing
  and phone tilt. Without a common backbone, our own data would need a rewrite.
- Decisions this builds on:
  - D-024: ANSUR II approved;
  - D-025: bust-point lengths "not available";
  - D-026: SMPL-X set aside.

## Inputs and outputs

| In | Out |
| --- | --- |
| A person: front and side silhouettes (photos after S7a), height (measured or stated), weight (optional, scale or stated), sex, age range, optional clothing and phone tilt | One JSON result in format `anthro-output/1`: every measurement in the person's set, each with value, source, quality, definition match and expected error |
| An app export folder (`anthro-export-YYYY-MM-DD-HH-MM/`) | The same internal table the BodyM loader produces, in catalogue codes |

## Method and why

### 1. The catalogue (`catalog/measurements.json`)

A copy of the app's `CATALOGUE` and sets from `src/lib/measurements.ts` (AnthropometryDC, branch
`f8-admin-export`), stamped with the source commit. For each of the 25 codes:
- code, label and how-text, exactly as in the app;
- type (girth or length);
- the app's adult plausibility range;
- whether it is taken twice;
- whether it is "personal";
- which sexes get it.

The two adult sets are generated from the app's rules: women 24 (all but `chest_girth`); men 18 (no bust
measurements, `high_hip_girth`, `waist_to_knee` or `shoulder_to_knee`). Children are excluded.

- **Why a copy, not a live link:** the model must be reproducible against a fixed list. A test fails if
  the copy and its stamp disagree.
- **Rejected:** keeping BodyM's names and translating at output time. That would leave the code shaped
  around BodyM.

### 2. BodyM translated at the door

BodyM's names are mapped to catalogue codes. Each mapping carries a `definition_match`: same, presumed
or differs.

| BodyM | Catalogue code | Match |
| --- | --- | --- |
| height | `height` | same |
| waist | `waist_girth` | same (narrowest point) |
| chest | `chest_girth` (men), `bust_girth` (women) | presumed |
| hip | `hip_girth` | differs (fixed height, not fullest) |
| shoulder-breadth | `shoulder_width` | differs (path unknown; the app's runs over the back) |
| arm-length | `arm_length` | presumed |
| bicep | `upper_arm_girth` | presumed |
| wrist, thigh, calf | `wrist_girth`, `thigh_girth`, `calf_girth` | presumed |

BodyM's ankle, forearm, leg-length and shoulder-to-crotch are kept internally, as inputs for ANSUR
formulas (S5a), but are not reported: the app doesn't collect them. Leg-length is not mapped to `inseam`
until its definition is checked.

### 3. The output format `anthro-output/1`

```json
{
  "schema": "anthro-output/1",
  "catalogue": "anthropometrydc-measurements@<commit>",
  "model_version": "v1",
  "created_at": "2026-12-01T10:00:00Z",
  "person": {"sex": "female", "age_range": "25-34", "height_cm": 165.0, "height_kind": "measured",
             "weight_kg": 62.0, "weight_kind": "stated", "clothing": "light"},
  "retake": {"flag": false, "reasons": []},
  "warnings": [],
  "measurements": {
    "waist_girth": {"value_cm": 74.3, "source": "photo_model", "quality": "measured",
                    "definition_match": "same", "expected_error_cm": 2.4,
                    "error_basis": "BodyM Test-B, per-person MAE, run r4"},
    "bust_point_distance": {"value_cm": null, "source": null, "quality": "not_available",
                            "definition_match": null, "expected_error_cm": null, "error_basis": null}
  }
}
```

- **Only the person's set** appears: 24 codes for women, 18 for men.
- **`quality`:**
  - `measured`: from the photo model;
  - `derived`: from formulas (ANSUR);
  - `rough`: from outline geometry, unchecked;
  - `not_available`.
- **`expected_error_cm`** comes only from a recorded evaluation report; otherwise `null`, never guessed.
  For the photo model it uses Test-B per-person average error, the honest figure for varied photos.
- **`retake`** carries the retake warning and its reasons (geometry and network disagree; geometry
  failed; later, phone tilt).
- **People under 18 are refused**, with a message, when the age range says so.
- **Why a versioned format:** the app or a demo reading it never breaks silently. A change means
  `anthro-output/2`.

### 4. Suppliers and routing

A **supplier** is a small module with a name, the codes it can supply, and
`estimate(case) → {code: Estimate}`.
- **Built now:** `photo_model`, wrapping the existing checkpoint, and `not_available`.
- **Later, each its own spec:**
  - `ansur` (S5a);
  - `geometry` (S5b);
  - `calibration` (S6);
  - `own_model` (S8).

`routing/default.json` lists, for each code, the suppliers to try in order. The first value found wins.
Codes with no supplier are `not_available`, so today 9–10 women's codes (photo model) are filled and the
rest are clearly marked.

- **Why:** replacing a source with our own model becomes a one-line routing change, testable side by
  side.
- **Rejected:** one large function with special cases per measurement.

### 5. App export loader (`anthro.export_loader`)

Reads an export folder: `participants.csv`, `measurements.csv`, `photos.csv`, `images/`, `README.txt`.
It returns:
- one row per participant;
- measurement columns in catalogue codes, empty where skipped or not in the person's set;
- both readings kept, for measurer repeatability;
- stated and scale weight kept separately;
- clothing, phone tilt and head-to-heel pixels kept;
- photo paths.

Checks:
- **Format stamp:** the export carries none yet. The loader accepts an unstamped export as format 0
  with a warning, accepts `anthro-export/1` once the app adds it, and refuses any other version.
- **Practice:** practice rows are refused unless `--include-practice` (then marked).
- **Columns:** missing columns are refused.

The loader is **tested only on a fake export** made by the tests. It never touches the app's database
or storage (policy).

## Out of scope here (each its own spec)

- **S7a:** photo → silhouette (needs a segmentation model, a new dependency).
- **S5a:** ANSUR formulas (licence check first).
- **S5b:** outline-geometry rough estimates.
- **S8:** training on the app's data (catalogue-driven outputs, missing values in the loss, both
  weights).

## Acceptance criteria (testable)

Built 25 Nov 2026 (tests/test_s7_0.py). File locations changed from this spec's first draft: the
catalogue, schema and routing live in `src/anthro/data/` (`measurements.json`,
`anthro-output-1.schema.json`, `routing_default.json`) so they install with the package on Colab;
`scripts/sync_catalog.py` refreshes the catalogue from the app.

- [x] `catalog/measurements.json` holds the 25 codes exactly as the app; generated sets are 24 (women)
      and 18 (men); a test compares them with the app's rules.
- [x] `anthro predict-silhouettes --front F --side S --height H [--weight W] --sex female --age-range 25-34 --ckpt C`
      writes one `anthro-output/1` JSON with every code of the person's set: photo-model codes filled,
      the others `not_available`.
- [x] The output validates against a JSON schema file kept in the repo (`schemas/anthro-output-1.json`).
- [x] `expected_error_cm` is filled only from an evaluation report; `null` otherwise.
- [x] Under-18 age ranges are refused.
- [x] Changing `routing/default.json` changes which supplier fills a code, with no code change (test).
- [x] The export loader reads a fake export into the internal table; refuses unknown format versions,
      missing columns and practice rows (unless asked).
- [x] r1–r4 behaviour, tests and results unchanged; coverage stays ≥ 80% (98%).
- [ ] Ryan runs notebooks/04_s7_predict.ipynb with r4 on one Test-B woman and man (first real check).

## How it is scored

No new accuracy numbers: this is structure. Existing per-person results are reused as the error
figures in the output.

## Security, licence and privacy checks

- No app data, photos or exports are committed. The loader is tested on generated fake folders only.
- Nothing here reads the AnthropometryDC database or storage. Only Ryan's own export folders, on his
  machine or Drive, are read, and only when he runs the loader.
- The output carries no name or ID beyond what the caller passes. Age is a range, never an exact age.
- No new dependency: a small schema check is written by hand rather than adding a library.

## Accessibility and fairness checks

- Every value carries its quality and definition match, so a tailor never mistakes a rough estimate for
  a measured one.
- The output states known weak spots as warnings: BMI 35+ less accurate (RESULTS), everyday clothing
  not tested, BodyM not Ghanaian.

## Tests to write

| Test | Allow or refuse |
| --- | --- |
| Catalogue: 25 codes; women 24, men 18; matches the app's rules | allow |
| BodyM → codes mapping covers the 14 and marks matches | allow |
| Output JSON has exactly the person's set and validates against the schema | allow |
| Missing weight → typical weight for height and sex, warning, values marked `weight_assumed` (D-027) | allow |
| Under-18 age range | refuse |
| Routing change switches the supplier | allow |
| Fake export loads; readings and both weights kept | allow |
| Unknown format, missing column, practice row without the flag | refuse |

## Ryan's answers (24 Nov 2026)

1. Approved.
2. Missing weight: use a typical weight for height and sex (straight-line fit per sex on BodyM training people), with a warning and a `weight_assumed` mark on affected values (D-027).
3. Errors: Test-B (D-028).

## Questions as asked

1. Approve this foundation (catalogue from the app's codes, `anthro-output/1`, suppliers and routing,
   export loader)?
2. **Missing weight:** the current photo model always needs a weight. Until S8 teaches a model to work
   without it, should a person with no weight be (a) refused, with a clear message (recommended), or
   (b) given a typical weight for their height and sex, with a warning?
3. **Error shown to users:** use Test-B figures (varied photos, more honest; recommended) or Test-A
   (controlled photos, smaller numbers)?
