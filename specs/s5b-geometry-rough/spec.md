# S5b: rough estimates for the last four measurements (under-bust, front length, high hip, knee)

**Status:** approved by Ryan, 25 Nov 2026 (seated front length accepted as `differs`, D-033; rough values
show no error until tape data exists, D-034)
**Branch:** `s5b-geometry-rough` (on top of `s5a-ansur`)

## Goal (one sentence)

Fill the last rows that BodyM, ANSUR and the photo model cannot fill (women 4, men 2) with honest estimates,
so the demo shows a number for 22 of 24 rows (women) and 18 of 18 (men).

## Where we are

| Code | Who | Today | Why not yet |
| --- | --- | --- | --- |
| underbust_girth | women | not_available | not in BodyM; ANSUR II column to be checked |
| front_length | both | not_available | not in BodyM; ANSUR II has a seated version only (to be checked) |
| high_hip_girth | women | not_available | not in BodyM or ANSUR II |
| knee_girth | both | not_available | not in BodyM; ANSUR II column to be checked |
| bust_point_distance, shoulder_to_bust_point | women | not_available | stays so (D-025): a silhouette has no bust points |

## Method and why: three sources, in order of trust

### 1. ANSUR first, where it has the same measurement

The build starts by printing every ANSUR II column whose name contains `breast`, `bust`, `knee`, `front`
or `hip`, with its definition from the ANSUR II report. A column whose definition matches the app's is fitted
exactly like S5a (basic / extended, held-out error, Black participants separately, the Black-fitted
formula if it wins: D-032 logic). Those values are `derived` with an error.

- **Why first:** a real tape measurement with a measured error beats any geometry guess.
- If a column exists but its posture or landmark differs (for example front length measured **sitting**),
  it is used only if Ryan accepts it as `differs` (question 2).

### 2. Outline geometry for what ANSUR lacks (`rough`)

The same full-resolution geometry as S3b, at new heights:

- **Under-bust:** ellipse from front width and side depth a short distance below the chest level, times
  the chest correction k from S3b (the nearest fitted level). The offset below the chest is taken from
  ANSUR's chest-to-under-breast height gap if ANSUR has it, otherwise a fixed fraction of height (stated).
- **High hip:** halfway (in height) between the waist and hip levels, times the average of the waist and hip
  corrections.
- **Knee girth:** at knee height (ANSUR's knee height / stature ratio), one leg only: width of one leg
  in the front view, depth of the legs in the side view, ellipse, no correction (none can be fitted).
- **Front length (if ANSUR can't):** the length of the side view's front edge, from the base of the neck
  down over the chest to the waist level: a surface path, not a straight drop.

Geometry values are `rough`, `expected_error_cm: null`, with a flag naming the method.

- **Why rough with no error:** nothing in BodyM or ANSUR lets us check these numbers. Printing an error we
  never measured would be invented. Ryan's tape data (S8) is the first real check.
- **Rejected:** making up a correction for knee or high hip from other levels without evidence.

### 3. Not available

If the outline can't be measured at that height (arms touching, legs merged, out of frame), the row stays
`not_available` and the retake reasons say why.

## Checks that are possible without new data

- **Plausibility:** each rough value must fall inside the app's adult range for that code (from the
  catalogue); outside → `not_available` with a reason.
- **Order:** under-bust < bust, waist < high hip < hip (fullest) for each person; a violation is flagged.
- **Spread on BodyM:** the rough values across BodyM's 202 validation people are summarised (median, 5th–95th
  percentile) beside ANSUR's spread for the same code where ANSUR has it, as a sanity check, not a score.

## Inputs and outputs

| In | Out |
| --- | --- |
| Full-resolution front and side silhouettes, height, the S3b geometry fit, the S5a formula file | Estimates for the four codes: `derived` (ANSUR) or `rough` (geometry) |

## Commands

```bash
anthro ansur-columns --ansur DRIVE/ansur2          # step 1: candidate columns, counts and medians only
anthro geometry-rough --data bodym --ckpt runs/r4/best.pt   # plausibility, order and spread on validation people
```

`predict-silhouettes` fills the codes automatically (supplier `geometry` in routing, already listed).

## Acceptance criteria (testable)

Built 25 Nov 2026 (tests/test_s5b.py). Front length: ANSUR `waistfrontlengthsitting`, used only if the column
exists. Under-bust 4.5% of height below the chest level; knee height from ANSUR (textbook 0.285 until refit).


- [x] Candidate ANSUR columns printed; any accepted column fitted and scored as in S5a.
- [x] Geometry gives under-bust, high hip and knee girth (and front length if ANSUR can't) from a silhouette
      pair, each `rough` with `expected_error_cm: null` and a method flag.
- [x] Values outside the catalogue's adult range, or out of order, become `not_available` or are flagged.
- [ ] Women get 22 of 24 rows filled, men 18 of 18, on a pair the outline can measure.
- [x] Tests on fake silhouettes with known widths (the ellipse and the edge-length maths checked by hand).
- [x] Coverage stays ≥ 80%.

- [ ] Ryan runs notebooks/06_s5b_rough.ipynb; the new formula file is committed.

## How it is scored

It isn't, yet: no data has these measurements and a photo. Only the ANSUR-derived ones get a held-out
error. Reported: spread on validation people and the share of people each code could be measured for.

## Security, licence and privacy checks

- No new data or dependency. ANSUR stays on Drive; only coefficients and summaries leave it.
- No silhouettes or images are saved by these commands; only numbers.

## Accessibility and fairness checks

- Share of validation people each rough code could be measured for, by sex and BMI band (merged arms and
  thighs are more common at higher BMI, so the gaps are reported, not hidden).

## Tests to write

| Test | Allow or refuse |
| --- | --- |
| Ellipse at a known width and depth gives the hand-computed girth | allow |
| Front-edge length of a known staircase outline equals its hand-computed length | allow |
| Knee measured on one leg, not both | allow |
| Value outside the adult range | refuse (not_available with reason) |
| Under-bust above bust, or high hip outside waist–hip | flagged |
| Merged legs at knee height | not_available |

## Ryan's answers (25 Nov 2026)

1. Approved. 2. ANSUR's seated front length accepted, marked `differs` (D-033). 3. Rough values show no ± error
until tape data exists (D-034).

## Questions as asked

1. **Approve S5b as written?**
2. **Seated front length:** if ANSUR's only front length is measured sitting, accept it as `differs`, or use
   the outline path instead?
3. **No error shown for rough values:** agree that `rough` rows show no ± error until your tape data exists
   (honest), rather than a guessed one?
