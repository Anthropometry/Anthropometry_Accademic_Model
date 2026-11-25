# S5a: ANSUR II supplier (formulas for the measurements BodyM doesn't have)

**Status:** draft, 25 Nov 2026. Waiting for Ryan's approval.
**Branch:** `s5a-ansur`

## Goal (one sentence)

Fill the app measurements BodyM lacks (9 for women, 7 for men) with formulas learned on ANSUR II tape
measurements, each with an honest expected error, so the demo shows `derived` values instead of `not_available`.

## Why ANSUR, and what it can and cannot do

- **What it is:** US Army 2012 survey of 4,082 men and 1,986 women; 93 tape and calliper measurements in mm
  (weight in tenths of a kg), plus self-reported height and weight. Released publicly in 2017.
- **Licence (D-024, checked 25 Nov 2026):** the Army's memorandum for the data files states they were reviewed
  and cleared for unlimited public release; the methods report (Gordon et al., 2014, NATICK/TR-15/007) is
  "approved for public release; distribution is unlimited". No licence conditions; we cite both. The 3D scans
  are not public (participant privacy); we don't need them.
- **What it cannot do:** it has no photos. So it cannot measure a person; it can only turn numbers we already
  have (height, weight, sex, photo-model measurements) into numbers we don't. Its people are US soldiers, not
  Ghanaian civilians: every formula must be re-checked on Ryan's tape data (S6/S8).

## Inputs and outputs

| In | Out |
| --- | --- |
| `ANSUR II MALE Public.csv`, `ANSUR II FEMALE Public.csv` (Ryan downloads once to Drive; never committed) | `ansur_formulas.json`: per code and sex, the chosen formula (plain numbers), its inputs and its held-out error |
| At prediction: the person (height, weight, sex) and the photo model's 14 values | `derived` estimates for the codes below, with `expected_error_cm` and `definition_match` |

## The measurements and their ANSUR source

Column names are from the ANSUR II data memorandum; the build's first step prints the real CSV headers and
stops if any is missing (that code then stays `not_available`). "Vertical" means ANSUR gives heights from the
floor, so a length is a difference of two heights, not a tape laid along the body.

| App code | App definition | ANSUR source | Match |
| --- | --- | --- | --- |
| neck_girth | around the base of the neck | `neckcircumferencebase` | same |
| shoulder_length | side of the neck to the shoulder point | `shoulderlength` | same |
| hip_girth | around the fullest part of the hips | `buttockcircumference` | same |
| back_length | nape down to the waist | `waistbacklength` | differs: ANSUR's waist is at the navel, the app's at the narrowest point |
| inseam | crotch down the inside leg to the ankle bone | `crotchheight` − `lateralmalleolusheight` (vertical) | presumed |
| outseam | waist down the side of the leg to the ankle bone | `waistheightomphalion` − `lateralmalleolusheight` (vertical) | differs: navel waist, vertical |
| waist_to_hip | at the side, waist to fullest hip | `waistheightomphalion` − `buttockheight` (vertical) | differs: navel waist, vertical |
| waist_to_knee (women) | waist to the middle of the knee | `waistheightomphalion` − `kneeheightmidpatella` (vertical) | differs: navel waist |
| shoulder_to_knee (women) | side of the neck at the shoulder to the knee | `acromialheight` − `kneeheightmidpatella` (vertical) | differs: ANSUR has no neck-side height; shoulder point is lower (question 3) |

`hip_girth` changes from the photo model's fixed-height hip (`differs`) to ANSUR's fullest hip (`same`), which
is why routing already lists `ansur` before `photo_model` for it.

## Method and why

1. **Two kinds of formula per code and sex, both straight-line (linear regression, scikit-learn, already a
   dependency):**
   - **Basic:** height, weight, BMI. Uses only what the person types, so it can't inherit photo-model error.
   - **Extended:** basic + the photo model's measurements that ANSUR also has in a comparable form
     (chest, waist, shoulder breadth, thigh, calf, wrist; the photo model's hip for codes other than hip).
   - **Why straight lines:** explainable in the writeup (one coefficient per input), hard to overfit with
     2,000–4,000 people, and stored as plain numbers. Rejected: gradient boosting or neural nets (opaque,
     little expected gain on smooth body-size relations).
2. **Honest error, chosen per code on people the formula never saw:** 5-fold split by person, inside ANSUR.
   For the extended formula, the inputs are first **blurred with noise the size of the photo model's
   Test-B error** (from r4's report), because at prediction time those inputs are photo-model estimates, not
   tape. The formula with the lower held-out error is kept; ties go to basic (simpler).
   - **Why the blur:** without it the extended formula looks far better than it will be in real use, since
     ANSUR's own inputs are exact tape measurements.
3. **Weight:** formulas are fitted on ANSUR's measured weight. A second check refits using ANSUR's
   *self-reported* height and weight, to show how much a stated weight (as many app users will give) costs.
4. **Expected error shown to users:** the chosen formula's held-out per-person average error (with the blur),
   labelled `ANSUR II held-out, photo-model-sized input noise`. It does not include the definition gaps
   above or the US-to-Ghana gap; the warnings already say so.
5. **Where formulas live:** `anthro ansur-fit` writes `ansur_formulas.json` (coefficients, inputs, errors,
   ANSUR file fingerprints, catalogue version). Prediction reads it with `--ansur FILE`.

## Commands

```bash
anthro ansur-fit --ansur DRIVE/ansur2 --report runs/r4/eval_report.json --out ansur_formulas.json
anthro predict-silhouettes ... --ansur ansur_formulas.json
```

Notebook `notebooks/05_s5a_ansur.ipynb`: check headers, fit, print the scores, run one prediction.

## Acceptance criteria (testable)

- [ ] Reads both ANSUR files; converts mm → cm and tenths of kg → kg; refuses missing columns, naming them.
- [ ] Fits basic and extended formulas per code and sex with a 5-fold split by person; held-out errors printed
      next to a "sex average" baseline.
- [ ] `ansur_formulas.json` holds plain numbers only and refuses to load for a different catalogue version.
- [ ] With `--ansur`, a woman's result has the 9 codes above as `derived` (7 for a man), with
      `expected_error_cm` and `definition_match`; without it, unchanged from S7-0.
- [ ] `weight_assumed` is carried onto derived values that used the weight.
- [ ] No ANSUR rows, only coefficients and summary errors, are written anywhere in the repository.
- [ ] Tests on a generated fake ANSUR; coverage stays ≥ 80%.

## How it is scored

Per person (one row each in ANSUR), held out by person, 5 folds. Shown per code and sex: average error,
90th-percentile error and % within ±1.5 cm, for the sex-average baseline, basic and extended (blurred and
unblurred). Lengths and girths separate. These numbers come from Ryan's Colab run only.

## Security, licence and privacy checks

- Data never committed: ANSUR files stay on Drive; `.gitignore` already blocks data folders; the loader
  refuses a path inside the repository.
- Safe loading: CSV only (no pickle); formulas are JSON.
- Inputs validated: unit ranges checked (e.g. neck 25–60 cm) so a units mistake is caught.
- ANSUR contains no names; its subject IDs are never printed.

## Accessibility and fairness checks

- Results by sex and by BMI band (< 25, 25–30, 30–35, 35+).
- ANSUR records self-reported race. Reporting errors for ANSUR's Black participants separately would show
  whether the formulas hold for African-descent bodies (still US soldiers, not Ghanaians). See question 2.
- Who this does not serve: anyone outside ANSUR's ranges (very small or very large bodies) gets the value with
  a "outside the formula's range" warning.

## Tests to write

| Test | Allow or refuse |
| --- | --- |
| Fake ANSUR loads; units converted | allow |
| Missing column | refuse, named |
| Fold split never puts one person in two folds | allow |
| Formula file for another catalogue version | refuse |
| Woman gets 9 derived codes, man 7; values in plausible ranges | allow |
| Derived values carry `weight_assumed` when weight was assumed | allow |
| Repository path given as the ANSUR folder | refuse |

## Open questions for Ryan

1. **Approve S5a as written?**
2. **Fairness check:** also report the errors for ANSUR's Black participants separately (yes, recommended /
   no)? A follow-up could fit formulas on them only, if it helps.
3. **Shoulder to knee:** ANSUR has no "side of the neck" height, so the formula measures from the shoulder
   point (a little lower, so the result would be a little short). Accept as `differs`, or leave shoulder to
   knee `not_available` until your tape data exists?
4. **Formula file:** commit `ansur_formulas.json` into the package after your run (coefficients only, from
   public data; then the demo works without the ANSUR files), or keep it on Drive only?
