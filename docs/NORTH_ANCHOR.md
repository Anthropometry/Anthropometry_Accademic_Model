# North Anchor — Anthropometry Academic Model

15 Nov 2026 · Ryan Tangu Mbun Tangwe · student research project at Ashesi University

This is the single source the model work is built from. When a question about scope, method or
success comes up, this file answers it; when it doesn't, Claude Code asks Ryan and records the
answer in [DISCUSSION.md](../DISCUSSION.md).

## Mission

Build and evaluate a system that takes **two photos (front and side) plus a person's height** and
returns **body measurements in centimetres**, accurate enough for tailoring and checked on
**Ghanaian** bodies.

## Relation to the other repository

| Repository | Purpose | Lifetime |
| --- | --- | --- |
| `Anthropometry/AnthropometryDC` (Anthro Collect) | Field app: consent, tape measurements, face-blurred photos from ~900 participants | Keeps running after the class |
| `Anthropometry/Anthropometry_Accademic_Model` (this repo) | The model: training, evaluation, calibration, demo | The semester project; reused by the larger build |

The data-collection app produces the Ghanaian ground truth this repo calibrates against (step 6).
No participant data is ever copied into this repository.

## Success criteria

The supervisor's target is a discrepancy of about **±1.5 cm** (D-002). Agreed reading:

1. **Headline:** % of people within ±1.5 cm, per measurement.
2. **Companion:** mean absolute error (MAE) per measurement ≤ 1.5 cm.
3. Scored **per person** (a person's photos averaged), on people the model never saw.
4. Reported **separately for lengths and girths**, and **by sex and body-size band**.
5. Compared against two references: the **height + weight baseline** (what photos must beat)
   and **two human measurers' disagreement** (the floor no method can be judged beyond).

Evidence so far (docs/RESULTS.md): height + weight alone already meets MAE ≤ 1.5 cm on 9 of 14
BodyM measurements; waist, chest, hip, leg length and thigh do not. Published BMnet results
suggest photos roughly halve those errors in the lab, but chest (1.88) and waist (1.58) stay
above 1.5 cm even there. **Girths are the hard part, and are reported honestly as such.**

## The method (D-004)

```
front + side photo → silhouettes → BMnet-style network (+ height, weight, sex) → 14 measurements
                                                       ↓
                               Ghanaian correction (one offset per measurement, step 6)
                                                       ↓
                 SMPL-X body fitted to the corrected numbers → extra tailoring measurements
                 (neck, armhole, back length, rise) + a "this looks inconsistent" flag
```

- The network does the core measurements. SMPL-X **extends and checks** them; fed only the
  network's numbers it cannot make them more accurate (no independent information).
- Users need **no tape**. The team tapes ~40+ people once, to calibrate and to prove accuracy.

## Principles, in priority order

1. **Honest evaluation** over impressive numbers: per person, held-out people, baselines shown.
2. **Privacy and licences respected** by design (docs/SECURITY.md).
3. **Works for every body it claims to serve**, and says clearly when it doesn't
   (docs/ACCESSIBILITY.md).
4. **Reproducible**: anyone with the repo and BodyM can rerun every number.
5. **Free to run**: Google Colab free tier, no paid services.

## In scope / out of scope

In scope: BodyM training and evaluation; SMPL-X extension; calibration on a small Ghanaian sample;
a demo that takes two photos and a height.

Out of scope for the semester: children (BodyM is adults only); commercial use (BodyM and SMPL-X
are non-commercial); seeing body shape under loose clothing (D-003); app-store release.

## Licences that bind this work (D-014)

| Asset | Licence | Consequence |
| --- | --- | --- |
| BodyM | CC BY-NC 4.0 | School project only; never committed to this repo |
| SMPL-X | Non-commercial research licence | Model files downloaded by each user after registering; never committed |
| This repo's code | No licence yet (all rights reserved) | Others may read but not reuse |
