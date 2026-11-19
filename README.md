# Anthropometry Academic Model

**Body measurements from two photos and a height, checked on Ghanaian bodies.**
Student research project at Ashesi University. The field data-collection app lives in
[`Anthropometry/AnthropometryDC`](https://github.com/Anthropometry/AnthropometryDC).

> **Input:** front photo + side photo + height (weight and sex optional)
> **Output:** body measurements in cm, each flagged when it should be confirmed with a tape
> **Target:** within about ±1.5 cm (supervisor's target)

## Where the project stands

| Step | What | Status |
| --- | --- | --- |
| S1 | Load and understand BodyM (2,505 people) | ✅ Done |
| S2 | Height + weight baseline, scored per person | ✅ Done (Test-A and Test-B) |
| S3 | Train a BMnet-style photo model | ✅ Run r1: on par with published BMnet on Test-A |
| S3b | Geometric girth step + training fixes | ✅ Final r4: Test-A 84.0% within ±1.5 cm (MAE 0.84), Test-B 74.2% (unchanged) |
| S4 | Compare photo model vs baseline | ✅ Done: see docs/RESULTS.md |
| S5 | SMPL-X fit for extra tailoring measurements | ⏳ After S4 |
| S6 | Ghanaian calibration (photos + tape, 40+ people) | ⏳ Needs ethics approval |
| S7 | Demo: upload two photos + height, get measurements | ⏳ Final |

Latest numbers: [docs/RESULTS.md](docs/RESULTS.md). Full plan: [specs/roadmap.md](specs/roadmap.md).

## Key findings so far

Out of every 100 people, how many get a measurement within ±1.5 cm (average of all 14 measurements):

| | Baseline (height + weight + sex) | Final photo model (r4) |
| --- | ---: | ---: |
| Test-A (controlled photos, 87 people) | 67 | 84 |
| Test-B (varied photos, 400 people) | 64 | 74 |

- On controlled photos the model matches the published BMnet (average error 0.84 cm against 0.92 cm).
- Chest, waist and hip are not yet at the target (Test-A: 57, 70 and 63 in 100; Test-B: 42–51).
- Gains on controlled photos did not carry over to varied photos; a retake check flags those photos.
- Larger bodies (BMI 35+) are served worse (about 54 in 100 on Test-B).

## For the supervisor: where to read

| Question | Document |
| --- | --- |
| What is this project, and what counts as success? | [specs/mission.md](specs/mission.md) |
| What is the plan, step by step? | [specs/roadmap.md](specs/roadmap.md) |
| What is it built with, and why? | [specs/tech-stack.md](specs/tech-stack.md) |
| What rules does the work follow? | [specs/policy.md](specs/policy.md) |
| What decisions were made, and why? | [specs/discussions.md](specs/discussions.md) |
| What went wrong, and how was it fixed? | [docs/CHALLENGES.md](docs/CHALLENGES.md) |
| What are the results? | [docs/RESULTS.md](docs/RESULTS.md) |
| How is personal data and model safety handled? | [docs/SECURITY.md](docs/SECURITY.md) |
| Who does the model serve, and who not yet? | [docs/ACCESSIBILITY.md](docs/ACCESSIBILITY.md) |
| What data is used, under which licence? | [docs/DATA.md](docs/DATA.md) |

## Run it

**In Google Colab (recommended):** open `notebooks/03_s3b_geometry.ipynb` (current model) or
`notebooks/02_colab_pipeline.ipynb` (first model), choose a T4 GPU, and run the cells in order.

**On a laptop:**

```bash
pip install -e ".[dev]"
ruff check . && pytest --cov                             # fake data, no GPU; coverage must stay ≥ 80%
aws s3 sync --no-sign-request s3://amazon-bodym/ bodym   # BodyM, no AWS account needed
anthro baseline --data bodym
anthro train    --data bodym --out runs/r1               # GPU strongly recommended
anthro evaluate --data bodym --ckpt runs/r1/best.pt
# S3b: geometry, cm loss, sharper images, hybrid model, retake check
anthro geometry     --data bodym --out geo
anthro geometry-fit --data bodym --cache geo --out geo/geo_fit.json
anthro train        --data bodym --out runs/r4 --loss cm --img-h 512 --img-w 256 --geo-fit geo/geo_fit.json --geo-cache geo
anthro evaluate     --data bodym --ckpt runs/r4/best.pt --geo-cache geo --bootstrap 1000
anthro geometry-shape --data bodym --cache geo --fit geo/geo_fit.json   # S3c superellipse check
```

## Tests

58 tests run on a small fake dataset shaped like BodyM, on a CPU, in under a minute. They include
refusal cases (bad files, impossible heights, unsafe checkpoints, mismatched data splits).
**Test coverage: about 97% of the code** (CI fails below 80%; run `pytest --cov` to see it per file).

## Repository layout

```
README.md            this page: for people
CLAUDE.md            start-here instructions for Claude Code
specs/               spec-driven development
  mission.md         what the project is for and what counts as success (North Anchor)
  roadmap.md         steps S1–S7 and their status
  tech-stack.md      what it is built with, and why
  policy.md          rules the work follows
  discussions.md     every decision, who made it, and why (D-001 …)
  _template/         template for a new feature spec
  s1-load-bodym/     one folder per feature spec (spec.md)
  s2-baseline/
  s3-photo-model/
  s3b-geometric-girth/   spec.md + execute-prompt.md
  s3c-shape-check/
src/anthro/          the package: config · data · baseline · preprocess · dataset · model · train ·
                     evaluate · checkpoint · geometry · geometry_fit · geometry_shape · cli
tests/               fake-BodyM tests (unit and end to end)
notebooks/           Colab drivers: 01 explore BodyM · 02 first model · 03 S3b experiments
docs/                results, challenges, data, security, accessibility
.github/             CI (lint, tests + coverage, notebook check, secret scan) and PR template
```

## Licences

BodyM (CC BY-NC 4.0) and SMPL-X (non-commercial) are used for this school project only and are
never stored in this repository. The code has no licence yet: all rights reserved.
