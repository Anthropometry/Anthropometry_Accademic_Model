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
| S2 | Height + weight baseline, scored per person | ✅ Done on Test-A |
| S3 | Train a BMnet-style photo model | 🟡 Code ready and tested; first real run next |
| S4 | Compare photo model vs baseline | ⏳ After S3 |
| S5 | SMPL-X fit for extra tailoring measurements | ⏳ After S4 |
| S6 | Ghanaian calibration (photos + tape, 40+ people) | ⏳ Needs ethics approval |
| S7 | Demo: upload two photos + height, get measurements | ⏳ Final |

Latest numbers: [docs/RESULTS.md](docs/RESULTS.md). Full plan: [docs/ROADMAP.md](docs/ROADMAP.md).

## Key finding so far

Height, weight and sex alone predict 9 of BodyM's 14 measurements within 1.5 cm on average. The
five they miss — **waist (3.3 cm), chest (2.9), hip (2.0), leg length (1.8), thigh (1.7)** — are
where people of the same size differ in shape. Those are what the photo model must improve.

## For the supervisor: where to read

| Question | Document |
| --- | --- |
| What is this project, and what counts as success? | [docs/NORTH_ANCHOR.md](docs/NORTH_ANCHOR.md) |
| What decisions were made, and why? | [DISCUSSION.md](DISCUSSION.md) |
| What went wrong, and how was it fixed? | [docs/CHALLENGES.md](docs/CHALLENGES.md) |
| What are the results? | [docs/RESULTS.md](docs/RESULTS.md) |
| How is personal data and model safety handled? | [docs/SECURITY.md](docs/SECURITY.md) |
| Who does the model serve, and who not yet? | [docs/ACCESSIBILITY.md](docs/ACCESSIBILITY.md) |
| What data is used, under which licence? | [docs/DATA.md](docs/DATA.md) |

## Run it

**In Google Colab (recommended):** open `notebooks/02_colab_pipeline.ipynb`, choose a T4 GPU, Run all.

**On a laptop:**

```bash
pip install -e ".[dev]"
ruff check . && pytest                                   # ~10 s, fake data, no GPU needed
aws s3 sync --no-sign-request s3://amazon-bodym/ bodym   # BodyM, no AWS account needed
anthro baseline --data bodym
anthro train    --data bodym --out runs/r1               # GPU strongly recommended
anthro evaluate --data bodym --ckpt runs/r1/best.pt
```

## Repository layout

```
src/anthro/     config · data · baseline · preprocess · dataset · model · train · evaluate · checkpoint · cli
tests/          fake-BodyM tests: loading, scoring, preprocessing, training, safe loading
notebooks/      Colab drivers: 01 explore BodyM, 02 baseline → train → evaluate
specs/          one spec per step, approved before building
docs/           North Anchor, roadmap, results, challenges, data, security, accessibility
```

## Licences

BodyM (CC BY-NC 4.0) and SMPL-X (non-commercial) are used for this school project only and are
never stored in this repository. The code has no licence yet: all rights reserved.
