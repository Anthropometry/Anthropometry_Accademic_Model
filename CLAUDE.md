# Anthropometry Academic Model — start here

The **model** for a student research project at Ashesi University: two photos (front, side) plus
height in, body measurements in cm out, accurate for Ghanaian bodies to about ±1.5 cm. The field
data-collection app is a separate repository (`Anthropometry/AnthropometryDC`).

## Every session

1. Read [specs/policy.md](specs/policy.md). Its rules override convenience; if one blocks a task, stop and ask Ryan.
2. Read [specs/mission.md](specs/mission.md), [specs/roadmap.md](specs/roadmap.md),
   [specs/tech-stack.md](specs/tech-stack.md) and the newest 5 rows of
   [specs/discussions.md](specs/discussions.md).
3. State the current step, the next task and any open questions.
4. For a step, read its `specs/<step>/spec.md`. No spec marked **approved** → write or update it from
   [specs/_template/spec.md](specs/_template/spec.md) (new folder `specs/<step>/`) and stop.

**Branches and pull requests only.** Never push to `main`; never merge without Ryan's "merge it".
Commits are authored as Patience Sombang (D-013).

## Fixed decisions (changed only by Ryan, via specs/discussions.md)

- Score **per person**, never per photo; split train/validation **by person** (D-007, D-008).
- Height is a **required, measured** input; weight and sex optional but used when given (D-006).
- Never use the scan `height` column as an input: it is an answer, not an input.
- Checkpoints hold only tensors and plain values and load with `weights_only=True` (D-012).
- BodyM, SMPL-X files, Ghanaian data, photos and checkpoints are **never committed** (D-014).

## Writing code here

- Every module starts with a docstring: **what** it does, **why** each decision was made, and
  **challenges met and fixed**. Inline comments explain *why*, not *what*.
- Decisions get a `D-###` row in specs/discussions.md; problems and fixes go in docs/CHALLENGES.md.
- New dependency or service → specs/discussions.md entry approved by Ryan first.
- Every change includes tests, including refusal cases. Run `ruff check .` and `pytest --cov` before
  "done"; coverage must stay at or above 80% (D-021).
- Results that go in docs/RESULTS.md come from Ryan's real runs only; never invent or estimate numbers.

## Commands

```bash
pip install -e ".[dev]"                                   # install with test tools
ruff check . && pytest --cov                              # lint + tests + coverage (fake data, CPU)
anthro baseline --data bodym --test testA testB           # step 2
anthro train    --data bodym --out runs/r1 --cache DIR    # step 3 (GPU)
anthro evaluate --data bodym --ckpt runs/r1/best.pt       # steps 3–4 (--test val: validation people)
anthro geometry     --data bodym --out GEO                # S3b geometry cache
anthro geometry-fit --data bodym --cache GEO --out FIT    # S3b levels + correction
anthro geometry-shape --data bodym --cache GEO --fit FIT  # S3c superellipse check
anthro predict-silhouettes --front F --side S --height H --sex female --ckpt runs/r4/best.pt  # S7-0
anthro read-export --folder anthro-export-...             # S7-0: check an app export (counts only)
anthro ansur-fit --ansur DRIVE/ansur2 --report runs/r4/eval_report.json --out FORMULAS  # S5a
anthro ansur-mix --data bodym --ckpt runs/r4/best.pt --formulas FORMULAS                  # S5a mix test
python scripts/sync_catalog.py APP/src/lib/measurements.ts --source "repo branch@commit"   # S7-0
```

## Reporting

End each task in the [pull request template](.github/pull_request_template.md) order: what changed;
tests; security and licence checks; results (real runs only); what Ryan needs to decide.
