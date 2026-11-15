# Anthropometry Academic Model — start here

The **model** for a student research project at Ashesi University: two photos (front, side) plus
height in, body measurements in cm out, accurate for Ghanaian bodies to about ±1.5 cm. The field
data-collection app is a separate repository (`Anthropometry/AnthropometryDC`).

## Every session

1. Read [POLICY.md](POLICY.md). Its rules override convenience; if one blocks a task, stop and ask Ryan.
2. Read [docs/NORTH_ANCHOR.md](docs/NORTH_ANCHOR.md), [docs/ROADMAP.md](docs/ROADMAP.md) and the
   newest 5 rows of [DISCUSSION.md](DISCUSSION.md).
3. State the current step, the next task and any open questions.
4. For a step, read its `specs/S#-*.md`. No spec marked **approved** → write or update it from
   [specs/_TEMPLATE.md](specs/_TEMPLATE.md) and stop.

**Branches and pull requests only.** Never push to `main`; never merge without Ryan's "merge it".
Commits are authored as Patience Sombang (D-013).

## Fixed decisions (changed only by Ryan, via DISCUSSION.md)

- Score **per person**, never per photo; split train/validation **by person** (D-007, D-008).
- Height is a **required, measured** input; weight and sex optional but used when given (D-006).
- Never use the scan `height` column as an input: it is an answer, not an input.
- Checkpoints hold only tensors and plain values and load with `weights_only=True` (D-012).
- BodyM, SMPL-X files, Ghanaian data, photos and checkpoints are **never committed** (D-014).

## Writing code here

- Every module starts with a docstring: **what** it does, **why** each decision was made, and
  **challenges met and fixed**. Inline comments explain *why*, not *what*.
- Decisions get a `D-###` row in DISCUSSION.md; problems and fixes go in docs/CHALLENGES.md.
- New dependency or service → DISCUSSION.md entry approved by Ryan first.
- Every change includes tests, including refusal cases. Run `ruff check .` and `pytest` before "done".
- Results that go in docs/RESULTS.md come from Ryan's real runs only; never invent or estimate numbers.

## Commands

```bash
pip install -e ".[dev]"                                   # install with test tools
ruff check . && pytest                                    # lint + tests (fake data, CPU, ~10 s)
anthro baseline --data bodym --test testA testB           # step 2
anthro train    --data bodym --out runs/r1 --cache DIR    # step 3 (GPU)
anthro evaluate --data bodym --ckpt runs/r1/best.pt       # steps 3–4
```

## Reporting

End each task in the [pull request template](.github/pull_request_template.md) order: what changed;
tests; security and licence checks; results (real runs only); what Ryan needs to decide.
