# Prompt: execute S3b (geometric girth step + training fixes)

Paste everything below the line into a new Claude Code session that has this repository attached.
Use it only after Ryan has marked `specs/s3b-geometric-girth/spec.md` as **approved**.

---

You are working in the GitHub repository `Anthropometry/Anthropometry_Accademic_Model`. It is a
student research project at Ashesi University (Ghana). The model takes a front and a side body
silhouette plus height, weight and sex, and outputs 14 body measurements in cm. The target is about
±1.5 cm. The headline figure is "% of people within ±1.5 cm".

## 0. Before writing any code

1. Read `CLAUDE.md`, `specs/policy.md`, `specs/mission.md`, `specs/roadmap.md`, the newest 8 rows of
   `specs/discussions.md`, `docs/RESULTS.md` (run r1) and `specs/s3b-geometric-girth/spec.md`.
2. Check the spec's **Status** line. If it does not say approved by Ryan, stop. Tell Ryan which of
   D-017 to D-020 still need his answer. Write no code.
3. Check out branch `s3b-geometric-girth` (create it from the newest `s1-s3-model-pipeline` if it
   is missing). Never commit to `main`. Never merge.
4. State, in a few lines: the current step (S3b), the parts approved, and your plan.

## 1. Rules that apply throughout

- Commit author and committer: `Patience Sombang <92175572+Patience3@users.noreply.github.com>`.
  Commit messages say what changed and why. End each message with the attribution lines your
  session tells you to use.
- Never commit BodyM, masks, images, caches (`.npy`, `.npz`), checkpoints (`.pt`), CSV exports or
  secrets. Check `git status` before every commit.
- Every module starts with a docstring covering: what it does; why each decision was made; challenges
  met and fixed. Comments explain why, not what.
- Every change ships with tests, including refusal cases, on **fake data** (`tests/conftest.py`).
  Run `ruff check . && pytest` before each commit; both must pass.
- Do not invent results. Only numbers from Ryan's Colab runs go in `docs/RESULTS.md`.
- Choose everything on **validation people or person-grouped CV on train**. Test-A and Test-B are
  scored once, at the end, on the chosen configuration only.
- Only approved parts of the spec are built. If a part is not approved, skip it and say so.
- No new dependencies (numpy, pandas, scikit-learn, torch, torchvision, pillow, tqdm only).

## 2. Build, in this order (one commit per item)

**A1: Loss in cm** (if D-017 approved)
- Add `TrainConfig.loss: str = "std"` with options `"std"` (r1 behaviour, unchanged) and `"cm"`.
- In `train.py`, for `"cm"`, weight the per-measurement L1 by the training std (from `y_scaler`) as
  a constant tensor, so the loss equals mean absolute error in cm. Add CLI `--loss {std,cm}`.
- Test: on a toy batch, `"cm"` loss equals hand-computed cm MAE. `"std"` is unchanged.

**A2: 512×256 images** (if D-018 approved)
- Already supported by `--img-h/--img-w`. Check that the cache name includes the size. Add a test
  that a 512×256 pair has the right shape and is not stretched. Note the T4 batch-size advice in
  the notebook (32 if out of memory).

**B1: `src/anthro/geometry.py`: measurement**
- `row_profile(front_mask, side_mask, height_cm)`. Works at full mask resolution and never
  resizes. Per row it returns:
  - front torso width in cm: the run containing the torso centre column, with arms excluded;
  - side depth in cm, with front and back extents measured from the torso midline;
  - the arm–torso gap flag.
- Scale = height_cm ÷ silhouette pixel height.
- Use `preprocess.load_mask` and `check_silhouette`. Refuse, with a clear `GeometryError`, any mask
  with no torso run.
- `ramanujan(a, b)`.
- `find_levels(profile, bands, chest_offset)`:
  - hip = maximum side depth in its band;
  - waist = minimum front width in its band;
  - chest = `chest_offset` below the first row with an arm–torso gap.

  Refuse a level that falls outside its band.
- `level_features(...)`: returns W, D, C₀, D/W, the front/back split and flags (`arm_merge`,
  `edge_soft`, `mask_px_height`) for chest, waist and hip.
- `fullness(profile, weight_kg)` = (weight ÷ 1.03) ÷ Σ(π/4 · W · D · Δh).
- Tests with synthetic masks: a disc and an ellipse recover W and D within 1 px; Ramanujan on a
  circle gives 2πr; an arms-touching torso sets `arm_merge`; empty and no-torso masks are refused.

**B2: `anthro geometry` CLI and cache**
- `anthro geometry --data bodym --split train testA testB --out DIR` computes features for every
  photo pair. It saves `{split}_geometry_{fingerprint}.npz` (with `allow_pickle=False`), prints E0
  diagnostics, and runs on CPU.
- Add `*.npz` to `.gitignore`. Test on fake BodyM.

**B3: Level search and correction (E1, E2)**
- `anthro geometry-fit --data bodym --cache DIR`:
  - 5-fold person-grouped CV on train only.
  - Search band limits (fractions of silhouette height) and the chest offset, to minimise CV error
    of C₀ against truth per level.
  - Fit k = f(D/W, BMI, sex, fullness) per level by linear regression.
  - Print E1: C₀ alone vs truth (MAE, % within ±1.5 cm, k mean and spread by sex × BMI band).
  - Print E2: Ĉ = k·C₀, next to the S2 baseline, on CV folds.
  - Save the chosen bands and coefficients as JSON (plain values) on Drive.
- Test: on fake data the fit runs, CV folds never share a person, and the JSON reloads.

**B4: Hybrid model (E4)** (if D-019 approved)
- `--geo-features PATH_TO_FIT_JSON`: in `model.py`, add an optional second input of standardised
  geometric features. It is joined to the ResNet's pooled features before a new final layer.
  Without the flag the model is exactly r1's.
- Store the geometric scaler in the checkpoint as plain lists. Loading must still use
  `weights_only=True`.
- `evaluate` must work for both kinds of checkpoint.
- Tests:
  - a 1-epoch smoke run with geometric features;
  - save, then safe reload;
  - an r1-style checkpoint still evaluates.

**B5: Retake flag (E5)** (D-020 sets the share; default 10%)
- In `evaluate`, compute |Ĉ − prediction| per level. Choose the threshold on validation people so
  that it fires for the approved share.
- Report MAE and % within ±1.5 cm for flagged vs unflagged people, by sex × BMI band.
- Test: the threshold reproduces the chosen share on a toy set.

**C: Final reporting support**
- `evaluate --bootstrap 1000` adds 95% confidence intervals by resampling people.
- Lengths and girths are reported in separate tables.

## 3. Notebook and documents

- Update `notebooks/02_colab_pipeline.ipynb`. Keep it with no saved outputs, and set
  `BRANCH = "s3b-geometric-girth"`. Add cells in experiment order:
  - E0/E1/E2: `anthro geometry` + `geometry-fit`, CPU;
  - E3a: `anthro train --loss cm --out runs/r2`;
  - E3b: `--loss cm --img-h 512 --img-w 256 --out runs/r3`;
  - E4: best of E3 + `--geo-features --out runs/r4`;
  - final: `anthro evaluate --bootstrap 1000` on the chosen run.

  Each cell says what to paste back.
- `specs/roadmap.md`: tick items as they are built; mark experiments "waiting for Ryan's run".
- `docs/CHALLENGES.md`: record each problem met and its fix.
- `specs/discussions.md`: no new decisions without Ryan. Record his answers to D-017 to D-020 when given.
- `docs/RESULTS.md`: leave empty for S3b until Ryan pastes real outputs.

## 4. Finish

1. Run `ruff check . && pytest`; both must pass.
2. Push `s3b-geometric-girth`. Open or update the pull request with the repo's template:
   - base `s1-s3-model-pipeline` if PR #1 is unmerged, otherwise `main`;
   - list what was built and what was skipped (unapproved);
   - list the tests, including refusal cases, and the security checks;
   - Results: "pending Ryan's Colab run";
   - say exactly which notebook cells Ryan runs and what he pastes back.
3. Do not merge. Stop and report to Ryan in plain, non-technical language.
