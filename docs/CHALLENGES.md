# Challenges and fixes

Problems met while building, what caused them and what fixed them. Newest first.

| Date | Challenge | Cause | Fix | Where |
| --- | --- | --- | --- | --- |
| 15 Nov 2026 | Consistency check printed `NaN` | Spread needs 2+ photos of a person; some splits have only one | Clear message instead of `NaN` | src/anthro/cli.py |
| 15 Nov 2026 | First notebook loaded checkpoints with `weights_only=False` | Normalisation statistics were stored as numpy arrays | Store them as plain lists; load with `weights_only=True`; test that a code-carrying file is refused | src/anthro/checkpoint.py, tests/test_train_smoke.py |
| 15 Nov 2026 | Baseline fix "didn't work": identical numbers after the change | The fix lines ran *before* the line that reassigned the tables, so they were overwritten | Rebuilt the cell with the de-duplication after table selection, plus a printed people count as a check | notebooks, src/anthro/baseline.py |
| 15 Nov 2026 | Baseline scored per photo, not per person | Joining the tables gives one row per photo pair; Test-A has ~19 per person | Score on one row per person; average a person's photo predictions (D-007) | src/anthro/evaluate.py |
| 15 Nov 2026 | Two "height" columns | `height_cm` is the entered height (input); `height` is scan head-to-heel (answer) | Scan height is never an input; documented in data.py | src/anthro/data.py |
| 15 Nov 2026 | Data re-downloaded after every Colab disconnect | Colab storage is wiped when the runtime ends | Keep one `.tar` on Drive, unpack per session (D-010) | notebooks/ |
| 15 Nov 2026 | Slow image loading | ~12,000 PNGs read every epoch | Prepare once, cache as one `.npy` per split, keyed by size and data fingerprint | src/anthro/dataset.py |
| 14 Nov 2026 | No BMnet model to load | Authors released the dataset but not code or weights | Train our own BMnet-style model (D-005) | src/anthro/model.py |
| 14 Nov 2026 | File names unknown before download | BodyM's file layout isn't documented | Exploration notebook discovers and prints the layout; code written against what it found | notebooks/01_explore_bodym.ipynb |
| 14 Nov 2026 | Our build environment couldn't reach Amazon storage | Network allowlist | Notebook runs in Colab; tests use a fake BodyM with the same layout | tests/conftest.py |
