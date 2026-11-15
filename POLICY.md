# POLICY — rules for Claude Code

These rules override convenience; when one blocks a task, Claude Code stops and asks rather than
working around it. This is a student research project at Ashesi University.

## Process

1. Spec first. No code for a step until its `specs/S#` file is marked approved by Ryan.
2. One step or fix per branch and pull request. Never push to `main`; never merge without Ryan's "merge it".
3. Every change includes tests, refusal cases included. Lint and tests pass before saying "done".
4. Unsure about scope, method, data or a trade-off? Ask Ryan, then record the answer in
   `DISCUSSION.md`. Never guess.
5. Commit messages say what changed and why, in plain words (`type(scope): summary`).

## Data, licences and privacy

6. Never commit datasets, images, silhouettes, checkpoints, caches, SMPL/SMPL-X files or
   participant data. `.gitignore` blocks them; never weaken it.
7. Never read, query or export the AnthropometryDC database or storage. Ghanaian data reaches this
   project only as a de-identified export Ryan prepares, and stays out of the repository.
8. BodyM and SMPL-X are non-commercial: no commercial use, and no redistribution.
9. Never put secrets (tokens, keys, passwords) in code, commits, notebooks or logs.
10. No names, faces or photos in logs, notebooks' saved outputs or reports.

## Honest results

11. Every reported number comes from a real run, with its seed, data fingerprint and commit.
12. Test sets are never used for training or for choosing settings.
13. Always show the baseline next to the model, and lengths and girths separately.
14. Never weaken a check (input validation, safe loading, per-person scoring) to make a number look better.

## Keeping Ryan in the loop

15. Start each session by stating the current step, the next task and open questions.
16. End each task with: done, tested, security and licence checked, waiting on Ryan for …
