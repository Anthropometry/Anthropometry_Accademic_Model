# Security for this model

## What "security" means here

Security means different things for different kinds of AI. For a chat model (an LLM) or an AI agent,
the big risks are prompt injection, leaking a system prompt, or an agent taking harmful actions with
its tools. **None of those apply here.** This project is a small image network (a CNN regressor):
numbers and silhouettes go in, 14 numbers come out. It cannot be talked into anything and takes no
actions.

For this model, security means six things:

| # | Risk | What could go wrong | Control in this repo |
| --- | --- | --- | --- |
| 1 | **People's data** | Participants' bodies, photos or identities leak | Silhouettes only (no faces or skin); no data ever committed; no names or photos in logs or notebook outputs |
| 2 | **Unsafe files** | A downloaded checkpoint runs code when loaded (PyTorch files are pickles) | Load only with `weights_only=True`; cached images with `allow_pickle=False`; a test proves a code-carrying file is refused |
| 3 | **Bad inputs** | An empty, partial or wrong image, or a mistyped height, gives a confident but meaningless answer | Silhouette checks, hard limits on height and weight, warnings outside the training range |
| 4 | **Data integrity** | BodyM changes or a file is corrupted, and results silently stop being comparable | Data fingerprint stored in every checkpoint; `evaluate` warns on mismatch; tables validated on load |
| 5 | **Supply chain** | A compromised package or model download | Only well-known packages, listed in `pyproject.toml`; pre-trained weights only from torchvision; new dependencies need Ryan's approval; secret scan in CI |
| 6 | **Licences and misuse** | BodyM or SMPL-X used or redistributed against their licences | Never committed (`.gitignore`); non-commercial use documented in specs/mission.md |

## Threats specific to trained models, and our level of concern

- **Membership inference** (guessing whether someone was in the training data from the model's
  answers): low risk for BodyM, which is public. Higher once Ghanaian participants are involved;
  so the calibration step only learns **one offset per measurement**, from which no individual can
  be recovered. Per-participant results are never published.
- **Model inversion** (reconstructing a body from the model): not meaningful for a model that
  outputs 14 numbers.
- **Data poisoning** (corrupted training data): BodyM is trusted, published data. For Ghanaian data,
  the collection app's double readings and range checks are the defence, and every export is
  reviewed by Ryan before use.
- **Adversarial images** (images crafted to fool the model): possible in principle; in the demo the
  defence is input checks and a "confirm with tape" flag on uncertain results, not a promise of
  robustness.

## Rules for the demo (step 7)

- Accept PNG/JPEG only, refuse oversized images, process in memory, **store nothing**.
- Never log uploaded images, heights or weights.
- Show results with their uncertainty and the "confirm with tape" flag.

## Reporting a problem

Open a private security advisory on the repository, or tell Ryan directly. Don't open a public issue
for anything involving personal data.
