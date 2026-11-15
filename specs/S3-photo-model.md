# S3: BMnet-style photo model

**Status:** approved by Ryan, 15 Nov 2026 — code built and tested on fake data; first real run next
**Branch:** `s3-photo-model`

## Goal (one sentence)

Train a network that turns a front and side silhouette plus height, weight and sex into the 14 BodyM
measurements, and beats the S2 baseline on waist, chest, hip, leg length and thigh.

## Inputs and outputs

| In | Out |
| --- | --- |
| front + side silhouette (256×128 each), height cm, weight kg, sex | 14 measurements in cm |

## Method and why

| Choice | Why | Rejected alternative |
| --- | --- | --- |
| ResNet-18, ImageNet pre-trained | ~2,000 people is too few to learn image features from zero; small enough for a free T4 | Training from scratch (needs far more data); larger networks (slower, little gain at this size) |
| 5 input channels (2 images + 3 constant channels) | BMnet's method: numbers usable from the first layer | Joining numbers only at the end (network learns shape without scale) |
| Crop to body, pad to 2:1, resize | Removes camera distance; height supplies real size (D-011) | Full frame (distance confounds size) |
| L1 loss on standardised targets | Matches "average error in cm"; robust to unusual bodies | Squared error (dominated by a few large errors) |
| AdamW + one-cycle, 25 epochs, lr 3e-4 | Standard, robust fine-tuning recipe with few settings | Hand-tuned schedules (more to tune, small data to tune on) |
| Best epoch chosen on 10% of training people | Test sets stay untouched (POLICY 12) | Choosing on Test-A (optimistic, dishonest) |
| ±3% shift and scale augmentation | Framing varies between photos | Flips and rotations (change pose; break level-camera assumption) |

## Acceptance criteria

- [ ] Runs end to end on a free Colab T4 in under ~1 hour
- [ ] Per-person Test-A and Test-B tables next to the baseline (`anthro evaluate`)
- [ ] Beats the baseline's MAE on waist, chest and hip on both test sets
- [ ] Results by sex and BMI band; consistency across a person's photos
- [ ] Checkpoint loads with `weights_only=True`; records config, data fingerprint and version

## Security, licence and privacy checks

- BodyM, caches and checkpoints live on Ryan's Drive; `.gitignore` blocks them.
- Silhouettes and inputs validated (preprocess.py); out-of-training-range height or weight flagged.

## Accessibility and fairness checks

- Report % within ±1.5 cm by sex × BMI band; name any band with fewer than 20 people as unreliable.
- BodyM's 30–35 and 35+ BMI bands are thin; expect larger errors there and say so.

## Tests

`tests/test_train_smoke.py` (1 epoch on fake data, safe reload, refusal of a code-carrying file, CLI
evaluate); `tests/test_preprocess.py` (shape, distance removed, no stretching, refusals).

## Ryan's answers (15 Nov 2026)

1. Network and training settings (D-015): **approved as proposed**.
2. Headline figure: **% of people within ±1.5 cm** (D-016); MAE shown alongside.
