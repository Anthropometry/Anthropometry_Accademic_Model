# DISCUSSION — Ryan's decisions, newest first

Each entry: ID, date, topic, decision, why, files changed. Only Ryan makes or changes a decision;
"proposed" rows wait for his approval.

| ID | Date | Topic | Decision | Why | Changed |
| --- | --- | --- | --- | --- | --- |
| D-016 | 15 Nov 2026 | Headline metric | **% of people within ±1.5 cm** is the headline figure in every report; MAE is shown alongside | Clearer to readers than an average, which can hide people outside tolerance | specs/S3-photo-model.md, docs/RESULTS.md |
| D-015 | 15 Nov 2026 | Network | ResNet-18 pre-trained on ImageNet, first layer widened to 5 channels (front, side, height, weight, sex), 14 outputs; L1 loss; AdamW + one-cycle; 256×128 images; best epoch chosen on 10% of training people. **Approved by Ryan, 15 Nov 2026** | Small enough for free Colab; pre-training compensates for ~2,000 people; mirrors BMnet's design | src/anthro/model.py, train.py |
| D-014 | 15 Nov 2026 | Licences | BodyM (CC BY-NC 4.0) and SMPL-X (non-commercial) are used for the school project only and never committed; each user downloads them | Licence terms; keeps the repository safe to make public | .gitignore, POLICY.md |
| D-013 | 15 Nov 2026 | Repository | Public repo `Anthropometry/Anthropometry_Accademic_Model`; commits authored as Patience Sombang; no code licence for now | Supervisor can follow by link; same authorship as AnthropometryDC; keeps the business option open | all |
| D-012 | 15 Nov 2026 | Safe checkpoints | Checkpoints hold only tensors and plain values and load with `torch.load(weights_only=True)` | Loading a pickle can run code; the first notebook used `weights_only=False` | src/anthro/checkpoint.py |
| D-011 | 15 Nov 2026 | Image preparation | Crop each silhouette to the body, pad to 2:1 without stretching, resize to 256×128; real size comes from the height input | Removes camera distance; keeps body proportions | src/anthro/preprocess.py |
| D-010 | 15 Nov 2026 | Colab storage | BodyM kept on Google Drive as one `.tar`, unpacked per session; prepared images cached as `.npy` | Reading ~18,000 small files from Drive is very slow; Colab disconnects often | notebooks/ |
| D-009 | 11 Nov 2026 | Ground truth | BodyM measurements are a virtual tape on SMPL-registered scans, not a tailor's tape; the gap is measured and corrected in step 6 | Definitions differ; consistent offsets are correctable (ISO 20685 agreement after constant offsets) | docs/NORTH_ANCHOR.md |
| D-008 | 15 Nov 2026 | Validation split | Hold back whole people, never single photos | One person's photos on both sides inflates the score | src/anthro/data.py |
| D-007 | 15 Nov 2026 | Scoring | Score per person, averaging a person's photo pairs | Test-A has ~19 pairs per person; per-photo scoring let a few people dominate | src/anthro/evaluate.py, baseline.py |
| D-006 | 11 Nov 2026 | Inputs | Two photos + measured height required; weight and sex optional extras | Height sets the scale (2% height error ≈ 2% on every measurement); weight improves girths (BMnet: hip error 2.83→0.97 cm with height+weight) | src/anthro/preprocess.py |
| D-005 | 14 Nov 2026 | BMnet availability | BMnet code and weights were never released; we train our own BMnet-style model on BodyM | Searched GitHub, Hugging Face, Papers with Code, Amazon Science | src/anthro/model.py |
| D-004 | 11 Nov 2026 | Method | BMnet-style network for the 14 core measurements; SMPL-X only for extra measurements and a consistency check; Ghanaian calibration by per-measurement offset. An ensemble of the network with SMPL-X fed by the network's own numbers is **rejected** | SMPL-X fed by the network has no independent information, so it can't add accuracy on shared measurements | docs/NORTH_ANCHOR.md |
| D-003 | 11 Nov 2026 | Clothing | No camera that tries to see body shape under clothing; fitted clothing required for accurate results | No phone camera can do it; loose clothing inflates girths | docs/NORTH_ANCHOR.md |
| D-002 | 11 Nov 2026 | Target | Supervisor's target: discrepancy about ±1.5 cm. Reported as % of people within ±1.5 cm (headline) and MAE (companion), per measurement | An average can hide many people outside tolerance | src/anthro/config.py, evaluate.py |
| D-001 | 11 Nov 2026 | Scope | The class deliverable is a system: photos + height in, measurements out, relevant to Ghanaians. Data collection is a separate, longer-running build | Separate lifetimes and audiences | docs/NORTH_ANCHOR.md |
