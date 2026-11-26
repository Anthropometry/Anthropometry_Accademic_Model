# Tech stack

What the model is built with, and why each piece was chosen. Adding or replacing any of it needs a
row in [discussions.md](discussions.md) approved by Ryan ([policy.md](policy.md), rule on dependencies).

## Language and runtime

| Piece | Version | Why |
| --- | --- | --- |
| Python | 3.10 or newer (CI: 3.11; Colab currently 3.13) | The language of the ML tools below; Colab provides it free |
| Package `anthro` (`src/anthro`) | 0.1.0 | All logic lives in one tested package; notebooks only call it |

## Libraries (runtime)

| Library | Range | Used for | Why this one |
| --- | --- | --- | --- |
| PyTorch | ≥ 2.2 | Training and running the photo model | Standard deep-learning library; Colab ships it with GPU support |
| torchvision | ≥ 0.17 | ResNet-18 with ImageNet starting weights | A small, proven image network that trains in minutes on a free T4 |
| NumPy | ≥ 1.24 | Arrays, geometry (width, depth, ellipse) | Fast whole-array maths |
| pandas | ≥ 2.0 | BodyM tables, per-person scoring, reports | Joins and group-by for per-person and sex × BMI results |
| scikit-learn | ≥ 1.3 | Baseline (linear regression), ellipse correction, person-grouped cross-validation | Simple, explainable models; `GroupKFold` keeps a person's photos in one fold |
| Pillow | ≥ 10.0 | Reading silhouette PNGs; refusing oversized images | Small, safe image reader |
| tqdm | ≥ 4.66 | Progress bars in Colab | Long steps show progress instead of looking frozen |
| SciPy (`scipy.ndimage`) | installed with scikit-learn | S7a: connected parts, hole filling and outlines of a photo's body mask | Already present (scikit-learn requires it); no new install |
| torchvision DeepLabV3-ResNet101 weights | with torchvision | S7a tool A: person segmentation | No new package; weights (~170 MB) download once into torch's cache (D-043) |
| rembg + onnxruntime (optional, `.[photos]`) | ≥ 2.0.50 / ≥ 1.16 | S7a tool B: person segmentation (`u2net_human_seg`) | Made for cutting people out; MIT licences; tested against tool A on Ryan's photos (D-043) |

Ranges, not exact pins: Colab ships its own PyTorch build, and forcing another wastes minutes per session.

## Development tools

| Tool | Used for |
| --- | --- |
| pytest | Unit and end-to-end tests on a fake, BodyM-shaped dataset (CPU, about a minute) |
| pytest-cov | Test coverage; CI fails below 80% (currently about 97%) |
| ruff | Linting, import order and security rules (`S` rules) |

## Compute, storage and data

| Piece | Used for | Notes |
| --- | --- | --- |
| Google Colab, T4 GPU | Training and evaluation | Free tier; a 512×256 run takes about 20 minutes |
| Google Drive | BodyM archive, image and geometry caches, checkpoints, run reports | Never the repository (policy: data is never committed) |
| BodyM (AWS Open Data) | Training and testing | CC BY-NC 4.0: research only; see [../docs/DATA.md](../docs/DATA.md) |
| File formats | `.npy` / `.npz` caches (loaded without pickle); `.pt` checkpoints (loaded with `weights_only=True`); JSON reports | Nothing that can run code is ever loaded from disk |

## Automation

| Piece | Checks |
| --- | --- |
| GitHub Actions (`.github/workflows/ci.yml`) | ruff; pytest with coverage (fails under 80%); notebooks committed without outputs; gitleaks secret scan |
| Pull request template | What changed, tests, security and licence checks, real results only, what Ryan decides |

## Planned, not yet used

| Piece | Step | Notes |
| --- | --- | --- |
| SMPL-X | S5 (armholes, neck, back length) | Research-only licence from the Max Planck Institute; files never committed |
| Ghanaian data from the AnthropometryDC app | S6 | Separate repository; its database is never read from here (policy) |
