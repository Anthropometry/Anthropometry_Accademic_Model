"""Shared constants and training settings.

Why one file: the measurement list, tolerance and input limits are used by data loading,
training, evaluation and (later) the demo. Defining them once means a change cannot leave
two modules disagreeing about, say, which 14 measurements exist.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# The 14 BodyM measurements, in the column order of measurements.csv.
# All in cm, computed by BodyM on 3D scans registered to the SMPL mesh — a "virtual tape",
# not a tailor's tape. Step 6 measures and corrects the gap to real tape (DISCUSSION D-009).
MEASUREMENTS: tuple[str, ...] = (
    "ankle", "arm-length", "bicep", "calf", "chest", "forearm", "height",
    "hip", "leg-length", "shoulder-breadth", "shoulder-to-crotch", "thigh", "waist", "wrist",
)

# Supervisor's target: predictions within about ±1.5 cm (DISCUSSION D-002).
TOLERANCE_CM = 1.5

# Input columns as they appear in hwg_metadata.csv.
HEIGHT_COL, WEIGHT_COL, SEX_COL, SUBJECT_COL, PHOTO_COL = (
    "height_cm", "weight_kg", "gender", "subject_id", "photo_id",
)

# Hard limits: values outside these are rejected as typing or sensor errors.
HEIGHT_LIMITS_CM = (100.0, 230.0)
WEIGHT_LIMITS_KG = (20.0, 250.0)

# BodyM's training range (from the step-1 exploration). Inside the hard limits but outside
# this range the model is extrapolating, so predictions are flagged, not trusted silently.
HEIGHT_TRAIN_RANGE_CM = (141.0, 197.2)
WEIGHT_TRAIN_RANGE_KG = (28.7, 184.5)


@dataclass
class TrainConfig:
    """Every setting that changes the trained model. Saved inside each checkpoint so any
    result can be traced back to exactly how it was produced."""

    img_h: int = 256          # tall, narrow frame for a standing person (2:1)
    img_w: int = 128
    epochs: int = 25
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    val_fraction: float = 0.1  # share of training PEOPLE held back to choose the best epoch
    pretrained: bool = True    # start from ImageNet weights: faster, more stable on ~2k people
    augment: bool = True
    seed: int = 0
    num_workers: int = 2

    def to_dict(self) -> dict:
        return asdict(self)
