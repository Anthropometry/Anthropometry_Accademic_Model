"""S7a — person segmentation: a phone photo in, a body-probability map out.

specs/s7a-photo-silhouette/spec.md (approved by Ryan, 25 Nov 2026; tools: D-043).

What it does
------------
One interface, `Segmenter`: `seg(rgb) -> prob`, where `rgb` is an H x W x 3 uint8 photo and `prob` an
H x W float array in [0, 1] (1 = body). Two tools behind it, tested side by side on Ryan's photos:

* **A: `torchvision`** DeepLabV3-ResNet101, COCO "person" class. No new package; the weights (~170 MB)
  download on first use into torch's cache (put `TORCH_HOME` on Drive to keep them between sessions).
* **B: `rembg`** with its `u2net_human_seg` model, made for cutting people out. Needs `rembg` and
  `onnxruntime` (`pip install -e ".[photos]"`); the weights download on first use.

Why these decisions
-------------------
* One interface: the rest of the pipeline (cleaning, checks, BodyM-like silhouette, prediction) is the
  same for every tool, so the comparison measures the tool alone; a third tool is one class.
* Imports happen inside each class: CI and people without the optional packages can still use the rest.
* The probability is kept (not just a yes/no mask), so the saved silhouette keeps soft edges like BodyM's;
  the hybrid model's edge-softness feature (S3b) was tuned on those.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Segmenter(Protocol):
    name: str

    def version(self) -> str: ...

    def __call__(self, rgb: np.ndarray) -> np.ndarray: ...


SEG_SHORT_SIDE = 1024   # px, short side of the photo as seen by DeepLab (was 520)


class TorchvisionSegmenter:
    """A: DeepLabV3-ResNet101 (torchvision), probability of COCO class 15 'person'."""

    name = "torchvision-deeplabv3"
    PERSON = 15

    def __init__(self, device: str | None = None):
        import torch
        from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights, deeplabv3_resnet101
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = DeepLabV3_ResNet101_Weights.DEFAULT
        self.model = deeplabv3_resnet101(weights=self.weights).eval().to(self.device)
        # The weights' own transform shrinks the photo to 520 px on the short side; DeepLab then works on a
        # grid 8x coarser still, and a narrow gap between arm and waist closes, so the arm merges into the
        # torso outline (D-054, seen on real photos). Run at SEG_SHORT_SIDE instead, normalised the same way.
        self.prep = self.weights.transforms(resize_size=SEG_SHORT_SIDE)

    def version(self) -> str:
        import torchvision
        return f"torchvision {torchvision.__version__}, {self.weights}"

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        torch = self._torch
        x = torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1)   # writable copy
        h, w = rgb.shape[:2]
        with torch.no_grad():
            out = self.model(self.prep(x)[None].to(self.device))["out"]
            out = torch.nn.functional.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
            prob = out.softmax(1)[0, self.PERSON]
        return prob.cpu().numpy().astype(np.float32)


class RembgSegmenter:
    """B: rembg with the u2net_human_seg model; its alpha matte is the body probability."""

    name = "rembg-u2net_human_seg"

    def __init__(self, model: str = "u2net_human_seg"):
        try:
            import rembg
        except ImportError as e:        # optional dependency (D-043)
            raise ImportError('rembg is not installed: pip install -e ".[photos]"') from e
        self._rembg = rembg
        self.model = model
        self.session = rembg.new_session(model)

    def version(self) -> str:
        return f"rembg {getattr(self._rembg, '__version__', '?')}, {self.model}"

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        from PIL import Image
        alpha = self._rembg.remove(Image.fromarray(rgb), session=self.session, only_mask=True)
        return np.asarray(alpha.convert("L"), np.float32) / 255.0


TOOLS = {"torchvision": TorchvisionSegmenter, "rembg": RembgSegmenter}


def get(name: str) -> Segmenter:
    if name not in TOOLS:
        raise ValueError(f"unknown segmentation tool {name!r}; choose {', '.join(TOOLS)}")
    return TOOLS[name]()
