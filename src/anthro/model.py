"""The BMnet-style network (specs/S3-photo-model.md, approved 15 Nov 2026).

BMnet itself was never released (searched GitHub, Hugging Face, Papers with Code and Amazon
Science, 14 Nov 2026; DISCUSSION D-005), so we rebuild the idea, not the exact network:

* Backbone: ResNet-18 pre-trained on ImageNet.
  Why: ~2,000 training people is small for learning image features from zero; a pre-trained
  network already detects edges and shapes, so it trains faster and more reliably. ResNet-18
  is small enough (11M parameters) for a free Colab T4 GPU.
* First layer widened from 3 to 5 channels (front, side, height, weight, sex). Its new
  weights start as the average of the pre-trained colour weights, so the pre-training is
  kept rather than thrown away.
* Last layer replaced: 14 outputs (one per measurement) instead of 1,000 ImageNet classes.

S3b hybrid (D-019, approved 17 Nov 2026): with `n_geo > 0` the ResNet's 512 pooled image
features are joined with the geometric features (width, depth, ellipse and corrected perimeter
per level, quality flags) and a new final layer maps all of them to the 14 outputs. Why join
them there and not as extra image channels (how height and weight enter): 25 numbers spread over
whole image channels is wasteful, and at the end the network can learn directly how much to trust
the geometry for each measurement. With `n_geo = 0` the network is exactly run r1's, so old
checkpoints still load.
"""

from __future__ import annotations

import torch
import torchvision

from .config import MEASUREMENTS

IN_CHANNELS = 5


def _resnet(pretrained: bool) -> torch.nn.Module:
    weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
    net = torchvision.models.resnet18(weights=weights)
    old = net.conv1
    net.conv1 = torch.nn.Conv2d(IN_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        net.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True).repeat(1, IN_CHANNELS, 1, 1))
    net.fc = torch.nn.Linear(net.fc.in_features, len(MEASUREMENTS))
    return net


class Hybrid(torch.nn.Module):
    def __init__(self, pretrained: bool, n_geo: int):
        super().__init__()
        self.backbone = _resnet(pretrained)
        n_img = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Identity()
        self.head = torch.nn.Linear(n_img + n_geo, len(MEASUREMENTS))

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.backbone(x), g.to(x.dtype)], 1))


def build(pretrained: bool = True, n_geo: int = 0) -> torch.nn.Module:
    return Hybrid(pretrained, n_geo) if n_geo > 0 else _resnet(pretrained)


def forward(net: torch.nn.Module, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """One call site for both kinds of network, so training and evaluation can't drift apart."""
    return net(x, g) if isinstance(net, Hybrid) else net(x)
