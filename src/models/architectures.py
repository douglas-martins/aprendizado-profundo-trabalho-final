"""Define the three segmentation architectures used by the experiment."""

import segmentation_models_pytorch as smp
import torch
from torch import nn


class _ConvBlock(nn.Module):
    """Two 3x3 conv + BatchNorm + ReLU layers -- the plain building block E1 is made of."""

    def __init__(self, in_channels: int, out_channels: int):
        """Build the two-layer conv block."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply both conv layers."""
        return self.net(x)


class TinyUNet(nn.Module):
    """E1: a small, from-scratch U-Net -- no pretrained encoder, the course's baseline.

    `base=8` gives 487,361 parameters -- chosen for a three-way size
    spread against E2 (~6.6M) and E3 (~0.857M): E1 (small, unpretrained)
    vs. E3 (small, pretrained) isolates the pretraining effect at matched
    scale, while E1 vs. E2 isolates scale and pretraining combined.
    """

    def __init__(self, in_channels: int = 4, base: int = 8, out_channels: int = 1):
        """Build a 4-level U-Net with `base` channels at the first level."""
        super().__init__()
        self.enc1 = _ConvBlock(in_channels, base)
        self.enc2 = _ConvBlock(base, base * 2)
        self.enc3 = _ConvBlock(base * 2, base * 4)
        self.enc4 = _ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _ConvBlock(base * 8, base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size=2, stride=2)
        self.dec4 = _ConvBlock(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = _ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = _ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = _ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode with 4 downsampling stages, decode with skip connections, return raw logits."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bottleneck = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(bottleneck), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def build_e1() -> nn.Module:
    """Build E1: TinyUNet, base=8, no pretraining."""
    return TinyUNet(in_channels=4, base=8, out_channels=1)


def build_e2(pretrained: bool = True) -> nn.Module:
    """Build E2: U-Net + MobileNetV2, 4-channel input via smp's default first-conv adaptation.

    `smp`'s `patch_first_conv` cyclically reuses the pretrained 3-channel
    kernel weights across all 4 channels (the mag1c channel gets a copy of
    the first RGB channel's weights, rescaled) -- accepted as-is to preserve
    the encoder's configured input adaptation.
    """
    return smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet" if pretrained else None,
        in_channels=4,
        classes=1,
    )


def build_e3(pretrained: bool = True) -> nn.Module:
    """Build E3: LinkNet + MobileNetV3-small-minimal, reproduced from Herec et al. (2026).

    `tu-tf_mobilenetv3_small_minimal_100` bridges to `timm`'s exact
    "minimal" MobileNetV3-small variant the paper uses -- there is no
    native (non-`timm`) `mobilenet_v3` encoder in `segmentation-models-pytorch`.
    """
    return smp.Linknet(
        encoder_name="tu-tf_mobilenetv3_small_minimal_100",
        encoder_weights="imagenet" if pretrained else None,
        in_channels=4,
        classes=1,
    )
