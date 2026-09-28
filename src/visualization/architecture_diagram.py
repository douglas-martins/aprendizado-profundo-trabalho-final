"""Render the architecture overview for the three segmentation models."""

from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
from matplotlib.patches import FancyArrowPatch, Rectangle

from models.architectures import TinyUNet, build_e1, build_e2, build_e3
from visualization.figure_common import (
    ARCHITECTURE_COLORS,
    FIGURES_DIR,
    IEEE_FULL_WIDTH_INCHES,
    apply_ieee_style,
    format_integer_ptbr,
)

_IMAGE_SIZE = 128


_TITLES = {
    "E1": "E1: U-Net pequena, blocos convolucionais simples",
    "E2": "E2: U-Net com encoder MobileNetV2",
    "E3": "E3: LinkNet com encoder MobileNetV3-small",
}
_PRETRAINED = {"E1": False, "E2": True, "E3": True}
_SKIP_TEXT = {"concat": "saltos por concatenação", "soma": "saltos por soma"}


_ROW_HEIGHT = 2.5
_CENTER_OFFSET = 0.85
_BOX_WIDTH = 0.66
_MIN_BOX_HEIGHT = 0.16
_MAX_BOX_HEIGHT = 0.9
_SKIP_CURVATURE = 0.15


@dataclass(frozen=True)
class Stage:
    """One feature map in the network: channel count and side length in pixels."""

    channels: int
    size: int


@dataclass(frozen=True)
class NetworkPlan:
    """What to draw for one configuration.

    `down` are the encoder stages (for `TinyUNet` the last one is the bottleneck), `up` the
    decoder stages; `skips` pairs an index in `down` with an index in `up` of the same size.
    """

    architecture: str
    down: list[Stage]
    up: list[Stage]
    skips: list[tuple[int, int]]
    skip_kind: str
    parameters: int
    in_channels: int
    out_channels: int


def parameter_count(model: torch.nn.Module) -> int:
    """Total number of parameters of `model`."""
    return sum(parameter.numel() for parameter in model.parameters())


def _last_conv_channels(module: torch.nn.Module) -> int:
    """Output channels of the last convolution inside `module` (a block's own output width)."""
    return [m for m in module.modules() if isinstance(m, torch.nn.Conv2d)][-1].out_channels


def _skip_pairs(down_count: int) -> list[tuple[int, int]]:
    """Deepest encoder stage feeds the first decoder stage, and so on outwards.

    The deepest encoder stage (the last of `down`) is the decoder's input, not a skip.
    """
    return [(down_count - 2 - index, index) for index in range(down_count - 1)]


def network_plan(architecture: str, model: torch.nn.Module, image_size: int = _IMAGE_SIZE):
    """Read `model` into the `NetworkPlan` the diagram draws; `TypeError` for other models."""
    if isinstance(model, TinyUNet):
        down_blocks = [model.enc1, model.enc2, model.enc3, model.enc4, model.bottleneck]
        up_blocks = [model.dec4, model.dec3, model.dec2, model.dec1]
        down = [
            Stage(_last_conv_channels(block), image_size // 2**index)
            for index, block in enumerate(down_blocks)
        ]
        up = [
            Stage(_last_conv_channels(block), image_size // 2 ** (len(up_blocks) - 1 - index))
            for index, block in enumerate(up_blocks)
        ]
        return NetworkPlan(
            architecture=architecture,
            down=down,
            up=up,
            skips=_skip_pairs(len(down)),
            skip_kind="concat",
            parameters=parameter_count(model),
            in_channels=model.enc1.net[0].in_channels,
            out_channels=model.head.out_channels,
        )
    if isinstance(model, (smp.Unet, smp.Linknet)):
        encoder_channels = model.encoder.out_channels
        down = [
            Stage(channels, image_size // 2 ** (index + 1))
            for index, channels in enumerate(encoder_channels[1:])
        ]
        blocks = list(model.decoder.blocks)
        up = [
            Stage(_last_conv_channels(block), image_size // 2 ** (len(blocks) - 1 - index))
            for index, block in enumerate(blocks)
        ]
        return NetworkPlan(
            architecture=architecture,
            down=down,
            up=up,
            skips=_skip_pairs(len(down)),
            skip_kind="soma" if isinstance(model, smp.Linknet) else "concat",
            parameters=parameter_count(model),
            in_channels=encoder_channels[0],
            out_channels=model.segmentation_head[0].out_channels,
        )
    raise TypeError(f"{architecture}: cannot plan a {type(model).__name__}")


def plan_labels(plan: NetworkPlan) -> tuple[str, str]:
    """Title and subtitle printed above one configuration's row (pt-BR, "encoder"/"decoder")."""
    pretraining = (
        "encoder pré-treinado no ImageNet, 1.ª convolução de 3 para 4 canais"
        if _PRETRAINED[plan.architecture]
        else "sem pré-treino"
    )
    subtitle = (
        f"{format_integer_ptbr(plan.parameters)} parâmetros · {pretraining}"
        f" · {_SKIP_TEXT[plan.skip_kind]}"
    )
    return _TITLES[plan.architecture], subtitle


def _box_height(size: int, image_size: int) -> float:
    """Box height shrinking with the resolution, so the U shape is visible."""
    return _MIN_BOX_HEIGHT + (_MAX_BOX_HEIGHT - _MIN_BOX_HEIGHT) * sqrt(size / image_size)


def _draw_network(ax, plan: NetworkPlan, row_bottom: float, image_size: int) -> None:
    """Draw one configuration's row: title, boxes, arrows and skip arcs."""
    color = ARCHITECTURE_COLORS[plan.architecture]
    y_center = row_bottom + _CENTER_OFFSET
    slots = (
        [("io", Stage(plan.in_channels, image_size))]
        + [("down", stage) for stage in plan.down]
        + [("up", stage) for stage in plan.up]
        + [("io", Stage(plan.out_channels, image_size))]
    )
    fill = {"io": ("0.9", 1.0), "down": (color, 0.35), "up": (color, 0.65)}
    tops = []
    for x, (kind, stage) in enumerate(slots):
        height = _box_height(stage.size, image_size)
        face, alpha = fill[kind]
        ax.add_patch(
            Rectangle(
                (x - _BOX_WIDTH / 2, y_center - height / 2),
                _BOX_WIDTH,
                height,
                facecolor=face,
                alpha=alpha,
                edgecolor="black",
                linewidth=0.5,
            )
        )
        ax.text(x, y_center, str(stage.channels), ha="center", va="center", fontsize=6.5)
        ax.text(
            x,
            y_center - _MAX_BOX_HEIGHT / 2 - 0.13,
            f"{stage.size}²",
            ha="center",
            va="top",
            fontsize=6,
            color="0.3",
        )
        tops.append((x, y_center + height / 2))
        if x + 1 < len(slots):
            ax.add_patch(
                FancyArrowPatch(
                    (x + _BOX_WIDTH / 2, y_center),
                    (x + 1 - _BOX_WIDTH / 2, y_center),
                    arrowstyle="-|>",
                    mutation_scale=5,
                    color="0.25",
                    linewidth=0.6,
                )
            )
    for down_index, up_index in plan.skips:
        start = tops[1 + down_index]
        end = tops[1 + len(plan.down) + up_index]
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                connectionstyle=f"arc3,rad=-{_SKIP_CURVATURE}",
                arrowstyle="-|>",
                mutation_scale=5,
                linestyle="--",
                color="0.35",
                linewidth=0.6,
            )
        )
    title, subtitle = plan_labels(plan)
    ax.text(-0.45, row_bottom + 2.3, title, fontsize=8, fontweight="bold")
    ax.text(-0.45, row_bottom + 2.08, subtitle, fontsize=7, color="0.25")


def plot_architectures(plans: list[NetworkPlan], out_path: Path) -> None:
    """Save the stacked diagram, first plan on top."""
    apply_ieee_style()
    slot_count = max(len(plan.down) + len(plan.up) + 2 for plan in plans)
    fig, ax = plt.subplots(figsize=(IEEE_FULL_WIDTH_INCHES, 4.1))
    for row, plan in enumerate(plans):
        row_bottom = (len(plans) - 1 - row) * _ROW_HEIGHT
        _draw_network(ax, plan, row_bottom, _IMAGE_SIZE)
    ax.set_xlim(-0.6, slot_count - 0.4)
    ax.set_ylim(0, len(plans) * _ROW_HEIGHT)
    ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    """Build the three configurations untrained (no weight download) and draw them."""
    plans = [
        network_plan("E1", build_e1()),
        network_plan("E2", build_e2(pretrained=False)),
        network_plan("E3", build_e3(pretrained=False)),
    ]
    for plan in plans:
        print(
            f"{plan.architecture}: {plan.parameters:,} params, "
            f"down {[s.channels for s in plan.down]}, up {[s.channels for s in plan.up]}, "
            f"skips {plan.skip_kind}"
        )
    out_path = FIGURES_DIR / "architecture_overview.png"
    plot_architectures(plans, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
