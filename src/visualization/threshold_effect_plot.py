"""Threshold-effect figure for the eight models scored on
the 342 real test scenes, the pooled F1 and the tile false-positive rate at the fixed 0.5
threshold against the threshold picked on the validation scenes.

Moving from 0.5 to the validation threshold raises the pooled F1 of the `raw` models by
0.16-0.27 and cuts the tile FPR from 83-94% to 27-60%, more than any architecture or tier
change at a fixed threshold. A paired plot makes this comparison explicit.

Every model is scored on the same scenes (`<arch>-<tier>-on-raw-full-scene-full_scene`
runs), so the models differ by architecture and training tier only, never by test set.

`threshold_effect_rows` and `load_scene_metrics` are the tested logic (a missing metric names
the model and the key instead of plotting a blank); `plot_threshold_effect` (matplotlib) is thin
glue, exercised by running `main()` and inspecting `figures/threshold_effect.png`.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from visualization.figure_common import (
    ARCHITECTURE_COLORS,
    ARCHITECTURE_MARKERS,
    FIGURES_DIR,
    IEEE_FULL_WIDTH_INCHES,
    apply_ieee_style,
    format_general_ptbr,
    latest_metric,
    open_client,
    resolve_run_id,
)

SCENE_RUN_NAMES = [
    ("E2 (R3)", "E2-raw-full-on-raw-full-scene-full_scene"),
    ("E3 (R3)", "E3-raw-full-on-raw-full-scene-full_scene"),
    ("E1 (R2)", "E1-r2-on-raw-full-scene-full_scene"),
    ("E2 (R2)", "E2-r2-on-raw-full-scene-full_scene"),
    ("E3 (R2)", "E3-r2-on-raw-full-scene-full_scene"),
    ("E1 (mini)", "E1-mini-on-raw-full-scene-full_scene"),
    ("E2 (mini)", "E2-mini-on-raw-full-scene-full_scene"),
    ("E3 (mini)", "E3-mini-on-raw-full-scene-full_scene"),
]


_METRIC_KEYS = {
    "f1_fixed": "at_0p5_all_scenes_f1",
    "f1_validation": "at_val_pick_all_scenes_f1",
    "fpr_fixed": "at_0p5_tile_fpr",
    "fpr_validation": "at_val_pick_tile_fpr",
}


@dataclass(frozen=True)
class ThresholdEffect:
    """One model's pooled F1 and tile FPR at the fixed and at the validation threshold."""

    label: str
    f1_fixed: float
    f1_validation: float
    fpr_fixed: float
    fpr_validation: float

    @property
    def f1_gain(self) -> float:
        """Pooled F1 with the validation threshold minus pooled F1 at 0.5."""
        return self.f1_validation - self.f1_fixed

    @property
    def fpr_change(self) -> float:
        """Tile FPR with the validation threshold minus tile FPR at 0.5 (negative is better)."""
        return self.fpr_validation - self.fpr_fixed


def threshold_effect_rows(metrics_by_label: dict[str, dict[str, float]]) -> list[ThresholdEffect]:
    """One `ThresholdEffect` per model, in the order of `metrics_by_label`.

    Raises `KeyError` naming the model and the field when a metric is missing.
    """
    rows = []
    for label, metrics in metrics_by_label.items():
        values = {}
        for field in _METRIC_KEYS:
            if field not in metrics:
                raise KeyError(f"{label}: missing metric {field}")
            values[field] = metrics[field]
        rows.append(ThresholdEffect(label=label, **values))
    return rows


def load_scene_metrics(client, run_id: str) -> dict[str, float]:
    """The four whole-scene metrics of `run_id`, keyed by `ThresholdEffect` field name."""
    return {field: latest_metric(client, run_id, key) for field, key in _METRIC_KEYS.items()}


def plot_threshold_effect(rows: list[ThresholdEffect], out_path: Path) -> None:
    """Save the paired plot: hollow marker at the fixed 0.5 threshold, filled at the validation
    threshold."""
    apply_ieee_style()
    fig, (f1_axis, fpr_axis) = plt.subplots(
        1, 2, figsize=(IEEE_FULL_WIDTH_INCHES, 2.9), sharey=True
    )
    positions = list(range(len(rows)))[::-1]
    for position, row in zip(positions, rows, strict=True):
        architecture = row.label.split(" ")[0]
        color = ARCHITECTURE_COLORS[architecture]
        marker = ARCHITECTURE_MARKERS[architecture]
        for axis, fixed, validation in (
            (f1_axis, row.f1_fixed, row.f1_validation),
            (fpr_axis, row.fpr_fixed, row.fpr_validation),
        ):
            axis.annotate(
                "",
                xy=(validation, position),
                xytext=(fixed, position),
                arrowprops={"arrowstyle": "->", "color": "0.45", "linewidth": 0.8},
            )
            axis.plot(fixed, position, marker=marker, markerfacecolor="white", color=color)
            axis.plot(validation, position, marker=marker, color=color, linestyle="none")
    f1_axis.set_yticks(positions, [row.label for row in rows])
    f1_axis.set_xlabel("F1 agrupado (342 cenas de teste)")
    fpr_axis.set_xlabel("Taxa de falsos positivos por tile (cenas sem pluma)")
    for axis in (f1_axis, fpr_axis):
        axis.set_xlim(0, 1)
        axis.grid(axis="x", alpha=0.3, linewidth=0.4)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda v, _: format_general_ptbr(v)))
    handles = [
        Line2D(
            [],
            [],
            color="0.3",
            marker="o",
            markerfacecolor="white",
            linestyle="none",
            label="limiar 0,5 fixo",
        ),
        Line2D(
            [], [], color="0.3", marker="o", linestyle="none", label="limiar escolhido na validação"
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    """Read the eight whole-scene runs from `mlflow.db` and write `threshold_effect.png`."""
    client = open_client()
    metrics_by_label = {
        label: load_scene_metrics(client, resolve_run_id(client, run_name))
        for label, run_name in SCENE_RUN_NAMES
    }
    rows = threshold_effect_rows(metrics_by_label)
    for row in rows:
        print(
            f"{row.label:10s} F1 {row.f1_fixed:.4f} -> {row.f1_validation:.4f} "
            f"({row.f1_gain:+.4f})  FPR {row.fpr_fixed:.4f} -> {row.fpr_validation:.4f}"
        )
    out_path = FIGURES_DIR / "threshold_effect.png"
    plot_threshold_effect(rows, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
