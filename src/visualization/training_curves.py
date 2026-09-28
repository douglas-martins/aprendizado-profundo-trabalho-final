"""Training-history figure: `val_loss` and `val_f1` per
epoch for every trained configuration, one column per tier (`mini`, R2, R3).

The stored runs have tables of best epochs but no curve. This is also the only overfitting
evidence available: the *training* loss was never logged to `mlflow.db`, so validation curves
plus the early-stopping point are what there is.

`per_epoch_series` and `load_training_curve` are the tested logic. The loader refuses a run
whose scalars disagree with its curves (a `best_epoch` that is not a logged epoch, an
`epochs_run` different from the last epoch, `val_loss`/`val_f1` over different epochs)
instead of plotting a marker in the wrong place. Runs extended by an exact resume log
`best_epoch`/`epochs_run` twice; `figure_common.latest_metric` takes the final one.
`plot_training_curves` (matplotlib) is thin glue, exercised by running `main()` and
inspecting `figures/training_curves.png`, the same split as `pr_curve_plots.py`.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from visualization.figure_common import (
    ARCHITECTURE_COLORS,
    ARCHITECTURE_MARKERS,
    FIGURES_DIR,
    IEEE_FULL_WIDTH_INCHES,
    TIER_LABELS,
    apply_ieee_style,
    format_general_ptbr,
    latest_metric,
    open_client,
    resolve_run_id,
)

TRAINING_RUN_NAMES = {
    "mini": {"E1": "E1-mini", "E2": "E2-mini", "E3": "E3-mini"},
    "r2": {"E1": "E1-r2", "E2": "E2-r2", "E3": "E3-r2"},
    "raw-full": {"E2": "E2-raw-full", "E3": "E3-raw-full"},
}


_F1_AXIS_LIMIT = 1.0
_MARKERS_PER_LINE = 8


@dataclass(frozen=True)
class TrainingCurve:
    """One run's validation curves and where its training ended."""

    epochs: list[int]
    val_loss: list[float]
    val_f1: list[float]
    best_epoch: int
    epochs_run: int
    stopped_early: bool


def per_epoch_series(history) -> tuple[list[int], list[float]]:
    """`(epochs, values)` ascending by epoch from MLflow `Metric` objects.

    An epoch logged more than once (redone after a fault) appears once, with the value of its
    latest timestamp.
    """
    latest_by_epoch = {}
    for metric in sorted(history, key=lambda metric: metric.timestamp):
        latest_by_epoch[metric.step] = metric.value
    epochs = sorted(latest_by_epoch)
    return epochs, [latest_by_epoch[epoch] for epoch in epochs]


def load_training_curve(client, run_id: str) -> TrainingCurve:
    """Read `run_id`'s validation curves and end-of-training scalars, checking they agree."""
    epochs, val_loss = per_epoch_series(client.get_metric_history(run_id, "val_loss"))
    val_f1_epochs, val_f1 = per_epoch_series(client.get_metric_history(run_id, "val_f1"))
    if not epochs:
        raise ValueError(f"run {run_id} logged no val_loss history")
    if val_f1_epochs != epochs:
        raise ValueError(f"run {run_id}: val_loss and val_f1 cover different epochs")

    best_epoch = int(latest_metric(client, run_id, "best_epoch"))
    epochs_run = int(latest_metric(client, run_id, "epochs_run"))
    if best_epoch not in epochs:
        raise ValueError(f"run {run_id}: best_epoch {best_epoch} is not a logged epoch")
    if epochs_run != epochs[-1]:
        raise ValueError(
            f"run {run_id}: epochs_run {epochs_run} differs from the last logged epoch {epochs[-1]}"
        )
    return TrainingCurve(
        epochs=epochs,
        val_loss=val_loss,
        val_f1=val_f1,
        best_epoch=best_epoch,
        epochs_run=epochs_run,
        stopped_early=bool(latest_metric(client, run_id, "stopped_early")),
    )


def plot_training_curves(curves: dict[str, dict[str, TrainingCurve]], out_path: Path) -> None:
    """Save the 2 x 3 figure: rows `val_loss` (log) and `val_f1`, columns the tiers in `curves`."""
    apply_ieee_style()
    tiers = list(curves)
    fig, axes = plt.subplots(
        2, len(tiers), figsize=(IEEE_FULL_WIDTH_INCHES, 3.9), sharex="col", squeeze=False
    )
    for column, tier in enumerate(tiers):
        loss_axis, f1_axis = axes[0][column], axes[1][column]
        for architecture, curve in curves[tier].items():
            color = ARCHITECTURE_COLORS[architecture]
            best_index = curve.epochs.index(curve.best_epoch)
            for axis, values in ((loss_axis, curve.val_loss), (f1_axis, curve.val_f1)):
                axis.plot(
                    curve.epochs,
                    values,
                    color=color,
                    linewidth=0.9,
                    marker=ARCHITECTURE_MARKERS[architecture],
                    markersize=2.5,
                    markevery=max(1, len(curve.epochs) // _MARKERS_PER_LINE),
                )
                axis.plot(
                    curve.best_epoch,
                    values[best_index],
                    marker="*",
                    markersize=7,
                    color=color,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    linestyle="none",
                )
                if curve.stopped_early:
                    axis.plot(
                        curve.epochs_run,
                        values[-1],
                        marker="x",
                        markersize=5,
                        color=color,
                        linestyle="none",
                    )
        loss_axis.set_yscale("log")
        loss_axis.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 3.0, 5.0), numticks=12))
        loss_axis.yaxis.set_minor_formatter(NullFormatter())
        f1_axis.set_ylim(0, _F1_AXIS_LIMIT)
        loss_axis.set_title(f"({chr(ord('a') + column)}) {TIER_LABELS[tier]}")
        f1_axis.set_xlabel("Época")
        for axis in (loss_axis, f1_axis):
            axis.grid(alpha=0.3, linewidth=0.4)
            axis.yaxis.set_major_formatter(FuncFormatter(lambda v, _: format_general_ptbr(v)))
    axes[0][0].set_ylabel("Perda de validação (escala logarítmica)")
    axes[1][0].set_ylabel("F1 de validação")

    handles = [
        Line2D([], [], color=ARCHITECTURE_COLORS[a], marker=ARCHITECTURE_MARKERS[a], label=a)
        for a in ARCHITECTURE_COLORS
    ]
    handles += [
        Line2D(
            [],
            [],
            color="0.5",
            marker="*",
            markersize=7,
            markeredgecolor="black",
            markeredgewidth=0.5,
            linestyle="none",
            label="melhor época",
        ),
        Line2D(
            [],
            [],
            color="0.5",
            marker="x",
            markersize=5,
            linestyle="none",
            label="fim do treino (parada antecipada)",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    """Read the eight final training runs from `mlflow.db` and write `training_curves.png`."""
    client = open_client()
    curves = {
        tier: {
            architecture: load_training_curve(client, resolve_run_id(client, run_name))
            for architecture, run_name in by_architecture.items()
        }
        for tier, by_architecture in TRAINING_RUN_NAMES.items()
    }
    out_path = FIGURES_DIR / "training_curves.png"
    plot_training_curves(curves, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
