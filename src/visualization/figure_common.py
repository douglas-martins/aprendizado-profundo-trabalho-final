"""Shared MLflow and plotting helpers for generated figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow

from evaluation.threshold_calibration import find_latest_run_id

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "figures"
TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
EXPERIMENT_NAME = "dl-final-project"


IEEE_COLUMN_WIDTH_INCHES = 3.5
IEEE_FULL_WIDTH_INCHES = 7.16


ARCHITECTURE_COLORS = {"E1": "#0072B2", "E2": "#D55E00", "E3": "#009E73"}
ARCHITECTURE_MARKERS = {"E1": "o", "E2": "s", "E3": "^"}


TIER_LABELS = {"mini": "mini", "r2": "R2", "raw-full": "R3"}
TIER_COLORS = {"mini": "#E69F00", "r2": "#56B4E9", "raw-full": "#CC79A7"}
TIER_MARKERS = {"mini": "o", "r2": "s", "raw-full": "^"}


def open_client(tracking_uri: str = TRACKING_URI) -> mlflow.MlflowClient:
    """An `MlflowClient` on this project's own SQLite store (never the repo-root `mlruns/`)."""
    return mlflow.MlflowClient(tracking_uri=tracking_uri)


def final_metric_value(history) -> float:
    """The last logged value in `history` (MLflow `Metric` objects with step/value/timestamp).

    Latest timestamp wins, and a timestamp tie goes to the larger step -- the same rule
    MLflow's own `run.data.metrics` applies, made explicit so it does not depend on the order
    a backend happens to return the history in. Raises `ValueError` on an empty history: a
    figure silently missing a value is worse than a crash.
    """
    if not history:
        raise ValueError("metric history is empty: the run never logged this metric")
    return max(history, key=lambda metric: (metric.timestamp, metric.step)).value


def latest_metric(client: mlflow.MlflowClient, run_id: str, key: str) -> float:
    """The final logged value of metric `key` on `run_id` (see `final_metric_value`)."""
    return final_metric_value(client.get_metric_history(run_id, key))


def resolve_run_id(client: mlflow.MlflowClient, run_name: str) -> str:
    """Run id of the newest FINISHED, non-superseded, non-throwaway run named `run_name`."""
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    return find_latest_run_id(client, experiment.experiment_id, run_name)


def format_decimal_ptbr(value: float, decimals: int) -> str:
    """`value` with `decimals` places and a decimal comma: `6,3`."""
    return f"{value:.{decimals}f}".replace(".", ",")


def format_general_ptbr(value: float) -> str:
    """`value` with only the decimals it needs and a decimal comma: `0,6`, `2`, `0,05`.

    For axis tick labels, where a fixed number of decimals would print `2,0` next to `0,5`.
    """
    return f"{value:g}".replace(".", ",")


def format_integer_ptbr(value: int) -> str:
    """`value` with a dot between thousands: `487.361`."""
    return f"{value:,}".replace(",", ".")


def apply_ieee_style() -> None:
    """8 pt Times-like serif text, as the IEEE template asks of figure labels."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.0,
            "savefig.dpi": 300,
        }
    )
