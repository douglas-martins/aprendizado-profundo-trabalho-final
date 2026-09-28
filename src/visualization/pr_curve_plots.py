"""Generate precision-recall comparison figures from recorded MLflow artifacts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow

from evaluation.metrics import sort_points_by_recall
from evaluation.threshold_calibration import latest_finished_run_id

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MLFLOW_TRACKING_URI = f"sqlite:///{_PROJECT_ROOT / 'mlflow.db'}"
_MLFLOW_EXPERIMENT = "dl-final-project"


def curve_points_from_artifact(artifact: dict) -> list[tuple[float, float]]:
    """Extract, sort, and clean `(recall, precision)` points from a loaded PR-curve JSON artifact.

    `evaluate.py` logs `{"precision_recall_curve": [[recall, precision], ...]}`
    in threshold order -- JSON round-trips each pair as a list, and plotting
    wants tuples sorted ascending by recall (matching `average_precision_from_sweep`'s
    own convention), so both conversions happen here in one place.

    The sweep's highest threshold(s) near 1.0 predict nothing by construction
    (`probs > threshold` is never true there): `metrics.py`'s
    `precision_recall_points_from_sweep` logs each as the synthetic pair
    `(recall=0.0, precision=0.0)` -- a zero-width anchor
    `average_precision_from_sweep` needs for its own math, but never an
    observed operating point. Once sorted, it lands first (recall=0 is the
    minimum); connecting it to the next point draws a straight line that is
    not an observed trajectory, so every leading `(0.0, 0.0)` is dropped
    here, plotting-only.
    """
    points = [
        (float(recall), float(precision))
        for recall, precision in artifact["precision_recall_curve"]
    ]
    sorted_points = sort_points_by_recall(points)
    while sorted_points and sorted_points[0] == (0.0, 0.0):
        sorted_points = sorted_points[1:]
    return sorted_points


def load_precision_recall_curve(run_name: str, split: str = "test") -> list[tuple[float, float]]:
    """Find the `mlflow.db` run named `run_name` and return its `split`'s PR curve.

    Uses the newest FINISHED run of that name that is not superseded/throwaway
    (`threshold_calibration.latest_finished_run_id`). Raises `ValueError` if there
    is none -- silently plotting an empty curve, or a stale run's, would be a worse
    failure than crashing here.
    """
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(_MLFLOW_EXPERIMENT)
    runs = client.search_runs([experiment.experiment_id])
    run_id = latest_finished_run_id(runs, run_name)
    artifact = mlflow.artifacts.load_dict(f"runs:/{run_id}/{split}_precision_recall_curve.json")
    return curve_points_from_artifact(artifact)


def plot_pr_curve_comparison(
    curves: dict[str, list[tuple[float, float]]], title: str, out_path: Path
) -> None:
    """Save a precision-recall line plot with one line per `curves` entry (label -> points)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, points in curves.items():
        recalls, precisions = zip(*points, strict=True)
        ax.plot(recalls, precisions, marker=".", markersize=3, label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precisão")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Generate the two precision-recall figures from already-recorded evaluation runs."""
    mlflow.set_tracking_uri(_MLFLOW_TRACKING_URI)
    figures_dir = _PROJECT_ROOT / "figures"

    within_tier_curves = {
        architecture: load_precision_recall_curve(f"{architecture}-mini-eval")
        for architecture in ["E1", "E2", "E3"]
    }
    plot_pr_curve_comparison(
        within_tier_curves,
        "E1 vs. E2 vs. E3 --- escala mini, split de teste",
        figures_dir / "pr_curve_within_tier_mini.png",
    )

    across_tier_curves = {
        "mini (cross-tier)": load_precision_recall_curve("E2-mini-on-raw-full-eval"),
        "R2": load_precision_recall_curve("E2-r2-eval"),
        "R3": load_precision_recall_curve("E2-raw-full-eval"),
    }
    plot_pr_curve_comparison(
        across_tier_curves,
        "E2 entre escalas --- split de teste de raw",
        figures_dir / "pr_curve_across_tiers_e2.png",
    )

    print(f"wrote {figures_dir / 'pr_curve_within_tier_mini.png'}")
    print(f"wrote {figures_dir / 'pr_curve_across_tiers_e2.png'}")


if __name__ == "__main__":
    main()
