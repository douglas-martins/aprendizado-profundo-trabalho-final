"""**Oracle** threshold calibration for this project's already-logged
evaluation runs (follow-up to PR-curve/threshold-calibration
finding: raw-tier models are recall-heavy at the fixed 0.5 threshold because
`pos_weight` scales with tier -- already names this and says PR-AUC/curves
should lead the raw-tier comparison).

**Read this before citing any number from here.** The F1-optimal threshold is
picked from the *test* split's own precision-recall curve and then scored on that
same split, so the result is an optimistic upper bound (an oracle), not an
operating point a deployed model could actually use. A threshold chosen on the
validation split and applied to test once is `threshold_selection.py`'s job; the
difference between the two is the optimism this module bakes in. Every metric
this logs keeps its `test_calibrated_*` name for traceability to older run ids,
and each run also gets the tag `test_calibrated_is_oracle=true`.

This reports the oracle operating point per run *in addition to* the
fixed-0.5 metrics, at zero retraining cost: every `evaluate.py` run already
logs a full 101-point precision-recall sweep as a
`{split}_precision_recall_curve.json` artifact, so the calibrated point is
just a different read of data already on disk.

Thin glue over already-tested logic (`metrics.py`'s `best_f1_operating_point`)
-- exercised by running `main()` for real against existing runs, not deep unit
coverage, matching this project's established "thin glue vs. tested logic"
pattern (`train.py`, `evaluate.py`, `pr_curve_plots.py`).
"""

from pathlib import Path

import mlflow
import torch

from evaluation.metrics import best_f1_operating_point

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MLFLOW_TRACKING_URI = f"sqlite:///{_PROJECT_ROOT / 'mlflow.db'}"
_MLFLOW_EXPERIMENT = "dl-final-project"


_DEFAULT_THRESHOLDS = torch.linspace(0.0, 1.0, steps=101).tolist()


RUN_NAMES = [
    "E1-mini-eval",
    "E2-mini-eval",
    "E3-mini-eval",
    "E1-mini-on-raw-full-eval",
    "E2-mini-on-raw-full-eval",
    "E3-mini-on-raw-full-eval",
    "E1-r2-eval",
    "E2-r2-eval",
    "E3-r2-eval",
    "E2-raw-full-eval",
    "E3-raw-full-eval",
]


def raw_precision_recall_points_from_artifact(artifact: dict) -> list[tuple[float, float]]:
    """Extract `(recall, precision)` points from a loaded PR-curve JSON artifact, unsorted.

    Deliberately *not* sorted by recall (unlike `pr_curve_plots.py`'s own
    `curve_points_from_artifact`, built for plotting) -- calibration needs
    each point paired positionally with `_DEFAULT_THRESHOLDS`, which is only
    true in the artifact's original, as-logged sweep order.
    """
    return [
        (float(recall), float(precision))
        for recall, precision in artifact["precision_recall_curve"]
    ]


def raw_precision_recall_points(run_id: str, split: str = "test") -> list[tuple[float, float]]:
    """Load `run_id`'s `{split}_precision_recall_curve.json` artifact, unsorted."""
    artifact = mlflow.artifacts.load_dict(f"runs:/{run_id}/{split}_precision_recall_curve.json")
    return raw_precision_recall_points_from_artifact(artifact)


def latest_finished_run_id(runs, run_name: str) -> str:
    """The newest usable run named `run_name` among `runs` (MLflow `Run` objects).

    Usable means `FINISHED` and not tagged `superseded=true` or `throwaway=true`: a run
    name is reused every time a configuration is re-evaluated or retrained, and picking
    an orphan RUNNING/FAILED attempt or a superseded pre-retrain run by name alone would
    silently feed stale numbers into a table. Newest is by `start_time`, not by the order
    the backend happens to return.
    """
    named = [run for run in runs if run.data.tags.get("mlflow.runName") == run_name]
    usable = [
        run
        for run in named
        if run.info.status == "FINISHED"
        and run.data.tags.get("superseded") != "true"
        and run.data.tags.get("throwaway") != "true"
    ]
    if not usable:
        if named:
            raise ValueError(
                f"{len(named)} run(s) named {run_name!r} exist but were not FINISHED "
                "or were superseded/throwaway"
            )
        raise ValueError(f"no run named {run_name!r} found")
    return max(usable, key=lambda run: run.info.start_time).info.run_id


def find_latest_run_id(client: mlflow.MlflowClient, experiment_id: str, run_name: str) -> str:
    """`latest_finished_run_id` over the runs of `experiment_id` (soft-deleted runs excluded)."""
    return latest_finished_run_id(client.search_runs([experiment_id]), run_name)


def main() -> None:
    """Compute and log the **oracle** F1-optimal operating point for every run in `RUN_NAMES`.

    Logs `{split}_calibrated_threshold/precision/recall/f1` onto the *same*
    run id the fixed-0.5 numbers already live on (via `MlflowClient.log_metric`,
    not a new `mlflow.start_run()` context) -- so each configuration retains one
    `run_id` for both the fixed and calibrated numbers, preserving traceability.
    """
    mlflow.set_tracking_uri(_MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(_MLFLOW_EXPERIMENT)

    for run_name in RUN_NAMES:
        run_id = find_latest_run_id(client, experiment.experiment_id, run_name)
        points = raw_precision_recall_points(run_id, split="test")
        calibrated = best_f1_operating_point(points, _DEFAULT_THRESHOLDS)

        client.log_metric(run_id, "test_calibrated_threshold", calibrated["threshold"])
        client.log_metric(run_id, "test_calibrated_precision", calibrated["precision"])
        client.log_metric(run_id, "test_calibrated_recall", calibrated["recall"])
        client.log_metric(run_id, "test_calibrated_f1", calibrated["f1"])
        client.set_tag(run_id, "test_calibrated_is_oracle", "true")

        fixed_f1 = client.get_run(run_id).data.metrics.get("test_f1")
        print(
            f"{run_name} ({run_id}): fixed@0.5 f1={fixed_f1:.4f} -> "
            f"ORACLE (test-picked threshold)@{calibrated['threshold']:.2f} "
            f"precision={calibrated['precision']:.4f} recall={calibrated['recall']:.4f} "
            f"f1={calibrated['f1']:.4f}"
        )


if __name__ == "__main__":
    main()
