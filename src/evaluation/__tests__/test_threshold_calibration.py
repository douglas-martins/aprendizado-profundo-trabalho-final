from types import SimpleNamespace

import pytest

import evaluation.threshold_calibration as threshold_calibration
from evaluation.threshold_calibration import (
    RUN_NAMES,
    latest_finished_run_id,
    raw_precision_recall_points_from_artifact,
)


class TestRawPrecisionRecallPointsFromArtifact:
    def test_loads_the_requested_run_artifact(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            threshold_calibration.mlflow.artifacts,
            "load_dict",
            lambda uri: calls.append(uri) or {"precision_recall_curve": [["1", "0.5"]]},
        )

        assert threshold_calibration.raw_precision_recall_points("run-1", split="val") == [
            (1.0, 0.5)
        ]
        assert threshold_calibration.raw_precision_recall_points("run-1") == [(1.0, 0.5)]
        assert calls == [
            "runs:/run-1/val_precision_recall_curve.json",
            "runs:/run-1/test_precision_recall_curve.json",
        ]

    def test_converts_json_lists_to_float_tuples_without_reordering(self):

        artifact = {"precision_recall_curve": [[1.0, 0.5], [0.5, 0.5], [0.0, 0.0]]}

        points = raw_precision_recall_points_from_artifact(artifact)

        assert points == [(1.0, 0.5), (0.5, 0.5), (0.0, 0.0)]
        assert all(isinstance(point, tuple) for point in points)


def _run(run_id, name, *, status="FINISHED", start_time=0, **tags):
    """The slice of an MLflow `Run` that `latest_finished_run_id` reads."""
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id, status=status, start_time=start_time),
        data=SimpleNamespace(tags={"mlflow.runName": name, **tags}),
    )


class TestLatestFinishedRunId:
    def test_picks_the_most_recent_finished_run_of_that_name(self):
        runs = [
            _run("old", "E2-raw-full-eval", start_time=100),
            _run("new", "E2-raw-full-eval", start_time=300),
            _run("mid", "E2-raw-full-eval", start_time=200),
        ]

        assert latest_finished_run_id(runs, "E2-raw-full-eval") == "new"

    def test_ignores_other_run_names(self):
        runs = [
            _run("other", "E3-raw-full-eval", start_time=900),
            _run("mine", "E2-raw-full-eval", start_time=100),
        ]

        assert latest_finished_run_id(runs, "E2-raw-full-eval") == "mine"

    @pytest.mark.parametrize("status", ["RUNNING", "FAILED", "KILLED", "SCHEDULED"])
    def test_a_run_that_did_not_finish_is_never_picked(self, status):
        runs = [
            _run("good", "E2-raw-full-eval", start_time=100),
            _run("bad", "E2-raw-full-eval", status=status, start_time=999),
        ]

        assert latest_finished_run_id(runs, "E2-raw-full-eval") == "good"

    def test_a_superseded_run_is_never_picked(self):
        runs = [
            _run("current", "E2-raw-full-eval", start_time=100),
            _run("stale", "E2-raw-full-eval", start_time=999, superseded="true"),
        ]

        assert latest_finished_run_id(runs, "E2-raw-full-eval") == "current"

    def test_a_throwaway_run_is_never_picked(self):
        runs = [
            _run("current", "E2-raw-full-eval", start_time=100),
            _run("scratch", "E2-raw-full-eval", start_time=999, throwaway="true"),
        ]

        assert latest_finished_run_id(runs, "E2-raw-full-eval") == "current"

    def test_a_tag_that_is_not_true_does_not_exclude_a_run(self):
        runs = [_run("kept", "E2-raw-full-eval", start_time=100, superseded="false")]

        assert latest_finished_run_id(runs, "E2-raw-full-eval") == "kept"

    def test_no_match_at_all_is_an_error_naming_the_run(self):
        with pytest.raises(ValueError, match="no run named 'E9-eval' found"):
            latest_finished_run_id([_run("a", "E2-eval")], "E9-eval")

    def test_only_unusable_matches_says_how_many_were_excluded(self):
        runs = [
            _run("a", "E2-eval", status="FAILED"),
            _run("b", "E2-eval", superseded="true"),
        ]

        with pytest.raises(
            ValueError, match="2 run.*named 'E2-eval'.*not FINISHED or were superseded"
        ):
            latest_finished_run_id(runs, "E2-eval")


class TestRunNames:
    def test_r3_runs_use_the_plain_names_the_legacy_evaluation_targets_produce(self):

        assert "E2-raw-full-eval" in RUN_NAMES
        assert "E3-raw-full-eval" in RUN_NAMES
        assert not any(name.endswith("-cuda") or name.endswith("-cpu") for name in RUN_NAMES)

    def test_lists_the_eleven_legacy_evaluation_runs(self):
        assert len(RUN_NAMES) == 11
        assert len(set(RUN_NAMES)) == 11


class TestFindLatestRunId:
    def test_searches_the_experiment_and_delegates_to_the_finished_run_filter(self):
        runs = [_run("bad", "E2-eval", status="FAILED", start_time=9), _run("ok", "E2-eval")]

        class _Client:
            def search_runs(self, experiment_ids):
                assert experiment_ids == ["7"]
                return runs

        assert threshold_calibration.find_latest_run_id(_Client(), "7", "E2-eval") == "ok"
