import visualization.pr_curve_plots as pr_curve_plots
from visualization.pr_curve_plots import curve_points_from_artifact


class TestLoadPrecisionRecallCurve:
    def test_uses_the_default_test_artifact(self, monkeypatch):
        calls = []

        class Client:
            def get_experiment_by_name(self, name):
                assert name == "dl-final-project"
                return type("Experiment", (), {"experiment_id": "1"})()

            def search_runs(self, experiment_ids):
                assert experiment_ids == ["1"]
                return []

        selection_calls = []
        monkeypatch.setattr(pr_curve_plots.mlflow, "MlflowClient", Client)
        monkeypatch.setattr(
            pr_curve_plots,
            "latest_finished_run_id",
            lambda runs, name: selection_calls.append((runs, name)) or "run-1",
        )
        monkeypatch.setattr(
            pr_curve_plots.mlflow.artifacts,
            "load_dict",
            lambda uri: calls.append(uri) or {"precision_recall_curve": [[1.0, 0.5]]},
        )

        assert pr_curve_plots.load_precision_recall_curve("E1-mini-eval") == [(1.0, 0.5)]
        assert len(selection_calls) == 1
        assert selection_calls[0][0] == []
        assert selection_calls[0][1] == "E1-mini-eval"
        assert calls == ["runs:/run-1/test_precision_recall_curve.json"]


class TestCurvePointsFromArtifact:
    def test_converts_json_lists_to_float_tuples(self):

        artifact = {"precision_recall_curve": [[1.0, 0.5], [0.2, 0.3]]}

        points = curve_points_from_artifact(artifact)

        assert points == [(0.2, 0.3), (1.0, 0.5)]
        assert all(isinstance(point, tuple) for point in points)

    def test_sorts_ascending_by_recall(self):

        artifact = {"precision_recall_curve": [[1.0, 0.5], [0.5, 0.5], [0.2, 0.1]]}

        points = curve_points_from_artifact(artifact)

        assert points == [(0.2, 0.1), (0.5, 0.5), (1.0, 0.5)]

    def test_drops_the_unreached_highest_threshold_point(self):

        artifact = {"precision_recall_curve": [[1.0, 0.5], [0.5, 0.5], [0.0, 0.0]]}

        points = curve_points_from_artifact(artifact)

        assert points == [(0.5, 0.5), (1.0, 0.5)]

    def test_drops_every_leading_zero_point_not_just_one(self):

        artifact = {"precision_recall_curve": [[1.0, 0.5], [0.0, 0.0], [0.0, 0.0]]}

        points = curve_points_from_artifact(artifact)

        assert points == [(1.0, 0.5)]

    def test_keeps_a_point_that_is_only_zero_in_one_coordinate(self):

        artifact = {"precision_recall_curve": [[1.0, 0.5], [0.0, 0.3]]}

        points = curve_points_from_artifact(artifact)

        assert points == [(0.0, 0.3), (1.0, 0.5)]
