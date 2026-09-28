from collections import namedtuple

import pytest

from visualization.figure_common import EXPERIMENT_NAME, open_client
from visualization.training_curves import (
    TRAINING_RUN_NAMES,
    TrainingCurve,
    load_training_curve,
    per_epoch_series,
)

_Metric = namedtuple("_Metric", ["step", "value", "timestamp"])


class TestPerEpochSeries:
    def test_returns_epochs_ascending_with_their_values(self):
        history = [_Metric(3, 0.3, 30), _Metric(1, 0.1, 10), _Metric(2, 0.2, 20)]

        assert per_epoch_series(history) == ([1, 2, 3], [0.1, 0.2, 0.3])

    def test_keeps_the_latest_logged_value_when_an_epoch_was_logged_twice(self):

        history = [_Metric(2, 0.9, 500), _Metric(1, 0.1, 10), _Metric(2, 0.2, 20)]

        assert per_epoch_series(history) == ([1, 2], [0.1, 0.9])

    def test_returns_empty_series_for_an_empty_history(self):
        assert per_epoch_series([]) == ([], [])


class TestTrainingRunNames:
    def test_r3_has_no_e1_run_because_e1_was_not_trained_at_full_scale(self):
        assert set(TRAINING_RUN_NAMES["raw-full"]) == {"E2", "E3"}

    def test_mini_and_r2_cover_all_three_architectures(self):
        assert set(TRAINING_RUN_NAMES["mini"]) == {"E1", "E2", "E3"}
        assert set(TRAINING_RUN_NAMES["r2"]) == {"E1", "E2", "E3"}

    def test_every_run_name_is_architecture_dash_tier(self):
        for tier, by_architecture in TRAINING_RUN_NAMES.items():
            for architecture, run_name in by_architecture.items():
                assert run_name == f"{architecture}-{tier}"


@pytest.fixture
def client_and_run(tmp_path):
    client = open_client(f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment_id = client.create_experiment(EXPERIMENT_NAME)
    run = client.create_run(experiment_id, tags={"mlflow.runName": "E2-r2"})
    return client, run.info.run_id


def _log_epochs(client, run_id, val_loss, val_f1, first_timestamp=1000):
    for epoch, (loss, f1) in enumerate(zip(val_loss, val_f1, strict=True), start=1):
        client.log_metric(run_id, "val_loss", loss, timestamp=first_timestamp + epoch, step=epoch)
        client.log_metric(run_id, "val_f1", f1, timestamp=first_timestamp + epoch, step=epoch)


def _log_summary(client, run_id, best_epoch, epochs_run, stopped_early=1.0, timestamp=2000):
    client.log_metric(run_id, "best_epoch", float(best_epoch), timestamp=timestamp)
    client.log_metric(run_id, "epochs_run", float(epochs_run), timestamp=timestamp)
    client.log_metric(run_id, "stopped_early", stopped_early, timestamp=timestamp)


class TestLoadTrainingCurve:
    def test_reads_the_curves_and_the_final_best_epoch_of_a_run_extended_by_a_resume(
        self, client_and_run
    ):
        client, run_id = client_and_run
        _log_epochs(client, run_id, [0.9, 0.5, 0.4, 0.6], [0.1, 0.3, 0.4, 0.2])

        _log_summary(client, run_id, best_epoch=3, epochs_run=3, timestamp=2000)
        _log_summary(client, run_id, best_epoch=3, epochs_run=4, timestamp=3000)

        curve = load_training_curve(client, run_id)

        assert curve == TrainingCurve(
            epochs=[1, 2, 3, 4],
            val_loss=[0.9, 0.5, 0.4, 0.6],
            val_f1=[0.1, 0.3, 0.4, 0.2],
            best_epoch=3,
            epochs_run=4,
            stopped_early=True,
        )

    def test_uses_the_last_logged_best_epoch_not_the_first(self, client_and_run):
        client, run_id = client_and_run
        _log_epochs(client, run_id, [0.9, 0.5, 0.4], [0.1, 0.3, 0.4])
        _log_summary(client, run_id, best_epoch=2, epochs_run=3, timestamp=2000)
        _log_summary(client, run_id, best_epoch=3, epochs_run=3, timestamp=3000)

        assert load_training_curve(client, run_id).best_epoch == 3

    def test_raises_when_best_epoch_is_not_one_of_the_logged_epochs(self, client_and_run):
        client, run_id = client_and_run
        _log_epochs(client, run_id, [0.9, 0.5], [0.1, 0.3])
        _log_summary(client, run_id, best_epoch=7, epochs_run=2)

        with pytest.raises(ValueError, match="best_epoch"):
            load_training_curve(client, run_id)

    def test_raises_when_epochs_run_disagrees_with_the_last_logged_epoch(self, client_and_run):
        client, run_id = client_and_run
        _log_epochs(client, run_id, [0.9, 0.5, 0.4], [0.1, 0.3, 0.4])
        _log_summary(client, run_id, best_epoch=3, epochs_run=5)

        with pytest.raises(ValueError, match="epochs_run 5 differs from the last logged epoch 3"):
            load_training_curve(client, run_id)

    def test_raises_when_val_loss_and_val_f1_cover_different_epochs(self, client_and_run):
        client, run_id = client_and_run
        _log_epochs(client, run_id, [0.9, 0.5, 0.4], [0.1, 0.3, 0.4])
        client.log_metric(run_id, "val_loss", 0.3, timestamp=4000, step=4)
        _log_summary(client, run_id, best_epoch=3, epochs_run=4)

        with pytest.raises(ValueError, match="val_f1"):
            load_training_curve(client, run_id)

    def test_reports_a_run_that_hit_the_epoch_cap_as_not_stopped_early(self, client_and_run):
        client, run_id = client_and_run
        _log_epochs(client, run_id, [0.9, 0.5, 0.4], [0.1, 0.3, 0.4])
        _log_summary(client, run_id, best_epoch=3, epochs_run=3, stopped_early=0.0)

        assert load_training_curve(client, run_id).stopped_early is False

    def test_raises_when_the_run_logged_no_epochs(self, client_and_run):
        client, run_id = client_and_run
        _log_summary(client, run_id, best_epoch=1, epochs_run=1)

        with pytest.raises(ValueError, match="no val_loss"):
            load_training_curve(client, run_id)
