from collections import namedtuple

import pytest

from visualization.figure_common import (
    ARCHITECTURE_COLORS,
    ARCHITECTURE_MARKERS,
    EXPERIMENT_NAME,
    TIER_COLORS,
    TIER_LABELS,
    TIER_MARKERS,
    final_metric_value,
    format_decimal_ptbr,
    format_general_ptbr,
    format_integer_ptbr,
    latest_metric,
    open_client,
    resolve_run_id,
)

_Metric = namedtuple("_Metric", ["step", "value", "timestamp"])


@pytest.fixture
def client(tmp_path):
    return open_client(f"sqlite:///{tmp_path / 'mlflow.db'}")


@pytest.fixture
def experiment_id(client):
    return client.create_experiment(EXPERIMENT_NAME)


def _log_run(client, experiment_id, run_name, start_time, metrics=(), status="FINISHED", tags=None):
    run = client.create_run(
        experiment_id, start_time=start_time, tags={"mlflow.runName": run_name, **(tags or {})}
    )
    for key, value, step, timestamp in metrics:
        client.log_metric(run.info.run_id, key, value, timestamp=timestamp, step=step)
    client.set_terminated(run.info.run_id, status)
    return run.info.run_id


class TestFinalMetricValue:
    def test_returns_the_value_with_the_latest_timestamp_regardless_of_list_order(self):
        history = [_Metric(0, 64.0, 2000), _Metric(0, 47.0, 1000)]

        assert final_metric_value(history) == 64.0

    def test_breaks_a_timestamp_tie_with_the_larger_step(self):
        history = [_Metric(3, 30.0, 1000), _Metric(7, 70.0, 1000), _Metric(5, 50.0, 1000)]

        assert final_metric_value(history) == 70.0

    def test_prefers_a_later_timestamp_over_a_larger_step(self):
        history = [_Metric(9, 90.0, 1000), _Metric(1, 10.0, 2000)]

        assert final_metric_value(history) == 10.0

    def test_raises_on_an_empty_history(self):
        with pytest.raises(ValueError, match="^metric history is empty"):
            final_metric_value([])


class TestLatestMetric:
    def test_returns_the_final_value_of_a_metric_logged_twice_by_a_resumed_run(
        self, client, experiment_id
    ):

        run_id = _log_run(
            client,
            experiment_id,
            "E2-r2",
            start_time=1,
            metrics=[("best_epoch", 47.0, 0, 1000), ("best_epoch", 64.0, 0, 2000)],
        )

        assert latest_metric(client, run_id, "best_epoch") == 64.0

    def test_raises_when_the_run_never_logged_the_metric(self, client, experiment_id):
        run_id = _log_run(client, experiment_id, "E1-mini", start_time=1)

        with pytest.raises(ValueError, match="^metric history is empty"):
            latest_metric(client, run_id, "best_epoch")


class TestResolveRunId:
    def test_picks_the_newest_finished_run_of_that_name(self, client, experiment_id):
        _log_run(client, experiment_id, "E1-mini", start_time=1000)
        newest = _log_run(client, experiment_id, "E1-mini", start_time=2000)
        _log_run(client, experiment_id, "E2-mini", start_time=3000)

        assert resolve_run_id(client, "E1-mini") == newest

    def test_skips_a_newer_run_tagged_superseded(self, client, experiment_id):
        kept = _log_run(client, experiment_id, "E1-mini", start_time=1000)
        _log_run(client, experiment_id, "E1-mini", start_time=2000, tags={"superseded": "true"})

        assert resolve_run_id(client, "E1-mini") == kept

    def test_raises_for_a_name_no_run_carries(self, client, experiment_id):
        with pytest.raises(ValueError, match="no run named"):
            resolve_run_id(client, "E9-mini")


class TestOpenClient:
    def test_uses_the_tracking_uri_it_is_given(self, tmp_path):
        uri = f"sqlite:///{tmp_path / 'other.db'}"

        client = open_client(uri)

        assert client.tracking_uri == uri


class TestFormatDecimalPtbr:
    def test_uses_a_decimal_comma(self):
        assert format_decimal_ptbr(6.3199, 1) == "6,3"

    def test_pads_to_the_requested_number_of_decimals(self):
        assert format_decimal_ptbr(0.5, 4) == "0,5000"

    def test_keeps_the_sign_of_a_negative_value(self):
        assert format_decimal_ptbr(-0.25, 2) == "-0,25"


class TestFormatGeneralPtbr:
    def test_uses_a_decimal_comma_for_a_fractional_tick(self):
        assert format_general_ptbr(0.6) == "0,6"

    def test_drops_the_decimal_part_of_a_whole_tick(self):
        assert format_general_ptbr(2.0) == "2"

    def test_keeps_the_leading_zeros_of_a_small_tick(self):
        assert format_general_ptbr(0.05) == "0,05"

    def test_writes_zero_as_a_plain_zero(self):
        assert format_general_ptbr(0.0) == "0"


class TestFormatIntegerPtbr:
    def test_groups_thousands_with_a_dot(self):
        assert format_integer_ptbr(487361) == "487.361"

    def test_groups_millions_with_two_dots(self):
        assert format_integer_ptbr(6629233) == "6.629.233"

    def test_leaves_a_three_digit_number_alone(self):
        assert format_integer_ptbr(999) == "999"


class TestFigureStyleTables:
    def test_every_architecture_has_a_color_and_a_marker(self):
        assert set(ARCHITECTURE_COLORS) == {"E1", "E2", "E3"}
        assert set(ARCHITECTURE_MARKERS) == set(ARCHITECTURE_COLORS)

    def test_architectures_are_told_apart_by_both_color_and_marker(self):

        assert len(set(ARCHITECTURE_COLORS.values())) == 3
        assert len(set(ARCHITECTURE_MARKERS.values())) == 3

    def test_every_tier_has_a_label_a_color_and_a_marker(self):
        assert set(TIER_LABELS) == {"mini", "r2", "raw-full"}
        assert set(TIER_COLORS) == set(TIER_LABELS)
        assert set(TIER_MARKERS) == set(TIER_LABELS)

    def test_tiers_are_told_apart_by_both_color_and_marker(self):
        assert len(set(TIER_COLORS.values())) == 3
        assert len(set(TIER_MARKERS.values())) == 3

    def test_tier_colors_do_not_reuse_an_architecture_color(self):

        assert set(TIER_COLORS.values()).isdisjoint(ARCHITECTURE_COLORS.values())

    def test_raw_full_is_shown_as_r3_in_figures(self):

        assert TIER_LABELS["raw-full"] == "R3"
