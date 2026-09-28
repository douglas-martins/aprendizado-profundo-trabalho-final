import inspect

import numpy as np
import pytest

from evaluation.paper_protocol import LOGIT_GRID, logit_grid
from evaluation.threshold_selection import (
    ThresholdChoice,
    apply_threshold,
    oracle_threshold,
    select_threshold,
    select_with_widening,
)

THRESHOLDS = np.array([0.1, 0.5, 0.9], dtype=np.float32)


def _counts(*scenes):
    return np.array(scenes, dtype=np.int64)


def _validation_counts():
    """A strong scene and a plume-free scene at thresholds 0.1 / 0.5 / 0.9.

    Pooled over both: t=0.1 -> tp90 fp500 fn10 (F1 0.261); t=0.5 -> tp80 fp20 fn20 (F1 0.8);
    t=0.9 -> tp40 fp0 fn60 (F1 0.571). Plume scenes only: 0.947 / 0.8 / 0.571.
    """
    strong = [[90, 0, 10, 900], [80, 0, 20, 900], [40, 0, 60, 900]]
    plume_free = [[0, 500, 0, 500], [0, 20, 0, 980], [0, 0, 0, 1000]]
    return _counts(strong, plume_free), ["strong", "plume_free"]


class TestSelectThreshold:
    def test_picks_the_threshold_with_the_best_pooled_f1_over_all_scenes(self):
        counts, buckets = _validation_counts()

        choice = select_threshold(counts, THRESHOLDS, buckets)

        assert choice.threshold == pytest.approx(0.5)
        assert choice.score == pytest.approx(160 / 200)
        assert choice.objective == "pooled_f1"
        assert choice.n_scenes == 2

    def test_the_default_objective_counts_false_positives_in_plume_free_scenes(self):
        counts, buckets = _validation_counts()

        pooled = select_threshold(counts, THRESHOLDS, buckets, objective="pooled_f1")
        plume_only = select_threshold(counts, THRESHOLDS, buckets, objective="plume_scenes_f1")

        assert pooled.threshold == pytest.approx(0.5)
        assert plume_only.threshold == pytest.approx(0.1)
        assert plume_only.score == pytest.approx(180 / 190)
        assert plume_only.n_scenes == 1

    def test_an_interior_pick_is_not_flagged_as_a_grid_edge(self):
        counts, buckets = _validation_counts()

        assert select_threshold(counts, THRESHOLDS, buckets).at_edge is False

    def test_a_pick_on_the_first_or_last_grid_point_is_flagged(self):
        counts, buckets = _validation_counts()

        first = select_threshold(counts, THRESHOLDS, buckets, objective="plume_scenes_f1")
        last = select_threshold(
            counts[:, ::-1], THRESHOLDS[::-1], buckets, objective="plume_scenes_f1"
        )

        assert first.at_edge is True
        assert last.at_edge is True

    def test_ties_go_to_the_lowest_threshold(self):
        row = [50, 10, 50, 890]
        counts = _counts([row, row, row])

        choice = select_threshold(counts, THRESHOLDS, ["strong"])

        assert choice.threshold == pytest.approx(0.1)

    def test_a_model_with_no_true_positives_scores_zero_and_sits_on_the_edge(self):
        counts = _counts([[0, 5, 50, 945], [0, 3, 50, 947], [0, 1, 50, 949]])

        choice = select_threshold(counts, THRESHOLDS, ["strong"])

        assert choice.score == 0.0
        assert choice.at_edge is True

    def test_an_unknown_objective_is_rejected(self):
        counts, buckets = _validation_counts()

        with pytest.raises(ValueError, match="objective.*best_f1"):
            select_threshold(counts, THRESHOLDS, buckets, objective="best_f1")

    def test_the_thresholds_must_match_the_counts(self):
        counts, buckets = _validation_counts()

        with pytest.raises(ValueError, match="3 thresholds but 2 were given"):
            select_threshold(counts, THRESHOLDS[:2], buckets)

    def test_one_bucket_per_scene_is_required(self):
        counts, _ = _validation_counts()

        with pytest.raises(ValueError, match="one bucket per scene"):
            select_threshold(counts, THRESHOLDS, ["strong"])

    def test_an_objective_population_without_scenes_is_rejected(self):
        counts, _ = _validation_counts()

        with pytest.raises(ValueError, match="no scenes for the plume_scenes_f1 objective"):
            select_threshold(
                counts, THRESHOLDS, ["plume_free", "plume_free"], objective="plume_scenes_f1"
            )

    def test_float64_thresholds_are_reported_as_the_float32_values_that_were_scored(self):
        counts, buckets = _validation_counts()

        choice = select_threshold(counts, np.array([0.1, 0.5, 0.9]), buckets)

        assert choice.threshold == float(np.float32(0.5))
        first = select_threshold(
            counts, np.array([0.1, 0.5, 0.9]), buckets, objective="plume_scenes_f1"
        )
        assert first.threshold == float(np.float32(0.1))

    def test_a_threshold_with_nothing_predicted_and_nothing_to_find_scores_zero(self):

        counts = _counts([[0, 5, 0, 995], [0, 1, 0, 999], [0, 0, 0, 1000]])

        choice = select_threshold(counts, THRESHOLDS, ["plume_free"])

        assert choice.score == 0.0

    def test_the_choice_records_the_grid_size(self):
        counts, buckets = _validation_counts()

        assert select_threshold(counts, THRESHOLDS, buckets).grid_size == 3

    def test_only_validation_data_can_be_passed_in(self):

        parameters = inspect.signature(select_threshold).parameters

        assert list(parameters) == ["val_counts", "thresholds", "val_buckets", "objective"]
        assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
        assert list(inspect.signature(select_with_widening).parameters) == [
            "rescore",
            "val_buckets",
            "objective",
            "half_ranges",
        ]


def _synthetic_counts(grid, best_logit):
    """Counts whose pooled F1 rises with the threshold until `best_logit`, then collapses.

    True positives are all found up to `best_logit` and none above; false positives shrink
    strictly as the threshold rises, so the F1 peak sits exactly at `best_logit` (no plateau).
    """
    logits = np.log(grid.astype(np.float64) / (1 - grid.astype(np.float64)))
    rows = []
    for logit in logits:
        tp = 100 if logit <= best_logit else 0
        fp = int(round(1e9 * np.exp(-logit)))
        rows.append([tp, fp, 100 - tp, 10_000])
    return _counts(rows)


class TestSelectWithWidening:
    def test_an_interior_pick_is_returned_without_widening(self):
        calls = []

        def rescore(grid):
            calls.append(len(grid))
            return _synthetic_counts(grid, best_logit=4.0)

        choice = select_with_widening(rescore, ["strong"])

        assert calls == [len(LOGIT_GRID)]
        assert choice.at_edge is False
        assert choice.n_widenings == 0
        assert choice.grid_size == len(LOGIT_GRID)

    def test_a_pick_on_the_edge_widens_the_grid_and_reselects(self):
        calls = []

        def rescore(grid):
            calls.append(grid)
            return _synthetic_counts(grid, best_logit=11.5)

        choice = select_with_widening(rescore, ["strong"])

        assert [len(grid) for grid in calls] == [81, len(logit_grid(12.5))]
        assert choice.n_widenings == 1
        assert choice.at_edge is False
        assert choice.threshold == pytest.approx(1 / (1 + np.exp(-11.5)), rel=1e-6)
        assert choice.grid_size == len(logit_grid(12.5))

    def test_it_stops_at_the_widest_grid_and_says_the_pick_is_still_on_the_edge(self):
        calls = []

        def rescore(grid):
            calls.append(len(grid))
            return _synthetic_counts(grid, best_logit=50.0)

        choice = select_with_widening(rescore, ["strong"])

        assert calls == [len(logit_grid(h)) for h in (10.0, 12.5, 15.0)]
        assert choice.n_widenings == 2
        assert choice.at_edge is True

    def test_the_half_ranges_can_be_chosen(self):
        calls = []

        def rescore(grid):
            calls.append(len(grid))
            return _synthetic_counts(grid, best_logit=50.0)

        select_with_widening(rescore, ["strong"], half_ranges=(4.0, 6.0))

        assert calls == [len(logit_grid(4.0)), len(logit_grid(6.0))]

    def test_the_objective_is_passed_through(self):
        def rescore(grid):
            return _synthetic_counts(grid, best_logit=4.0)

        choice = select_with_widening(rescore, ["strong"], objective="plume_scenes_f1")

        assert choice.objective == "plume_scenes_f1"

    def test_the_grids_are_handed_to_rescore_as_float32_ascending(self):
        seen = []

        def rescore(grid):
            seen.append(grid)
            return _synthetic_counts(grid, best_logit=4.0)

        select_with_widening(rescore, ["strong"])

        assert seen[0].dtype == np.float32
        assert (np.diff(seen[0]) > 0).all()

    def test_at_least_one_grid_is_required(self):
        with pytest.raises(
            ValueError, match="^half_ranges must hold at least one grid half-range$"
        ):
            select_with_widening(lambda grid: None, ["strong"], half_ranges=())


class TestApplyThreshold:
    def _test_counts(self):

        strong = [[80, 20, 20, 900], [60, 5, 40, 900]]
        weak = [[20, 30, 30, 900], [10, 10, 40, 900]]
        plume_free = [[0, 11, 0, 989], [0, 4, 0, 996]]
        return _counts(strong, weak, plume_free), ["strong", "weak", "plume_free"]

    def test_reports_the_paper_metrics_at_the_chosen_threshold_only(self):
        counts, buckets = self._test_counts()
        thresholds = np.array([0.5, 0.9], dtype=np.float32)

        result = apply_threshold(counts, thresholds, buckets, 0.5)

        assert result["threshold"] == pytest.approx(0.5)
        assert result["n_scenes"] == 3
        assert result["metrics"]["strong"]["f1"] == pytest.approx(160 / 200)
        assert result["metrics"]["weak"]["f1"] == pytest.approx(40 / (40 + 30 + 30))
        assert result["tile_fpr"] == 1.0
        assert result["captured"]["plume_scenes"] == {"captured": 2, "total": 2, "rate": 1.0}

        at_high = apply_threshold(counts, thresholds, buckets, 0.9)
        assert at_high["tile_fpr"] == 0.0
        assert at_high["metrics"]["strong"]["f1"] == pytest.approx(120 / (120 + 5 + 40))

    def test_counts_the_scenes_of_each_bucket(self):
        counts, buckets = self._test_counts()

        result = apply_threshold(counts, np.array([0.5, 0.9], dtype=np.float32), buckets, 0.5)

        assert result["n_scenes_by_bucket"] == {"strong": 1, "weak": 1, "plume_free": 1}

    def test_a_split_without_plume_free_scenes_has_no_tile_fpr(self):

        counts = _counts([[80, 20, 20, 900], [60, 5, 40, 900]])

        result = apply_threshold(counts, np.array([0.5, 0.9], dtype=np.float32), ["weak"], 0.5)

        assert result["tile_fpr"] is None
        assert result["n_scenes_by_bucket"] == {"strong": 0, "weak": 1, "plume_free": 0}

    def test_the_layout_of_the_counts_is_checked(self):
        counts, buckets = self._test_counts()

        with pytest.raises(ValueError, match="one bucket per scene"):
            apply_threshold(counts, np.array([0.5, 0.9], dtype=np.float32), buckets[:2], 0.5)

    def test_the_reported_threshold_is_the_float32_value_that_was_scored(self):
        counts, buckets = self._test_counts()

        result = apply_threshold(counts, np.array([0.1, 0.5]), buckets, 0.1)

        assert result["threshold"] == float(np.float32(0.1))

    def test_a_threshold_the_counts_were_not_computed_at_is_an_error(self):
        counts, buckets = self._test_counts()

        with pytest.raises(ValueError, match="not on the threshold grid"):
            apply_threshold(counts, np.array([0.5, 0.9], dtype=np.float32), buckets, 0.7)

    def test_a_threshold_chosen_on_validation_is_looked_up_exactly(self):
        counts, buckets = self._test_counts()
        thresholds = LOGIT_GRID[:2]
        chosen = float(LOGIT_GRID[1])

        assert apply_threshold(counts, thresholds, buckets, chosen)["threshold"] == pytest.approx(
            chosen
        )


class TestOracleThreshold:
    def test_is_the_same_search_run_on_the_scored_split_itself(self):
        counts, buckets = _validation_counts()

        oracle = oracle_threshold(counts, THRESHOLDS, buckets)

        assert oracle == select_threshold(counts, THRESHOLDS, buckets)

    def test_the_objective_is_passed_through(self):
        counts, buckets = _validation_counts()

        oracle = oracle_threshold(counts, THRESHOLDS, buckets, objective="plume_scenes_f1")

        assert oracle.objective == "plume_scenes_f1"
        assert oracle.threshold == pytest.approx(0.1)

    def test_the_result_is_a_threshold_choice(self):
        counts, buckets = _validation_counts()

        assert isinstance(oracle_threshold(counts, THRESHOLDS, buckets), ThresholdChoice)
