import numpy as np
import pytest
import torch

from evaluation.paper_protocol import (
    ALL_THRESHOLDS,
    AUPRC_GRIDS,
    GRID_101,
    LOGIT_GRID,
    MAX_LOGIT_HALF_RANGE,
    PAPER_THRESHOLDS,
    TILE_PIXEL_THRESHOLD,
    auprc,
    auprc_by_grid,
    bucket_confusion,
    bucket_metrics,
    captured_plumes,
    confusion_metrics,
    logit_grid,
    select_thresholds,
    threshold_index,
    tile_fpr,
)


def _counts(*scenes):
    """`(n_scenes, n_thresholds, 4)` int64 from per-scene lists of `[tp, fp, fn, tn]` rows."""
    return np.array(scenes, dtype=np.int64)


class TestGrids:
    def test_the_paper_grid_is_the_sixteen_thresholds_of_starcops_validation_code(self):
        expected = [0, 1e-3, 1e-2, *np.arange(0.5, 0.96, 0.05).tolist(), 0.99, 0.995, 0.999]

        assert len(PAPER_THRESHOLDS) == 16
        assert PAPER_THRESHOLDS.dtype == np.float32
        assert np.allclose(PAPER_THRESHOLDS, expected, atol=1e-7)
        assert (np.diff(PAPER_THRESHOLDS) > 0).all()

    def test_our_grid_is_the_101_evenly_spaced_points_evaluate_py_sweeps(self):
        assert GRID_101.dtype == np.float32
        assert np.array_equal(GRID_101, torch.linspace(0.0, 1.0, steps=101).numpy())

    def test_the_logit_grid_is_uniform_in_logit_space_and_dense_near_both_ends(self):
        logits = np.log(LOGIT_GRID.astype(np.float64) / (1 - LOGIT_GRID.astype(np.float64)))

        assert LOGIT_GRID.dtype == np.float32
        assert (LOGIT_GRID > 0).all() and (LOGIT_GRID < 1).all()
        assert (np.diff(LOGIT_GRID) > 0).all()
        assert np.allclose(np.diff(logits), 0.25, atol=1e-2)

        assert np.diff(LOGIT_GRID)[-1] < np.diff(LOGIT_GRID)[len(LOGIT_GRID) // 2] / 10
        assert 0.5 in LOGIT_GRID

    def test_the_default_logit_grid_is_the_module_constant(self):
        assert np.array_equal(logit_grid(), LOGIT_GRID)

    def test_a_wider_logit_grid_extends_the_range_at_the_same_step(self):
        wide = logit_grid(13.0)

        assert len(wide) == 105
        assert np.isin(LOGIT_GRID, wide).all()
        assert wide[0] < LOGIT_GRID[0] and wide[-1] > LOGIT_GRID[-1]
        assert (np.diff(wide) > 0).all()

    def test_the_step_can_be_chosen(self):
        assert len(logit_grid(2.0, step=1.0)) == 5

    def test_the_widest_allowed_logit_grid_still_has_distinct_float32_thresholds(self):
        widest = logit_grid(MAX_LOGIT_HALF_RANGE)

        assert (np.diff(widest) > 0).all()
        assert widest[-1] < 1.0

    def test_the_logit_grid_cannot_go_where_float32_thresholds_would_repeat(self):
        with pytest.raises(ValueError, match="half_range"):
            logit_grid(MAX_LOGIT_HALF_RANGE + 0.5)

    def test_a_half_range_below_one_is_allowed(self):
        assert len(logit_grid(0.5)) == 5

    def test_a_non_positive_half_range_is_rejected(self):
        with pytest.raises(ValueError, match="half_range"):
            logit_grid(0.0)

    def test_the_union_grid_holds_every_grid_sorted_without_duplicates(self):
        assert ALL_THRESHOLDS.dtype == np.float32
        assert (np.diff(ALL_THRESHOLDS) > 0).all()
        for grid in AUPRC_GRIDS.values():
            assert np.isin(grid, ALL_THRESHOLDS).all()

    def test_the_named_auprc_grids_are_the_three_supported_grids(self):
        assert list(AUPRC_GRIDS) == ["paper16", "grid101", "logit"]


class TestThresholdLookupPrecision:
    def test_float64_thresholds_are_matched_after_casting_to_float32(self):

        thresholds = np.array([0.1, 0.5, 0.9])

        assert threshold_index(thresholds, 0.1) == 0

    def test_select_thresholds_accepts_float64_on_both_sides_and_returns_float32(self):
        thresholds = np.array([0.1, 0.5, 0.9])
        counts = np.arange(2 * 3 * 4).reshape(2, 3, 4)

        subset, kept = select_thresholds(counts, thresholds, np.array([0.9, 0.1]))

        assert kept.dtype == np.float32
        assert np.array_equal(subset[:, 0], counts[:, 2])
        assert np.array_equal(subset[:, 1], counts[:, 0])

    def test_the_error_names_every_missing_threshold_rounded_to_six_places(self):
        thresholds = np.array([0.1, 0.5], dtype=np.float32)

        with pytest.raises(ValueError) as error:
            select_thresholds(
                np.zeros((1, 2, 4)), thresholds, np.array([0.5, 0.7, 0.1234567], dtype=np.float32)
            )

        assert "0.7" in str(error.value)
        assert "0.123457" in str(error.value)
        assert "0.1234567" not in str(error.value)


class TestThresholdLookup:
    def test_finds_the_exact_float32_threshold(self):
        thresholds = np.array([0.1, 0.5, 0.9], dtype=np.float32)

        assert threshold_index(thresholds, 0.5) == 1

    def test_a_threshold_that_is_not_on_the_grid_is_an_error(self):
        with pytest.raises(ValueError, match="0.6.*not on the threshold grid"):
            threshold_index(np.array([0.1, 0.5], dtype=np.float32), 0.6)

    def test_select_thresholds_keeps_the_matching_columns_in_the_requested_order(self):
        thresholds = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        counts = np.arange(2 * 3 * 4).reshape(2, 3, 4)

        subset, kept = select_thresholds(counts, thresholds, np.array([0.9, 0.1], dtype=np.float32))

        assert kept.tolist() == pytest.approx([0.9, 0.1])
        assert np.array_equal(subset[:, 0], counts[:, 2])
        assert np.array_equal(subset[:, 1], counts[:, 0])

    def test_select_thresholds_names_every_missing_threshold(self):
        thresholds = np.array([0.1, 0.5], dtype=np.float32)

        with pytest.raises(ValueError, match="0.7"):
            select_thresholds(np.zeros((1, 2, 4)), thresholds, np.array([0.5, 0.7]))


class TestConfusionMetrics:
    def test_precision_recall_and_f1_from_a_confusion_vector(self):

        metrics = confusion_metrics(np.array([80, 20, 20, 900]))

        assert metrics == pytest.approx({"precision": 0.8, "recall": 0.8, "f1": 0.8})

    def test_f1_is_the_harmonic_mean_written_from_counts(self):

        metrics = confusion_metrics(np.array([100, 50, 100, 0]))

        assert metrics["f1"] == pytest.approx(200 / 350)

    def test_empty_denominators_give_zero_not_nan(self):
        assert confusion_metrics(np.array([0, 0, 0, 500])) == {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
        assert confusion_metrics(np.array([0, 5, 0, 500]))["recall"] == 0.0
        assert confusion_metrics(np.array([0, 0, 5, 500]))["precision"] == 0.0


class TestConfusionMetricsBothWays:
    def test_precision_and_recall_differ_when_false_positives_and_negatives_differ(self):

        metrics = confusion_metrics(np.array([3, 5, 7, 100]))

        assert metrics == pytest.approx({"precision": 3 / 8, "recall": 3 / 10, "f1": 6 / 18})

    def test_a_single_true_positive_and_nothing_else_is_perfect(self):

        assert confusion_metrics(np.array([1, 0, 0, 50])) == pytest.approx(
            {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        )

    def test_a_single_false_positive_gives_zero_precision(self):
        metrics = confusion_metrics(np.array([0, 1, 0, 50]))

        assert metrics == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_true_positives_below_false_positives_still_have_a_precision(self):
        assert confusion_metrics(np.array([1, 9, 0, 0]))["precision"] == pytest.approx(0.1)

    def test_true_positives_below_false_negatives_still_have_a_recall(self):
        assert confusion_metrics(np.array([1, 0, 9, 0]))["recall"] == pytest.approx(0.1)


class TestBucketConfusion:
    def setup_method(self):

        self.counts = _counts(
            [[80, 20, 20, 100], [10, 1, 90, 119]],
            [[20, 30, 80, 70], [5, 2, 95, 98]],
            [[5, 5, 5, 85], [1, 0, 9, 90]],
            [[0, 9, 0, 91], [0, 1, 0, 99]],
        )
        self.buckets = ["strong", "strong", "weak", "plume_free"]

    def test_pools_the_counts_of_every_scene_in_the_bucket_at_one_threshold(self):
        assert bucket_confusion(self.counts, self.buckets, "strong", 0).tolist() == [
            100,
            50,
            100,
            170,
        ]
        assert bucket_confusion(self.counts, self.buckets, "strong", 1).tolist() == [
            15,
            3,
            185,
            217,
        ]

    def test_accepts_several_buckets_at_once(self):
        both = bucket_confusion(self.counts, self.buckets, ("strong", "weak"), 0)

        assert both.tolist() == [105, 55, 105, 255]

    def test_an_unknown_bucket_name_is_rejected(self):
        with pytest.raises(ValueError, match="unknown bucket.*stong"):
            bucket_confusion(self.counts, self.buckets, "stong", 0)

    def test_one_bucket_label_per_scene_is_required(self):
        with pytest.raises(ValueError, match="one bucket per scene is required: 3 buckets for 4"):
            bucket_confusion(self.counts, self.buckets[:3], "strong", 0)

    def test_an_empty_bucket_pools_to_zeros(self):
        empty = bucket_confusion(self.counts[:1], ["strong"], "weak", 0)

        assert empty.tolist() == [0, 0, 0, 0]

    def test_pooled_f1_is_not_the_mean_of_per_scene_f1(self):

        metrics = bucket_metrics(self.counts, self.buckets, 0)

        assert metrics["strong"]["f1"] == pytest.approx(200 / 350)
        assert metrics["strong"]["f1"] != pytest.approx((0.8 + 40 / 150) / 2)

    def test_reports_strong_weak_the_pooled_plume_scenes_and_every_scene(self):
        metrics = bucket_metrics(self.counts, self.buckets, 0)

        assert list(metrics) == ["strong", "weak", "plume_scenes", "all_scenes"]
        assert metrics["weak"]["f1"] == pytest.approx(10 / 20)
        assert metrics["plume_scenes"]["f1"] == pytest.approx(210 / (210 + 55 + 105))
        assert metrics["all_scenes"]["f1"] == pytest.approx(210 / (210 + 64 + 105))


class TestTileFalsePositiveRate:
    def _plume_free(self, *predicted_pixels):

        return _counts(*[[[0, p, 0, 1000 - p]] for p in predicted_pixels])

    def test_a_tile_is_positive_only_above_ten_active_pixels(self):
        assert TILE_PIXEL_THRESHOLD == 10
        counts = self._plume_free(11, 10, 0)

        assert tile_fpr(counts, ["plume_free"] * 3, 0) == pytest.approx(1 / 3)

    def test_counts_true_and_false_positive_pixels_together_as_active_pixels(self):
        counts = _counts([[6, 5, 0, 989]])

        assert tile_fpr(counts, ["plume_free"], 0) == 1.0

    def test_only_plume_free_scenes_count(self):
        counts = _counts([[50, 60, 0, 100]], [[0, 0, 0, 1000]])

        assert tile_fpr(counts, ["strong", "plume_free"], 0) == 0.0

    def test_no_plume_free_scene_means_no_rate(self):
        with pytest.raises(ValueError, match="^no plume-free scenes: the tile FPR is undefined$"):
            tile_fpr(_counts([[1, 1, 1, 1]]), ["strong"], 0)

    def test_uses_the_requested_threshold_column(self):

        counts = _counts([[10, 10, 0, 980], [1, 2, 0, 997]])

        assert tile_fpr(counts, ["plume_free"], 0) == 1.0
        assert tile_fpr(counts, ["plume_free"], 1) == 0.0

    def test_the_pixel_threshold_can_be_overridden(self):
        counts = self._plume_free(11, 10, 0)

        assert tile_fpr(counts, ["plume_free"] * 3, 0, min_pixels=5) == pytest.approx(2 / 3)


class TestCapturedPlumes:
    def _scenes(self):

        return _counts(
            [[1, 10, 40, 900]],
            [[5, 5, 40, 900]],
            [[0, 50, 45, 900]],
            [[9, 30, 3, 900]],
        )

    def test_a_plume_needs_more_than_ten_active_pixels_and_at_least_one_overlapping_pixel(self):
        result = captured_plumes(self._scenes(), ["strong", "strong", "weak", "weak"], 0)

        assert result["strong"] == {"captured": 1, "total": 2, "rate": 0.5}
        assert result["weak"] == {"captured": 1, "total": 2, "rate": 0.5}

    def test_reports_the_two_plume_buckets_together(self):
        result = captured_plumes(self._scenes(), ["strong", "strong", "weak", "weak"], 0)

        assert result["plume_scenes"] == {"captured": 2, "total": 4, "rate": 0.5}

    def test_plume_free_scenes_are_ignored(self):
        counts = np.concatenate([self._scenes(), _counts([[0, 99, 0, 900]])])

        result = captured_plumes(counts, ["strong", "strong", "weak", "weak", "plume_free"], 0)

        assert result["plume_scenes"]["total"] == 4

    def test_a_bucket_without_scenes_has_a_zero_total_and_no_rate(self):
        counts = self._scenes()[:2]

        result = captured_plumes(counts, ["strong", "strong"], 0)

        assert result["weak"] == {"captured": 0, "total": 0, "rate": None}


class TestAuprc:
    def test_a_perfect_ranker_scores_one(self):
        counts = _counts([[5, 0, 5, 90], [10, 0, 0, 90]])
        thresholds = np.array([0.9, 0.1], dtype=np.float32)

        assert auprc(counts, thresholds, ["strong"]) == pytest.approx(1.0)

    def test_a_hand_worked_curve(self):

        counts = _counts([[5, 0, 5, 90], [10, 10, 0, 80]])
        thresholds = np.array([0.9, 0.1], dtype=np.float32)

        assert auprc(counts, thresholds, ["strong"]) == pytest.approx(0.75)

    def test_the_order_of_the_thresholds_does_not_matter(self):
        counts = _counts([[5, 0, 5, 90], [10, 10, 0, 80]])

        forward = auprc(counts, np.array([0.9, 0.1], dtype=np.float32), ["strong"])
        backward = auprc(counts[:, ::-1], np.array([0.1, 0.9], dtype=np.float32), ["strong"])

        assert forward == pytest.approx(backward)

    def test_all_scenes_pool_their_pixels_before_the_curve_is_built(self):
        strong = [[5, 0, 5, 90], [10, 0, 0, 90]]
        plume_free = [[0, 10, 0, 90], [0, 20, 0, 80]]
        counts = _counts(strong, plume_free)
        thresholds = np.array([0.9, 0.1], dtype=np.float32)

        assert auprc(counts, thresholds, ["strong", "plume_free"]) == pytest.approx(
            0.5 / 3 + 0.5 / 3
        )

    def test_the_plume_scenes_population_leaves_plume_free_scenes_out(self):
        strong = [[5, 0, 5, 90], [10, 0, 0, 90]]
        plume_free = [[0, 10, 0, 90], [0, 20, 0, 80]]
        counts = _counts(strong, plume_free)
        thresholds = np.array([0.9, 0.1], dtype=np.float32)

        value = auprc(counts, thresholds, ["strong", "plume_free"], population="plume_scenes")

        assert value == pytest.approx(1.0)

    def test_pooled_counts_beyond_float32_precision_are_not_rounded(self):

        counts = _counts([[5, 2**24 + 1, 5, 0], [10, 2**25 + 1, 0, 0]])
        thresholds = np.array([0.9, 0.1], dtype=np.float32)
        expected = 0.5 * (5 / (5 + 2**24 + 1)) + 0.5 * (10 / (10 + 2**25 + 1))

        assert auprc(counts, thresholds, ["strong"]) == pytest.approx(expected, rel=1e-12, abs=0)

    def test_the_number_of_thresholds_must_match_the_counts(self):
        counts = _counts([[5, 0, 5, 90], [10, 0, 0, 90]])

        with pytest.raises(ValueError, match="2 thresholds but 3 were given"):
            auprc(counts, np.array([0.9, 0.5, 0.1], dtype=np.float32), ["strong"])

    def test_an_unknown_population_is_rejected(self):
        with pytest.raises(ValueError, match="population"):
            auprc(
                np.zeros((1, 1, 4)), np.array([0.5], dtype=np.float32), ["strong"], population="x"
            )

    def test_a_population_without_scenes_is_rejected(self):
        with pytest.raises(ValueError, match="no scenes"):
            auprc(
                _counts([[1, 1, 1, 1]]),
                np.array([0.5], dtype=np.float32),
                ["plume_free"],
                population="plume_scenes",
            )

    def test_agrees_with_the_torch_average_precision_the_older_evaluation_uses(self):
        from evaluation.metrics import average_precision_from_sweep

        counts = _counts([[3, 1, 7, 89], [6, 4, 4, 86], [9, 20, 1, 70]])
        thresholds = np.array([0.9, 0.5, 0.1], dtype=np.float32)

        expected = average_precision_from_sweep(torch.tensor(counts[0], dtype=torch.float32))

        assert auprc(counts, thresholds, ["strong"]) == pytest.approx(expected)


class TestAuprcByGrid:
    def _counts_on(self, thresholds):

        rows = []
        for threshold in thresholds:
            tp = int(round(100 * (1 - float(threshold))))
            rows.append([tp, 10, 100 - tp, 890])
        return _counts(rows)

    def test_reports_every_grid_for_both_populations(self):
        counts = self._counts_on(ALL_THRESHOLDS)

        table = auprc_by_grid(counts, ALL_THRESHOLDS, ["strong"])

        assert list(table) == ["paper16", "grid101", "logit"]
        for populations in table.values():
            assert list(populations) == ["all_scenes", "plume_scenes"]
            assert all(0.0 <= value <= 1.0 for value in populations.values())

    def test_each_grid_is_evaluated_on_its_own_thresholds_only(self):
        counts = self._counts_on(ALL_THRESHOLDS)

        table = auprc_by_grid(counts, ALL_THRESHOLDS, ["strong"])
        direct_counts, direct_thresholds = select_thresholds(
            counts, ALL_THRESHOLDS, PAPER_THRESHOLDS
        )

        assert table["paper16"]["all_scenes"] == pytest.approx(
            auprc(direct_counts, direct_thresholds, ["strong"])
        )

    def test_the_populations_are_evaluated_separately_on_every_grid(self):

        strong = [
            [int(round(100 * (1 - float(t)))), 0, 100 - int(round(100 * (1 - float(t)))), 900]
            for t in ALL_THRESHOLDS
        ]
        plume_free = [[0, 500, 0, 500] for _ in ALL_THRESHOLDS]
        counts = _counts(strong, plume_free)

        table = auprc_by_grid(counts, ALL_THRESHOLDS, ["strong", "plume_free"])

        for populations in table.values():
            assert populations["plume_scenes"] > populations["all_scenes"] + 0.05

    def test_a_grid_the_counts_were_not_computed_on_is_an_error(self):
        counts = self._counts_on(GRID_101)

        with pytest.raises(ValueError, match="not on the threshold grid"):
            auprc_by_grid(counts, GRID_101, ["strong"])
