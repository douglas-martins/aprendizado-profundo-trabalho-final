import json

import numpy as np
import pytest

from evaluation.evaluate_scenes import (
    EVAL_SOURCES,
    build_run_name,
    flatten_metrics,
    score_grid,
    summarize_test,
)
from evaluation.paper_protocol import ALL_THRESHOLDS
from evaluation.threshold_selection import ThresholdChoice


class TestBuildRunName:
    def test_names_the_model_the_evaluated_tier_and_the_mode(self):
        assert (
            build_run_name("E2", "raw-full", "raw-full", "full_scene")
            == "E2-raw-full-on-raw-full-scene-full_scene"
        )

    def test_a_cross_tier_model_keeps_its_own_training_tier_in_the_name(self):
        assert (
            build_run_name("E1", "mini", "raw-full", "patches")
            == "E1-mini-on-raw-full-scene-patches"
        )

    def test_an_explicit_device_is_appended_so_a_comparison_run_keeps_its_own_identity(self):
        assert (
            build_run_name("E3", "raw-full", "raw-full", "full_scene", device_override="cpu")
            == "E3-raw-full-on-raw-full-scene-full_scene-cpu"
        )

    def test_an_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="mode"):
            build_run_name("E2", "raw-full", "raw-full", "stitched")


class TestEvalSources:
    def test_the_raw_tier_has_test_and_validation_sources(self):
        source = EVAL_SOURCES["raw-full"]

        assert source["dataset"] == "starcop_raw"
        assert source["test"]["labels"] == "data/starcop_raw/test.csv"
        assert source["test"]["patches"].endswith("starcop_raw/patches/test_tiled_128_128.csv")
        assert source["val"]["labels"] == "data/starcop_raw/train.csv"
        assert source["val"]["patches"].endswith("starcop_raw/patches/val_tiled_128_128.csv")

    def test_the_mini_tier_has_a_test_source_and_no_validation_source(self):
        source = EVAL_SOURCES["mini"]

        assert source["dataset"] == "starcop_mini"
        assert source["test"]["labels"] == "data/processed/starcop_mini/splits/test.csv"
        assert source["val"] is None


class TestScoreGrid:
    def test_union_is_every_threshold_any_analysis_needs(self):
        assert np.array_equal(score_grid("union"), ALL_THRESHOLDS)

    def test_single_is_just_the_fixed_half_threshold(self):
        grid = score_grid("single")

        assert grid.dtype == np.float32
        assert grid.tolist() == [0.5]

    def test_anything_else_is_rejected(self):
        with pytest.raises(ValueError, match="thresholds"):
            score_grid("many")


def _test_counts():
    """3 scenes (strong, weak, plume_free) at thresholds 0.5 / 0.9 (a tiny grid)."""
    strong = [[80, 20, 20, 880], [60, 5, 40, 890]]
    weak = [[20, 30, 30, 920], [10, 10, 40, 935]]
    plume_free = [[0, 11, 0, 989], [0, 4, 0, 991]]
    return np.array([strong, weak, plume_free], dtype=np.int64), ["strong", "weak", "plume_free"]


THRESHOLDS = np.array([0.5, 0.9], dtype=np.float32)


class TestSummarizeTest:
    def test_a_fixed_threshold_run_reports_only_the_half_threshold(self):
        counts, buckets = _test_counts()

        summary = summarize_test(counts, THRESHOLDS, buckets, chosen=None)

        assert summary["at_0p5"]["metrics"]["strong"]["f1"] == pytest.approx(160 / 200)
        assert summary["at_0p5"]["tile_fpr"] == 1.0
        assert summary["at_validation_pick"] is None
        assert summary["validation_choice"] is None
        assert summary["optimism_gap_pooled_f1"] is None

    def test_a_validation_pick_is_applied_once_and_compared_with_the_oracle(self):
        counts, buckets = _test_counts()
        chosen = ThresholdChoice(0.9, 0.55, "pooled_f1", 2, 5, False)

        summary = summarize_test(counts, THRESHOLDS, buckets, chosen=chosen)

        assert summary["at_validation_pick"]["threshold"] == pytest.approx(0.9)
        assert summary["at_validation_pick"]["tile_fpr"] == 0.0
        assert summary["validation_choice"]["threshold"] == 0.9
        assert summary["validation_choice"]["objective"] == "pooled_f1"
        picked = summary["at_validation_pick"]["metrics"]["all_scenes"]["f1"]
        oracle = summary["oracle_at_threshold"]["metrics"]["all_scenes"]["f1"]
        assert summary["oracle_choice"]["score"] == pytest.approx(oracle)
        assert summary["optimism_gap_pooled_f1"] == pytest.approx(oracle - picked)
        assert summary["optimism_gap_pooled_f1"] >= 0

    def test_the_summary_always_has_the_same_keys_picked_or_not(self):
        counts, buckets = _test_counts()
        chosen = ThresholdChoice(0.9, 0.55, "pooled_f1", 2, 5, False)
        expected = {
            "n_scenes",
            "pixels_per_scene",
            "pixel_total",
            "at_0p5",
            "validation_choice",
            "at_validation_pick",
            "oracle_choice",
            "oracle_at_threshold",
            "optimism_gap_pooled_f1",
            "auprc",
        }

        assert set(summarize_test(counts, THRESHOLDS, buckets, chosen=None)) == expected
        assert set(summarize_test(counts, THRESHOLDS, buckets, chosen=chosen)) == expected

    def test_a_fixed_threshold_run_leaves_the_oracle_fields_empty(self):
        counts, buckets = _test_counts()

        summary = summarize_test(counts, THRESHOLDS, buckets, chosen=None)

        assert summary["oracle_choice"] is None
        assert summary["oracle_at_threshold"] is None

    def test_the_oracle_searches_the_same_objective_the_validation_pick_used(self):
        counts, buckets = _test_counts()
        chosen = ThresholdChoice(0.9, 0.5, "plume_scenes_f1", 2, 2, False)

        summary = summarize_test(counts, THRESHOLDS, buckets, chosen=chosen)

        assert summary["oracle_choice"]["objective"] == "plume_scenes_f1"

    def test_the_pixel_totals_and_scene_counts_are_recorded(self):
        counts, buckets = _test_counts()

        summary = summarize_test(counts, THRESHOLDS, buckets, chosen=None)

        assert summary["n_scenes"] == 3
        assert summary["pixels_per_scene"] == [1000, 1000, 1000]
        assert summary["pixel_total"] == 3000

    def test_auprc_is_included_only_when_every_grid_was_scored(self):
        counts, buckets = _test_counts()

        assert summarize_test(counts, THRESHOLDS, buckets, chosen=None)["auprc"] is None

    def test_the_summary_is_json_serialisable(self):
        counts, buckets = _test_counts()
        chosen = ThresholdChoice(0.9, 0.55, "pooled_f1", 2, 5, False)

        text = json.dumps(summarize_test(counts, THRESHOLDS, buckets, chosen=chosen))

        assert "optimism_gap_pooled_f1" in text

    def test_the_full_grid_adds_the_three_grid_auprc_table(self):
        counts = np.stack(
            [
                np.stack(
                    [
                        [100 - int(30 * t), int(200 * (1 - t)) + 5, int(30 * t), 5000]
                        for t in ALL_THRESHOLDS
                    ]
                )
                for _ in range(3)
            ]
        ).astype(np.int64)

        summary = summarize_test(
            counts, ALL_THRESHOLDS, ["strong", "weak", "plume_free"], chosen=None
        )

        assert list(summary["auprc"]) == ["paper16", "grid101", "logit"]
        assert summary["auprc"]["grid101"]["all_scenes"] > 0


def _union_counts():
    return np.stack(
        [
            np.stack(
                [
                    [100 - int(30 * t), int(200 * (1 - t)) + 5, int(30 * t), 5000]
                    for t in ALL_THRESHOLDS
                ]
            )
            for _ in range(3)
        ]
    ).astype(np.int64)


class TestFlattenMetrics:
    def test_the_scene_and_pixel_totals_are_reported(self):
        counts, buckets = _test_counts()
        flat = flatten_metrics(summarize_test(counts, THRESHOLDS, buckets, chosen=None))

        assert flat["n_scenes"] == 3.0
        assert flat["pixel_total"] == 3000.0

    def test_a_validation_pick_reports_its_widenings_edge_flag_and_the_oracle_score(self):
        counts, buckets = _test_counts()
        chosen = ThresholdChoice(0.9, 0.55, "pooled_f1", 2, 5, True, n_widenings=2)
        summary = summarize_test(counts, THRESHOLDS, buckets, chosen=chosen)

        flat = flatten_metrics(summary)

        assert flat["val_pick_widenings"] == 2.0
        assert flat["val_pick_at_edge"] == 1.0
        assert flat["oracle_pooled_f1"] == pytest.approx(summary["oracle_choice"]["score"])

    def test_the_auprc_table_becomes_one_metric_per_grid_and_population(self):
        summary = summarize_test(
            _union_counts(), ALL_THRESHOLDS, ["strong", "weak", "plume_free"], chosen=None
        )

        flat = flatten_metrics(summary)

        for grid in ("paper16", "grid101", "logit"):
            for population in ("all_scenes", "plume_scenes"):
                assert flat[f"auprc_{grid}_{population}"] == pytest.approx(
                    summary["auprc"][grid][population]
                )

    def test_a_run_without_the_union_grid_has_no_auprc_metrics(self):
        counts, buckets = _test_counts()

        flat = flatten_metrics(summarize_test(counts, THRESHOLDS, buckets, chosen=None))

        assert not any(key.startswith("auprc") for key in flat)

    def _summary(self, with_choice=True):
        counts, buckets = _test_counts()
        chosen = ThresholdChoice(0.9, 0.55, "pooled_f1", 2, 5, False) if with_choice else None
        return summarize_test(counts, THRESHOLDS, buckets, chosen=chosen)

    def test_flattens_the_half_threshold_metrics_with_safe_metric_names(self):
        flat = flatten_metrics(self._summary())

        assert flat["at_0p5_strong_f1"] == pytest.approx(0.8)
        assert flat["at_0p5_all_scenes_f1"] > 0
        assert flat["at_0p5_tile_fpr"] == 1.0
        assert flat["at_0p5_captured_plume_scenes"] == 1.0
        assert all(key.replace("_", "").isalnum() for key in flat), list(flat)

    def test_adds_the_validation_pick_and_the_oracle_gap_when_a_pick_was_made(self):
        flat = flatten_metrics(self._summary())

        assert flat["val_pick_threshold"] == pytest.approx(0.9)
        assert flat["val_pick_score"] == pytest.approx(0.55)
        assert flat["at_val_pick_tile_fpr"] == 0.0
        assert "oracle_threshold" in flat
        assert "optimism_gap_pooled_f1" in flat

    def test_leaves_out_what_was_not_computed(self):
        flat = flatten_metrics(self._summary(with_choice=False))

        assert not any(key.startswith(("val_pick", "at_val_pick", "oracle")) for key in flat)
        assert "optimism_gap_pooled_f1" not in flat

    def test_every_value_is_a_plain_float(self):
        flat = flatten_metrics(self._summary())

        assert all(type(value) is float for value in flat.values())

    def test_a_split_without_plume_free_scenes_has_no_fpr_metric(self):
        counts = np.array([[[80, 20, 20, 900], [60, 5, 40, 900]]], dtype=np.int64)
        summary = summarize_test(counts, THRESHOLDS, ["weak"], chosen=None)

        assert "at_0p5_tile_fpr" not in flatten_metrics(summary)
