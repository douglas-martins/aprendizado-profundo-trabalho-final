import pytest
import torch

from evaluation.metrics import (
    add_counts,
    average_precision_from_sweep,
    best_f1_operating_point,
    confusion_matrix_counts,
    confusion_matrix_from_counts,
    detection_rate_from_counts,
    f1_from_counts,
    patch_detection_counts,
    pixel_f1,
    precision_from_counts,
    precision_recall_points_from_sweep,
    recall_from_counts,
    sort_points_by_recall,
    sweep_confusion_counts,
)


class TestPixelF1:
    def test_perfect_prediction_scores_one(self):
        predictions = torch.tensor([1.0, 0.0, 1.0, 0.0])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert pixel_f1(predictions, targets) == 1.0

    def test_all_wrong_scores_zero(self):
        predictions = torch.tensor([1.0, 1.0])
        targets = torch.tensor([0.0, 0.0])
        assert pixel_f1(predictions, targets) == 0.0

    def test_no_predicted_or_true_positives_scores_zero_not_nan(self):

        predictions = torch.zeros(4)
        targets = torch.zeros(4)
        result = pixel_f1(predictions, targets)
        assert result == 0.0
        assert not torch.isnan(torch.tensor(result))

    def test_matches_hand_computed_precision_recall(self):

        predictions = torch.tensor([1.0, 1.0, 0.0])
        targets = torch.tensor([1.0, 0.0, 1.0])
        assert pixel_f1(predictions, targets) == 0.5


class TestConfusionMatrixCounts:
    def test_perfect_prediction_counts_only_tp_and_tn(self):
        predictions = torch.tensor([1.0, 0.0, 1.0, 0.0])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert confusion_matrix_counts(predictions, targets) == {
            "tp": 2.0,
            "fp": 0.0,
            "fn": 0.0,
            "tn": 2.0,
        }

    def test_matches_hand_computed_case(self):

        predictions = torch.tensor([1.0, 1.0, 0.0])
        targets = torch.tensor([1.0, 0.0, 1.0])
        assert confusion_matrix_counts(predictions, targets) == {
            "tp": 1.0,
            "fp": 1.0,
            "fn": 1.0,
            "tn": 0.0,
        }


class TestAddCounts:
    def test_sums_every_key(self):
        first = {"tp": 1.0, "fp": 2.0, "fn": 0.0, "tn": 3.0}
        second = {"tp": 4.0, "fp": 0.0, "fn": 1.0, "tn": 2.0}
        assert add_counts(first, second) == {"tp": 5.0, "fp": 2.0, "fn": 1.0, "tn": 5.0}


class TestPrecisionRecallF1FromCounts:
    def test_matches_hand_computed_case(self):
        counts = {"tp": 1.0, "fp": 1.0, "fn": 1.0, "tn": 0.0}
        assert precision_from_counts(counts) == 0.5
        assert recall_from_counts(counts) == 0.5
        assert f1_from_counts(counts) == 0.5

    def test_perfect_prediction_scores_one(self):
        counts = {"tp": 4.0, "fp": 0.0, "fn": 0.0, "tn": 6.0}
        assert precision_from_counts(counts) == 1.0
        assert recall_from_counts(counts) == 1.0
        assert f1_from_counts(counts) == 1.0

    def test_no_predicted_positives_precision_is_zero_not_nan(self):
        counts = {"tp": 0.0, "fp": 0.0, "fn": 5.0, "tn": 10.0}
        result = precision_from_counts(counts)
        assert result == 0.0
        assert not torch.isnan(torch.tensor(result))

    def test_no_true_positives_recall_is_zero_not_nan(self):
        counts = {"tp": 0.0, "fp": 3.0, "fn": 0.0, "tn": 10.0}
        result = recall_from_counts(counts)
        assert result == 0.0
        assert not torch.isnan(torch.tensor(result))

    def test_all_zero_counts_f1_is_zero_not_nan(self):
        counts = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 10.0}
        result = f1_from_counts(counts)
        assert result == 0.0
        assert not torch.isnan(torch.tensor(result))

    def test_denominator_of_exactly_one_is_still_a_valid_denominator(self):

        assert precision_from_counts({"tp": 1.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}) == 1.0
        assert recall_from_counts({"tp": 1.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}) == 1.0

        assert f1_from_counts({"tp": 0.5, "fp": 0.0, "fn": 0.0, "tn": 0.0}) == 1.0


class TestConfusionMatrixFromCounts:
    def test_layout_is_tn_fp_fn_tp(self):

        counts = {"tp": 1.0, "fp": 2.0, "fn": 3.0, "tn": 4.0}
        matrix = confusion_matrix_from_counts(counts)
        assert torch.equal(matrix, torch.tensor([[4.0, 2.0], [3.0, 1.0]]))


class TestSweepConfusionCounts:
    def test_matches_hand_computed_counts_per_threshold(self):

        probs = torch.tensor([0.9, 0.4, 0.1, 0.6])
        targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
        thresholds = torch.tensor([0.0, 0.5, 1.0])

        counts = sweep_confusion_counts(probs, targets, thresholds)

        assert torch.equal(
            counts, torch.tensor([[2.0, 2.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 2.0, 2.0]])
        )

    def test_accepts_any_matching_shape_not_only_1d(self):
        probs = torch.tensor([[0.9, 0.1], [0.4, 0.6]])
        targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        thresholds = torch.tensor([0.5])

        counts = sweep_confusion_counts(probs, targets, thresholds)

        assert torch.equal(counts, torch.tensor([[1.0, 1.0, 1.0, 1.0]]))

    def test_probability_exactly_at_threshold_is_not_predicted_positive(self):

        probs = torch.tensor([0.5])
        targets = torch.tensor([1.0])
        thresholds = torch.tensor([0.5])

        counts = sweep_confusion_counts(probs, targets, thresholds)

        assert torch.equal(counts, torch.tensor([[0.0, 0.0, 1.0, 0.0]]))


class TestPrecisionRecallPointsFromSweep:
    def test_matches_hand_computed_points(self):
        sweep_counts = torch.tensor(
            [[2.0, 2.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 2.0, 2.0]]
        )

        points = precision_recall_points_from_sweep(sweep_counts)

        assert points == pytest.approx([(1.0, 0.5), (0.5, 0.5), (0.0, 0.0)])

    def test_all_zero_row_is_zero_not_nan(self):
        sweep_counts = torch.tensor([[0.0, 0.0, 0.0, 5.0]])

        points = precision_recall_points_from_sweep(sweep_counts)

        assert points[0] == (0.0, 0.0)

    def test_denominator_of_exactly_one_is_still_a_valid_denominator(self):

        sweep_counts = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        points = precision_recall_points_from_sweep(sweep_counts)

        assert points == [(1.0, 1.0)]


class TestSortPointsByRecall:
    def test_sorts_ascending_by_recall(self):

        points = [(1.0, 0.5), (0.5, 0.5), (0.0, 0.0)]

        assert sort_points_by_recall(points) == [(0.0, 0.0), (0.5, 0.5), (1.0, 0.5)]

    def test_already_sorted_input_is_unchanged(self):
        points = [(0.0, 0.0), (0.5, 0.5), (1.0, 0.5)]

        assert sort_points_by_recall(points) == points

    def test_sorts_only_by_recall_ties_keep_original_relative_order(self):

        points = [(0.5, 0.9), (0.5, 0.1)]

        assert sort_points_by_recall(points) == [(0.5, 0.9), (0.5, 0.1)]


class TestAveragePrecisionFromSweep:
    def test_matches_hand_computed_non_interpolated_average_precision(self):

        sweep_counts = torch.tensor(
            [[2.0, 2.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 2.0, 2.0]]
        )

        assert average_precision_from_sweep(sweep_counts) == pytest.approx(0.5)

    def test_all_zero_counts_is_zero_not_nan(self):
        sweep_counts = torch.tensor([[0.0, 0.0, 0.0, 5.0]])

        result = average_precision_from_sweep(sweep_counts)

        assert result == 0.0
        assert not torch.isnan(torch.tensor(result))

    def test_first_recall_step_is_measured_from_zero(self):

        sweep_counts = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

        assert average_precision_from_sweep(sweep_counts) == pytest.approx(0.5)


class TestPatchDetectionCounts:
    def test_counts_a_correctly_detected_positive_patch(self):
        predictions = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]])
        targets = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]])

        counts = patch_detection_counts(predictions, targets)

        assert counts == {"positive_patches": 1, "detected_patches": 1}

    def test_a_missed_positive_patch_is_not_counted_as_detected(self):
        predictions = torch.zeros(1, 1, 2, 2)
        targets = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])

        counts = patch_detection_counts(predictions, targets)

        assert counts == {"positive_patches": 1, "detected_patches": 0}

    def test_a_false_positive_on_a_negative_patch_is_not_counted(self):

        predictions = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
        targets = torch.zeros(1, 1, 2, 2)

        counts = patch_detection_counts(predictions, targets)

        assert counts == {"positive_patches": 0, "detected_patches": 0}

    def test_counts_per_patch_across_a_multi_channel_batch(self):

        predictions = torch.zeros(4, 2, 3, 3)
        targets = torch.zeros(4, 2, 3, 3)
        targets[0, 0, 0, 0] = 1.0
        predictions[0, 0, 0, 0] = 1.0
        targets[1, 1, 1, 1] = 1.0
        targets[2, 0, 2, 2] = 1.0
        targets[2, 1, 0, 1] = 1.0
        predictions[2, 0, 2, 2] = 1.0

        counts = patch_detection_counts(predictions, targets)

        assert counts == {"positive_patches": 3, "detected_patches": 2}


class TestBestF1OperatingPoint:
    def test_picks_the_threshold_with_the_highest_f1(self):

        points = [(1.0, 0.5), (1.0, 1.0), (0.0, 0.0)]
        thresholds = [0.0, 0.5, 1.0]

        result = best_f1_operating_point(points, thresholds)

        assert result == {"threshold": 0.5, "recall": 1.0, "precision": 1.0, "f1": 1.0}

    def test_ties_return_the_first_matching_threshold_in_input_order(self):

        points = [(1.0, 1.0), (1.0, 1.0)]
        thresholds = [0.2, 0.7]

        result = best_f1_operating_point(points, thresholds)

        assert result["threshold"] == 0.2

    def test_all_zero_precision_and_recall_is_zero_f1_not_nan(self):

        points = [(0.0, 0.0)]
        thresholds = [1.0]

        result = best_f1_operating_point(points, thresholds)

        assert result == {"threshold": 1.0, "recall": 0.0, "precision": 0.0, "f1": 0.0}
        assert not torch.isnan(torch.tensor(result["f1"]))

    def test_low_precision_and_recall_summing_to_at_most_one_still_scores_nonzero_f1(self):

        points = [(0.0, 0.0), (0.3, 0.2), (0.1, 0.1)]
        thresholds = [0.1, 0.5, 0.9]

        result = best_f1_operating_point(points, thresholds)

        assert result["threshold"] == 0.5
        assert result["f1"] == pytest.approx(0.24)

    def test_matches_a_realistic_sweep_shaped_curve(self):

        points = [(0.98, 0.06), (0.85, 0.20), (0.50, 0.60), (0.10, 0.90), (0.0, 0.0)]
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

        result = best_f1_operating_point(points, thresholds)

        assert result["threshold"] == 0.5
        assert result["f1"] == pytest.approx(0.5454545454545454)

    def test_computes_the_first_point_f1_with_the_harmonic_mean_formula(self):
        result = best_f1_operating_point([(0.4, 0.6), (0.1, 0.1)], [0.1, 0.2])

        assert result == {"threshold": 0.1, "recall": 0.4, "precision": 0.6, "f1": 0.48}


class TestDetectionRateFromCounts:
    def test_matches_hand_computed_fraction(self):
        counts = {"positive_patches": 4, "detected_patches": 3}
        assert detection_rate_from_counts(counts) == 0.75

    def test_no_positive_patches_returns_zero_not_nan(self):
        counts = {"positive_patches": 0, "detected_patches": 0}
        result = detection_rate_from_counts(counts)
        assert result == 0.0
        assert not torch.isnan(torch.tensor(result))

    def test_total_of_exactly_one_is_still_a_valid_denominator(self):
        counts = {"positive_patches": 1, "detected_patches": 1}
        assert detection_rate_from_counts(counts) == 1.0
