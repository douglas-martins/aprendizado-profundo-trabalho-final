import pandas as pd
import pytest
import torch

from models.losses import build_loss, compute_pos_weight, is_degenerate


class TestComputePosWeight:
    def test_uses_the_mean_frac_positives_across_equal_size_patches(self):

        df = pd.DataFrame({"frac_positives": [0.1, 0.1, 0.1]})
        assert compute_pos_weight(df) == pytest.approx(9.0)

    def test_matches_the_known_mini_imbalance_ratio(self):

        df = pd.DataFrame({"frac_positives": [72764 / 6422528] * 392})
        assert compute_pos_weight(df) == pytest.approx(87.26518608102909, rel=1e-4)

    def test_raises_on_all_zero_frac_positives(self):
        df = pd.DataFrame({"frac_positives": [0.0, 0.0]})
        with pytest.raises(ValueError, match="no positive pixels"):
            compute_pos_weight(df)

    def test_raises_with_the_exact_documented_message(self):
        df = pd.DataFrame({"frac_positives": [0.0]})
        with pytest.raises(
            ValueError, match="^patches_df has no positive pixels at all; pos_weight is undefined$"
        ):
            compute_pos_weight(df)


class TestBuildLoss:
    def test_returns_a_bce_with_logits_loss_carrying_pos_weight(self):
        loss_fn = build_loss(pos_weight=9.0)
        assert isinstance(loss_fn, torch.nn.BCEWithLogitsLoss)
        assert loss_fn.pos_weight.item() == pytest.approx(9.0)

    def test_pos_weight_actually_penalizes_missed_positives_more(self):

        logits = torch.tensor([-2.0, -2.0])
        target_positive = torch.tensor([1.0, 0.0])
        weighted = build_loss(pos_weight=20.0)
        unweighted = build_loss(pos_weight=1.0)
        assert weighted(logits, target_positive) > unweighted(logits, target_positive)


class TestIsDegenerate:
    def test_flags_all_zero_predictions(self):
        predictions = torch.zeros(4, 1, 8, 8)
        assert is_degenerate(predictions) is True

    def test_flags_all_one_predictions(self):
        predictions = torch.ones(4, 1, 8, 8)
        assert is_degenerate(predictions) is True

    def test_does_not_flag_a_mixed_prediction(self):
        predictions = torch.zeros(4, 1, 8, 8)
        predictions[0, 0, 0, 0] = 1.0
        assert is_degenerate(predictions) is False

    def test_value_exactly_at_the_lower_epsilon_boundary_is_not_flagged(self):

        predictions = torch.full((4, 1, 8, 8), 1e-6)
        assert is_degenerate(predictions, epsilon=1e-6) is False

    def test_value_exactly_at_the_upper_epsilon_boundary_is_not_flagged(self):

        predictions = torch.full((4, 1, 8, 8), 1 - 1e-6)
        assert is_degenerate(predictions, epsilon=1e-6) is False
