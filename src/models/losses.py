"""Loss functions and non-degeneracy checks for segmentation training."""

import pandas as pd
import torch


def compute_pos_weight(patches_df: pd.DataFrame) -> float:
    """Return background:positive pixel ratio from `frac_positives`, for `BCEWithLogitsLoss`.

    All patches are the same size, so the plain mean of each patch's own
    positive-pixel fraction equals the overall fraction across the whole
    set -- no need to re-read label rasters; `patch_extract.py` already
    computed `frac_positives` once per patch.
    """
    positive_fraction = patches_df["frac_positives"].mean()
    if positive_fraction == 0:
        raise ValueError("patches_df has no positive pixels at all; pos_weight is undefined")
    return (1 - positive_fraction) / positive_fraction


def build_loss(pos_weight: float) -> torch.nn.BCEWithLogitsLoss:
    """Build `BCEWithLogitsLoss` weighted by `pos_weight` (background:positive ratio)."""
    return torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))


def is_degenerate(sigmoid_predictions: torch.Tensor, epsilon: float = 1e-6) -> bool:
    """True if `sigmoid_predictions` (post-sigmoid, in [0, 1]) collapsed to all-0 or all-1.

    This guards against a model that gives up and predicts every pixel
    negative (or, less likely but still checked, every pixel positive) given
    the severe class imbalance.
    """
    return bool((sigmoid_predictions < epsilon).all() or (sigmoid_predictions > 1 - epsilon).all())
