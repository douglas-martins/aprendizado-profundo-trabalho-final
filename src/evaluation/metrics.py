"""Pixel-level segmentation metrics and threshold-sweep utilities."""

import torch

Counts = dict[str, float]


def pixel_f1(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Pixel-level F1 between binary `predictions` and `targets` (any matching shape).

    Returns 0.0, not NaN, when there are no predicted and no true positives
    -- the degenerate all-negative case explicitly checked here.
    """
    return f1_from_counts(confusion_matrix_counts(predictions, targets))


def confusion_matrix_counts(predictions: torch.Tensor, targets: torch.Tensor) -> Counts:
    """Pixel-level `{tp, fp, fn, tn}` between binary `predictions` and `targets`.

    Returns plain floats (not tensors) so counts from many batches can be
    summed with `add_counts` and logged as MLflow metrics directly.
    """
    return {
        "tp": (predictions * targets).sum().item(),
        "fp": (predictions * (1 - targets)).sum().item(),
        "fn": ((1 - predictions) * targets).sum().item(),
        "tn": ((1 - predictions) * (1 - targets)).sum().item(),
    }


def add_counts(first: Counts, second: Counts) -> Counts:
    """Sum two `{tp, fp, fn, tn}` count dicts key-wise -- pools counts across batches.

    Pooling counts before computing precision/recall/F1 (rather than
    averaging per-batch scores) is required for a correct metric: a batch
    with no positives at all has an undefined per-batch score, and
    averaging per-batch F1 does not equal the F1 of the pooled confusion
    matrix in general.
    """
    return {key: first[key] + second[key] for key in first}


def precision_from_counts(counts: Counts) -> float:
    """TP / (TP + FP). Returns 0.0, not NaN, when nothing was predicted positive."""
    denominator = counts["tp"] + counts["fp"]
    return counts["tp"] / denominator if denominator > 0 else 0.0


def recall_from_counts(counts: Counts) -> float:
    """TP / (TP + FN). Returns 0.0, not NaN, when there are no true positives."""
    denominator = counts["tp"] + counts["fn"]
    return counts["tp"] / denominator if denominator > 0 else 0.0


def f1_from_counts(counts: Counts) -> float:
    """Harmonic mean of precision/recall from pooled counts. Returns 0.0, not NaN, if both are 0."""
    denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
    return (2 * counts["tp"]) / denominator if denominator > 0 else 0.0


def confusion_matrix_from_counts(counts: Counts) -> torch.Tensor:
    """`[[TN, FP], [FN, TP]]` tensor from a pooled counts dict."""
    return torch.tensor([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])


def sweep_confusion_counts(
    probs: torch.Tensor, targets: torch.Tensor, thresholds: torch.Tensor
) -> torch.Tensor:
    """`(len(thresholds), 4)` tensor of summed `[tp, fp, fn, tn]` counts, one row per threshold.

    Vectorized over the threshold dimension so a full precision-recall sweep
    costs one pass over `probs`/`targets` per batch, not one pass per
    threshold -- the same incremental-accumulation reasoning as
    `confusion_matrix_counts`: rows from many batches sum directly (plain
    tensor addition), so PR-AUC over a large split never requires holding
    every prediction in memory at once.
    """

    flat_probs = probs.reshape(1, -1)
    flat_targets = targets.reshape(1, -1)
    predicted = (flat_probs > thresholds.reshape(-1, 1)).float()
    true_positive = (predicted * flat_targets).sum(dim=1)
    false_positive = (predicted * (1 - flat_targets)).sum(dim=1)
    false_negative = ((1 - predicted) * flat_targets).sum(dim=1)
    true_negative = ((1 - predicted) * (1 - flat_targets)).sum(dim=1)
    return torch.stack([true_positive, false_positive, false_negative, true_negative], dim=1)


def precision_recall_points_from_sweep(sweep_counts: torch.Tensor) -> list[tuple[float, float]]:
    """`(recall, precision)` per row of a `sweep_confusion_counts` tensor, in the given row order.

    Each row's precision/recall is 0.0, not NaN, when its denominator is
    zero -- same convention as `precision_from_counts`/`recall_from_counts`.
    """
    true_positive, false_positive, false_negative, _true_negative = sweep_counts.unbind(dim=1)
    precision = torch.where(
        true_positive + false_positive > 0,
        true_positive / (true_positive + false_positive),
        torch.zeros_like(true_positive),
    )
    recall = torch.where(
        true_positive + false_negative > 0,
        true_positive / (true_positive + false_negative),
        torch.zeros_like(true_positive),
    )

    return list(zip(recall.tolist(), precision.tolist(), strict=True))


def sort_points_by_recall(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sort `(recall, precision)` points ascending by recall.

    `precision_recall_points_from_sweep` returns points in threshold order
    (high threshold / low recall first), which both `average_precision_from_sweep`
    and any PR-curve plot need ascending instead -- factored out here so both
    share one definition of "ascending" rather than each re-sorting inline.
    """
    return sorted(points, key=lambda point: point[0])


def average_precision_from_sweep(sweep_counts: torch.Tensor) -> float:
    """Non-interpolated (step-function) average precision from a threshold sweep.

    Same convention this project already uses for the thesis's own AUPRC
    (`src/baselines/starcop/evaluation/paper_metrics.py`'s
    `non_interpolated_average_precision`/`derive_auprc`, not imported here --
    see this module's docstring for the reuse-vs-reimplement call): sort
    points ascending by recall, sum `(recall_n - recall_{n-1}) * precision_n`
    with `recall_0 = 0`, matching `sklearn.metrics.average_precision_score`'s
    convention -- no interpolation between points, no artificial endpoint at
    recall=1 if the curve doesn't reach it.
    """
    points = sort_points_by_recall(precision_recall_points_from_sweep(sweep_counts))
    average_precision = 0.0
    previous_recall = 0.0
    for recall, precision in points:
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
    return average_precision


def best_f1_operating_point(
    points: list[tuple[float, float]], thresholds: list[float]
) -> dict[str, float]:
    """Pick the `(threshold, recall, precision)` triple maximizing F1 from a threshold sweep.

    `points` is `(recall, precision)` per threshold, in the same order as
    `thresholds` -- exactly `evaluate.py`'s raw `precision_recall_curve`
    artifact order (unsorted, matching its own `pr_thresholds` sweep), not
    `sort_points_by_recall`'s ascending-by-recall order used for plotting.
    Own finding motivates this: the fixed 0.5 threshold is
    poorly calibrated for `pos_weight`-heavy raw-tier models, so reporting
    the F1-optimal point alongside the fixed-threshold one tells a more
    honest story without retraining anything.

    Ties keep the first-encountered threshold in the given input order --
    deterministic, and irrelevant for the default ascending-threshold sweep
    used everywhere in this project (a tie there would be a real plateau in
    the curve, not resolved by picking a particular side of it).
    """

    best_index = 0
    best_f1 = -1.0
    for index, (recall, precision) in enumerate(points):
        denominator = precision + recall
        f1 = (2 * precision * recall) / denominator if denominator > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_index = index
    best_recall, best_precision = points[best_index]
    return {
        "threshold": thresholds[best_index],
        "recall": best_recall,
        "precision": best_precision,
        "f1": best_f1,
    }


def patch_detection_counts(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, int]:
    """Per-patch "detected at all" counts: among patches with at least
    one true positive pixel, how many got at least one predicted positive pixel.

    `predictions`/`targets` are a batch of patches, `(B, ...)` -- every
    dimension after the batch dimension belongs to one patch. A false
    positive on an otherwise-negative patch is deliberately not counted: this
    metric answers "does the model notice a real plume at all," not overall
    precision, which the pixel-level confusion matrix already covers.
    """
    per_patch_dims = tuple(range(1, predictions.dim()))
    has_true_positive = targets.sum(dim=per_patch_dims) > 0
    has_predicted_positive = predictions.sum(dim=per_patch_dims) > 0
    return {
        "positive_patches": int(has_true_positive.sum().item()),
        "detected_patches": int((has_true_positive & has_predicted_positive).sum().item()),
    }


def detection_rate_from_counts(counts: dict[str, int]) -> float:
    """Fraction of positive patches detected. Returns 0.0, not NaN, if there are none."""
    total = counts["positive_patches"]
    return counts["detected_patches"] / total if total > 0 else 0.0
