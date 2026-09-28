"""Implement the scene-level evaluation protocol over thresholded counts."""

import numpy as np
import torch

from evaluation.metrics import average_precision_from_sweep

BUCKETS = ("strong", "weak", "plume_free")
PLUME_BUCKETS = ("strong", "weak")
TILE_PIXEL_THRESHOLD = 10


PAPER_THRESHOLDS = np.sort(
    np.array(
        [0.0, 1e-3, 1e-2, *np.arange(0.5, 0.96, 0.05).tolist(), 0.99, 0.995, 0.999],
        dtype=np.float32,
    )
)

GRID_101 = torch.linspace(0.0, 1.0, steps=101).numpy()


MAX_LOGIT_HALF_RANGE = 15.0


def logit_grid(half_range: float = 10.0, step: float = 0.25) -> np.ndarray:
    """Float32 probabilities that are uniform in logit space over `[-half_range, half_range]`.

    Dense near both 0 and 1, where a model trained with a large `pos_weight` puts its useful
    thresholds (the best F1 used to sit on the 0.99 edge of `GRID_101`). Wider ranges are for
    when a selected threshold lands on the edge; at the default 0.25 step the values stay
    distinct float32s up to `MAX_LOGIT_HALF_RANGE`, which is therefore the limit.
    """
    if not 0 < half_range <= MAX_LOGIT_HALF_RANGE:
        raise ValueError(f"half_range must be in (0, {MAX_LOGIT_HALF_RANGE}], got {half_range}")
    logits = np.linspace(-half_range, half_range, int(round(2 * half_range / step)) + 1)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


LOGIT_GRID = logit_grid()

ALL_THRESHOLDS = np.unique(np.concatenate([PAPER_THRESHOLDS, GRID_101, LOGIT_GRID]))
AUPRC_GRIDS = {"paper16": PAPER_THRESHOLDS, "grid101": GRID_101, "logit": LOGIT_GRID}
AUPRC_POPULATIONS = ("all_scenes", "plume_scenes")


def threshold_index(thresholds, value: float) -> int:
    """Index of `value` in `thresholds`, compared exactly as float32 (the dtype scoring used)."""
    matches = np.flatnonzero(np.asarray(thresholds, dtype=np.float32) == np.float32(value))
    if len(matches) == 0:
        raise ValueError(f"threshold {value!r} is not on the threshold grid")
    return int(matches[0])


def select_thresholds(counts: np.ndarray, thresholds, wanted) -> tuple[np.ndarray, np.ndarray]:
    """`(counts, thresholds)` restricted to the `wanted` thresholds, in the order requested.

    Every wanted threshold must be one the counts were computed at (exact float32 match);
    otherwise a `ValueError` names all the missing ones.
    """
    available = np.asarray(thresholds, dtype=np.float32)
    wanted = np.asarray(wanted, dtype=np.float32)
    missing = [round(float(value), 6) for value in wanted if value not in available]
    if missing:
        raise ValueError(f"thresholds not on the threshold grid: {missing}")
    indices = [threshold_index(available, value) for value in wanted]
    return counts[:, indices, :], available[indices]


def confusion_metrics(confusion) -> dict[str, float]:
    """Precision, recall and F1 from a `[tp, fp, fn, tn]` vector; 0.0 (not NaN) when undefined."""
    tp, fp, fn = int(confusion[0]), int(confusion[1]), int(confusion[2])
    predicted, actual, f1_denominator = tp + fp, tp + fn, 2 * tp + fp + fn
    return {
        "precision": tp / predicted if predicted else 0.0,
        "recall": tp / actual if actual else 0.0,
        "f1": 2 * tp / f1_denominator if f1_denominator else 0.0,
    }


def _scene_mask(buckets, names) -> np.ndarray:
    """Boolean mask of the scenes whose bucket is one of `names` (a name or a tuple of names)."""
    names = (names,) if isinstance(names, str) else tuple(names)
    unknown = [name for name in names if name not in BUCKETS]
    if unknown:
        raise ValueError(f"unknown bucket(s): {unknown}; expected some of {BUCKETS}")
    return np.isin(np.asarray(buckets), names)


def _check_one_bucket_per_scene(counts: np.ndarray, buckets) -> None:
    if len(buckets) != counts.shape[0]:
        raise ValueError(
            f"one bucket per scene is required: {len(buckets)} buckets for {counts.shape[0]} scenes"
        )


def check_counts_layout(counts: np.ndarray, thresholds, buckets) -> None:
    """Raise `ValueError` unless `counts` has a row per bucket label and a column per threshold."""
    _check_one_bucket_per_scene(counts, buckets)
    if counts.shape[1] != len(thresholds):
        raise ValueError(
            f"counts hold {counts.shape[1]} thresholds but {len(thresholds)} were given"
        )


def bucket_confusion(counts: np.ndarray, buckets, bucket, t_index: int) -> np.ndarray:
    """`[tp, fp, fn, tn]` pooled over the scenes in `bucket` (a name or tuple) at one threshold."""
    _check_one_bucket_per_scene(counts, buckets)
    return counts[_scene_mask(buckets, bucket), t_index, :].sum(axis=0)


def bucket_metrics(counts: np.ndarray, buckets, t_index: int) -> dict[str, dict[str, float]]:
    """Precision/recall/F1 for `strong`, `weak`, `plume_scenes` (both pooled) and `all_scenes`.

    `plume_scenes` is the assumed population behind a paper's single "F1 (all plumes)" figure
    when it is not stated; `all_scenes` pools everything, plume-free scenes included, like this
    project's older patch-pooled protocol.
    """
    groups = {
        "strong": "strong",
        "weak": "weak",
        "plume_scenes": PLUME_BUCKETS,
        "all_scenes": BUCKETS,
    }
    return {
        name: confusion_metrics(bucket_confusion(counts, buckets, bucket, t_index))
        for name, bucket in groups.items()
    }


def _active_pixels(counts: np.ndarray, mask: np.ndarray, t_index: int) -> np.ndarray:
    """Predicted-positive pixels (`tp + fp`) of the scenes in `mask` at one threshold."""
    return counts[mask, t_index, 0] + counts[mask, t_index, 1]


def tile_fpr(
    counts: np.ndarray, buckets, t_index: int, *, min_pixels: int = TILE_PIXEL_THRESHOLD
) -> float:
    """Fraction of plume-free scenes with more than `min_pixels` predicted-positive pixels."""
    _check_one_bucket_per_scene(counts, buckets)
    mask = _scene_mask(buckets, "plume_free")
    if not mask.any():
        raise ValueError("no plume-free scenes: the tile FPR is undefined")
    return float((_active_pixels(counts, mask, t_index) > min_pixels).mean())


def captured_plumes(
    counts: np.ndarray, buckets, t_index: int, *, min_pixels: int = TILE_PIXEL_THRESHOLD
) -> dict[str, dict]:
    """Plume scenes whose tile is marked (> `min_pixels` active pixels) and overlaps the label.

    Returns `{"strong"|"weak"|"plume_scenes": {"captured", "total", "rate"}}`; `rate` is `None`
    for a bucket with no scenes.
    """
    _check_one_bucket_per_scene(counts, buckets)
    result = {}
    for name, bucket in (
        ("strong", "strong"),
        ("weak", "weak"),
        ("plume_scenes", PLUME_BUCKETS),
    ):
        mask = _scene_mask(buckets, bucket)
        marked = _active_pixels(counts, mask, t_index) > min_pixels
        overlapping = counts[mask, t_index, 0] >= 1
        captured = int((marked & overlapping).sum())
        total = int(mask.sum())
        result[name] = {
            "captured": captured,
            "total": total,
            "rate": captured / total if total else None,
        }
    return result


def auprc(counts: np.ndarray, thresholds, buckets, *, population: str = "all_scenes") -> float:
    """Non-interpolated average precision of the pixel counts pooled over `population`.

    `population` is `"all_scenes"` (every scene, as the paper's validation code pools them) or
    `"plume_scenes"` (strong and weak scenes only). The grid is whatever `thresholds` holds;
    see `auprc_by_grid` for the three grids used by the comparison.
    """
    if population not in AUPRC_POPULATIONS:
        raise ValueError(f"population must be one of {AUPRC_POPULATIONS}, got {population!r}")
    check_counts_layout(counts, thresholds, buckets)
    mask = _scene_mask(buckets, BUCKETS if population == "all_scenes" else PLUME_BUCKETS)
    if not mask.any():
        raise ValueError(f"no scenes in the {population} population")
    pooled = counts[mask].sum(axis=0)
    return average_precision_from_sweep(torch.tensor(pooled, dtype=torch.float64))


def auprc_by_grid(counts: np.ndarray, thresholds, buckets) -> dict[str, dict[str, float]]:
    """AUPRC on each of `AUPRC_GRIDS` for each population: `{grid: {population: value}}`.

    `counts` must have been computed on a grid containing all of them (`ALL_THRESHOLDS`).
    """
    table = {}
    for grid_name, grid in AUPRC_GRIDS.items():
        subset, kept = select_thresholds(counts, thresholds, grid)
        table[grid_name] = {
            population: auprc(subset, kept, buckets, population=population)
            for population in AUPRC_POPULATIONS
        }
    return table
