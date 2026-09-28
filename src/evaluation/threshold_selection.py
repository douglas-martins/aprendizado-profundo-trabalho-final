"""Select a validation threshold from pooled scene-level predictions."""

from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from evaluation.paper_protocol import (
    BUCKETS,
    PLUME_BUCKETS,
    _scene_mask,
    bucket_metrics,
    captured_plumes,
    check_counts_layout,
    confusion_metrics,
    logit_grid,
    threshold_index,
    tile_fpr,
)

OBJECTIVES = {"pooled_f1": BUCKETS, "plume_scenes_f1": PLUME_BUCKETS}
DEFAULT_HALF_RANGES = (10.0, 12.5, 15.0)


class ThresholdChoice(NamedTuple):
    """A selected threshold and how it was found.

    `score` is the objective at `threshold`; `at_edge` says the pick sits on the first or last
    point of the final grid (so a better threshold may lie beyond it); `n_widenings` counts how
    many times `select_with_widening` had to widen the grid first.
    """

    threshold: float
    score: float
    objective: str
    grid_size: int
    n_scenes: int
    at_edge: bool
    n_widenings: int = 0


def select_threshold(val_counts, thresholds, val_buckets, *, objective="pooled_f1"):
    """The threshold on the grid `thresholds` that maximises `objective` on validation.

    `val_counts` is `(n_scenes, n_thresholds, 4)` `[tp, fp, fn, tn]` from `score_scenes` on the
    validation scenes, `val_buckets` one bucket label per scene. Ties go to the lowest threshold.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {tuple(OBJECTIVES)}, got {objective!r}")
    check_counts_layout(val_counts, thresholds, val_buckets)
    mask = _scene_mask(val_buckets, OBJECTIVES[objective])
    if not mask.any():
        raise ValueError(f"no scenes for the {objective} objective")
    pooled = val_counts[mask].sum(axis=0)
    scores = np.array([confusion_metrics(row)["f1"] for row in pooled])
    best = int(np.argmax(scores))
    return ThresholdChoice(
        threshold=float(np.asarray(thresholds, dtype=np.float32)[best]),
        score=float(scores[best]),
        objective=objective,
        grid_size=len(scores),
        n_scenes=int(mask.sum()),
        at_edge=best in (0, len(scores) - 1),
    )


def select_with_widening(
    rescore: Callable[[np.ndarray], np.ndarray],
    val_buckets,
    *,
    objective="pooled_f1",
    half_ranges=DEFAULT_HALF_RANGES,
) -> ThresholdChoice:
    """`select_threshold` on logit grids of growing `half_ranges` until the pick is interior.

    `rescore(grid)` must return the validation counts at the float32 thresholds `grid`. Stops at
    the first grid whose pick is not on its edge; if even the widest grid ends on an edge the
    result says so (`at_edge=True`) rather than pretending it found an optimum.
    """
    if len(half_ranges) == 0:
        raise ValueError("half_ranges must hold at least one grid half-range")
    for widening, half_range in enumerate(half_ranges):
        grid = logit_grid(half_range)
        choice = select_threshold(rescore(grid), grid, val_buckets, objective=objective)
        choice = choice._replace(n_widenings=widening)
        if not choice.at_edge:
            break
    return choice


def apply_threshold(counts, thresholds, buckets, threshold: float) -> dict:
    """The paper-protocol metrics of `counts` at one already-chosen `threshold`.

    Meant to be called once per model on the test scenes, with a threshold chosen on validation.
    `threshold` must be one of the thresholds the counts were computed at (exact float32 match).
    The tile FPR is `None` when the scenes include no plume-free ones (e.g. mini's test split).
    """
    check_counts_layout(counts, thresholds, buckets)
    t_index = threshold_index(thresholds, threshold)
    n_by_bucket = {name: int(_scene_mask(buckets, name).sum()) for name in BUCKETS}
    return {
        "threshold": float(np.asarray(thresholds, dtype=np.float32)[t_index]),
        "n_scenes": int(counts.shape[0]),
        "n_scenes_by_bucket": n_by_bucket,
        "metrics": bucket_metrics(counts, buckets, t_index),
        "tile_fpr": tile_fpr(counts, buckets, t_index) if n_by_bucket["plume_free"] else None,
        "captured": captured_plumes(counts, buckets, t_index),
    }


def oracle_threshold(counts, thresholds, buckets, *, objective="pooled_f1") -> ThresholdChoice:
    """The best threshold on the *same* scenes it will be scored on: an optimistic oracle.

    Never a headline number. It exists to measure how much a validation-picked threshold gives up
    (the "optimism gap") and to relabel the older test-curve calibration honestly.
    """
    return select_threshold(counts, thresholds, buckets, objective=objective)
