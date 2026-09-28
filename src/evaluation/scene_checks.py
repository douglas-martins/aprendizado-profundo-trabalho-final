"""Helpers for the regression checks of the paper-protocol evaluation.

The `patches` diagnostic of `scene_inference` sums per-patch counts, so a pixel is counted once for
every 128x128 patch (stride 64) that contains it. These functions compute, from the labels alone
and independently of any model, how many times each pixel is counted -- which pins down exactly
what the patch-mode positive and total pixel counts of a scene must be.
"""

import numpy as np


def _axis_cover(scene_size: int, patch_size: int, stride: int) -> np.ndarray:
    """How many windows contain each coordinate along one axis."""
    if (scene_size - patch_size) % stride != 0:
        raise ValueError(
            f"windows of {patch_size} with stride {stride} do not tile a scene of {scene_size}"
        )
    cover = np.zeros(scene_size, dtype=np.int64)
    for start in range(0, scene_size - patch_size + 1, stride):
        cover[start : start + patch_size] += 1
    return cover


def patch_cover_counts(
    scene_size: int = 512, patch_size: int = 128, stride: int = 64
) -> np.ndarray:
    """`(scene_size, scene_size)` array: the number of patch windows that contain each pixel."""
    axis = _axis_cover(scene_size, patch_size, stride)
    return np.outer(axis, axis)


def expected_patch_positive_pixels(label: np.ndarray, cover: np.ndarray) -> int:
    """Positive label pixels of one scene as the `patches` mode counts them (with multiplicity).

    The patch-mode `tp + fn` of that scene, at any threshold, must equal this exactly.
    """
    if label.shape != cover.shape:
        raise ValueError(f"label shape {label.shape} != cover shape {cover.shape}")
    return int(((label > 0) * cover).sum())


def compare_pooled_counts(first, second) -> dict:
    """Compare two `[tp, fp, fn, tn]` vectors: `identical` and each count's `first - second`."""
    if len(first) != 4 or len(second) != 4:
        raise ValueError("both inputs need exactly four counts: tp, fp, fn, tn")
    differences = {
        name: int(a) - int(b) for name, a, b in zip(("tp", "fp", "fn", "tn"), first, second)
    }
    return {"identical": not any(differences.values()), "differences": differences}
