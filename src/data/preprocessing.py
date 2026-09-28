"""Normalize model inputs using the fixed band offset and factor contract.

The function covers inputs only. The binary label remains a raw 0/1 value and
is not passed through this normalization.
"""

import numpy as np

BAND_NORMALIZATION = {
    "mag1c": {"offset": 0, "factor": 1750, "clip": (0, 2)},
    "TOA_AVIRIS_640nm": {"offset": 0, "factor": 60, "clip": (0, 2)},
    "TOA_AVIRIS_550nm": {"offset": 0, "factor": 60, "clip": (0, 2)},
    "TOA_AVIRIS_460nm": {"offset": 0, "factor": 60, "clip": (0, 2)},
}


def normalize_band(
    array: np.ndarray, offset: float, factor: float, clip: tuple[float, float]
) -> np.ndarray:
    """Apply STARCOP's fixed (array - offset) / factor, clipped to `clip`.

    Same formula `DataNormalizer.normalize_x` applies via `torch.clamp` --
    this is why the real `mag1c` sentinel value (~100000) is not a special
    case: at this factor it already lands far outside `clip` and gets clamped
    like any other out-of-range value, regardless of its raw magnitude.
    """
    clip_min, clip_max = clip
    scaled = (array.astype("float32") - offset) / factor
    return np.clip(scaled, clip_min, clip_max).astype("float32")
