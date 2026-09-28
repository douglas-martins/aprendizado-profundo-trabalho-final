"""Flat, reduced-precision on-disk cache for decoded/normalized patches.

`PatchDataset`'s RAM-bound LRU cache (see `dataset.py`) helps `starcop_mini`
(its whole ~125 MB train split fits in one worker's budget) but barely
moves `starcop_raw`'s 141k-patch train split, where each worker's bounded
cache covers only a fraction of its per-epoch share and the dataset is
overwhelmingly disk-I/O-bound regardless. This module is the on-disk
counterpart: a one-time precompute (`precompute_patch_cache.py`) writes
every patch's decoded, normalized (pre-augmentation) pair once, as flat
`float16` input / `uint8` output arrays -- unlike the RAM cache, this isn't
duplicated per worker process and isn't bounded by per-worker memory, so it
gives `starcop_raw` the same "cold once, then fast" behavior `starcop_mini`
already gets from RAM caching alone. Benchmarked at ~95x faster than the
live GeoTIFF windowed read it replaces (single process, real raw patches).

`float16` is safe here because every input band is clipped to a small fixed
range (`BAND_NORMALIZATION`'s `clip` in `preprocessing.py`, e.g. `(0, 2)`)
before this cache ever sees it -- nowhere near float16's overflow point,
and with more precision than the normalization itself preserves. The label
is exact 0/1, so `uint8` is lossless.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

_IDENTITY_COLUMNS = ["folder", "window_col_off", "window_row_off", "window_width", "window_height"]
_MANIFEST_NAME = "manifest.json"
_INPUTS_NAME = "inputs.npy"
_OUTPUTS_NAME = "outputs.npy"


def fingerprint(patches_df: pd.DataFrame) -> str:
    """Order-sensitive content fingerprint of `patches_df`'s patch-identifying columns.

    Cache array position `i` corresponds to `patches_df.iloc[i]`, so a
    fingerprint must change if the rows are reordered, resized, or point at
    different windows -- not just if their *set* of contents differs.
    Restricted to the columns `PatchDataset.__getitem__` actually reads,
    so unrelated columns (e.g. `has_plume`) can't cause a false mismatch.
    """

    identity_frame = patches_df[_IDENTITY_COLUMNS].reset_index(drop=True)
    hashes = pd.util.hash_pandas_object(identity_frame)
    return hashlib.sha256(hashes.to_numpy().tobytes()).hexdigest()


def is_valid(cache_dir: Path, patches_df: pd.DataFrame) -> bool:
    """Whether `cache_dir` holds a manifest and arrays matching `patches_df` exactly."""
    manifest_path = Path(cache_dir) / _MANIFEST_NAME
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    return (
        manifest.get("num_patches") == len(patches_df)
        and manifest.get("fingerprint") == fingerprint(patches_df)
        and (Path(cache_dir) / _INPUTS_NAME).exists()
        and (Path(cache_dir) / _OUTPUTS_NAME).exists()
    )


def write(
    cache_dir: Path, patches_df: pd.DataFrame, inputs: np.ndarray, outputs: np.ndarray
) -> None:
    """Write `inputs` (float16) / `outputs` (uint8) plus a manifest fingerprinting `patches_df`."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / _INPUTS_NAME, inputs.astype(np.float16))
    np.save(cache_dir / _OUTPUTS_NAME, outputs.astype(np.uint8))
    manifest = {
        "num_patches": len(patches_df),
        "fingerprint": fingerprint(patches_df),
    }
    (cache_dir / _MANIFEST_NAME).write_text(json.dumps(manifest))


def resolve_cache_dir(
    cache_root: Path, dataset: str, split: str, patches_df: pd.DataFrame
) -> Path | None:
    """Return `cache_root/dataset/split` if it's a valid cache for `patches_df`, else `None`.

    The wiring layer (`train.py`, `evaluate.py`) calls this to decide
    *whether* to opt into the on-disk cache at all -- tolerant by design,
    unlike `PatchDataset`'s own strict `cache_dir` contract: a tier whose
    DataFrame doesn't match what was cached (e.g. an r2 subsample, or
    raw-smoke's sliced val split) just falls back to live reads, since
    those callers never explicitly asked for a cache.
    """
    candidate = Path(cache_root) / dataset / split
    return candidate if is_valid(candidate, patches_df) else None


def load(cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (inputs, outputs) as read-only memmaps over `cache_dir`'s arrays."""
    cache_dir = Path(cache_dir)
    inputs = np.load(cache_dir / _INPUTS_NAME, mmap_mode="r")
    outputs = np.load(cache_dir / _OUTPUTS_NAME, mmap_mode="r")
    return inputs, outputs
