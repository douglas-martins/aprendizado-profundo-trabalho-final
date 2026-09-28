"""One-time precompute of the on-disk patch cache (see `patch_cache.py`).

Reads every patch through `PatchDataset`'s existing live read path
(`augment=False` -- same pre-augmentation contract as its own RAM cache)
exactly once, and writes the flat float16/uint8 arrays `PatchDataset(...,
cache_dir=...)` reads back. Uses a plain `DataLoader` purely to parallelize
that one-time read across `num_workers` processes (matching `train.py`'s
own real-run `num_workers`) -- `shuffle=False` keeps batch order aligned
with `patches_df`'s row order, which the cache's array position depends on.

Usage: python precompute_patch_cache.py dataset=starcop_raw [splits=train,val,test]
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from data import patch_cache
from data.dataset import PatchDataset

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CACHE_ROOT = _PROJECT_ROOT / "patch_cache"


def build_cache_for_split(
    dataset: str,
    patches_df: pd.DataFrame,
    cache_dir: Path,
    num_workers: int = 4,
    batch_size: int = 256,
) -> None:
    """Read every patch in `patches_df` once and write it to `cache_dir` (see `patch_cache.py`)."""
    live_dataset = PatchDataset(patches_df, dataset=dataset, augment=False, max_cache_bytes=0)
    loader = DataLoader(live_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    n = len(live_dataset)
    inputs = None

    outputs = None
    offset = 0
    for batch in loader:
        batch_input_array = batch["input"].numpy()
        batch_output_array = batch["output"].numpy()
        if inputs is None:
            _, c, h, w = batch_input_array.shape

            inputs = np.empty((n, c, h, w), dtype=np.float16)
            outputs = np.empty((n, 1, h, w), dtype=np.uint8)
        b = batch_input_array.shape[0]
        inputs[offset : offset + b] = batch_input_array
        outputs[offset : offset + b] = batch_output_array
        offset += b
    patch_cache.write(cache_dir, patches_df, inputs, outputs)


def _parse_kv_args(argv: list[str]) -> dict[str, str]:
    """Parse command-line `key=value` arguments."""
    parsed = {}
    for arg in argv:
        if "=" in arg:
            key, value = arg.split("=", 1)
            parsed[key] = value
    return parsed


def main() -> None:
    """CLI entry point -- see this module's own docstring for usage."""
    args = _parse_kv_args(sys.argv[1:])
    dataset = args.get("dataset", "starcop_mini")
    splits = args.get("splits", "train,val,test").split(",")
    patches_root = _REPO_ROOT / "data" / "processed" / dataset / "patches"

    for split in splits:
        csv_path = patches_root / f"{split}_tiled_128_128.csv"
        patches_df = pd.read_csv(csv_path, low_memory=False)
        cache_dir = _CACHE_ROOT / dataset / split
        start = time.perf_counter()
        build_cache_for_split(dataset, patches_df, cache_dir)
        elapsed = time.perf_counter() - start
        print(f"{dataset}/{split}: {len(patches_df)} patches -> {cache_dir} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
