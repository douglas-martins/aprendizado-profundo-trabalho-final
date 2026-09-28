"""R2 subsample construction: a seeded, scene-disjoint slice of
`starcop_raw` for the training/evaluation stages, without reprocessing anything.

Sampling unit is the flightline (`name` column) -- the same unit
`split.py`'s `stratify_by` already splits `starcop_raw`'s own train/val/test
on, so R2's scene-disjointness from raw's val/test is inherited for free
rather than re-derived here. This is a pure row filter over
`patch_extract@starcop_raw`'s already-built patches CSV -- no new versionamento externo
stage, no `configs/dataset/*.yaml` entry.
"""

import numpy as np
import pandas as pd


def sample_flightlines(patches_df: pd.DataFrame, n_flightlines: int, seed: int) -> list[str]:
    """Seeded, duplicate-free sample of `n_flightlines` distinct `name` values."""
    unique_flightlines = patches_df["name"].unique()
    if len(unique_flightlines) < n_flightlines:
        raise ValueError(
            f"Not enough flightlines: need {n_flightlines}, have {len(unique_flightlines)}"
        )
    rng = np.random.default_rng(seed)
    selected = rng.choice(unique_flightlines, size=n_flightlines, replace=False)
    return sorted(selected.tolist())


def build_r2_manifest(patches_df: pd.DataFrame, n_flightlines: int, seed: int) -> pd.DataFrame:
    """Filter `patches_df` down to the rows from a seeded sample of `n_flightlines`."""
    selected = sample_flightlines(patches_df, n_flightlines, seed)
    return patches_df[patches_df["name"].isin(selected)].reset_index(drop=True)
