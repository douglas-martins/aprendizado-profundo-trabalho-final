"""Build and persist the R2 subsample manifests.

Run once; the resulting CSVs are committed and reused by every R2 run, not
regenerated per run (a redrawn subsample would silently make R2 runs
non-comparable to each other).

`sampling.py`'s `build_r2_manifest` already has full test coverage; this
script is the one real, recorded invocation that produces the actual
artifact, the same "thin glue exercised by a real run" pattern as
`confirm_raw.py`.
"""

from pathlib import Path

import pandas as pd

from data.sampling import build_r2_manifest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SEED = 42
_TRAIN_FLIGHTLINES = 10
_VAL_FLIGHTLINES = 5


def main() -> None:
    """Build the train and val R2 manifests from starcop_raw's real patches CSVs."""
    patches_root = _REPO_ROOT / "data" / "processed" / "starcop_raw" / "patches"

    train_df = pd.read_csv(patches_root / "train_tiled_128_128.csv", low_memory=False)
    train_manifest = build_r2_manifest(train_df, n_flightlines=_TRAIN_FLIGHTLINES, seed=_SEED)
    train_manifest.to_csv(
        _PROJECT_ROOT / "data" / "manifests" / "r2_manifest_train.csv", index=False
    )
    print(
        f"train: {len(train_manifest)} patches, {train_manifest['name'].nunique()} flightlines, "
        f"has_plume={train_manifest['has_plume'].mean():.2%}"
    )

    val_df = pd.read_csv(patches_root / "val_tiled_128_128.csv", low_memory=False)
    val_manifest = build_r2_manifest(val_df, n_flightlines=_VAL_FLIGHTLINES, seed=_SEED)
    val_manifest.to_csv(_PROJECT_ROOT / "data" / "manifests" / "r2_manifest_val.csv", index=False)
    print(
        f"val: {len(val_manifest)} patches, {val_manifest['name'].nunique()} flightlines, "
        f"has_plume={val_manifest['has_plume'].mean():.2%}"
    )


if __name__ == "__main__":
    main()
