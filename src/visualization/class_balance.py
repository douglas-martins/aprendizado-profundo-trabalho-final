"""Class balance of plume pixels and plume patches, per data scale and partition.

Two criteria, deliberately reported side by side because they answer different questions:

- *pixels*: the plume fraction of the scenes' own label rasters, each pixel counted **once**.
  The patch tables cannot give this number: patches overlap by half (stride 64 over 128), so
  a pixel sits in up to four of them and the pooled patch fraction over-counts whatever the
  patches share.
- *patches*: how many patches STARCOP flags `has_plume`, which is `frac_positives > 0.00244`
  (`configs/data.yaml`), i.e. **40 plume pixels or more** of a 128 x 128 patch, not "at least
  one pixel".

The three data scales are `mini` (its own splits), R3 (the full `raw` splits) and R2 (a fixed
sample of `raw` scenes, taken from its manifests; its test set is the `raw` test set).
"""

import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import rasterio

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_ROOT = Path("data") / "processed"
MINI_DATASET = "starcop_mini"
RAW_DATASET = "starcop_raw"
SPLITS_DIR = "splits"
PATCHES_DIR = "patches"
SPLIT_FILE = "{split}.csv"
PATCH_FILE = "{split}_tiled_128_128.csv"
LABEL_FILE = "labelbinary.tif"
PARTITIONS = ("train", "val", "test")


def count_scene_pixels(scene_folders: Iterable[Path]) -> tuple[int, int]:
    """Return `(plume pixels, total pixels)` over every scene's `labelbinary.tif`, once each."""
    folders = list(scene_folders)
    if not folders:
        raise ValueError("no scenes to count")
    plume = 0
    total = 0
    for folder in folders:
        with rasterio.open(Path(folder) / LABEL_FILE) as label:
            plume += int((label.read(1) == 1).sum())
            total += label.height * label.width
    return plume, total


def patch_plume_balance(patches_df: pd.DataFrame) -> dict:
    """Return `{plume_patches, total_patches}`, counting patches STARCOP flags `has_plume`."""
    if len(patches_df) == 0:
        raise ValueError("no patches to count")
    return {
        "plume_patches": int(patches_df["has_plume"].sum()),
        "total_patches": len(patches_df),
    }


def partition_balance(scene_folders: list[Path], patches_df: pd.DataFrame) -> dict:
    """Return both balance criteria, and their fractions, for one scale/partition."""
    plume_pixels, total_pixels = count_scene_pixels(scene_folders)
    patches = patch_plume_balance(patches_df)
    return {
        "scenes": len(scene_folders),
        "plume_pixels": plume_pixels,
        "total_pixels": total_pixels,
        "pixel_fraction": plume_pixels / total_pixels,
        **patches,
        "patch_fraction": patches["plume_patches"] / patches["total_patches"],
    }


def _dataset_partitions(repo_root: Path, dataset: str) -> dict:
    """Balance of each split of `dataset`, from its splits and tiled-patch tables."""
    base = repo_root / PROCESSED_ROOT / dataset
    result = {}
    for split in PARTITIONS:
        scenes = pd.read_csv(base / SPLITS_DIR / SPLIT_FILE.format(split=split))["folder"]
        patches = pd.read_csv(base / PATCHES_DIR / PATCH_FILE.format(split=split))
        result[split] = partition_balance([repo_root / folder for folder in scenes], patches)
    return result


def _manifest_partition(repo_root: Path, manifest: Path) -> dict:
    """Balance of an R2 manifest: its patches as listed, its scenes once each."""
    rows = pd.read_csv(manifest)
    scenes = rows["folder"].drop_duplicates()
    return partition_balance([repo_root / folder for folder in scenes], rows)


def build_class_balance(repo_root: Path, r2_train: Path, r2_val: Path) -> dict:
    """Balance per scale (`mini`, `R2`, `R3`) and partition, ready for `json.dump`."""
    raw = _dataset_partitions(repo_root, RAW_DATASET)
    return {
        "mini": _dataset_partitions(repo_root, MINI_DATASET),
        "R2": {
            "train": _manifest_partition(repo_root, r2_train),
            "val": _manifest_partition(repo_root, r2_val),
            "test": raw["test"],
        },
        "R3": raw,
    }


def write_class_balance(repo_root: Path, r2_train: Path, r2_val: Path, out_path: Path) -> Path:
    """Write `build_class_balance` to `out_path` as JSON (creating its folder) and return it."""
    balance = build_class_balance(repo_root, r2_train, r2_val)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(balance, indent=2) + "\n")
    return out_path


def main() -> None:
    """Write `figures/class_balance.json`; run from the repository root."""
    here = _PROJECT_ROOT
    out_path = write_class_balance(
        _PROJECT_ROOT,
        here / "data" / "manifests" / "r2_manifest_train.csv",
        here / "data" / "manifests" / "r2_manifest_val.csv",
        here / "figures" / "class_balance.json",
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
