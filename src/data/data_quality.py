"""Measure data-quality properties of the processed scenes and write a JSON summary.

The checks cover missing values, noise, outliers, duplicate scenes, resolution,
and spatial overlap between dataset partitions.
"""

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from data.preprocessing import BAND_NORMALIZATION
from visualization.class_balance import (
    LABEL_FILE,
    MINI_DATASET,
    PARTITIONS,
    PROCESSED_ROOT,
    RAW_DATASET,
    SPLIT_FILE,
    SPLITS_DIR,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

MAG1C_BAND = "mag1c"
BANDS = tuple(BAND_NORMALIZATION)

CLIP_LIMITS = {band: spec["factor"] * spec["clip"][1] for band, spec in BAND_NORMALIZATION.items()}
NOISE_THRESHOLD = 500.0
NODATA_BAND = "TOA_AVIRIS_640nm"
DUPLICATE_FILES = ("TOA_AVIRIS_640nm.tif", "mag1c.tif", LABEL_FILE)
BLOCK_PIXELS = 16
CELL_METERS = 100.0
PARTITION_PAIRS = (("train", "val"), ("train", "test"), ("val", "test"))
PAIR_CLASSES = (
    "same_flight_line",
    "different_flight_line_same_partition",
    "different_partitions",
)


def _read_band(folder: Path, band: str) -> np.ndarray:
    """First band of `<folder>/<band>.tif`."""
    with rasterio.open(Path(folder) / f"{band}.tif") as src:
        return src.read(1)


def scene_stats(folder: Path) -> dict:
    """Per-scene counts: pixels, plume pixels, per-band defects and mag1c against its thresholds."""
    with rasterio.open(Path(folder) / LABEL_FILE) as src:
        label = src.read(1) == 1
    arrays = {band: _read_band(folder, band) for band in BANDS}
    bands = {
        band: {
            "nan": int(np.isnan(values).sum()),
            "inf": int(np.isinf(values).sum()),
            "zeros": int((values == 0).sum()),
            "negative": int((values < 0).sum()),
            "above_clip": int((values > CLIP_LIMITS[band]).sum()),
        }
        for band, values in arrays.items()
    }
    mag1c = arrays[MAG1C_BAND]
    above_noise = int((mag1c > NOISE_THRESHOLD).sum())
    above_clip_in_label = int(((mag1c > CLIP_LIMITS[MAG1C_BAND]) & label).sum())
    return {
        "pixels": int(label.size),
        "plume_pixels": int(label.sum()),
        "bands": bands,
        "mag1c_above_noise": above_noise,
        "mag1c_above_clip_in_label": above_clip_in_label,
    }


def summarize_quality(stats: Sequence[dict]) -> dict:
    """Aggregate `scene_stats`: totals, fractions and how many scenes each defect touches."""
    if not stats:
        raise ValueError("no scenes to summarize")
    pixels = sum(s["pixels"] for s in stats)
    bands = {}
    for band in BANDS:
        total = {key: sum(s["bands"][band][key] for s in stats) for key in stats[0]["bands"][band]}
        bands[band] = {
            **total,
            "zero_fraction": total["zeros"] / pixels,
            "above_clip_fraction": total["above_clip"] / pixels,
            "scenes_with_zeros": sum(1 for s in stats if s["bands"][band]["zeros"] > 0),
            "scenes_with_negative": sum(1 for s in stats if s["bands"][band]["negative"] > 0),
            "scenes_above_clip": sum(1 for s in stats if s["bands"][band]["above_clip"] > 0),
        }
    unlabelled = [s for s in stats if s["plume_pixels"] == 0]
    above_clip = bands[MAG1C_BAND]["above_clip"]
    return {
        "scenes": len(stats),
        "pixels": pixels,
        "bands": bands,
        "unlabelled_scenes": len(unlabelled),
        "unlabelled_noise_fraction": (
            sum(s["mag1c_above_noise"] for s in unlabelled) / sum(s["pixels"] for s in unlabelled)
            if unlabelled
            else None
        ),
        "unlabelled_scenes_with_noise": sum(1 for s in unlabelled if s["mag1c_above_noise"] > 0),
        "mag1c_above_clip_in_label_share": (
            sum(s["mag1c_above_clip_in_label"] for s in stats) / above_clip if above_clip else None
        ),
    }


def label_discordance(has_plume: Sequence[bool], plume_pixels: Sequence[int]) -> dict:
    """Scenes whose scene-level label and pixel-level label disagree, by kind."""
    if len(has_plume) != len(plume_pixels):
        raise ValueError("has_plume and plume_pixels differ in length")
    pairs = list(zip(has_plume, plume_pixels))
    return {
        "plume_without_pixels": sum(1 for flag, pixels in pairs if flag and pixels == 0),
        "pixels_without_plume": sum(1 for flag, pixels in pairs if not flag and pixels > 0),
    }


def _content_digest(folder: Path) -> str:
    """SHA-256 of the scene's TOA 640 nm, mag1c and label rasters, in `DUPLICATE_FILES` order."""
    digest = hashlib.sha256()
    for name in DUPLICATE_FILES:
        with rasterio.open(Path(folder) / name) as src:
            digest.update(src.read(1).tobytes())
    return digest.hexdigest()


def duplicate_scenes(folders: Sequence[Path]) -> list[list[Path]]:
    """Groups of scenes with identical pixel content, in order of first appearance."""
    groups = defaultdict(list)
    for folder in folders:
        groups[_content_digest(folder)].append(folder)
    return [group for group in groups.values() if len(group) > 1]


def pixel_size(folder: Path) -> tuple[float, float]:
    """Ground size in metres of one column step and one row step, from the affine matrix.

    Scenes are georeferenced with a rotation, so `transform.a` is only the cosine part of the
    column step: the size is the length of each column of the matrix.
    """
    with rasterio.open(Path(folder) / f"{MAG1C_BAND}.tif") as src:
        transform = src.transform
    return math.hypot(transform.a, transform.d), math.hypot(transform.b, transform.e)


def scene_cells(
    folder: Path, block: int = BLOCK_PIXELS, cell: float = CELL_METERS
) -> set[tuple[int, int]]:
    """Ground cells of `cell` metres that hold data, one per `block` x `block` pixel block.

    A block holds data when any pixel of the `NODATA_BAND` is non-zero (zero is `nodata`). Each
    such block is placed by the ground position of its centre, through the full affine matrix.
    """
    with rasterio.open(Path(folder) / f"{NODATA_BAND}.tif") as src:
        values = src.read(1)
        transform = src.transform
    rows, cols = values.shape[0] // block, values.shape[1] // block
    blocks = values[: rows * block, : cols * block].reshape(rows, block, cols, block)
    block_rows, block_cols = np.nonzero((blocks != 0).any(axis=(1, 3)))
    centre_rows = block_rows * block + block / 2
    centre_cols = block_cols * block + block / 2
    x = transform.c + transform.a * centre_cols + transform.b * centre_rows
    y = transform.f + transform.d * centre_cols + transform.e * centre_rows
    return set(zip((x // cell).astype(int).tolist(), (y // cell).astype(int).tolist()))


def coverage_report(covers: dict[str, list[set]], target: str, reference: str) -> dict:
    """How much of `target`'s ground cells the scenes of `reference` also cover."""
    scenes = covers[target]
    if not scenes:
        raise ValueError(f"no scenes in partition {target!r}")
    reference_cells = set().union(*covers[reference])
    fractions = []
    for cells in scenes:
        if not cells:
            raise ValueError("scene without valid cells")
        fractions.append(len(cells & reference_cells) / len(cells))
    target_cells = set().union(*scenes)
    return {
        "scenes": len(scenes),
        "cells_covered_fraction": len(target_cells & reference_cells) / len(target_cells),
        "median_scene_fraction": statistics.median(fractions),
        "scenes_half_covered": sum(1 for f in fractions if f >= 0.5),
        "scenes_nine_tenths_covered": sum(1 for f in fractions if f >= 0.9),
        "scenes_uncovered": sum(1 for f in fractions if f == 0),
    }


def overlap_pairs(covers: Sequence[set]) -> list[tuple[int, int, float]]:
    """Every scene pair sharing a cell, as `(i, j, shared cells / cells of the smaller scene)`."""
    members_by_cell = defaultdict(list)
    for index, cells in enumerate(covers):
        for cell in cells:
            members_by_cell[cell].append(index)
    shared = Counter()
    for members in members_by_cell.values():
        for position, first in enumerate(members):
            for second in members[position + 1 :]:
                shared[(first, second)] += 1
    return [
        (first, second, count / min(len(covers[first]), len(covers[second])))
        for (first, second), count in sorted(shared.items())
    ]


def classify_pairs(
    pairs: Sequence[tuple[int, int, float]], splits: Sequence[str], names: Sequence[str]
) -> dict:
    """Count pairs by kind (same flight line, other line in a partition, across partitions)."""
    result = {kind: {"pairs": 0, "half_or_more": 0} for kind in PAIR_CLASSES}
    for first, second, fraction in pairs:
        if splits[first] != splits[second]:
            kind = "different_partitions"
        elif names[first] == names[second]:
            kind = "same_flight_line"
        else:
            kind = "different_flight_line_same_partition"
        result[kind]["pairs"] += 1
        if fraction >= 0.5:
            result[kind]["half_or_more"] += 1
    return result


def shared_between_partitions(values: dict[str, Sequence]) -> dict[str, list]:
    """Sorted values that appear in both partitions of each pair (`train/val`, ...)."""
    return {f"{a}/{b}": sorted(set(values[a]) & set(values[b])) for a, b in PARTITION_PAIRS}


def date_ranges(dates: dict[str, Sequence[str]]) -> dict[str, dict]:
    """Number of distinct days and first and last day (ISO strings) per partition."""
    result = {}
    for partition, values in dates.items():
        days = sorted(set(values))
        if not days:
            raise ValueError(f"no dates for partition {partition!r}")
        result[partition] = {"days": len(days), "first": days[0], "last": days[-1]}
    return result


def _load_scenes(repo_root: Path, dataset: str) -> pd.DataFrame:
    """Every scene of `dataset` (train, val, test in that order) with its partition."""
    base = repo_root / PROCESSED_ROOT / dataset / SPLITS_DIR
    frames = [
        pd.read_csv(base / SPLIT_FILE.format(split=split)).assign(split=split)
        for split in PARTITIONS
    ]
    return pd.concat(frames, ignore_index=True)


def build_data_quality(
    repo_root: Path, *, block: int = BLOCK_PIXELS, cell: float = CELL_METERS
) -> dict:
    """Run every check on the processed scenes of `mini` and the full set (`raw`)."""
    mini = _load_scenes(repo_root, MINI_DATASET)
    raw = _load_scenes(repo_root, RAW_DATASET)
    raw_folders = [repo_root / folder for folder in raw["folder"]]
    raw_stats = [scene_stats(folder) for folder in raw_folders]
    splits = raw["split"].tolist()

    discordance = {}
    for split in PARTITIONS:
        chosen = [i for i, s in enumerate(splits) if s == split]
        discordance[split] = label_discordance(
            [bool(raw["has_plume"][i]) for i in chosen],
            [raw_stats[i]["plume_pixels"] for i in chosen],
        )

    sizes = [pixel_size(folder)[0] for folder in raw_folders]
    dates = {split: raw.loc[raw["split"] == split, "date"].tolist() for split in PARTITIONS}
    names = {split: raw.loc[raw["split"] == split, "name"].tolist() for split in PARTITIONS}
    cells = [scene_cells(folder, block, cell) for folder in raw_folders]
    covers = {split: [c for c, s in zip(cells, splits) if s == split] for split in PARTITIONS}
    return {
        "quality": {
            "mini": summarize_quality(
                [scene_stats(repo_root / folder) for folder in mini["folder"]]
            ),
            "raw": summarize_quality(raw_stats),
        },
        "label_discordance": discordance,
        "duplicate_scene_groups": len(duplicate_scenes(raw_folders)),
        "pixel_size_m": {
            "min": min(sizes),
            "median": statistics.median(sizes),
            "max": max(sizes),
        },
        "dates": date_ranges(dates),
        "shared_dates": shared_between_partitions(dates),
        "shared_flight_lines": shared_between_partitions(names),
        "coverage": {
            "test_by_train": coverage_report(covers, "test", "train"),
            "val_by_train": coverage_report(covers, "val", "train"),
        },
        "overlapping_pairs": classify_pairs(overlap_pairs(cells), splits, raw["name"].tolist()),
    }


def write_data_quality(
    repo_root: Path, out_path: Path, *, block: int = BLOCK_PIXELS, cell: float = CELL_METERS
) -> Path:
    """Write `build_data_quality` to `out_path` as JSON (creating its folder) and return it."""
    result = build_data_quality(repo_root, block=block, cell=cell)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    return out_path


def main() -> None:
    """Write `figures/data_quality.json`; run from the repository root (a few minutes)."""
    out_path = write_data_quality(_PROJECT_ROOT, _PROJECT_ROOT / "figures" / "data_quality.json")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
