"""Shared fixtures for the project test suite."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


def _find_repo_root_containing_configs(start: Path) -> Path:
    """Climb from `start` until the local dataset configurations are found."""
    candidate = start
    while not (candidate / "configs" / "dataset").is_dir():
        if candidate.parent == candidate:
            raise RuntimeError(f"no configs/dataset/ found above {start}")
        candidate = candidate.parent
    return candidate


@pytest.fixture(autouse=True)
def _fix_dataset_repo_root_for_copied_trees():
    """Use the repository root discovered from the test location."""
    from data import dataset as dataset_module

    original_repo_root = dataset_module._REPO_ROOT
    dataset_module._REPO_ROOT = _find_repo_root_containing_configs(Path(__file__).resolve().parent)
    yield
    dataset_module._REPO_ROOT = original_repo_root


@pytest.fixture
def tiny_geotiff_factory():
    """Write a tiny single-band GeoTIFF with a given pixel array."""

    def _make(path: Path, array: np.ndarray) -> Path:
        """Write `array` as a single-band GeoTIFF at `path`."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=array.shape[0],
            width=array.shape[1],
            count=1,
            dtype=array.dtype,
            crs="EPSG:4326",
            transform=from_origin(0, 0, 1, 1),
        ) as dst:
            dst.write(array, 1)
        return path

    return _make
