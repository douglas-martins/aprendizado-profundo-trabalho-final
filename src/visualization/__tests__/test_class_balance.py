import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

import visualization.class_balance as class_balance
from visualization.class_balance import (
    LABEL_FILE,
    build_class_balance,
    count_scene_pixels,
    partition_balance,
    patch_plume_balance,
    write_class_balance,
)


def _write_scene(scene_dir: Path, positives: int, shape: tuple[int, int] = (8, 8)) -> Path:
    """Write a real one-band uint8 `labelbinary.tif` with `positives` pixels set to 1."""
    scene_dir.mkdir(parents=True, exist_ok=True)
    label = np.zeros(shape, dtype=np.uint8)
    label.flat[:positives] = 1
    with rasterio.open(
        scene_dir / LABEL_FILE,
        "w",
        driver="GTiff",
        height=shape[0],
        width=shape[1],
        count=1,
        dtype="uint8",
        transform=from_origin(500000, 3600000, 1, 1),
    ) as dst:
        dst.write(label, 1)
    return scene_dir


def _write_two_band_scene(scene_dir: Path, first_band_positives: int) -> Path:
    """Write a two-band `labelbinary.tif` whose second band is all ones."""
    scene_dir.mkdir(parents=True, exist_ok=True)
    first = np.zeros((8, 8), dtype=np.uint8)
    first.flat[:first_band_positives] = 1
    with rasterio.open(
        scene_dir / LABEL_FILE,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=2,
        dtype="uint8",
        transform=from_origin(500000, 3600000, 1, 1),
    ) as dst:
        dst.write(first, 1)
        dst.write(np.ones((8, 8), dtype=np.uint8), 2)
    return scene_dir


def _patches(flags: list[bool]) -> pd.DataFrame:
    return pd.DataFrame({"has_plume": flags})


class TestCountScenePixels:
    def test_counts_plume_pixels_and_total_of_one_scene(self, tmp_path):
        scene = _write_scene(tmp_path / "a", positives=3)
        assert count_scene_pixels([scene]) == (3, 64)

    def test_sums_scenes_of_different_size(self, tmp_path):
        small = _write_scene(tmp_path / "small", positives=2, shape=(4, 4))
        wide = _write_scene(tmp_path / "wide", positives=5, shape=(8, 2))
        assert count_scene_pixels([small, wide]) == (7, 32)

    def test_scene_without_plume_adds_only_to_the_total(self, tmp_path):
        empty = _write_scene(tmp_path / "empty", positives=0)
        plume = _write_scene(tmp_path / "plume", positives=4)
        assert count_scene_pixels([empty, plume]) == (4, 128)

    def test_accepts_a_one_shot_iterator_of_scenes(self, tmp_path):
        scenes = iter([_write_scene(tmp_path / "a", positives=3)])
        assert count_scene_pixels(scenes) == (3, 64)

    def test_reads_only_the_first_band(self, tmp_path):
        scene = _write_two_band_scene(tmp_path / "two_bands", first_band_positives=3)
        assert count_scene_pixels([scene]) == (3, 64)

    def test_rejects_an_empty_scene_list(self):
        with pytest.raises(ValueError, match=r"^no scenes to count$"):
            count_scene_pixels([])

    def test_rejects_an_empty_one_shot_iterator(self):
        with pytest.raises(ValueError, match=r"^no scenes to count$"):
            count_scene_pixels(iter([]))

    def test_missing_label_raster_raises(self, tmp_path):
        (tmp_path / "no_label").mkdir()
        with pytest.raises(rasterio.errors.RasterioIOError):
            count_scene_pixels([tmp_path / "no_label"])


class TestPatchPlumeBalance:
    def test_counts_patches_flagged_with_plume(self):
        result = patch_plume_balance(_patches([True, False, True, False, False]))
        assert result == {"plume_patches": 2, "total_patches": 5}

    def test_no_flagged_patch_gives_zero(self):
        result = patch_plume_balance(_patches([False, False]))
        assert result == {"plume_patches": 0, "total_patches": 2}

    def test_rejects_an_empty_patch_table(self):
        with pytest.raises(ValueError, match=r"^no patches to count$"):
            patch_plume_balance(_patches([]))


class TestPartitionBalance:
    def test_reports_both_criteria_with_their_fractions(self, tmp_path):
        scenes = [
            _write_scene(tmp_path / "a", positives=8),
            _write_scene(tmp_path / "b", positives=0),
        ]
        result = partition_balance(scenes, _patches([True, False, False, False]))
        assert result == {
            "scenes": 2,
            "plume_pixels": 8,
            "total_pixels": 128,
            "pixel_fraction": pytest.approx(8 / 128),
            "plume_patches": 1,
            "total_patches": 4,
            "patch_fraction": pytest.approx(1 / 4),
        }

    def test_counts_a_pixel_once_even_when_patches_overlap(self, tmp_path):

        scene = _write_scene(tmp_path / "a", positives=1)
        result = partition_balance([scene], _patches([True, True, True, True]))
        assert result["plume_pixels"] == 1
        assert result["pixel_fraction"] == pytest.approx(1 / 64)
        assert result["plume_patches"] == 4


def _write_dataset(root: Path, dataset: str, splits: dict[str, list[tuple[str, int, bool]]]):
    """Write `splits/<split>.csv` and `patches/<split>_tiled_128_128.csv` for a fake dataset.

    `splits` maps a split to `(scene name, positive pixels, patch has_plume)` triples; each
    scene gets one patch row carrying that flag.
    """
    for split, scenes in splits.items():
        folders = []
        for name, positives, _ in scenes:
            folder = f"data/processed/{dataset}/selected/{name}"
            _write_scene(root / folder, positives)
            folders.append(folder)
        base = root / f"data/processed/{dataset}"
        (base / "splits").mkdir(parents=True, exist_ok=True)
        (base / "patches").mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"folder": folders}).to_csv(base / "splits" / f"{split}.csv", index=False)
        pd.DataFrame({"has_plume": [flag for _, _, flag in scenes]}).to_csv(
            base / "patches" / f"{split}_tiled_128_128.csv"
        )


def _write_manifest(path: Path, rows: list[tuple[str, bool]]) -> Path:
    pd.DataFrame(
        {"folder": [folder for folder, _ in rows], "has_plume": [flag for _, flag in rows]}
    ).to_csv(path, index=False)
    return path


@pytest.fixture
def fake_repo(tmp_path):
    _write_dataset(
        tmp_path,
        "starcop_mini",
        {
            "train": [("m1", 4, True), ("m2", 0, False)],
            "val": [("m3", 2, True)],
            "test": [("m4", 16, True), ("m5", 0, False)],
        },
    )
    _write_dataset(
        tmp_path,
        "starcop_raw",
        {
            "train": [("r1", 1, True), ("r2", 0, False), ("r3", 0, False), ("r4", 3, True)],
            "val": [("r5", 0, False), ("r6", 8, True)],
            "test": [("r7", 32, True), ("r8", 0, False), ("r9", 0, False)],
        },
    )
    raw = "data/processed/starcop_raw/selected"

    r2_train = _write_manifest(
        tmp_path / "r2_train.csv",
        [(f"{raw}/r1", True), (f"{raw}/r1", False), (f"{raw}/r2", False), (f"{raw}/r2", False)],
    )
    r2_val = _write_manifest(tmp_path / "r2_val.csv", [(f"{raw}/r6", True), (f"{raw}/r6", True)])
    return tmp_path, r2_train, r2_val


class TestDataLayoutContract:
    def test_processed_root_is_data_processed(self):
        assert class_balance.PROCESSED_ROOT == Path("data") / "processed"

    def test_dataset_directory_names(self):
        assert class_balance.MINI_DATASET == "starcop_mini"
        assert class_balance.RAW_DATASET == "starcop_raw"

    def test_split_and_patch_table_locations(self):
        assert class_balance.SPLITS_DIR == "splits"
        assert class_balance.PATCHES_DIR == "patches"
        assert class_balance.SPLIT_FILE.format(split="val") == "val.csv"
        assert class_balance.PATCH_FILE.format(split="val") == "val_tiled_128_128.csv"

    def test_partitions_are_train_val_test_in_that_order(self):
        assert class_balance.PARTITIONS == ("train", "val", "test")

    def test_label_file_is_labelbinary(self):
        assert LABEL_FILE == "labelbinary.tif"


class TestBuildClassBalance:
    def test_covers_three_scales_and_three_partitions(self, fake_repo):
        root, r2_train, r2_val = fake_repo
        result = build_class_balance(root, r2_train, r2_val)
        assert list(result) == ["mini", "R2", "R3"]
        for scale in result.values():
            assert list(scale) == ["train", "val", "test"]

    def test_mini_reads_the_mini_splits(self, fake_repo):
        root, r2_train, r2_val = fake_repo
        mini = build_class_balance(root, r2_train, r2_val)["mini"]
        assert mini["train"]["scenes"] == 2
        assert mini["train"]["plume_pixels"] == 4
        assert mini["val"]["scenes"] == 1
        assert mini["test"]["plume_pixels"] == 16
        assert mini["test"]["total_pixels"] == 128

    def test_r3_reads_the_raw_splits(self, fake_repo):
        root, r2_train, r2_val = fake_repo
        r3 = build_class_balance(root, r2_train, r2_val)["R3"]
        assert r3["train"]["scenes"] == 4
        assert r3["train"]["plume_pixels"] == 4
        assert r3["train"]["plume_patches"] == 2
        assert r3["val"]["plume_pixels"] == 8
        assert r3["test"]["scenes"] == 3

    def test_r2_counts_manifest_scenes_once_and_keeps_manifest_patches(self, fake_repo):
        root, r2_train, r2_val = fake_repo
        r2 = build_class_balance(root, r2_train, r2_val)["R2"]
        assert r2["train"]["scenes"] == 2
        assert r2["train"]["plume_pixels"] == 1
        assert r2["train"]["total_pixels"] == 128
        assert r2["train"]["plume_patches"] == 1
        assert r2["train"]["total_patches"] == 4
        assert r2["val"]["scenes"] == 1
        assert r2["val"]["plume_pixels"] == 8
        assert r2["val"]["plume_patches"] == 2

    def test_r2_test_partition_is_the_raw_test_set(self, fake_repo):
        root, r2_train, r2_val = fake_repo
        result = build_class_balance(root, r2_train, r2_val)
        assert result["R2"]["test"] == result["R3"]["test"]

    def test_result_is_json_serialisable(self, fake_repo):
        root, r2_train, r2_val = fake_repo
        result = build_class_balance(root, r2_train, r2_val)
        assert json.loads(json.dumps(result)) == result


class TestWriteClassBalance:
    def test_writes_the_balance_as_json_and_returns_the_path(self, fake_repo, tmp_path):
        root, r2_train, r2_val = fake_repo
        out_path = tmp_path / "out" / "figures" / "class_balance.json"
        written = write_class_balance(root, r2_train, r2_val, out_path)
        assert written == out_path
        assert json.loads(out_path.read_text()) == build_class_balance(root, r2_train, r2_val)

    def test_json_file_is_two_space_indented_and_ends_with_a_newline(self, fake_repo, tmp_path):
        root, r2_train, r2_val = fake_repo
        out_path = tmp_path / "class_balance.json"
        write_class_balance(root, r2_train, r2_val, out_path)
        expected = json.dumps(build_class_balance(root, r2_train, r2_val), indent=2) + "\n"
        assert out_path.read_text() == expected
