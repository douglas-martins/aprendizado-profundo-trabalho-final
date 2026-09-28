import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import Affine, from_origin

import data.data_quality as data_quality
from data.data_quality import (
    build_data_quality,
    classify_pairs,
    coverage_report,
    date_ranges,
    duplicate_scenes,
    label_discordance,
    overlap_pairs,
    pixel_size,
    scene_cells,
    scene_stats,
    shared_between_partitions,
    summarize_quality,
    write_data_quality,
)

MAG1C = "mag1c"
TOA = ("TOA_AVIRIS_640nm", "TOA_AVIRIS_550nm", "TOA_AVIRIS_460nm")
NORTH_UP = from_origin(500000, 3600000, 1, 1)


def _write_band(path: Path, array: np.ndarray, transform=NORTH_UP) -> None:
    dtype = "uint8" if array.dtype == np.uint8 else "float32"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=dtype,
        transform=transform,
    ) as dst:
        dst.write(array.astype(dtype), 1)


def _write_two_bands(path: Path, first: np.ndarray, second: np.ndarray) -> None:
    """Write a two-band raster; the second band is what a `read(None)` would wrongly include."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=first.shape[0],
        width=first.shape[1],
        count=2,
        dtype="float32",
        transform=NORTH_UP,
    ) as dst:
        dst.write(first.astype("float32"), 1)
        dst.write(second.astype("float32"), 2)


def _write_scene(
    folder: Path,
    *,
    mag1c=None,
    toa=None,
    label=None,
    shape=(4, 4),
    transform=NORTH_UP,
) -> Path:
    """Write a scene folder with the four input bands and the label (all-ones TOA by default)."""
    folder.mkdir(parents=True, exist_ok=True)
    mag1c = np.full(shape, 10.0) if mag1c is None else np.asarray(mag1c, dtype="float64")
    label = np.zeros(shape, dtype=np.uint8) if label is None else np.asarray(label, dtype=np.uint8)
    _write_band(folder / "mag1c.tif", mag1c, transform)
    for band in TOA:
        toa_array = np.ones(shape) if toa is None else np.asarray(toa, dtype="float64")
        _write_band(folder / f"{band}.tif", toa_array, transform)
    _write_band(folder / "labelbinary.tif", label, transform)
    return folder


def _flat(values: list[float], shape=(4, 4), fill: float = 1.0) -> np.ndarray:
    """A `shape` array starting with `values`, padded with `fill`."""
    array = np.full(shape[0] * shape[1], fill)
    array[: len(values)] = values
    return array.reshape(shape)


class TestConstants:
    def test_clip_limits_are_factor_times_the_upper_clip(self):
        assert data_quality.CLIP_LIMITS == {
            "mag1c": 3500.0,
            "TOA_AVIRIS_640nm": 120.0,
            "TOA_AVIRIS_550nm": 120.0,
            "TOA_AVIRIS_460nm": 120.0,
        }

    def test_bands_start_with_mag1c_then_the_three_toa_bands(self):
        assert data_quality.BANDS == ("mag1c", *TOA)

    def test_noise_threshold_is_the_starcop_baseline_500_ppm_m(self):
        assert data_quality.NOISE_THRESHOLD == 500.0

    def test_nodata_and_duplicate_bands(self):
        assert data_quality.NODATA_BAND == "TOA_AVIRIS_640nm"
        assert data_quality.DUPLICATE_FILES == (
            "TOA_AVIRIS_640nm.tif",
            "mag1c.tif",
            "labelbinary.tif",
        )

    def test_spatial_grid_defaults(self):
        assert data_quality.BLOCK_PIXELS == 16
        assert data_quality.CELL_METERS == 100.0

    def test_partition_pairs_and_layout(self):
        assert data_quality.PARTITION_PAIRS == (
            ("train", "val"),
            ("train", "test"),
            ("val", "test"),
        )
        assert data_quality.PROCESSED_ROOT == Path("data") / "processed"
        assert data_quality.MINI_DATASET == "starcop_mini"
        assert data_quality.RAW_DATASET == "starcop_raw"
        assert data_quality.SPLITS_DIR == "splits"
        assert data_quality.SPLIT_FILE == "{split}.csv"


class TestSceneStats:
    def test_counts_pixels_and_plume_pixels(self, tmp_path):
        label = _flat([1, 1, 1], fill=0)
        scene = _write_scene(tmp_path / "s", label=label)
        stats = scene_stats(scene)
        assert stats["pixels"] == 16
        assert stats["plume_pixels"] == 3

    def test_counts_zeros_negatives_nan_and_inf_per_band(self, tmp_path):
        mag1c = _flat([0, 0, -1, np.nan, np.inf])
        scene = _write_scene(tmp_path / "s", mag1c=mag1c)
        band = scene_stats(scene)["bands"]["mag1c"]
        assert band["zeros"] == 2
        assert band["negative"] == 1
        assert band["nan"] == 1
        assert band["inf"] == 1

    def test_zero_is_not_counted_as_negative_and_negative_not_as_zero(self, tmp_path):
        scene = _write_scene(tmp_path / "s", mag1c=_flat([0, -2, -3]))
        band = scene_stats(scene)["bands"]["mag1c"]
        assert (band["zeros"], band["negative"]) == (1, 2)

    def test_above_clip_is_strict_and_uses_each_bands_own_limit(self, tmp_path):
        mag1c = _flat([3500, 3501, 5000])
        toa = _flat([120, 121, 500])
        scene = _write_scene(tmp_path / "s", mag1c=mag1c, toa=toa)
        bands = scene_stats(scene)["bands"]
        assert bands["mag1c"]["above_clip"] == 2
        for name in TOA:
            assert bands[name]["above_clip"] == 2

    def test_a_mag1c_value_that_is_only_above_the_toa_limit_is_not_above_clip(self, tmp_path):
        scene = _write_scene(tmp_path / "s", mag1c=_flat([121, 3000]))
        assert scene_stats(scene)["bands"]["mag1c"]["above_clip"] == 0

    def test_a_toa_value_that_is_only_above_the_mag1c_limit_counts_for_toa(self, tmp_path):
        scene = _write_scene(tmp_path / "s", toa=_flat([200]))
        bands = scene_stats(scene)["bands"]
        assert all(bands[name]["above_clip"] == 1 for name in TOA)

    def test_every_toa_band_is_read_from_its_own_file(self, tmp_path):
        scene = _write_scene(tmp_path / "s")
        _write_band(scene / "TOA_AVIRIS_550nm.tif", _flat([0, 0]))
        bands = scene_stats(scene)["bands"]
        assert bands["TOA_AVIRIS_640nm"]["zeros"] == 0
        assert bands["TOA_AVIRIS_550nm"]["zeros"] == 2
        assert bands["TOA_AVIRIS_460nm"]["zeros"] == 0

    def test_mag1c_above_noise_threshold_is_strict(self, tmp_path):
        scene = _write_scene(tmp_path / "s", mag1c=_flat([500, 501, 900]))
        assert scene_stats(scene)["mag1c_above_noise"] == 2

    def test_mag1c_above_clip_inside_the_label(self, tmp_path):
        mag1c = _flat([4000, 4000, 4000, 100])
        label = _flat([1, 0, 1, 1], fill=0)
        scene = _write_scene(tmp_path / "s", mag1c=mag1c, label=label)
        assert scene_stats(scene)["mag1c_above_clip_in_label"] == 2

    def test_mag1c_at_the_clip_limit_inside_the_label_is_not_above_clip(self, tmp_path):
        scene = _write_scene(tmp_path / "s", mag1c=_flat([3500, 3501]), label=_flat([1, 1], fill=0))
        assert scene_stats(scene)["mag1c_above_clip_in_label"] == 1

    def test_only_the_first_band_of_the_label_is_read(self, tmp_path):
        scene = _write_scene(tmp_path / "s")
        _write_two_bands(scene / "labelbinary.tif", _flat([1, 1, 1], fill=0), np.ones((4, 4)))
        stats = scene_stats(scene)
        assert stats["pixels"] == 16
        assert stats["plume_pixels"] == 3

    def test_only_the_first_band_of_an_input_raster_is_read(self, tmp_path):
        scene = _write_scene(tmp_path / "s")
        _write_two_bands(scene / "mag1c.tif", _flat([0, 0]), np.full((4, 4), 9000.0))
        band = scene_stats(scene)["bands"]["mag1c"]
        assert band["zeros"] == 2
        assert band["above_clip"] == 0

    def test_missing_band_file_raises(self, tmp_path):
        scene = _write_scene(tmp_path / "s")
        (scene / "mag1c.tif").unlink()
        with pytest.raises(rasterio.errors.RasterioIOError):
            scene_stats(scene)


def _stats(
    *, pixels=100, plume=0, zeros=0, negative=0, above=0, noise=0, in_label=0, nan=0, inf=0
) -> dict:
    band = {"nan": nan, "inf": inf, "zeros": zeros, "negative": negative, "above_clip": above}
    return {
        "pixels": pixels,
        "plume_pixels": plume,
        "bands": {name: dict(band) for name in data_quality.BANDS},
        "mag1c_above_noise": noise,
        "mag1c_above_clip_in_label": in_label,
    }


class TestSummarizeQuality:
    def test_totals_and_fractions_per_band(self):
        summary = summarize_quality([_stats(zeros=10, above=4), _stats(zeros=30, above=6)])
        band = summary["bands"]["TOA_AVIRIS_640nm"]
        assert summary["scenes"] == 2
        assert summary["pixels"] == 200
        assert band["zeros"] == 40
        assert band["zero_fraction"] == pytest.approx(0.2)
        assert band["above_clip"] == 10
        assert band["above_clip_fraction"] == pytest.approx(0.05)

    def test_counts_scenes_with_zeros_negatives_and_values_above_clip(self):
        summary = summarize_quality(
            [_stats(zeros=1), _stats(negative=2, above=3), _stats(), _stats(zeros=5, negative=1)]
        )
        band = summary["bands"]["mag1c"]
        assert band["scenes_with_zeros"] == 2
        assert band["scenes_with_negative"] == 2
        assert band["scenes_above_clip"] == 1

    def test_scene_counts_of_zero_negative_and_clip_are_independent(self):
        summary = summarize_quality([_stats(zeros=1), _stats(above=1), _stats(negative=1)])
        band = summary["bands"]["mag1c"]
        assert (band["scenes_with_zeros"], band["scenes_above_clip"]) == (1, 1)
        assert band["scenes_with_negative"] == 1

    def test_sums_nan_and_inf(self):
        summary = summarize_quality([_stats(nan=1, inf=2), _stats(nan=3, inf=4)])
        band = summary["bands"]["mag1c"]
        assert (band["nan"], band["inf"]) == (4, 6)

    def test_noise_fraction_uses_only_scenes_without_plume_pixels(self):
        stats = [
            _stats(pixels=100, plume=0, noise=2),
            _stats(pixels=300, plume=0, noise=6),
            _stats(pixels=1000, plume=50, noise=999),
        ]
        summary = summarize_quality(stats)
        assert summary["unlabelled_scenes"] == 2
        assert summary["unlabelled_noise_fraction"] == pytest.approx(8 / 400)
        assert summary["unlabelled_scenes_with_noise"] == 2

    def test_unlabelled_scenes_with_noise_ignores_clean_and_labelled_scenes(self):
        stats = [_stats(plume=0, noise=1), _stats(plume=0, noise=0), _stats(plume=9, noise=5)]
        assert summarize_quality(stats)["unlabelled_scenes_with_noise"] == 1

    def test_without_unlabelled_scenes_the_noise_fraction_is_none(self):
        summary = summarize_quality([_stats(plume=5, noise=3)])
        assert summary["unlabelled_scenes"] == 0
        assert summary["unlabelled_noise_fraction"] is None

    def test_share_of_above_clip_pixels_inside_the_label(self):
        stats = [_stats(above=6, in_label=2), _stats(above=4, in_label=2)]
        assert summarize_quality(stats)["mag1c_above_clip_in_label_share"] == pytest.approx(0.4)

    def test_share_is_none_when_nothing_is_above_clip(self):
        assert summarize_quality([_stats()])["mag1c_above_clip_in_label_share"] is None

    def test_rejects_an_empty_list(self):
        with pytest.raises(ValueError, match=r"^no scenes to summarize$"):
            summarize_quality([])

    def test_is_json_serialisable(self):
        summary = summarize_quality([_stats(zeros=1, above=2, in_label=1, plume=0, noise=1)])
        assert json.loads(json.dumps(summary)) == summary


class TestLabelDiscordance:
    def test_counts_both_kinds_of_disagreement(self):
        has_plume = [True, True, False, False, True]
        plume_pixels = [10, 0, 0, 5, 3]
        assert label_discordance(has_plume, plume_pixels) == {
            "plume_without_pixels": 1,
            "pixels_without_plume": 1,
        }

    def test_a_single_labelled_pixel_without_plume_flag_is_a_disagreement(self):
        assert label_discordance([False], [1]) == {
            "plume_without_pixels": 0,
            "pixels_without_plume": 1,
        }

    def test_agreement_gives_zero(self):
        assert label_discordance([True, False], [4, 0]) == {
            "plume_without_pixels": 0,
            "pixels_without_plume": 0,
        }

    def test_rejects_lists_of_different_length(self):
        with pytest.raises(ValueError, match=r"^has_plume and plume_pixels differ in length$"):
            label_discordance([True], [1, 2])


class TestDuplicateScenes:
    def test_identical_scenes_form_one_group(self, tmp_path):
        first = _write_scene(tmp_path / "a", mag1c=_flat([5, 6]))
        second = _write_scene(tmp_path / "b", mag1c=_flat([5, 6]))
        third = _write_scene(tmp_path / "c", mag1c=_flat([7, 8]))
        groups = duplicate_scenes([first, second, third])
        assert groups == [[first, second]]

    def test_distinct_scenes_give_no_group(self, tmp_path):
        first = _write_scene(tmp_path / "a", mag1c=_flat([5]))
        second = _write_scene(tmp_path / "b", mag1c=_flat([6]))
        assert duplicate_scenes([first, second]) == []

    def test_scenes_that_differ_only_in_the_label_are_not_duplicates(self, tmp_path):
        first = _write_scene(tmp_path / "a")
        second = _write_scene(tmp_path / "b", label=_flat([1], fill=0))
        assert duplicate_scenes([first, second]) == []

    def test_scenes_that_differ_only_in_toa_are_not_duplicates(self, tmp_path):
        first = _write_scene(tmp_path / "a")
        second = _write_scene(tmp_path / "b", toa=_flat([2]))
        assert duplicate_scenes([first, second]) == []

    def test_scenes_that_differ_only_in_mag1c_are_not_duplicates(self, tmp_path):
        first = _write_scene(tmp_path / "a", mag1c=_flat([5]))
        second = _write_scene(tmp_path / "b", mag1c=_flat([9]))
        assert duplicate_scenes([first, second]) == []

    def test_three_copies_form_a_single_group_in_input_order(self, tmp_path):
        folders = [_write_scene(tmp_path / name) for name in ("a", "b", "c")]
        assert duplicate_scenes(folders) == [folders]

    def test_only_the_first_band_of_each_raster_is_compared(self, tmp_path):
        first = _write_scene(tmp_path / "a")
        second = _write_scene(tmp_path / "b")
        _write_two_bands(first / "mag1c.tif", np.full((4, 4), 5.0), np.full((4, 4), 1.0))
        _write_two_bands(second / "mag1c.tif", np.full((4, 4), 5.0), np.full((4, 4), 2.0))
        assert duplicate_scenes([first, second]) == [[first, second]]

    def test_no_scenes_give_no_group(self):
        assert duplicate_scenes([]) == []


class TestPixelSize:
    def test_north_up_raster_returns_its_pixel_size(self, tmp_path):
        scene = _write_scene(tmp_path / "s", transform=from_origin(0, 100, 5, 7))
        assert pixel_size(scene) == (pytest.approx(5.0), pytest.approx(7.0))

    def test_rotated_raster_returns_the_true_size_not_the_a_term(self, tmp_path):

        transform = Affine.rotation(30) * Affine.scale(5, -5)
        scene = _write_scene(tmp_path / "s", transform=Affine.translation(1000, 2000) * transform)
        size_x, size_y = pixel_size(scene)
        assert size_x == pytest.approx(5.0)
        assert size_y == pytest.approx(5.0)

    def test_quarter_turn_raster(self, tmp_path):
        transform = Affine.translation(0, 0) * Affine.rotation(90) * Affine.scale(4, -4)
        scene = _write_scene(tmp_path / "s", transform=transform)
        assert pixel_size(scene) == (pytest.approx(4.0), pytest.approx(4.0))

    def test_each_axis_uses_its_own_pair_of_terms(self, tmp_path):

        scene = _write_scene(tmp_path / "s", transform=Affine(3, 1, 0, 4, 2, 0))
        assert pixel_size(scene) == (pytest.approx(5.0), pytest.approx(math.sqrt(5)))


class TestSceneCells:
    def test_fully_valid_scene_covers_every_block_centre_cell(self, tmp_path):

        scene = _write_scene(
            tmp_path / "s", shape=(8, 8), transform=from_origin(0, 8, 1, 1), toa=np.ones((8, 8))
        )
        assert scene_cells(scene, block=4, cell=4.0) == {(0, 1), (1, 1), (0, 0), (1, 0)}

    def test_all_zero_blocks_are_nodata_and_leave_no_cell(self, tmp_path):
        toa = np.ones((8, 8))
        toa[:4, :] = 0
        scene = _write_scene(
            tmp_path / "s", shape=(8, 8), transform=from_origin(0, 8, 1, 1), toa=toa
        )
        assert scene_cells(scene, block=4, cell=4.0) == {(0, 0), (1, 0)}

    def test_a_block_with_a_single_valid_pixel_still_counts(self, tmp_path):
        toa = np.zeros((8, 8))
        toa[0, 0] = 5
        scene = _write_scene(
            tmp_path / "s", shape=(8, 8), transform=from_origin(0, 8, 1, 1), toa=toa
        )
        assert scene_cells(scene, block=4, cell=4.0) == {(0, 1)}

    def test_cell_size_sets_the_grid(self, tmp_path):
        scene = _write_scene(
            tmp_path / "s", shape=(8, 8), transform=from_origin(0, 8, 1, 1), toa=np.ones((8, 8))
        )
        assert scene_cells(scene, block=4, cell=8.0) == {(0, 0)}

    def test_only_the_nodata_band_decides_validity(self, tmp_path):
        scene = _write_scene(
            tmp_path / "s", shape=(8, 8), transform=from_origin(0, 8, 1, 1), toa=np.zeros((8, 8))
        )
        _write_band(scene / "TOA_AVIRIS_640nm.tif", np.ones((8, 8)), from_origin(0, 8, 1, 1))
        assert len(scene_cells(scene, block=4, cell=4.0)) == 4

    def test_a_block_is_placed_by_the_ground_position_of_its_centre(self, tmp_path):

        scene = _write_scene(
            tmp_path / "s", shape=(8, 8), transform=from_origin(0, 8, 1, 1), toa=np.ones((8, 8))
        )
        assert scene_cells(scene, block=4, cell=0.5) == {(4, 12), (12, 12), (4, 4), (12, 4)}

    def test_rotated_scene_places_cells_along_its_own_axes(self, tmp_path):

        scene = _write_scene(
            tmp_path / "s",
            shape=(8, 4),
            transform=Affine.rotation(90),
            toa=np.ones((8, 4)),
        )
        assert scene_cells(scene, block=4, cell=1.0) == {(-2, 2), (-6, 2)}

    def test_negative_coordinates_are_floored_not_truncated(self, tmp_path):

        scene = _write_scene(
            tmp_path / "s", shape=(8, 8), transform=from_origin(-8, 0, 1, 1), toa=np.ones((8, 8))
        )
        assert scene_cells(scene, block=4, cell=4.0) == {(-2, -1), (-1, -1), (-2, -2), (-1, -2)}

    def test_cell_indices_are_integers(self, tmp_path):
        scene = _write_scene(
            tmp_path / "s", shape=(8, 8), transform=from_origin(0, 8, 1, 1), toa=np.ones((8, 8))
        )
        cells = scene_cells(scene, block=4, cell=4.0)
        assert all(type(index) is int for cell in cells for index in cell)

    def test_scene_smaller_than_a_block_has_no_cell(self, tmp_path):
        scene = _write_scene(tmp_path / "s", shape=(2, 2), toa=np.ones((2, 2)))
        assert scene_cells(scene, block=4, cell=4.0) == set()


class TestCoverageReport:
    def test_fraction_of_target_cells_covered_by_the_reference(self):
        covers = {
            "test": [{(0, 0), (1, 0)}, {(5, 5)}],
            "train": [{(0, 0)}, {(9, 9)}],
        }
        report = coverage_report(covers, "test", "train")
        assert report["cells_covered_fraction"] == pytest.approx(1 / 3)

    def test_per_scene_statistics(self):
        covers = {
            "test": [{(0, 0), (1, 0)}, {(5, 5)}, {(0, 0), (1, 0), (2, 0), (3, 0)}, {(8, 8)}],
            "train": [{(0, 0), (1, 0), (2, 0)}],
        }
        report = coverage_report(covers, "test", "train")
        assert report["scenes"] == 4
        assert report["scenes_half_covered"] == 2
        assert report["scenes_nine_tenths_covered"] == 1
        assert report["scenes_uncovered"] == 2
        assert report["median_scene_fraction"] == pytest.approx(0.375)

    def test_exactly_half_covered_counts_as_half_covered(self):
        covers = {"test": [{(0, 0), (1, 0)}], "train": [{(0, 0)}]}
        assert coverage_report(covers, "test", "train")["scenes_half_covered"] == 1

    def test_exactly_nine_tenths_counts(self):
        target = {(i, 0) for i in range(10)}
        covers = {"test": [target], "train": [{(i, 0) for i in range(9)}]}
        assert coverage_report(covers, "test", "train")["scenes_nine_tenths_covered"] == 1

    def test_reference_is_the_union_of_all_its_scenes(self):
        covers = {"test": [{(0, 0), (1, 0)}], "train": [{(0, 0)}, {(1, 0)}]}
        report = coverage_report(covers, "test", "train")
        assert report["cells_covered_fraction"] == pytest.approx(1.0)
        assert report["scenes_uncovered"] == 0

    def test_a_scene_without_cells_is_rejected(self):
        covers = {"test": [set()], "train": [{(0, 0)}]}
        with pytest.raises(ValueError, match=r"^scene without valid cells$"):
            coverage_report(covers, "test", "train")

    def test_a_partition_without_scenes_is_rejected(self):
        with pytest.raises(ValueError, match=r"^no scenes in partition 'test'$"):
            coverage_report({"test": [], "train": [{(0, 0)}]}, "test", "train")


class TestOverlapPairs:
    def test_reports_pairs_sharing_a_cell_with_the_fraction_of_the_smaller_scene(self):
        covers = [{(0, 0), (1, 0), (2, 0), (3, 0)}, {(2, 0), (3, 0)}, {(9, 9)}]
        assert overlap_pairs(covers) == [(0, 1, 1.0)]

    def test_fraction_is_relative_to_the_smaller_scene(self):
        covers = [{(0, 0), (1, 0)}, {(1, 0), (5, 5), (6, 6), (7, 7)}]
        assert overlap_pairs(covers) == [(0, 1, 0.5)]

    def test_scenes_without_shared_cells_give_no_pair(self):
        assert overlap_pairs([{(0, 0)}, {(1, 1)}]) == []

    def test_each_pair_is_reported_once_with_the_lower_index_first(self):
        covers = [{(0, 0), (1, 0)}, {(0, 0), (1, 0)}, {(0, 0)}]
        pairs = overlap_pairs(covers)
        assert [(i, j) for i, j, _ in pairs] == [(0, 1), (0, 2), (1, 2)]

    def test_shared_cells_are_counted_not_just_detected(self):
        covers = [{(0, 0), (1, 0), (2, 0), (3, 0)}, {(0, 0), (1, 0), (7, 7), (8, 8)}]
        assert overlap_pairs(covers) == [(0, 1, 0.5)]

    def test_empty_input_gives_no_pair(self):
        assert overlap_pairs([]) == []


class TestClassifyPairs:
    def test_splits_pairs_by_flight_line_and_partition(self):
        splits = ["train", "train", "train", "test"]
        names = ["a", "a", "b", "c"]
        pairs = [(0, 1, 0.6), (0, 2, 0.2), (2, 3, 0.9), (1, 3, 0.4)]
        result = classify_pairs(pairs, splits, names)
        assert result == {
            "same_flight_line": {"pairs": 1, "half_or_more": 1},
            "different_flight_line_same_partition": {"pairs": 1, "half_or_more": 0},
            "different_partitions": {"pairs": 2, "half_or_more": 1},
        }

    def test_exactly_half_counts_as_half_or_more(self):
        result = classify_pairs([(0, 1, 0.5)], ["train", "train"], ["a", "a"])
        assert result["same_flight_line"]["half_or_more"] == 1

    def test_different_partitions_wins_over_the_same_flight_line_name(self):

        result = classify_pairs([(0, 1, 0.1)], ["train", "test"], ["a", "a"])
        assert result["different_partitions"]["pairs"] == 1
        assert result["same_flight_line"]["pairs"] == 0

    def test_counts_every_half_covered_pair_of_a_kind(self):
        pairs = [(0, 1, 0.9), (0, 2, 0.6), (1, 2, 0.1)]
        result = classify_pairs(pairs, ["train"] * 3, ["a", "a", "a"])
        assert result["same_flight_line"] == {"pairs": 3, "half_or_more": 2}

    def test_no_pairs_gives_zero_counts(self):
        result = classify_pairs([], [], [])
        assert all(v == {"pairs": 0, "half_or_more": 0} for v in result.values())


class TestPartitionHelpers:
    def test_shared_values_between_each_pair_of_partitions(self):
        values = {"train": ["a", "b", "c"], "val": ["b", "x"], "test": ["c", "y"]}
        assert shared_between_partitions(values) == {
            "train/val": ["b"],
            "train/test": ["c"],
            "val/test": [],
        }

    def test_shared_values_are_sorted_and_unique(self):
        values = {"train": ["b", "a", "b"], "val": ["a", "b", "a"], "test": []}
        assert shared_between_partitions(values)["train/val"] == ["a", "b"]

    def test_date_ranges_give_count_first_and_last_day(self):
        ranges = date_ranges(
            {"train": ["2019-10-02", "2019-09-22", "2019-10-02"], "test": ["2019-10-18"]}
        )
        assert ranges["train"] == {"days": 2, "first": "2019-09-22", "last": "2019-10-02"}
        assert ranges["test"] == {"days": 1, "first": "2019-10-18", "last": "2019-10-18"}

    def test_date_ranges_reject_a_partition_without_dates(self):
        with pytest.raises(ValueError, match=r"^no dates for partition 'val'$"):
            date_ranges({"val": []})


def _write_dataset(root: Path, dataset: str, splits: dict[str, list[dict]]) -> None:
    """Write `splits/<split>.csv` (folder, name, date, has_plume) and the scene folders."""
    base = root / "data" / "processed" / dataset
    (base / "splits").mkdir(parents=True, exist_ok=True)
    for split, scenes in splits.items():
        rows = []
        for scene in scenes:
            folder = f"data/processed/{dataset}/selected/{scene['id']}"
            _write_scene(
                root / folder,
                mag1c=scene.get("mag1c"),
                toa=scene.get("toa"),
                label=scene.get("label"),
                shape=(8, 8),
                transform=scene.get("transform", from_origin(0, 8, 1, 1)),
            )
            rows.append(
                {
                    "folder": folder,
                    "name": scene["name"],
                    "date": scene["date"],
                    "has_plume": scene["has_plume"],
                }
            )
        pd.DataFrame(rows).to_csv(base / "splits" / f"{split}.csv", index=False)


def _label(positives: int) -> np.ndarray:
    array = np.zeros((8, 8), dtype=np.uint8)
    array.flat[:positives] = 1
    return array


@pytest.fixture
def fake_repo(tmp_path):
    def scene(id_, name, date, has_plume, *, label=None, mag1c=None, transform=None):
        entry = {"id": id_, "name": name, "date": date, "has_plume": has_plume}
        entry["label"] = _label(0) if label is None else label
        entry["mag1c"] = np.full((8, 8), 10.0) if mag1c is None else mag1c
        if transform is not None:
            entry["transform"] = transform
        return entry

    far = from_origin(10_000, 10_008, 2, 1)
    _write_dataset(
        tmp_path,
        "starcop_mini",
        {
            "train": [scene("m1", "f1", "2019-10-01", True, label=_label(4))],
            "val": [scene("m2", "f2", "2019-10-02", True, label=_label(2))],
            "test": [scene("m3", "f3", "2019-10-18", True, label=_label(8))],
        },
    )
    _write_dataset(
        tmp_path,
        "starcop_raw",
        {
            "train": [
                scene("r1", "f1", "2019-10-01", True, label=_label(4)),
                scene("r2", "f1", "2019-10-01", False, mag1c=np.full((8, 8), 600.0)),
                scene("r3", "f4", "2019-10-03", True),
            ],
            "val": [scene("r4", "f2", "2019-10-02", False, mag1c=np.full((8, 8), 20.0))],
            "test": [
                scene("r5", "f3", "2019-10-18", True, label=_label(8)),
                scene("r6", "f5", "2019-10-21", False, transform=far),
            ],
        },
    )
    return tmp_path


@pytest.fixture
def adjacent_repo(tmp_path):
    """Three raw scenes side by side: the validation one touches the train one but not overlaps."""

    def scene(id_, name, date, x_left):
        return {
            "id": id_,
            "name": name,
            "date": date,
            "has_plume": False,
            "transform": from_origin(x_left, 8, 1, 1),
        }

    _write_dataset(
        tmp_path,
        "starcop_mini",
        {
            "train": [scene("m1", "f1", "2019-10-01", 0)],
            "val": [scene("m2", "f2", "2019-10-02", 8)],
            "test": [scene("m3", "f3", "2019-10-18", 500)],
        },
    )
    _write_dataset(
        tmp_path,
        "starcop_raw",
        {
            "train": [scene("r1", "f1", "2019-10-01", 0)],
            "val": [scene("r2", "f2", "2019-10-02", 8)],
            "test": [scene("r3", "f3", "2019-10-18", 500)],
        },
    )
    return tmp_path


class TestBuildDataQuality:
    def _build(self, root):
        return build_data_quality(root, block=4, cell=4.0)

    def test_expected_fields_are_present(self, fake_repo):
        result = self._build(fake_repo)
        assert set(result) == {
            "quality",
            "label_discordance",
            "duplicate_scene_groups",
            "pixel_size_m",
            "dates",
            "shared_dates",
            "shared_flight_lines",
            "coverage",
            "overlapping_pairs",
        }

    def test_quality_summarises_mini_and_the_full_set_separately(self, fake_repo):
        quality = self._build(fake_repo)["quality"]
        assert quality["mini"]["scenes"] == 3
        assert quality["raw"]["scenes"] == 6
        assert quality["raw"]["pixels"] == 6 * 64

    def test_quality_of_the_full_set_counts_noise_in_unlabelled_scenes(self, fake_repo):
        raw = self._build(fake_repo)["quality"]["raw"]

        assert raw["unlabelled_scenes"] == 4
        assert raw["unlabelled_scenes_with_noise"] == 1
        assert raw["unlabelled_noise_fraction"] == pytest.approx(64 / (4 * 64))

    def test_label_discordance_is_counted_per_partition_of_the_full_set(self, fake_repo):
        discordance = self._build(fake_repo)["label_discordance"]
        assert discordance["train"] == {"plume_without_pixels": 1, "pixels_without_plume": 0}
        assert discordance["val"] == {"plume_without_pixels": 0, "pixels_without_plume": 0}
        assert discordance["test"] == {"plume_without_pixels": 0, "pixels_without_plume": 0}

    def test_pixel_size_range_of_the_full_set(self, fake_repo):
        size = self._build(fake_repo)["pixel_size_m"]

        assert size == {"min": 1.0, "median": 1.0, "max": 2.0}

    def test_dates_and_shared_values(self, fake_repo):
        result = self._build(fake_repo)
        assert result["dates"]["train"] == {"days": 2, "first": "2019-10-01", "last": "2019-10-03"}
        assert result["dates"]["test"]["days"] == 2
        assert result["shared_dates"] == {"train/val": [], "train/test": [], "val/test": []}
        assert result["shared_flight_lines"] == {"train/val": [], "train/test": [], "val/test": []}

    def test_identical_scenes_are_reported_as_a_duplicate_group(self, fake_repo):

        result = self._build(fake_repo)
        assert result["duplicate_scene_groups"] == 1

    def test_coverage_of_test_and_validation_by_training(self, fake_repo):
        coverage = self._build(fake_repo)["coverage"]
        assert set(coverage) == {"test_by_train", "val_by_train"}

        assert coverage["test_by_train"]["scenes"] == 2
        assert coverage["test_by_train"]["scenes_uncovered"] == 1
        assert coverage["test_by_train"]["scenes_nine_tenths_covered"] == 1
        assert coverage["val_by_train"]["scenes_half_covered"] == 1

    def test_overlapping_pairs_are_classified(self, fake_repo):
        pairs = self._build(fake_repo)["overlapping_pairs"]

        assert pairs["same_flight_line"] == {"pairs": 1, "half_or_more": 1}
        assert pairs["different_flight_line_same_partition"]["pairs"] == 2
        assert pairs["different_partitions"]["pairs"] == 7

    def test_the_cell_size_decides_which_scenes_share_ground(self, adjacent_repo):

        coverage = self._build(adjacent_repo)["coverage"]
        assert coverage["val_by_train"]["scenes_uncovered"] == 1
        assert (
            build_data_quality(adjacent_repo, block=4)["coverage"]["val_by_train"][
                "scenes_uncovered"
            ]
            == 0
        )

    def test_result_is_json_serialisable(self, fake_repo):
        result = self._build(fake_repo)
        assert json.loads(json.dumps(result)) == result


class TestWriteDataQuality:
    def test_creates_every_missing_parent_folder(self, fake_repo, tmp_path):
        out_path = tmp_path / "out" / "figures" / "data_quality.json"
        write_data_quality(fake_repo, out_path, block=4, cell=4.0)
        assert out_path.is_file()

    def test_writes_into_a_folder_that_already_exists(self, fake_repo, tmp_path):
        out_path = tmp_path / "data_quality.json"
        write_data_quality(fake_repo, out_path, block=4, cell=4.0)
        assert out_path.is_file()

    def test_passes_the_cell_size_through_to_the_checks(self, adjacent_repo, tmp_path):
        out_path = tmp_path / "data_quality.json"
        write_data_quality(adjacent_repo, out_path, block=4, cell=4.0)
        coverage = json.loads(out_path.read_text())["coverage"]
        assert coverage["val_by_train"]["scenes_uncovered"] == 1

    def test_writes_two_space_indented_json_and_returns_the_path(self, fake_repo, tmp_path):
        out_path = tmp_path / "out" / "data_quality.json"
        written = write_data_quality(fake_repo, out_path, block=4, cell=4.0)
        assert written == out_path
        expected = json.dumps(build_data_quality(fake_repo, block=4, cell=4.0), indent=2) + "\n"
        assert out_path.read_text() == expected
