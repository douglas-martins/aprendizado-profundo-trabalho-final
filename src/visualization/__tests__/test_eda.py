import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.windows import Window

import visualization.eda as eda
from visualization.eda import (
    compute_patch_level_balance,
    example_column_title,
    example_plume_pixels,
    partition_series,
    read_patch_bands,
    rgb_composite,
    select_example_patches,
)
from visualization.figure_common import TIER_COLORS


def _patches_df(has_plume: list[bool], frac_positives: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [f"patch_{i}" for i in range(len(has_plume))],
            "has_plume": has_plume,
            "frac_positives": frac_positives,
        }
    )


class TestComputePatchLevelBalance:
    def test_counts_positive_and_total(self):
        df = _patches_df([True, False, True, False, False], [0.1, 0.0, 0.02, 0.0, 0.0])
        result = compute_patch_level_balance(df)
        assert result == {"positive": 2, "total": 5, "positive_fraction": pytest.approx(0.4)}

    def test_all_negative_gives_zero_fraction(self):
        df = _patches_df([False, False], [0.0, 0.0])
        result = compute_patch_level_balance(df)
        assert result["positive"] == 0
        assert result["positive_fraction"] == 0.0

    def test_empty_dataframe_does_not_raise(self):
        df = _patches_df([], [])
        result = compute_patch_level_balance(df)
        assert result == {"positive": 0, "total": 0, "positive_fraction": 0.0}

    def test_an_empty_dataframe_does_not_need_a_has_plume_column(self):
        assert compute_patch_level_balance(pd.DataFrame()) == {
            "positive": 0,
            "total": 0,
            "positive_fraction": 0.0,
        }


class TestSelectExamplePatches:
    def test_returns_requested_counts(self):
        df = _patches_df(
            has_plume=[True, True, True, False, False, False],
            frac_positives=[0.05, 0.003, 0.2, 0.0, 0.0, 0.0],
        )
        selected = select_example_patches(df, n_positive=2, n_negative=2, seed=42)
        assert len(selected) == 4
        assert selected["has_plume"].sum() == 2
        assert (~selected["has_plume"]).sum() == 2

    def test_includes_the_faintest_plume_as_the_hard_example(self):
        df = _patches_df(
            has_plume=[True, True, True, False, False],
            frac_positives=[0.05, 0.003, 0.2, 0.0, 0.0],
        )
        selected = select_example_patches(df, n_positive=2, n_negative=1, seed=42)
        assert "patch_1" in selected["id"].to_numpy()

    def test_deterministic_given_same_seed(self):
        df = _patches_df(
            has_plume=[True, True, True, True, False, False, False, False],
            frac_positives=[0.05, 0.003, 0.2, 0.09, 0.0, 0.0, 0.0, 0.0],
        )
        first = select_example_patches(df, n_positive=2, n_negative=2, seed=7)
        second = select_example_patches(df, n_positive=2, n_negative=2, seed=7)
        assert list(first["id"]) == list(second["id"])

    def test_raises_when_not_enough_positive_patches(self):
        df = _patches_df(has_plume=[True, False, False], frac_positives=[0.1, 0.0, 0.0])
        with pytest.raises(ValueError, match="positive"):
            select_example_patches(df, n_positive=2, n_negative=1, seed=42)

    def test_raises_when_not_enough_negative_patches(self):
        df = _patches_df(has_plume=[True, True, False], frac_positives=[0.1, 0.2, 0.0])
        with pytest.raises(ValueError, match="negative"):
            select_example_patches(df, n_positive=1, n_negative=2, seed=42)

    def test_does_not_raise_when_exactly_enough_positive_patches(self):
        df = _patches_df(has_plume=[True, True, False], frac_positives=[0.1, 0.2, 0.0])
        selected = select_example_patches(df, n_positive=2, n_negative=1, seed=42)
        assert selected["has_plume"].sum() == 2

    def test_does_not_raise_when_exactly_enough_negative_patches(self):
        df = _patches_df(has_plume=[True, False, False], frac_positives=[0.1, 0.0, 0.0])
        selected = select_example_patches(df, n_positive=1, n_negative=2, seed=42)
        assert (~selected["has_plume"]).sum() == 2

    def test_result_has_a_clean_contiguous_index(self):

        df = _patches_df(
            has_plume=[True, True, True, False, False, False],
            frac_positives=[0.05, 0.003, 0.2, 0.0, 0.0, 0.0],
        )
        selected = select_example_patches(df, n_positive=2, n_negative=2, seed=42)
        assert list(selected.index) == list(range(len(selected)))

    def test_sampling_uses_the_correct_count_and_seed_for_each_group(self, monkeypatch):

        df = _patches_df(
            has_plume=[True, True, True, False, False, False],
            frac_positives=[0.05, 0.003, 0.2, 0.0, 0.0, 0.0],
        )
        captured = []
        original_sample = pd.DataFrame.sample

        def spy_sample(self, *args, **kwargs):
            captured.append(
                {
                    "n_rows": len(self),
                    "n": kwargs.get("n"),
                    "random_state": kwargs.get("random_state"),
                }
            )
            return original_sample(self, *args, **kwargs)

        monkeypatch.setattr(pd.DataFrame, "sample", spy_sample)

        select_example_patches(df, n_positive=3, n_negative=2, seed=99)

        assert {"n_rows": 2, "n": 2, "random_state": 99} in captured

        assert {"n_rows": 3, "n": 2, "random_state": 99} in captured


class TestReadPatchBands:
    def test_reads_the_requested_window_from_each_band(self, tmp_path, tiny_geotiff_factory):
        scene = tmp_path / "scene"
        band_a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32")
        band_b = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype="float32")
        tiny_geotiff_factory(scene / "band_a.tif", band_a)
        tiny_geotiff_factory(scene / "band_b.tif", band_b)

        window = Window(col_off=1, row_off=0, width=2, height=2)
        result = read_patch_bands(scene, window, ["band_a", "band_b"])

        assert set(result.keys()) == {"band_a", "band_b"}
        np.testing.assert_array_equal(result["band_a"], [[2.0, 3.0], [5.0, 6.0]])
        np.testing.assert_array_equal(result["band_b"], [[20.0, 30.0], [50.0, 60.0]])

    def test_missing_band_file_raises(self, tmp_path):
        scene = tmp_path / "scene"
        scene.mkdir()
        window = Window(col_off=0, row_off=0, width=1, height=1)
        with pytest.raises(rasterio.errors.RasterioIOError):
            read_patch_bands(scene, window, ["nonexistent_band"])


class TestRgbComposite:
    def test_stacks_three_bands_in_order(self):
        bands = {
            "r": np.array([[0.0, 10.0]], dtype="float32"),
            "g": np.array([[0.0, 20.0]], dtype="float32"),
            "b": np.array([[0.0, 30.0]], dtype="float32"),
        }
        composite = rgb_composite(bands, "r", "g", "b")
        assert composite.shape == (1, 2, 3)

        np.testing.assert_allclose(composite[0, 1], [1.0, 1.0, 1.0])
        np.testing.assert_allclose(composite[0, 0], [0.0, 0.0, 0.0])

    def test_constant_band_does_not_produce_nan(self):
        bands = {
            "r": np.array([[5.0, 5.0]], dtype="float32"),
            "g": np.array([[0.0, 1.0]], dtype="float32"),
            "b": np.array([[0.0, 1.0]], dtype="float32"),
        }
        composite = rgb_composite(bands, "r", "g", "b")
        assert not np.isnan(composite).any()

    def test_scales_relative_to_the_bands_own_min_not_from_zero(self):

        bands = {
            "r": np.array([[2.0, 6.0]], dtype="float32"),
            "g": np.array([[0.0, 1.0]], dtype="float32"),
            "b": np.array([[0.0, 1.0]], dtype="float32"),
        }
        composite = rgb_composite(bands, "r", "g", "b")
        np.testing.assert_allclose(composite[0, :, 0], [0.0, 1.0])

    def test_output_is_always_float32_regardless_of_input_dtype(self):

        bands = {
            "r": np.array([[5.0, 5.0]], dtype="float64"),
            "g": np.array([[0.0, 1.0]], dtype="float64"),
            "b": np.array([[0.0, 1.0]], dtype="float64"),
        }
        composite = rgb_composite(bands, "r", "g", "b")
        assert composite.dtype == np.float32


def _balance(train: float, val: float, test: float) -> dict:
    return {
        "train": {"pixel_fraction": train, "patch_fraction": train * 10},
        "val": {"pixel_fraction": val, "patch_fraction": val * 10},
        "test": {"pixel_fraction": test, "patch_fraction": test * 10},
    }


class TestPartitionSeries:
    def test_returns_percentages_in_train_val_test_order(self):
        balance = {"mini": _balance(0.01, 0.002, 0.03)}
        assert partition_series(balance, "pixel_fraction") == {"mini": [1.0, 0.2, 3.0]}

    def test_keeps_the_scales_in_their_given_order(self):
        balance = {"mini": _balance(0.1, 0.1, 0.1), "R2": _balance(0.2, 0.2, 0.2)}
        assert list(partition_series(balance, "pixel_fraction")) == ["mini", "R2"]

    def test_reads_the_requested_fraction_key(self):
        balance = {"R3": _balance(0.01, 0.02, 0.03)}
        assert partition_series(balance, "patch_fraction") == {"R3": [10.0, 20.0, 30.0]}

    def test_unknown_fraction_key_raises(self):
        with pytest.raises(KeyError):
            partition_series({"mini": _balance(0.1, 0.1, 0.1)}, "nonexistent")


class TestExamplePlumePixels:
    def test_rounds_to_the_nearest_pixel_not_down(self):

        assert example_plume_pixels(0.0028, 128, 128) == 46

    def test_a_patch_at_the_plume_threshold_is_forty_pixels(self):
        assert example_plume_pixels(0.00244, 128, 128) == 40

    def test_a_plume_free_patch_has_no_plume_pixels(self):
        assert example_plume_pixels(0.0, 128, 128) == 0

    def test_uses_both_window_sides(self):
        assert example_plume_pixels(0.25, 8, 4) == 8

    def test_returns_a_plain_int(self):
        assert type(example_plume_pixels(0.0028, 128, 128)) is int


class TestExampleColumnTitle:
    def test_plume_patch_names_scale_and_pixel_count(self):
        assert example_column_title("mini", True, 46) == "mini, com pluma\n46 pixels"

    def test_plume_free_patch(self):
        assert example_column_title("R3", False, 0) == "R3, sem pluma\n0 pixels"


class TestFigureText:
    def test_partition_labels_are_portuguese(self):
        assert eda.PARTITION_LABELS == {"train": "Treino", "val": "Validação", "test": "Teste"}

    def test_balance_panels_name_the_two_criteria(self):
        assert eda.BALANCE_PANELS == (
            ("pixel_fraction", "(a) Pixels de pluma (%)"),
            ("patch_fraction", "(b) Recortes com pluma (%)"),
        )

    def test_example_rows_are_portuguese(self):
        assert eda.EXAMPLE_ROW_LABELS == ("Composição RGB", "mag1c", "Rótulo")

    def test_mag1c_colour_scale_is_shared_and_labelled_with_its_unit(self):
        assert eda.MAG1C_COLOR_MAX == 2000
        assert eda.MAG1C_LABEL == "mag1c (ppm·m)"

    def test_no_figure_string_is_left_in_english(self):
        texts = [
            *eda.PARTITION_LABELS.values(),
            *(label for _, label in eda.BALANCE_PANELS),
            *eda.EXAMPLE_ROW_LABELS,
            eda.MAG1C_LABEL,
            example_column_title("mini", True, 46),
        ]
        english = ("patch", "Patches", "composite", "ground truth", "train", "positive", "negative")
        assert not [text for text in texts if any(word in text for word in english)]

    def test_scale_colours_reuse_the_tier_palette_in_scale_order(self):
        assert list(eda.SCALE_COLORS) == ["mini", "R2", "R3"]
        assert list(eda.SCALE_COLORS.values()) == [
            TIER_COLORS["mini"],
            TIER_COLORS["r2"],
            TIER_COLORS["raw-full"],
        ]
