from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from kornia.augmentation import AugmentationSequential
from omegaconf import OmegaConf

from data import patch_cache
from data.dataset import PatchDataset, _load_dataset_config


def _make_scene(tiny_geotiff_factory, tmp_path, name="scene", size=8):
    """Write a tiny 4-input-band + label scene, all-zero except where overridden."""
    folder = tmp_path / name
    bands = {
        "mag1c": np.zeros((size, size), dtype="float32"),
        "TOA_AVIRIS_640nm": np.zeros((size, size), dtype="float32"),
        "TOA_AVIRIS_550nm": np.zeros((size, size), dtype="float32"),
        "TOA_AVIRIS_460nm": np.zeros((size, size), dtype="float32"),
        "labelbinary": np.zeros((size, size), dtype="float32"),
    }
    return folder, bands


def _write_scene(tiny_geotiff_factory, folder, bands):
    for band_name, array in bands.items():
        tiny_geotiff_factory(folder / f"{band_name}.tif", array)


def _patch_row(folder: Path, size: int = 8) -> dict:
    return {
        "id": "patch_0",
        "name": "scene",
        "folder": str(folder),
        "window_col_off": 0,
        "window_row_off": 0,
        "window_width": size,
        "window_height": size,
        "has_plume": True,
    }


class TestPatchDatasetShapesAndDtypes:
    def test_input_and_output_have_the_documented_shapes(self, tmp_path, tiny_geotiff_factory):
        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder)])

        item = PatchDataset(df, dataset="starcop_mini", augment=False)[0]

        assert item["input"].shape == (4, 8, 8)
        assert item["input"].dtype == torch.float32
        assert item["output"].shape == (1, 8, 8)

    def test_reads_only_the_declared_window_not_the_whole_file(
        self, tmp_path, tiny_geotiff_factory
    ):

        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path, size=16)
        bands["mag1c"][12, 12] = 1e9
        _write_scene(tiny_geotiff_factory, folder, bands)
        row = _patch_row(folder, size=16)
        row["window_width"] = 8
        row["window_height"] = 8
        df = pd.DataFrame([row])

        item = PatchDataset(df, dataset="starcop_mini", augment=False)[0]

        assert item["input"].shape == (4, 8, 8)
        assert item["input"][0].max().item() < 1.0


class TestLoadDatasetConfig:
    def test_reads_from_the_exact_lowercase_configs_dataset_path(self, monkeypatch):

        captured = {}

        def fake_config_path(dataset):
            path = Path("configs") / "dataset" / f"{dataset}.yaml"
            captured["path"] = path
            return path

        monkeypatch.setattr("data.dataset._dataset_config_path", fake_config_path)

        def fake_load(path):
            return OmegaConf.create({"dataset_cfg": {"input_products": [], "output_products": []}})

        monkeypatch.setattr("data.dataset.OmegaConf.load", fake_load)

        assert _load_dataset_config("starcop_mini") == ([], [])
        assert captured["path"].parts[-3:] == ("configs", "dataset", "starcop_mini.yaml")


class TestPatchDatasetInit:
    def test_patches_df_index_is_reset(self, tmp_path, tiny_geotiff_factory):
        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder)], index=[7])

        dataset = PatchDataset(df, dataset="starcop_mini", augment=False)

        assert list(dataset.patches_df.index) == [0]
        assert "index" not in dataset.patches_df.columns


class TestPatchDatasetBandOrder:
    def test_channel_order_matches_the_dataset_config(self, tmp_path, tiny_geotiff_factory):

        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        bands["mag1c"][:] = 175.0
        bands["TOA_AVIRIS_640nm"][:] = 12.0
        bands["TOA_AVIRIS_550nm"][:] = 18.0
        bands["TOA_AVIRIS_460nm"][:] = 24.0
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder)])

        item = PatchDataset(df, dataset="starcop_mini", augment=False)[0]

        from conftest import _find_repo_root_containing_configs

        repo_root = _find_repo_root_containing_configs(Path(__file__).resolve().parent)
        expected_config = OmegaConf.load(repo_root / "configs" / "dataset" / "starcop_mini.yaml")
        input_products = list(expected_config.dataset_cfg.input_products)
        expected_means = {
            "mag1c": 0.1,
            "TOA_AVIRIS_640nm": 0.2,
            "TOA_AVIRIS_550nm": 0.3,
            "TOA_AVIRIS_460nm": 0.4,
        }
        for channel_idx, product in enumerate(input_products):
            np.testing.assert_allclose(
                item["input"][channel_idx].mean().item(), expected_means[product], atol=1e-5
            )


class TestPatchDatasetLabelPassthrough:
    def test_label_is_returned_unnormalized(self, tmp_path, tiny_geotiff_factory):
        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        bands["labelbinary"][0, 0] = 1.0
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder)])

        item = PatchDataset(df, dataset="starcop_mini", augment=False)[0]

        assert set(item["output"].unique().tolist()) <= {0.0, 1.0}
        assert item["output"][0, 0, 0].item() == 1.0


class TestPatchDatasetCaching:
    def test_reads_the_same_index_from_disk_only_once(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):

        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder)])
        dataset = PatchDataset(df, dataset="starcop_mini", augment=False)

        from data import dataset as dataset_module

        real_read = dataset_module.read_patch_bands
        calls = []

        def _spy(*args, **kwargs):
            calls.append(args)
            return real_read(*args, **kwargs)

        monkeypatch.setattr(dataset_module, "read_patch_bands", _spy)

        dataset[0]
        dataset[0]
        dataset[0]

        assert len(calls) == 1

    def test_different_indices_are_each_read_from_disk_once(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):
        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder), _patch_row(folder)])
        dataset = PatchDataset(df, dataset="starcop_mini", augment=False)

        from data import dataset as dataset_module

        real_read = dataset_module.read_patch_bands
        calls = []

        def _spy(*args, **kwargs):
            calls.append(args)
            return real_read(*args, **kwargs)

        monkeypatch.setattr(dataset_module, "read_patch_bands", _spy)

        dataset[0]
        dataset[1]
        dataset[0]
        dataset[1]

        assert len(calls) == 2

    def test_mutating_a_returned_tensor_does_not_leak_into_the_next_access(
        self, tmp_path, tiny_geotiff_factory
    ):

        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder)])
        dataset = PatchDataset(df, dataset="starcop_mini", augment=False)

        first = dataset[0]
        first["input"].fill_(999.0)
        first["output"].fill_(999.0)

        second = dataset[0]

        assert second["input"].max().item() != 999.0
        assert second["output"].max().item() != 999.0


class TestEntryBytes:
    def test_computes_the_exact_combined_byte_size_of_input_and_output(self):
        input_tensor = torch.zeros((4, 8, 8), dtype=torch.float32)
        output_tensor = torch.zeros((1, 8, 8), dtype=torch.float32)

        assert PatchDataset._entry_bytes(input_tensor, output_tensor) == 4 * 8 * 8 * 4 + 8 * 8 * 4

    def test_scales_with_input_channel_count_not_just_output(self):

        input_tensor = torch.zeros((2, 4, 4), dtype=torch.float32)
        output_tensor = torch.zeros((1, 4, 4), dtype=torch.float32)

        assert PatchDataset._entry_bytes(input_tensor, output_tensor) == 2 * 4 * 4 * 4 + 4 * 4 * 4


class TestPatchDatasetCacheEviction:
    _ONE_PATCH_BYTES = 1280

    def test_starts_with_an_empty_cache_and_zero_bytes_used(self, tmp_path, tiny_geotiff_factory):
        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder)])

        dataset = PatchDataset(df, dataset="starcop_mini", augment=False)

        assert dataset._cache == {}
        assert dataset._cache_bytes_used == 0

    def test_a_budget_exactly_equal_to_one_entry_still_caches_it(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):

        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder)])
        dataset = PatchDataset(
            df, dataset="starcop_mini", augment=False, max_cache_bytes=self._ONE_PATCH_BYTES
        )

        from data import dataset as dataset_module

        real_read = dataset_module.read_patch_bands
        calls = []

        def _spy(*args, **kwargs):
            calls.append(args)
            return real_read(*args, **kwargs)

        monkeypatch.setattr(dataset_module, "read_patch_bands", _spy)

        dataset[0]
        dataset[0]

        assert len(calls) == 1

    def test_a_budget_exactly_equal_to_two_entries_fits_both_without_evicting(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):

        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder), _patch_row(folder)])
        dataset = PatchDataset(
            df, dataset="starcop_mini", augment=False, max_cache_bytes=2 * self._ONE_PATCH_BYTES
        )

        from data import dataset as dataset_module

        real_read = dataset_module.read_patch_bands
        calls = []

        def _spy(*args, **kwargs):
            calls.append(args)
            return real_read(*args, **kwargs)

        monkeypatch.setattr(dataset_module, "read_patch_bands", _spy)

        dataset[0]
        dataset[1]
        dataset[0]
        dataset[1]

        assert len(calls) == 2

    def test_byte_accounting_stays_correct_across_repeated_evictions(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):

        small_folder, small_bands = _make_scene(tiny_geotiff_factory, tmp_path, name="small")
        _write_scene(tiny_geotiff_factory, small_folder, small_bands)
        big_folder, big_bands = _make_scene(tiny_geotiff_factory, tmp_path, name="big", size=16)
        _write_scene(tiny_geotiff_factory, big_folder, big_bands)

        medium_row = _patch_row(big_folder, size=16)
        medium_row["window_width"] = 10
        medium_row["window_height"] = 10
        df = pd.DataFrame(
            [_patch_row(small_folder), _patch_row(small_folder), _patch_row(small_folder)]
            + [medium_row]
        )

        dataset = PatchDataset(
            df, dataset="starcop_mini", augment=False, max_cache_bytes=3 * self._ONE_PATCH_BYTES
        )

        from data import dataset as dataset_module

        real_read = dataset_module.read_patch_bands
        calls = []

        def _spy(*args, **kwargs):
            calls.append(args)
            return real_read(*args, **kwargs)

        monkeypatch.setattr(dataset_module, "read_patch_bands", _spy)

        dataset[0]
        dataset[1]
        dataset[2]
        dataset[3]

        dataset[2]
        dataset[1]

        assert len(calls) == 5

    def test_exceeding_the_byte_budget_evicts_the_least_recently_used_entry(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):

        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder), _patch_row(folder)])
        dataset = PatchDataset(
            df, dataset="starcop_mini", augment=False, max_cache_bytes=self._ONE_PATCH_BYTES
        )

        from data import dataset as dataset_module

        real_read = dataset_module.read_patch_bands
        calls = []

        def _spy(*args, **kwargs):
            calls.append(args)
            return real_read(*args, **kwargs)

        monkeypatch.setattr(dataset_module, "read_patch_bands", _spy)

        dataset[0]
        dataset[1]
        dataset[0]

        assert len(calls) == 3

    def test_a_budget_that_fits_every_entry_never_evicts(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):
        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder), _patch_row(folder)])
        dataset = PatchDataset(
            df, dataset="starcop_mini", augment=False, max_cache_bytes=10 * self._ONE_PATCH_BYTES
        )

        from data import dataset as dataset_module

        real_read = dataset_module.read_patch_bands
        calls = []

        def _spy(*args, **kwargs):
            calls.append(args)
            return real_read(*args, **kwargs)

        monkeypatch.setattr(dataset_module, "read_patch_bands", _spy)

        dataset[0]
        dataset[1]
        dataset[0]
        dataset[1]

        assert len(calls) == 2

    def test_an_entry_larger_than_the_whole_budget_is_never_cached_but_still_returned(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):
        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder)])
        dataset = PatchDataset(df, dataset="starcop_mini", augment=False, max_cache_bytes=1)

        from data import dataset as dataset_module

        real_read = dataset_module.read_patch_bands
        calls = []

        def _spy(*args, **kwargs):
            calls.append(args)
            return real_read(*args, **kwargs)

        monkeypatch.setattr(dataset_module, "read_patch_bands", _spy)

        item = dataset[0]
        dataset[0]

        assert item["input"].shape == (4, 8, 8)
        assert len(calls) == 2

    def test_default_budget_keeps_starcop_minis_whole_dataset_cached(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):

        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder), _patch_row(folder)])
        dataset = PatchDataset(df, dataset="starcop_mini", augment=False)

        from data import dataset as dataset_module

        real_read = dataset_module.read_patch_bands
        calls = []

        def _spy(*args, **kwargs):
            calls.append(args)
            return real_read(*args, **kwargs)

        monkeypatch.setattr(dataset_module, "read_patch_bands", _spy)

        dataset[0]
        dataset[1]
        dataset[0]
        dataset[1]

        assert len(calls) == 2


class TestPatchDatasetAugmentation:
    def test_val_test_construction_has_no_augmenter(self, tmp_path, tiny_geotiff_factory):
        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder)])

        dataset = PatchDataset(df, dataset="starcop_mini", augment=False)

        assert dataset.augmenter is None

    def test_train_construction_has_a_kornia_augmenter(self, tmp_path, tiny_geotiff_factory):
        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder)])

        dataset = PatchDataset(df, dataset="starcop_mini", augment=True)

        assert isinstance(dataset.augmenter, AugmentationSequential)

    def test_augmentation_moves_the_image_marker_with_the_mask(
        self, tmp_path, tiny_geotiff_factory
    ):

        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path, size=8)
        bands["mag1c"][2, 5] = 1e9
        bands["labelbinary"][2, 5] = 1.0
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder, size=8)])
        dataset = PatchDataset(df, dataset="starcop_mini", augment=True)

        torch.manual_seed(0)
        for _ in range(20):
            item = dataset[0]
            mask = item["output"][0]
            assert mask.sum().item() == 1.0
            marker_row, marker_col = (mask == 1).nonzero(as_tuple=True)
            mag1c_channel = item["input"][0]
            assert mag1c_channel[marker_row, marker_col].item() == pytest.approx(2.0)

    def test_augmenter_includes_90_degree_rotation_not_only_flips(
        self, tmp_path, tiny_geotiff_factory
    ):

        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path, size=8)
        bands["mag1c"][1, 5] = 1e9
        bands["labelbinary"][1, 5] = 1.0
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder, size=8)])
        dataset = PatchDataset(df, dataset="starcop_mini", augment=True)

        torch.manual_seed(0)
        positions = set()
        for _ in range(200):
            item = dataset[0]
            row, col = (item["output"][0] == 1).nonzero(as_tuple=True)
            positions.add((row.item(), col.item()))

        assert len(positions) > 4


class TestPatchDatasetOnDiskCache:
    def test_reads_from_the_cache_without_touching_disk(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):

        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder), _patch_row(folder)])
        inputs = np.stack([np.full((4, 8, 8), 0.25), np.full((4, 8, 8), 0.75)]).astype(np.float16)
        outputs = np.stack([np.zeros((1, 8, 8)), np.ones((1, 8, 8))]).astype(np.uint8)
        cache_dir = tmp_path / "cache"
        patch_cache.write(cache_dir, df, inputs, outputs)

        from data import dataset as dataset_module

        calls = []
        monkeypatch.setattr(
            dataset_module, "read_patch_bands", lambda *a, **k: calls.append(a) or {}
        )

        dataset = PatchDataset(df, dataset="starcop_mini", augment=False, cache_dir=cache_dir)
        first = dataset[0]
        second = dataset[1]

        assert calls == []
        assert first["input"].shape == (4, 8, 8)
        assert torch.allclose(first["input"], torch.full((4, 8, 8), 0.25))
        assert torch.allclose(second["output"], torch.ones(1, 8, 8))

    def test_raises_when_the_cache_does_not_match_the_given_dataframe(
        self, tmp_path, tiny_geotiff_factory
    ):

        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        _write_scene(tiny_geotiff_factory, folder, bands)
        built_for = pd.DataFrame([_patch_row(folder)])
        inputs = np.zeros((1, 4, 8, 8), dtype=np.float16)
        outputs = np.zeros((1, 1, 8, 8), dtype=np.uint8)
        cache_dir = tmp_path / "cache"
        patch_cache.write(cache_dir, built_for, inputs, outputs)

        mismatched = pd.DataFrame([_patch_row(folder), _patch_row(folder)])

        with pytest.raises(ValueError, match="cache"):
            PatchDataset(mismatched, dataset="starcop_mini", augment=False, cache_dir=cache_dir)

    def test_cache_dir_none_keeps_using_live_reads(self, tmp_path, tiny_geotiff_factory):

        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path)
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder)])

        item = PatchDataset(df, dataset="starcop_mini", augment=False, cache_dir=None)[0]

        assert item["input"].shape == (4, 8, 8)

    def test_augmentation_still_applies_fresh_on_a_cache_hit(self, tmp_path, tiny_geotiff_factory):

        folder, bands = _make_scene(tiny_geotiff_factory, tmp_path, size=8)
        _write_scene(tiny_geotiff_factory, folder, bands)
        df = pd.DataFrame([_patch_row(folder, size=8)])
        inputs = np.zeros((1, 4, 8, 8), dtype=np.float16)
        outputs = np.zeros((1, 1, 8, 8), dtype=np.uint8)
        outputs[0, 0, 1, 5] = 1
        cache_dir = tmp_path / "cache"
        patch_cache.write(cache_dir, df, inputs, outputs)

        dataset = PatchDataset(df, dataset="starcop_mini", augment=True, cache_dir=cache_dir)

        torch.manual_seed(0)
        positions = set()
        for _ in range(200):
            item = dataset[0]
            row, col = (item["output"][0] == 1).nonzero(as_tuple=True)
            positions.add((row.item(), col.item()))

        assert len(positions) > 1


def _asymmetric_dataset(tiny_geotiff_factory, tmp_path, n_rows=1, size=8, **kwargs):
    """Augmenting dataset whose scene has no symmetry, so every flip/rotation is visible."""
    folder, bands = _make_scene(tiny_geotiff_factory, tmp_path, size=size)
    bands["mag1c"] = np.arange(size * size, dtype="float32").reshape(size, size)
    bands["labelbinary"][1, 5] = 1.0
    _write_scene(tiny_geotiff_factory, folder, bands)
    df = pd.DataFrame([_patch_row(folder, size=size) for _ in range(n_rows)])
    return PatchDataset(df, dataset="starcop_mini", augment=True, **kwargs)


def _fingerprint(item) -> tuple:
    return (item["input"].flatten().tolist(), item["output"].flatten().tolist())


class TestPatchDatasetDeterministicAugmentation:
    def test_the_same_seed_epoch_and_index_give_the_same_augmentation_in_any_instance(
        self, tmp_path, tiny_geotiff_factory
    ):
        first = _asymmetric_dataset(tiny_geotiff_factory, tmp_path, augment_seed=42)
        second = _asymmetric_dataset(tiny_geotiff_factory, tmp_path, augment_seed=42)
        first.set_epoch(3)
        second.set_epoch(3)

        torch.manual_seed(1)
        expected = _fingerprint(first[0])
        torch.manual_seed(999)

        assert _fingerprint(second[0]) == expected

    def test_repeated_reads_of_one_index_in_one_epoch_are_identical(
        self, tmp_path, tiny_geotiff_factory
    ):
        dataset = _asymmetric_dataset(tiny_geotiff_factory, tmp_path, augment_seed=42)

        assert _fingerprint(dataset[0]) == _fingerprint(dataset[0])

    def test_the_augmentation_changes_from_one_epoch_to_the_next(
        self, tmp_path, tiny_geotiff_factory
    ):
        dataset = _asymmetric_dataset(tiny_geotiff_factory, tmp_path, augment_seed=42)

        seen = set()
        for epoch in range(20):
            dataset.set_epoch(epoch)
            seen.add(str(_fingerprint(dataset[0])))

        assert len(seen) > 1

    def test_different_indices_in_one_epoch_draw_independently(
        self, tmp_path, tiny_geotiff_factory
    ):
        dataset = _asymmetric_dataset(tiny_geotiff_factory, tmp_path, n_rows=20, augment_seed=42)

        seen = {str(_fingerprint(dataset[index])) for index in range(20)}

        assert len(seen) > 1

    def test_the_seed_changes_the_augmentation(self, tmp_path, tiny_geotiff_factory):
        seen = set()
        for seed in range(20):
            dataset = _asymmetric_dataset(tiny_geotiff_factory, tmp_path, augment_seed=seed)
            seen.add(str(_fingerprint(dataset[0])))

        assert len(seen) > 1

    def test_reading_an_item_leaves_the_global_torch_rng_untouched(
        self, tmp_path, tiny_geotiff_factory
    ):
        dataset = _asymmetric_dataset(tiny_geotiff_factory, tmp_path, augment_seed=42)

        torch.manual_seed(5)
        untouched = torch.rand(3)
        torch.manual_seed(5)
        dataset[0]

        assert torch.equal(torch.rand(3), untouched)

    def test_set_epoch_reaches_persistent_dataloader_workers(self, tmp_path, tiny_geotiff_factory):

        from torch.utils.data import DataLoader

        worker_side = _asymmetric_dataset(tiny_geotiff_factory, tmp_path, augment_seed=42)
        in_process = _asymmetric_dataset(tiny_geotiff_factory, tmp_path, augment_seed=42)
        loader = DataLoader(worker_side, batch_size=1, num_workers=2, persistent_workers=True)
        list(loader)

        worker_side.set_epoch(3)
        in_process.set_epoch(3)
        (batch,) = list(loader)

        assert torch.equal(batch["input"][0], in_process[0]["input"])
        assert torch.equal(batch["output"][0], in_process[0]["output"])

    def test_without_an_augment_seed_the_global_rng_still_drives_augmentation(
        self, tmp_path, tiny_geotiff_factory
    ):

        dataset = _asymmetric_dataset(tiny_geotiff_factory, tmp_path)

        torch.manual_seed(7)
        first = _fingerprint(dataset[0])
        torch.manual_seed(7)
        second = _fingerprint(dataset[0])

        assert first == second
        seen = set()
        for _ in range(20):
            seen.add(str(_fingerprint(dataset[0])))
        assert len(seen) > 1
