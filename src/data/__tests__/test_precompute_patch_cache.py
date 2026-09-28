import numpy as np
import pandas as pd
import torch

from data import patch_cache
from data.precompute_patch_cache import build_cache_for_split


def _make_scene(tiny_geotiff_factory, tmp_path, name="scene", size=8, mag1c_value=0.5):
    folder = tmp_path / name
    bands = {
        "mag1c": np.full((size, size), mag1c_value, dtype="float32"),
        "TOA_AVIRIS_640nm": np.full((size, size), 0.25, dtype="float32"),
        "TOA_AVIRIS_550nm": np.full((size, size), 0.25, dtype="float32"),
        "TOA_AVIRIS_460nm": np.full((size, size), 0.25, dtype="float32"),
        "labelbinary": np.zeros((size, size), dtype="float32"),
    }
    bands["labelbinary"][0, 0] = 1.0
    for band_name, array in bands.items():
        tiny_geotiff_factory(folder / f"{band_name}.tif", array)
    return folder


def _patches_df(folder, n_rows=3, size=8) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": f"patch_{i}",
                "name": "scene",
                "folder": str(folder),
                "window_col_off": 0,
                "window_row_off": 0,
                "window_width": size,
                "window_height": size,
                "has_plume": True,
            }
            for i in range(n_rows)
        ]
    )


class TestBuildCacheForSplit:
    def test_writes_a_cache_valid_for_the_given_dataframe(self, tmp_path, tiny_geotiff_factory):
        folder = _make_scene(tiny_geotiff_factory, tmp_path)
        df = _patches_df(folder, n_rows=3)
        cache_dir = tmp_path / "cache_out"

        build_cache_for_split("starcop_mini", df, cache_dir, num_workers=0)

        assert patch_cache.is_valid(cache_dir, df) is True

    def test_cached_values_match_a_live_read_of_the_same_patch(
        self, tmp_path, tiny_geotiff_factory
    ):
        from data.dataset import PatchDataset

        folder = _make_scene(tiny_geotiff_factory, tmp_path)
        df = _patches_df(folder, n_rows=2)
        cache_dir = tmp_path / "cache_out"

        build_cache_for_split("starcop_mini", df, cache_dir, num_workers=0)

        live_item = PatchDataset(df, dataset="starcop_mini", augment=False)[1]
        cached_item = PatchDataset(df, dataset="starcop_mini", augment=False, cache_dir=cache_dir)[
            1
        ]

        assert torch.allclose(cached_item["input"], live_item["input"], atol=1e-3)
        torch.testing.assert_close(cached_item["output"], live_item["output"])

    def test_works_with_parallel_workers_too(self, tmp_path, tiny_geotiff_factory):
        folder = _make_scene(tiny_geotiff_factory, tmp_path)
        df = _patches_df(folder, n_rows=5)
        cache_dir = tmp_path / "cache_out"

        build_cache_for_split("starcop_mini", df, cache_dir, num_workers=2, batch_size=2)

        assert patch_cache.is_valid(cache_dir, df) is True
        inputs, outputs = patch_cache.load(cache_dir)
        assert inputs.shape == (5, 4, 8, 8)
        assert outputs.shape == (5, 1, 8, 8)

    def test_preserves_per_row_content_and_order_across_multiple_batches(
        self, tmp_path, tiny_geotiff_factory
    ):

        from data.dataset import PatchDataset

        folders = [
            _make_scene(tiny_geotiff_factory, tmp_path, name=f"scene{i}", mag1c_value=i * 0.1)
            for i in range(5)
        ]
        df = pd.DataFrame(
            [
                {
                    "id": f"patch_{i}",
                    "name": "scene",
                    "folder": str(folder),
                    "window_col_off": 0,
                    "window_row_off": 0,
                    "window_width": 8,
                    "window_height": 8,
                    "has_plume": True,
                }
                for i, folder in enumerate(folders)
            ]
        )
        cache_dir = tmp_path / "cache_out"

        build_cache_for_split("starcop_mini", df, cache_dir, num_workers=0, batch_size=2)

        for i in range(len(folders)):
            live_item = PatchDataset(df, dataset="starcop_mini", augment=False)[i]
            cached_item = PatchDataset(
                df, dataset="starcop_mini", augment=False, cache_dir=cache_dir
            )[i]
            assert torch.allclose(cached_item["input"], live_item["input"], atol=1e-3), (
                f"row {i} does not match its own patch's live read"
            )


class TestBuildCacheForSplitDefaults:
    def test_uses_documented_defaults_for_workers_batch_size_augment_and_ram_cache(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):

        import data.precompute_patch_cache as ppc

        folder = _make_scene(tiny_geotiff_factory, tmp_path)
        df = _patches_df(folder, n_rows=2)
        cache_dir = tmp_path / "cache_out"

        dataset_kwargs = {}
        real_patch_dataset = ppc.PatchDataset

        def spy_patch_dataset(*args, **kwargs):
            dataset_kwargs.update(kwargs)
            return real_patch_dataset(*args, **kwargs)

        loader_kwargs = {}
        real_data_loader = ppc.DataLoader

        def spy_data_loader(*args, **kwargs):
            loader_kwargs.update(kwargs)
            return real_data_loader(*args, **kwargs)

        write_call = {}
        real_write = ppc.patch_cache.write

        def spy_write(cache_dir, patches_df, inputs, outputs):
            write_call["inputs_dtype"] = inputs.dtype
            write_call["outputs_dtype"] = outputs.dtype
            return real_write(cache_dir, patches_df, inputs, outputs)

        monkeypatch.setattr(ppc, "PatchDataset", spy_patch_dataset)
        monkeypatch.setattr(ppc, "DataLoader", spy_data_loader)
        monkeypatch.setattr(ppc.patch_cache, "write", spy_write)

        build_cache_for_split("starcop_mini", df, cache_dir)

        assert dataset_kwargs == {
            "dataset": "starcop_mini",
            "augment": False,
            "max_cache_bytes": 0,
        }
        assert loader_kwargs == {"batch_size": 256, "shuffle": False, "num_workers": 4}
        assert write_call["inputs_dtype"] == np.float16
        assert write_call["outputs_dtype"] == np.uint8
