import numpy as np
import pandas as pd
import pytest
import torch

import evaluation.scene_inference as scene_inference
from data.precompute_patch_cache import build_cache_for_split
from evaluation.scene_inference import (
    SceneCounts,
    _loader,
    load_scene_counts,
    save_scene_counts,
    score_scenes,
)

THRESHOLDS = [0.25, 0.5, 0.75]


class _FixedLogits(torch.nn.Module):
    """Ignores its input and returns the same (H, W) logits for every item."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.logits = logits

    def forward(self, x):
        return self.logits.expand(x.shape[0], 1, *self.logits.shape)


class _PerPixel(torch.nn.Module):
    """Context-free: each output pixel depends only on that pixel's mag1c value."""

    def forward(self, x):
        return x[:, :1] * 3.0 - 1.0


class _RecordsTrainingFlag(_FixedLogits):
    def __init__(self, logits):
        super().__init__(logits)
        self.seen_training_flags = []

    def forward(self, x):
        self.seen_training_flags.append(self.training)
        return super().forward(x)


def _logit(probability):
    return float(np.log(probability / (1 - probability)))


def _write_scene(tiny_geotiff_factory, folder, label, mag1c=None):
    """One scene folder with the four input bands and a `labelbinary` raster."""
    size = label.shape[0]
    rng = np.random.default_rng(size)
    mag = mag1c if mag1c is not None else rng.uniform(0, 500, (size, size))
    bands = {
        "mag1c": mag.astype("float32"),
        "TOA_AVIRIS_640nm": rng.uniform(0, 40, (size, size)).astype("float32"),
        "TOA_AVIRIS_550nm": rng.uniform(0, 40, (size, size)).astype("float32"),
        "TOA_AVIRIS_460nm": rng.uniform(0, 40, (size, size)).astype("float32"),
        "labelbinary": label.astype("float32"),
    }
    for name, array in bands.items():
        tiny_geotiff_factory(folder / f"{name}.tif", array)
    return folder


def _scene_df(folders, size):
    return pd.DataFrame(
        [
            {
                "scene_id": folder.name,
                "folder": str(folder),
                "window_col_off": 0,
                "window_row_off": 0,
                "window_width": size,
                "window_height": size,
            }
            for folder in folders
        ]
    )


def _tiles_df(folders, size, tile):
    """Non-overlapping `tile` x `tile` patch windows for every scene folder."""
    rows = []
    for folder in folders:
        for row_off in range(0, size, tile):
            for col_off in range(0, size, tile):
                rows.append(
                    {
                        "id": f"{folder.name}_{row_off}_{col_off}",
                        "id_original": folder.name,
                        "folder": str(folder),
                        "window_col_off": col_off,
                        "window_row_off": row_off,
                        "window_width": tile,
                        "window_height": tile,
                    }
                )
    return pd.DataFrame(rows)


def _oracle_counts(probs, label, thresholds):
    """Independent numpy count of [tp, fp, fn, tn] per threshold."""
    rows = []
    for threshold in thresholds:
        predicted = probs > threshold
        actual = label > 0
        rows.append(
            [
                int((predicted & actual).sum()),
                int((predicted & ~actual).sum()),
                int((~predicted & actual).sum()),
                int((~predicted & ~actual).sum()),
            ]
        )
    return rows


@pytest.fixture
def two_scenes(tmp_path, tiny_geotiff_factory):
    """Two 4x4 scenes with different labels."""
    label_a = np.zeros((4, 4))
    label_a[0, 0] = label_a[0, 1] = label_a[3, 3] = 1
    label_b = np.zeros((4, 4))
    label_b[2, :] = 1
    folders = [
        _write_scene(tiny_geotiff_factory, tmp_path / "scene_a", label_a),
        _write_scene(tiny_geotiff_factory, tmp_path / "scene_b", label_b),
    ]
    return folders, [label_a, label_b], _scene_df(folders, 4)


PROBABILITIES = np.array(
    [
        [0.9, 0.6, 0.5, 0.1],
        [0.3, 0.76, 0.24, 0.8],
        [0.51, 0.49, 0.9, 0.05],
        [0.2, 0.7, 0.4, 0.95],
    ]
)
LOGITS = torch.tensor(
    [[_logit(p) if p != 0.5 else 0.0 for p in row] for row in PROBABILITIES], dtype=torch.float32
)


class TestScoreScenesFullScene:
    def test_counts_match_an_independent_count_for_every_scene_and_threshold(self, two_scenes):
        _, labels, scene_df = two_scenes

        result = score_scenes(
            _FixedLogits(LOGITS),
            scene_df,
            mode="full_scene",
            thresholds=THRESHOLDS,
            dataset="starcop_mini",
        )

        assert result.counts.shape == (2, 3, 4)
        for scene_index, label in enumerate(labels):
            expected = _oracle_counts(PROBABILITIES, label, THRESHOLDS)
            assert result.counts[scene_index].tolist() == expected

    def test_a_hand_worked_case(self, two_scenes):

        _, _, scene_df = two_scenes

        result = score_scenes(
            _FixedLogits(LOGITS),
            scene_df,
            mode="full_scene",
            thresholds=[0.5],
            dataset="starcop_mini",
        )

        assert result.counts[0, 0].tolist() == [3, 5, 0, 8]

    def test_a_probability_exactly_at_the_threshold_is_not_positive(self, two_scenes):

        _, _, scene_df = two_scenes
        at_threshold = torch.zeros(4, 4)

        result = score_scenes(
            _FixedLogits(at_threshold),
            scene_df,
            mode="full_scene",
            thresholds=[0.5],
            dataset="starcop_mini",
        )

        assert result.counts[:, 0, 0].tolist() == [0, 0]
        assert result.counts[:, 0, 1].tolist() == [0, 0]

    def test_every_scene_accounts_for_all_of_its_pixels(self, two_scenes):
        _, _, scene_df = two_scenes

        result = score_scenes(
            _FixedLogits(LOGITS),
            scene_df,
            mode="full_scene",
            thresholds=THRESHOLDS,
            dataset="starcop_mini",
        )

        assert (result.counts.sum(axis=2) == 16).all()

    def test_scene_ids_and_thresholds_are_returned_in_input_order(self, two_scenes):
        _, _, scene_df = two_scenes

        result = score_scenes(
            _FixedLogits(LOGITS),
            scene_df.iloc[::-1].reset_index(drop=True),
            mode="full_scene",
            thresholds=THRESHOLDS,
            dataset="starcop_mini",
        )

        assert result.scene_ids == ["scene_b", "scene_a"]
        assert result.thresholds.tolist() == THRESHOLDS
        assert result.mode == "full_scene"

    def test_the_batch_size_does_not_change_the_counts(self, two_scenes):
        _, _, scene_df = two_scenes
        kwargs = {"mode": "full_scene", "thresholds": THRESHOLDS, "dataset": "starcop_mini"}

        one = score_scenes(_FixedLogits(LOGITS), scene_df, batch_size=1, **kwargs)
        two = score_scenes(_FixedLogits(LOGITS), scene_df, batch_size=2, **kwargs)

        assert np.array_equal(one.counts, two.counts)

    def test_counts_are_exact_integers(self, two_scenes):
        _, _, scene_df = two_scenes

        result = score_scenes(
            _FixedLogits(LOGITS),
            scene_df,
            mode="full_scene",
            thresholds=THRESHOLDS,
            dataset="starcop_mini",
        )

        assert result.counts.dtype == np.int64

    def test_thresholds_may_be_a_list_an_array_or_a_tensor(self, two_scenes):
        _, _, scene_df = two_scenes
        kwargs = {"mode": "full_scene", "dataset": "starcop_mini"}

        from_list = score_scenes(_FixedLogits(LOGITS), scene_df, thresholds=THRESHOLDS, **kwargs)
        from_array = score_scenes(
            _FixedLogits(LOGITS), scene_df, thresholds=np.array(THRESHOLDS), **kwargs
        )
        from_tensor = score_scenes(
            _FixedLogits(LOGITS), scene_df, thresholds=torch.tensor(THRESHOLDS), **kwargs
        )

        assert np.array_equal(from_list.counts, from_array.counts)
        assert np.array_equal(from_list.counts, from_tensor.counts)

    def test_the_model_is_put_in_eval_mode(self, two_scenes):
        _, _, scene_df = two_scenes
        model = _RecordsTrainingFlag(LOGITS)
        model.train()

        score_scenes(
            model, scene_df, mode="full_scene", thresholds=THRESHOLDS, dataset="starcop_mini"
        )

        assert model.seen_training_flags == [False]

    def test_moves_thresholds_inputs_and_targets_to_the_requested_device(
        self, two_scenes, monkeypatch
    ):
        _, _, scene_df = two_scenes
        calls = []
        original_to = torch.Tensor.to

        def recording_to(tensor, *args, **kwargs):
            calls.append(args[0] if args else kwargs.get("device"))
            return original_to(tensor, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "to", recording_to)

        score_scenes(
            _FixedLogits(LOGITS),
            scene_df,
            mode="full_scene",
            thresholds=THRESHOLDS,
            dataset="starcop_mini",
            device="cpu",
        )

        assert calls.count("cpu") == 3
        assert None not in calls

    def test_an_unknown_mode_is_rejected(self, two_scenes):
        _, _, scene_df = two_scenes

        with pytest.raises(ValueError, match="mode"):
            score_scenes(
                _FixedLogits(LOGITS),
                scene_df,
                mode="stitched",
                thresholds=THRESHOLDS,
                dataset="starcop_mini",
            )

    def test_an_empty_threshold_list_is_rejected(self, two_scenes):
        _, _, scene_df = two_scenes

        with pytest.raises(ValueError) as error:
            score_scenes(
                _FixedLogits(LOGITS),
                scene_df,
                mode="full_scene",
                thresholds=[],
                dataset="starcop_mini",
            )

        assert str(error.value) == "at least one threshold is required"


class TestScoreScenesPatchesMode:
    def test_a_context_free_model_gives_the_same_counts_as_full_scene_on_non_overlapping_tiles(
        self, tmp_path, tiny_geotiff_factory
    ):
        rng = np.random.default_rng(3)
        labels = [(rng.random((16, 16)) < 0.2).astype(float) for _ in range(2)]
        folders = [
            _write_scene(tiny_geotiff_factory, tmp_path / f"s{i}", label)
            for i, label in enumerate(labels)
        ]
        scene_df = _scene_df(folders, 16)
        kwargs = {"thresholds": THRESHOLDS, "dataset": "starcop_mini"}

        full = score_scenes(_PerPixel(), scene_df, mode="full_scene", **kwargs)
        patches = score_scenes(
            _PerPixel(),
            scene_df,
            mode="patches",
            patches_df=_tiles_df(folders, 16, 8),
            **kwargs,
        )

        assert np.array_equal(full.counts, patches.counts)
        assert patches.mode == "patches"

    def test_overlapping_patches_count_a_pixel_once_per_patch_that_contains_it(
        self, tmp_path, tiny_geotiff_factory
    ):
        label = np.zeros((16, 16))
        folder = _write_scene(tiny_geotiff_factory, tmp_path / "s0", label)
        scene_df = _scene_df([folder], 16)
        rows = [
            {
                "id": f"p{k}",
                "id_original": folder.name,
                "folder": str(folder),
                "window_col_off": col_off,
                "window_row_off": 0,
                "window_width": 8,
                "window_height": 16,
            }
            for k, col_off in enumerate((0, 4, 8))
        ]

        result = score_scenes(
            _PerPixel(),
            scene_df,
            mode="patches",
            thresholds=THRESHOLDS,
            patches_df=pd.DataFrame(rows),
            dataset="starcop_mini",
        )

        assert result.counts.sum(axis=2)[0].tolist() == [3 * 8 * 16] * 3

    def test_only_the_requested_scenes_are_scored_and_in_the_requested_order(
        self, tmp_path, tiny_geotiff_factory
    ):
        folders = [
            _write_scene(tiny_geotiff_factory, tmp_path / name, np.zeros((16, 16)))
            for name in ("s0", "s1", "s2")
        ]
        patches_df = _tiles_df(folders, 16, 8)

        result = score_scenes(
            _PerPixel(),
            _scene_df([folders[2], folders[0]], 16),
            mode="patches",
            thresholds=THRESHOLDS,
            patches_df=patches_df,
            dataset="starcop_mini",
        )

        assert result.scene_ids == ["s2", "s0"]
        assert result.counts.shape == (2, 3, 4)

    def test_a_scene_without_patches_is_rejected(self, tmp_path, tiny_geotiff_factory):
        folders = [
            _write_scene(tiny_geotiff_factory, tmp_path / name, np.zeros((16, 16)))
            for name in ("s0", "s1")
        ]

        with pytest.raises(ValueError, match="no patches.*s1"):
            score_scenes(
                _PerPixel(),
                _scene_df(folders, 16),
                mode="patches",
                thresholds=THRESHOLDS,
                patches_df=_tiles_df(folders[:1], 16, 8),
                dataset="starcop_mini",
            )

    def test_patches_mode_requires_a_patch_manifest(self, two_scenes):
        _, _, scene_df = two_scenes

        with pytest.raises(ValueError) as error:
            score_scenes(
                _PerPixel(), scene_df, mode="patches", thresholds=THRESHOLDS, dataset="starcop_mini"
            )

        assert str(error.value) == "mode='patches' needs patches_df, the patch manifest"

    def test_a_matching_on_disk_cache_is_used_instead_of_live_reads(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):
        folder = _write_scene(tiny_geotiff_factory, tmp_path / "s0", np.zeros((16, 16)))
        scene_df = _scene_df([folder], 16)
        patches_df = _tiles_df([folder], 16, 8)
        cache_dir = tmp_path / "cache"
        build_cache_for_split("starcop_mini", patches_df, cache_dir, num_workers=0)
        from data import dataset as dataset_module

        live_reads = []
        monkeypatch.setattr(
            dataset_module, "read_patch_bands", lambda *a, **k: live_reads.append(a) or {}
        )

        result = score_scenes(
            _PerPixel(),
            scene_df,
            mode="patches",
            thresholds=THRESHOLDS,
            patches_df=patches_df,
            dataset="starcop_mini",
            cache_dir=cache_dir,
        )

        assert live_reads == []
        assert result.counts.sum(axis=2)[0].tolist() == [16 * 16] * 3


def _record_loaders(monkeypatch):
    real_loader = scene_inference.DataLoader
    captured = []

    def _recording(dataset, **kwargs):
        captured.append((dataset, kwargs))
        return real_loader(dataset, **kwargs)

    monkeypatch.setattr(scene_inference, "DataLoader", _recording)
    return captured


class TestLoaderConfiguration:
    def test_defaults_are_batches_of_four_in_the_main_process_without_shuffling(
        self, two_scenes, monkeypatch
    ):
        _, _, scene_df = two_scenes
        captured = _record_loaders(monkeypatch)

        score_scenes(
            _FixedLogits(LOGITS),
            scene_df,
            mode="full_scene",
            thresholds=THRESHOLDS,
            dataset="starcop_mini",
        )

        ((dataset, kwargs),) = captured
        assert kwargs["batch_size"] == 4
        assert kwargs["num_workers"] == 0
        assert kwargs["pin_memory"] is False
        assert not kwargs.get("shuffle", False)
        assert dataset.augmenter is None
        assert dataset._max_cache_bytes == 0

    def test_the_batch_size_and_worker_count_reach_the_loader_in_both_modes(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):
        folder = _write_scene(tiny_geotiff_factory, tmp_path / "s0", np.zeros((16, 16)))
        scene_df = _scene_df([folder], 16)
        patches_df = _tiles_df([folder], 16, 8)
        captured = _record_loaders(monkeypatch)

        for mode in ("full_scene", "patches"):
            score_scenes(
                _PerPixel(),
                scene_df,
                mode=mode,
                thresholds=THRESHOLDS,
                dataset="starcop_mini",
                patches_df=patches_df,
                batch_size=3,
                num_workers=2,
            )

        assert [(k["batch_size"], k["num_workers"]) for _, k in captured] == [(3, 2), (3, 2)]
        assert all(dataset.augmenter is None for dataset, _ in captured)

    def test_loader_disables_augmentation_and_cache(self, two_scenes, monkeypatch):
        _, _, scene_df = two_scenes
        captured = {}

        class Dataset:
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(scene_inference, "PatchDataset", Dataset)
        monkeypatch.setattr(
            scene_inference,
            "DataLoader",
            lambda dataset, **kwargs: (dataset, kwargs),
        )

        _loader(scene_df, "starcop_mini", "cpu", 2, 0)

        assert captured["augment"] is False
        assert captured["max_cache_bytes"] == 0

    def test_memory_is_pinned_only_for_a_cuda_device(self, two_scenes):
        _, _, scene_df = two_scenes

        assert _loader(scene_df, "starcop_mini", "cuda", 2, 0).pin_memory is True
        assert _loader(scene_df, "starcop_mini", "cpu", 2, 0).pin_memory is False

    def test_worker_tensors_travel_through_shared_files_not_file_descriptors(self, two_scenes):

        _, _, scene_df = two_scenes
        if "file_descriptor" not in torch.multiprocessing.get_all_sharing_strategies():
            pytest.skip("file_descriptor sharing is unavailable on this platform")
        previous = torch.multiprocessing.get_sharing_strategy()
        try:
            torch.multiprocessing.set_sharing_strategy("file_descriptor")

            _loader(scene_df, "starcop_mini", "cpu", 2, 2)

            assert torch.multiprocessing.get_sharing_strategy() == "file_system"
        finally:
            torch.multiprocessing.set_sharing_strategy(previous)

    def test_the_loader_does_not_reorder_the_windows(self, two_scenes):
        _, _, scene_df = two_scenes

        loader = _loader(scene_df, "starcop_mini", "cpu", 1, 0)

        assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)


class TestThresholdsAndMessages:
    def test_thresholds_are_stored_as_float32_whatever_their_input_type(self, two_scenes):
        _, _, scene_df = two_scenes

        result = score_scenes(
            _FixedLogits(LOGITS),
            scene_df,
            mode="full_scene",
            thresholds=np.array(THRESHOLDS, dtype=np.float64),
            dataset="starcop_mini",
        )

        assert result.thresholds.dtype == np.float32

    def test_the_missing_patches_error_names_at_most_five_scenes(
        self, tmp_path, tiny_geotiff_factory
    ):
        folders = [
            _write_scene(tiny_geotiff_factory, tmp_path / f"m{i}", np.zeros((16, 16)))
            for i in range(7)
        ]

        with pytest.raises(ValueError, match="no patches for scene") as error:
            score_scenes(
                _PerPixel(),
                _scene_df(folders, 16),
                mode="patches",
                thresholds=THRESHOLDS,
                patches_df=_tiles_df(folders[:1], 16, 8),
                dataset="starcop_mini",
            )

        assert "m5" in str(error.value)
        assert "m6" not in str(error.value)


class TestPatchesModeIsPositionAware:
    def test_patches_mode_moves_inputs_and_targets_to_the_requested_device(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):
        folder = _write_scene(tiny_geotiff_factory, tmp_path / "s0", np.zeros((16, 16)))
        scene_df = _scene_df([folder], 16)
        patches_df = _tiles_df([folder], 16, 8)
        calls = []
        dtypes = []
        original_to = torch.Tensor.to
        original_zeros = scene_inference.torch.zeros

        def recording_zeros(*args, **kwargs):
            dtypes.append(kwargs.get("dtype"))
            return original_zeros(*args, **kwargs)

        def recording_to(tensor, *args, **kwargs):
            calls.append(args[0] if args else kwargs.get("device"))
            return original_to(tensor, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "to", recording_to)
        monkeypatch.setattr(scene_inference.torch, "zeros", recording_zeros)

        score_scenes(
            _PerPixel(),
            scene_df,
            mode="patches",
            thresholds=THRESHOLDS,
            patches_df=patches_df,
            dataset="starcop_mini",
            device="cpu",
        )

        assert calls.count("cpu") == 3
        assert None not in calls
        assert dtypes[-1] == torch.float64

    def test_a_position_dependent_model_is_scored_on_the_unaugmented_tiles(
        self, tmp_path, tiny_geotiff_factory
    ):

        rng = np.random.default_rng(11)
        label = (rng.random((16, 16)) < 0.3).astype(float)
        folder = _write_scene(tiny_geotiff_factory, tmp_path / "s0", label)
        probabilities = rng.uniform(0.05, 0.95, (8, 8))
        logits = torch.tensor(np.log(probabilities / (1 - probabilities)), dtype=torch.float32)

        result = score_scenes(
            _FixedLogits(logits),
            _scene_df([folder], 16),
            mode="patches",
            thresholds=THRESHOLDS,
            patches_df=_tiles_df([folder], 16, 8),
            dataset="starcop_mini",
        )

        expected = np.zeros((3, 4), dtype=np.int64)
        for row_off in (0, 8):
            for col_off in (0, 8):
                tile = label[row_off : row_off + 8, col_off : col_off + 8]
                expected += np.array(_oracle_counts(probabilities, tile, THRESHOLDS))
        assert result.counts[0].tolist() == expected.tolist()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
class TestOnCuda:
    def test_full_scene_counts_on_the_gpu_equal_the_cpu_counts(self, two_scenes):
        _, _, scene_df = two_scenes
        kwargs = {"mode": "full_scene", "thresholds": THRESHOLDS, "dataset": "starcop_mini"}

        cpu = score_scenes(_FixedLogits(LOGITS), scene_df, **kwargs)
        gpu = score_scenes(_FixedLogits(LOGITS.cuda()), scene_df, device="cuda", **kwargs)

        assert np.array_equal(cpu.counts, gpu.counts)

    def test_patch_counts_on_the_gpu_equal_the_cpu_counts(self, tmp_path, tiny_geotiff_factory):
        rng = np.random.default_rng(5)
        label = (rng.random((16, 16)) < 0.2).astype(float)
        folder = _write_scene(tiny_geotiff_factory, tmp_path / "s0", label)
        kwargs = {
            "mode": "patches",
            "thresholds": THRESHOLDS,
            "dataset": "starcop_mini",
            "patches_df": _tiles_df([folder], 16, 8),
        }
        scene_df = _scene_df([folder], 16)

        cpu = score_scenes(_PerPixel(), scene_df, **kwargs)
        gpu = score_scenes(_PerPixel().cuda(), scene_df, device="cuda", **kwargs)

        assert np.array_equal(cpu.counts, gpu.counts)


class TestSaveAndLoadSceneCounts:
    def test_load_refuses_pickle_objects_by_default(self, monkeypatch, tmp_path):
        calls = []

        class Loaded:
            def __enter__(self):
                return {
                    "counts": np.zeros((1, 1, 4), dtype=np.int64),
                    "scene_ids": np.array(["a"]),
                    "thresholds": np.array([0.5], dtype=np.float32),
                    "mode": np.array("full_scene"),
                }

            def __exit__(self, *args):
                return False

        def load(path, **kwargs):
            calls.append(kwargs)
            return Loaded()

        monkeypatch.setattr(scene_inference.np, "load", load)
        load_scene_counts(tmp_path / "counts.npz")

        assert calls == [{"allow_pickle": False}]

    def test_a_saved_result_loads_back_identically(self, tmp_path):
        original = SceneCounts(
            counts=np.arange(2 * 3 * 4, dtype=np.int64).reshape(2, 3, 4),
            scene_ids=["scene_a", "scene_b"],
            thresholds=np.array([0.25, 0.5, 0.75], dtype=np.float32),
            mode="full_scene",
        )

        save_scene_counts(tmp_path / "out" / "counts.npz", original)
        loaded = load_scene_counts(tmp_path / "out" / "counts.npz")

        assert np.array_equal(loaded.counts, original.counts)
        assert loaded.counts.dtype == np.int64
        assert loaded.scene_ids == original.scene_ids
        assert loaded.thresholds.dtype == np.float32
        assert np.array_equal(loaded.thresholds, original.thresholds)
        assert loaded.mode == "full_scene"

    def _tiny_result(self):
        return SceneCounts(
            counts=np.zeros((1, 1, 4), dtype=np.int64),
            scene_ids=["a"],
            thresholds=np.array([0.5], dtype=np.float32),
            mode="patches",
        )

    def test_missing_parent_directories_are_created_at_any_depth(self, tmp_path):
        target = tmp_path / "deep" / "nested" / "counts.npz"

        save_scene_counts(target, self._tiny_result())

        assert target.is_file()

    def test_saving_into_an_existing_directory_and_over_an_existing_file_works(self, tmp_path):
        target = tmp_path / "out" / "counts.npz"
        save_scene_counts(target, self._tiny_result())

        save_scene_counts(target, self._tiny_result())

        assert load_scene_counts(target).scene_ids == ["a"]

    def test_a_file_holding_pickled_objects_is_refused(self, tmp_path):
        target = tmp_path / "pickled.npz"
        np.savez(
            target,
            counts=np.zeros((1, 1, 4), dtype=np.int64),
            scene_ids=np.array(["a", None], dtype=object),
            thresholds=np.array([0.5], dtype=np.float32),
            mode=np.array("full_scene"),
        )

        with pytest.raises(ValueError, match="pickle"):
            load_scene_counts(target)


def _repo_root_with_data():
    from pathlib import Path

    for parent in Path(__file__).resolve().parents:
        if (parent / "data" / "starcop_raw" / "test.csv").is_file():
            return parent
    return None


REPO_ROOT = _repo_root_with_data()


@pytest.mark.skipif(REPO_ROOT is None, reason="real STARCOP data is not available here")
class TestRealScenes:
    def test_real_512_scenes_account_for_every_pixel(self, monkeypatch):
        from data.scene_manifest import load_scene_manifest

        monkeypatch.chdir(REPO_ROOT)

        manifest = load_scene_manifest(
            REPO_ROOT / "data/starcop_raw/test.csv",
            REPO_ROOT / "data/processed/starcop_raw/patches/test_tiled_128_128.csv",
        ).head(2)

        result = score_scenes(
            _PerPixel(), manifest, mode="full_scene", thresholds=THRESHOLDS, dataset="starcop_raw"
        )

        assert result.counts.shape == (2, 3, 4)
        assert (result.counts.sum(axis=2) == 512 * 512).all()
        assert result.scene_ids == manifest["scene_id"].tolist()
