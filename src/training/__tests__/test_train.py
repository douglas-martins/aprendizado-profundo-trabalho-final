import numpy as np
import pandas as pd
import pytest
import segmentation_models_pytorch as smp
import torch

import training.train as train
from data.precompute_patch_cache import build_cache_for_split
from models.losses import build_loss
from training.train import (
    EpochShuffleSampler,
    _epoch_marker,
    _loader_kwargs,
    augment_for,
    build_model,
    count_parameters,
    evaluate,
    fit,
    library_versions,
    set_seed,
)


class TestBuildModel:
    def test_e1_is_built_with_no_pretrained_argument(self, monkeypatch):

        calls = []
        monkeypatch.setattr(
            train, "_ARCHITECTURE_BUILDERS", {"E1": lambda **kwargs: calls.append(kwargs)}
        )

        build_model("E1")

        assert calls == [{}]

    def test_e2_and_e3_are_built_with_pretrained_true(self, monkeypatch):

        for architecture in ("E2", "E3"):
            calls = []
            monkeypatch.setattr(
                train,
                "_ARCHITECTURE_BUILDERS",
                {architecture: lambda **kwargs: calls.append(kwargs)},
            )

            build_model(architecture)

            assert calls == [{"pretrained": True}]


class TestAugmentFor:
    def test_e1_gets_no_augmentation(self):

        assert augment_for("E1") is False

    def test_e2_and_e3_get_augmentation(self):
        assert augment_for("E2") is True
        assert augment_for("E3") is True


class _FixedLogitModel(torch.nn.Module):
    """Returns pre-baked logits regardless of input -- lets tests control predictions exactly."""

    def __init__(self, logits_by_call: list[torch.Tensor]):
        super().__init__()
        self._logits_by_call = iter(logits_by_call)
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return next(self._logits_by_call)


class TestEvaluate:
    def test_accumulates_across_batches_without_holding_the_full_set_in_memory(self):

        batches = [
            {
                "input": torch.zeros(2, 4, 2, 2),
                "output": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]]),
            },
            {
                "input": torch.zeros(2, 4, 2, 2),
                "output": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]]),
            },
        ]

        logits_batch1 = torch.tensor([[[[5.0, 5.0], [-5.0, -5.0]]], [[[-5.0, -5.0], [-5.0, -5.0]]]])

        logits_batch2 = torch.full((2, 1, 2, 2), -5.0)
        model = _FixedLogitModel([logits_batch1, logits_batch2])
        loss_fn = build_loss(pos_weight=1.0)

        result = evaluate(model, batches, loss_fn, device="cpu")

        assert result["f1"] == pytest.approx(0.5)
        assert result["degenerate"] is False

    def test_flags_all_zero_predictions_as_degenerate_across_batches(self):
        batches = [
            {"input": torch.zeros(2, 4, 2, 2), "output": torch.zeros(2, 1, 2, 2)},
            {"input": torch.zeros(2, 4, 2, 2), "output": torch.zeros(2, 1, 2, 2)},
        ]
        model = _FixedLogitModel([torch.full((2, 1, 2, 2), -20.0), torch.full((2, 1, 2, 2), -20.0)])
        loss_fn = build_loss(pos_weight=1.0)

        result = evaluate(model, batches, loss_fn, device="cpu")

        assert result["degenerate"] is True


def _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=0):
    bands = {
        "mag1c": np.random.default_rng(0).uniform(0, 500, (size, size)).astype("float32"),
        "TOA_AVIRIS_640nm": np.random.default_rng(1).uniform(0, 40, (size, size)).astype("float32"),
        "TOA_AVIRIS_550nm": np.random.default_rng(2).uniform(0, 40, (size, size)).astype("float32"),
        "TOA_AVIRIS_460nm": np.random.default_rng(3).uniform(0, 40, (size, size)).astype("float32"),
        "labelbinary": np.zeros((size, size), dtype="float32"),
    }
    bands["labelbinary"].flat[:positive_pixels] = 1.0
    for band_name, array in bands.items():
        tiny_geotiff_factory(folder / f"{band_name}.tif", array)


def _patches_df(folder, n_rows, size=16):
    rows = []
    for i in range(n_rows):
        rows.append(
            {
                "id": f"patch_{i}",
                "name": "scene0",
                "folder": str(folder),
                "window_col_off": 0,
                "window_row_off": 0,
                "window_width": size,
                "window_height": size,
                "has_plume": i % 3 == 0,
                "frac_positives": 0.05 if i % 3 == 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _canned_evaluate(pairs):
    """Fake `train.evaluate` returning successive `(loss, f1)` pairs, one per call."""
    calls = iter(pairs)

    def _evaluate(model, loader, loss_fn, device, max_batches=None):
        loss, f1 = next(calls)
        return {"loss": loss, "f1": f1, "degenerate": False, "positive_fraction": 0.0}

    return _evaluate


class TestFit:
    def test_runs_for_the_requested_epochs_and_saves_a_checkpoint(
        self, tmp_path, tiny_geotiff_factory
    ):
        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=6, size=16)
        val_df = _patches_df(folder, n_rows=4, size=16)
        checkpoint_path = tmp_path / "checkpoints" / "e1.pt"

        model = torch.nn.Sequential(
            torch.nn.Conv2d(4, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 1, 1),
        )
        result = fit(
            model,
            train_df,
            val_df,
            dataset="starcop_mini",
            lr=1e-3,
            batch_size=2,
            max_epochs=2,
            patience=10,
            checkpoint_path=checkpoint_path,
            log_to_mlflow=False,
        )

        assert result["epochs_run"] == 2
        assert checkpoint_path.exists()
        assert "loss" in result
        assert "f1" in result
        assert "degenerate" in result

    def test_resume_from_loads_the_checkpoint_into_the_model_before_training(
        self, tmp_path, tiny_geotiff_factory
    ):

        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=4, size=16)
        val_df = _patches_df(folder, n_rows=2, size=16)

        source_model = torch.nn.Conv2d(4, 1, 1)
        resume_path = tmp_path / "resume.pt"
        torch.save(source_model.state_dict(), resume_path)

        model = torch.nn.Conv2d(4, 1, 1)
        loaded_state_dicts = []
        original_load_state_dict = model.load_state_dict

        def _spy_load_state_dict(state_dict, *args, **kwargs):
            loaded_state_dicts.append(state_dict)
            return original_load_state_dict(state_dict, *args, **kwargs)

        model.load_state_dict = _spy_load_state_dict

        fit(
            model,
            train_df,
            val_df,
            dataset="starcop_mini",
            lr=1e-3,
            batch_size=2,
            max_epochs=1,
            patience=10,
            resume_from=resume_path,
            log_to_mlflow=False,
        )

        assert len(loaded_state_dicts) == 1
        for key, value in source_model.state_dict().items():
            assert torch.equal(loaded_state_dicts[0][key], value)

    def test_resume_from_none_does_not_touch_the_models_weights(
        self, tmp_path, tiny_geotiff_factory
    ):
        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=4, size=16)
        val_df = _patches_df(folder, n_rows=2, size=16)

        model = torch.nn.Conv2d(4, 1, 1)
        calls = []
        original_load_state_dict = model.load_state_dict
        model.load_state_dict = lambda state_dict, *a, **k: (
            calls.append(state_dict) or (original_load_state_dict(state_dict, *a, **k))
        )

        fit(
            model,
            train_df,
            val_df,
            dataset="starcop_mini",
            lr=1e-3,
            batch_size=2,
            max_epochs=1,
            patience=10,
            log_to_mlflow=False,
        )

        assert calls == []

    def test_resume_never_replaces_the_checkpoint_with_an_epoch_worse_than_the_resumed_weights(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):

        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=4, size=16)
        val_df = _patches_df(folder, n_rows=2, size=16)
        source_model = torch.nn.Conv2d(4, 1, 1)
        resume_path = tmp_path / "resume.pt"
        torch.save(source_model.state_dict(), resume_path)
        checkpoint_path = tmp_path / "out" / "ckpt.pt"

        monkeypatch.setattr(
            "training.train.evaluate", _canned_evaluate([(0.10, 0.9), (0.50, 0.2), (0.60, 0.2)])
        )

        result = fit(
            torch.nn.Conv2d(4, 1, 1),
            train_df,
            val_df,
            dataset="starcop_mini",
            lr=1e-3,
            batch_size=2,
            max_epochs=2,
            patience=2,
            monitor="val_f1",
            resume_from=resume_path,
            checkpoint_path=checkpoint_path,
            log_to_mlflow=False,
        )

        assert result["best_epoch"] == 0
        assert result["f1"] == 0.9
        saved = torch.load(checkpoint_path)
        for key, value in source_model.state_dict().items():
            assert torch.equal(saved[key], value)

    def test_resume_saves_an_epoch_that_beats_the_resumed_weights(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):
        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=4, size=16)
        val_df = _patches_df(folder, n_rows=2, size=16)
        source_model = torch.nn.Conv2d(4, 1, 1)
        resume_path = tmp_path / "resume.pt"
        torch.save(source_model.state_dict(), resume_path)
        checkpoint_path = tmp_path / "out" / "ckpt.pt"

        monkeypatch.setattr("training.train.evaluate", _canned_evaluate([(0.50, 0.3), (0.40, 0.8)]))

        result = fit(
            torch.nn.Conv2d(4, 1, 1),
            train_df,
            val_df,
            dataset="starcop_mini",
            lr=1e-3,
            batch_size=2,
            max_epochs=1,
            patience=5,
            monitor="val_f1",
            resume_from=resume_path,
            checkpoint_path=checkpoint_path,
            log_to_mlflow=False,
        )

        assert result["best_epoch"] == 1
        assert result["f1"] == 0.8
        saved = torch.load(checkpoint_path)
        assert any(
            not torch.equal(saved[key], value) for key, value in source_model.state_dict().items()
        )

    def test_returns_the_best_epochs_metrics_not_the_last_epochs(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):

        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=4, size=16)
        val_df = _patches_df(folder, n_rows=2, size=16)

        canned_losses = [0.5, 0.1, 0.4, 0.6]
        calls = iter(canned_losses)

        def _fake_evaluate(model, loader, loss_fn, device, max_batches=None):
            return {
                "loss": next(calls),
                "f1": 0.0,
                "degenerate": False,
                "positive_fraction": 0.0,
            }

        monkeypatch.setattr("training.train.evaluate", _fake_evaluate)

        model = torch.nn.Conv2d(4, 1, 1)
        result = fit(
            model,
            train_df,
            val_df,
            dataset="starcop_mini",
            lr=1e-3,
            batch_size=2,
            max_epochs=4,
            patience=100,
            log_to_mlflow=False,
        )

        assert result["loss"] == 0.1
        assert result["best_epoch"] == 2

    def test_monitor_val_f1_selects_the_best_f1_epoch_not_the_best_loss_epoch(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):

        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=4, size=16)
        val_df = _patches_df(folder, n_rows=2, size=16)

        canned = [(0.1, 0.2), (0.2, 0.5), (0.3, 0.1)]
        calls = iter(canned)

        def _fake_evaluate(model, loader, loss_fn, device, max_batches=None):
            loss, f1 = next(calls)
            return {"loss": loss, "f1": f1, "degenerate": False, "positive_fraction": 0.0}

        monkeypatch.setattr("training.train.evaluate", _fake_evaluate)

        model = torch.nn.Conv2d(4, 1, 1)
        result = fit(
            model,
            train_df,
            val_df,
            dataset="starcop_mini",
            lr=1e-3,
            batch_size=2,
            max_epochs=3,
            patience=100,
            monitor="val_f1",
            log_to_mlflow=False,
        )

        assert result["best_epoch"] == 2
        assert result["f1"] == 0.5

    def test_monitor_val_f1_does_not_stop_while_val_loss_is_still_improving(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):

        canned = [
            (0.50, 0.10),
            (0.30, 0.10),
            (0.20, 0.50),
            (0.19, 0.10),
            (0.18, 0.10),
            (0.17, 0.10),
            (0.16, 0.10),
            (0.15, 0.10),
            (0.14, 0.60),
            (0.13, 0.10),
        ]
        calls = iter(canned)

        def _fake_evaluate(model, loader, loss_fn, device, max_batches=None):
            loss, f1 = next(calls)
            return {"loss": loss, "f1": f1, "degenerate": False, "positive_fraction": 0.0}

        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=4, size=16)
        val_df = _patches_df(folder, n_rows=2, size=16)
        monkeypatch.setattr("training.train.evaluate", _fake_evaluate)

        result = fit(
            torch.nn.Conv2d(4, 1, 1),
            train_df,
            val_df,
            dataset="starcop_mini",
            lr=1e-3,
            batch_size=2,
            max_epochs=len(canned),
            patience=5,
            monitor="val_f1",
            log_to_mlflow=False,
        )

        assert result["epochs_run"] >= 9
        assert result["best_epoch"] == 9
        assert result["f1"] == 0.60

    def test_monitor_val_f1_still_stops_once_both_metrics_plateau(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):

        canned = [
            (0.50, 0.10),
            (0.30, 0.50),
            (0.31, 0.10),
            (0.32, 0.10),
            (0.33, 0.10),
        ]
        calls = iter(canned)

        def _fake_evaluate(model, loader, loss_fn, device, max_batches=None):
            loss, f1 = next(calls)
            return {"loss": loss, "f1": f1, "degenerate": False, "positive_fraction": 0.0}

        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=4, size=16)
        val_df = _patches_df(folder, n_rows=2, size=16)
        monkeypatch.setattr("training.train.evaluate", _fake_evaluate)

        result = fit(
            torch.nn.Conv2d(4, 1, 1),
            train_df,
            val_df,
            dataset="starcop_mini",
            lr=1e-3,
            batch_size=2,
            max_epochs=len(canned),
            patience=3,
            monitor="val_f1",
            log_to_mlflow=False,
        )

        assert result["epochs_run"] == 5
        assert result["best_epoch"] == 2
        assert result["stopped_early"] is True

    def test_monitor_val_loss_is_unaffected_by_the_dual_condition(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):

        canned = [
            (0.1, 0.9),
            (0.2, 0.1),
            (0.3, 0.1),
            (0.4, 0.1),
        ]
        calls = iter(canned)

        def _fake_evaluate(model, loader, loss_fn, device, max_batches=None):
            loss, f1 = next(calls)
            return {"loss": loss, "f1": f1, "degenerate": False, "positive_fraction": 0.0}

        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=4, size=16)
        val_df = _patches_df(folder, n_rows=2, size=16)
        monkeypatch.setattr("training.train.evaluate", _fake_evaluate)

        result = fit(
            torch.nn.Conv2d(4, 1, 1),
            train_df,
            val_df,
            dataset="starcop_mini",
            lr=1e-3,
            batch_size=2,
            max_epochs=len(canned),
            patience=3,
            log_to_mlflow=False,
        )

        assert result["epochs_run"] == 4
        assert result["best_epoch"] == 1
        assert result["loss"] == 0.1

    def test_monitor_defaults_to_val_loss_unchanged_behavior(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):

        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=4, size=16)
        val_df = _patches_df(folder, n_rows=2, size=16)

        canned = [(0.1, 0.2), (0.2, 0.5), (0.3, 0.1)]
        calls = iter(canned)

        def _fake_evaluate(model, loader, loss_fn, device, max_batches=None):
            loss, f1 = next(calls)
            return {"loss": loss, "f1": f1, "degenerate": False, "positive_fraction": 0.0}

        monkeypatch.setattr("training.train.evaluate", _fake_evaluate)

        model = torch.nn.Conv2d(4, 1, 1)
        result = fit(
            model,
            train_df,
            val_df,
            dataset="starcop_mini",
            lr=1e-3,
            batch_size=2,
            max_epochs=3,
            patience=100,
            log_to_mlflow=False,
        )

        assert result["best_epoch"] == 1
        assert result["loss"] == 0.1

    def test_max_steps_stops_training_early_regardless_of_max_epochs(
        self, tmp_path, tiny_geotiff_factory
    ):
        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=6, size=16)
        val_df = _patches_df(folder, n_rows=4, size=16)

        model = torch.nn.Sequential(torch.nn.Conv2d(4, 1, 1))
        result = fit(
            model,
            train_df,
            val_df,
            dataset="starcop_mini",
            lr=1e-3,
            batch_size=2,
            max_epochs=50,
            patience=1000,
            max_steps=3,
            log_to_mlflow=False,
        )

        assert result["steps_run"] == 3
        assert result["epochs_run"] < 50


class TestLoaderKwargs:
    def test_pins_memory_for_a_cuda_device(self):

        kwargs = _loader_kwargs("cuda", num_workers=0, seed=1, persistent_workers=True)
        assert kwargs["pin_memory"] is True

    def test_does_not_pin_memory_for_cpu(self):
        kwargs = _loader_kwargs("cpu", num_workers=0, seed=1, persistent_workers=True)
        assert kwargs["pin_memory"] is False

    def test_omits_worker_kwargs_when_num_workers_is_zero(self):

        kwargs = _loader_kwargs("cpu", num_workers=0, seed=1, persistent_workers=True)
        assert "num_workers" not in kwargs
        assert "persistent_workers" not in kwargs
        assert "worker_init_fn" not in kwargs

    def test_sets_persistent_workers_only_when_requested(self):

        persistent = _loader_kwargs("cpu", num_workers=4, seed=1, persistent_workers=True)
        not_persistent = _loader_kwargs("cpu", num_workers=4, seed=1, persistent_workers=False)

        assert persistent["persistent_workers"] is True
        assert "persistent_workers" not in not_persistent

        assert not_persistent["num_workers"] == 4

    def test_worker_init_fn_seeds_from_the_given_base_seed(self):
        kwargs = _loader_kwargs("cpu", num_workers=2, seed=99, persistent_workers=True)
        assert kwargs["worker_init_fn"].keywords == {"base_seed": 99}


class TestFitLoaderConcurrency:
    def test_only_the_train_loader_is_requested_with_persistent_workers(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):
        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=4, size=16)
        val_df = _patches_df(folder, n_rows=2, size=16)

        real_loader_kwargs = train._loader_kwargs
        captured = []

        def _recording_loader_kwargs(*args, **kwargs):
            captured.append(kwargs)
            return real_loader_kwargs(*args, **kwargs)

        monkeypatch.setattr(train, "_loader_kwargs", _recording_loader_kwargs)

        model = torch.nn.Conv2d(4, 1, 1)
        fit(
            model,
            train_df,
            val_df,
            dataset="starcop_mini",
            lr=1e-3,
            batch_size=2,
            max_epochs=1,
            patience=10,
            max_steps=1,
            log_to_mlflow=False,
        )

        train_call_kwargs, val_call_kwargs = captured
        assert train_call_kwargs["persistent_workers"] is True
        assert val_call_kwargs["persistent_workers"] is False


class TestFitOnDiskCache:
    def test_uses_a_matching_on_disk_cache_for_both_loaders(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):

        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=4, size=16)
        val_df = _patches_df(folder, n_rows=2, size=16)

        cache_root = tmp_path / "patch_cache"
        monkeypatch.setattr(train, "_CACHE_ROOT", cache_root)
        build_cache_for_split(
            "starcop_mini", train_df, cache_root / "starcop_mini" / "train", num_workers=0
        )
        build_cache_for_split(
            "starcop_mini", val_df, cache_root / "starcop_mini" / "val", num_workers=0
        )

        from data import dataset as dataset_module

        calls = []
        monkeypatch.setattr(
            dataset_module, "read_patch_bands", lambda *a, **k: calls.append(a) or {}
        )

        model = torch.nn.Conv2d(4, 1, 1)
        fit(
            model,
            train_df,
            val_df,
            dataset="starcop_mini",
            lr=1e-3,
            batch_size=2,
            max_epochs=1,
            patience=10,
            log_to_mlflow=False,
        )

        assert calls == []

    def test_falls_back_to_live_reads_when_no_cache_matches(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):

        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=4, size=16)
        val_df = _patches_df(folder, n_rows=2, size=16)

        cache_root = tmp_path / "patch_cache"
        monkeypatch.setattr(train, "_CACHE_ROOT", cache_root)

        other_df = _patches_df(folder, n_rows=99, size=16)
        build_cache_for_split(
            "starcop_mini", other_df, cache_root / "starcop_mini" / "train", num_workers=0
        )

        model = torch.nn.Conv2d(4, 1, 1)
        result = fit(
            model,
            train_df,
            val_df,
            dataset="starcop_mini",
            lr=1e-3,
            batch_size=2,
            max_epochs=1,
            patience=10,
            log_to_mlflow=False,
        )

        assert result["epochs_run"] == 1


class TestSetSeed:
    def test_sets_torch_and_numpy_rng_deterministically(self):

        set_seed(123)
        first_torch = torch.rand(3)
        first_numpy = np.random.rand(3)

        set_seed(123)
        second_torch = torch.rand(3)
        second_numpy = np.random.rand(3)

        assert torch.equal(first_torch, second_torch)
        assert np.array_equal(first_numpy, second_numpy)

    def test_enables_cudnn_determinism(self):

        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        set_seed(123)

        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False


class TestFitSeed:
    def test_same_seed_produces_identical_metrics_across_runs(self, tmp_path, tiny_geotiff_factory):

        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=6, size=16)
        val_df = _patches_df(folder, n_rows=4, size=16)

        def _run():
            torch.manual_seed(0)
            model = torch.nn.Sequential(
                torch.nn.Conv2d(4, 8, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(8, 1, 1),
            )
            return fit(
                model,
                train_df,
                val_df,
                dataset="starcop_mini",
                lr=1e-3,
                batch_size=2,
                max_epochs=2,
                patience=10,
                seed=42,
                augment=True,
                log_to_mlflow=False,
            )

        first = _run()
        second = _run()

        assert first["loss"] == second["loss"]
        assert first["f1"] == second["f1"]

    def test_seed_is_wired_into_the_train_loaders_sampler_and_augmentation(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):

        import training.train as train_module

        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=4, size=16)
        val_df = _patches_df(folder, n_rows=2, size=16)

        real_data_loader = train_module.DataLoader
        captured = []

        def _recording_data_loader(dataset, **kwargs):
            captured.append((dataset, kwargs))
            return real_data_loader(dataset, **kwargs)

        monkeypatch.setattr(train_module, "DataLoader", _recording_data_loader)

        model = torch.nn.Conv2d(4, 1, 1)
        fit(
            model,
            train_df,
            val_df,
            dataset="starcop_mini",
            lr=1e-3,
            batch_size=2,
            max_epochs=1,
            patience=10,
            max_steps=1,
            seed=7,
            log_to_mlflow=False,
        )

        train_dataset, train_kwargs = captured[0]
        assert isinstance(train_kwargs["sampler"], EpochShuffleSampler)
        assert train_kwargs["sampler"].seed == 7
        assert "generator" not in train_kwargs
        assert "shuffle" not in train_kwargs
        assert train_dataset.augment_seed == 7


class TestCountParameters:
    def test_counts_every_parameter_tensor_element(self):

        model = torch.nn.Linear(4, 2)
        assert count_parameters(model) == 10

    def test_matches_the_sum_of_numel_across_parameters(self):

        model = torch.nn.Sequential(torch.nn.Conv2d(4, 8, 3), torch.nn.Conv2d(8, 1, 1))
        expected = sum(p.numel() for p in model.parameters())
        assert count_parameters(model) == expected


class TestLibraryVersions:
    def test_reports_torch_smp_and_cuda_versions_for_mlflow_params(self):

        versions = library_versions()

        assert versions["torch_version"] == torch.__version__
        assert versions["smp_version"] == smp.__version__
        assert versions["cuda_version"] == (torch.version.cuda or "cpu")

    def test_reports_cpu_when_the_torch_build_has_no_cuda(self, monkeypatch):

        monkeypatch.setattr(torch.version, "cuda", None)

        assert library_versions()["cuda_version"] == "cpu"


class TestFitTiming:
    def test_returns_nonzero_wall_clock_and_seconds_per_epoch(self, tmp_path, tiny_geotiff_factory):

        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=4, size=16)
        val_df = _patches_df(folder, n_rows=2, size=16)

        model = torch.nn.Conv2d(4, 1, 1)
        result = fit(
            model,
            train_df,
            val_df,
            dataset="starcop_mini",
            lr=1e-3,
            batch_size=2,
            max_epochs=2,
            patience=10,
            log_to_mlflow=False,
        )

        assert result["wall_clock_seconds"] > 0
        assert result["seconds_per_epoch"] > 0

    def test_seconds_per_epoch_is_wall_clock_divided_by_epochs_run(
        self, tmp_path, tiny_geotiff_factory
    ):
        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=4, size=16)
        val_df = _patches_df(folder, n_rows=2, size=16)

        model = torch.nn.Conv2d(4, 1, 1)
        result = fit(
            model,
            train_df,
            val_df,
            dataset="starcop_mini",
            lr=1e-3,
            batch_size=2,
            max_epochs=3,
            patience=10,
            log_to_mlflow=False,
        )

        assert result["seconds_per_epoch"] == pytest.approx(
            result["wall_clock_seconds"] / result["epochs_run"]
        )


class TestEvaluateMaxBatches:
    def test_stops_after_max_batches_and_ignores_the_rest(self):

        batches = [
            {"input": torch.zeros(1, 4, 2, 2), "output": torch.zeros(1, 1, 2, 2)},
            {"input": torch.zeros(1, 4, 2, 2), "output": torch.zeros(1, 1, 2, 2)},
            {"input": torch.zeros(1, 4, 2, 2), "output": torch.zeros(1, 1, 2, 2)},
        ]
        logits = [
            torch.full((1, 1, 2, 2), -5.0),
            torch.full((1, 1, 2, 2), 0.0),
            torch.full((1, 1, 2, 2), 5.0),
        ]
        model = _FixedLogitModel(logits)
        loss_fn = build_loss(pos_weight=1.0)

        result = evaluate(model, batches, loss_fn, device="cpu", max_batches=2)

        target = torch.zeros(1, 1, 2, 2)
        expected_loss = (loss_fn(logits[0], target).item() + loss_fn(logits[1], target).item()) / 2
        assert result["loss"] == pytest.approx(expected_loss)

    def test_none_processes_every_batch_unchanged(self):

        batches = [
            {"input": torch.zeros(1, 4, 2, 2), "output": torch.zeros(1, 1, 2, 2)},
            {"input": torch.zeros(1, 4, 2, 2), "output": torch.zeros(1, 1, 2, 2)},
        ]
        model = _FixedLogitModel([torch.full((1, 1, 2, 2), -5.0), torch.full((1, 1, 2, 2), 5.0)])
        loss_fn = build_loss(pos_weight=1.0)

        result = evaluate(model, batches, loss_fn, device="cpu", max_batches=None)

        target = torch.zeros(1, 1, 2, 2)
        expected_loss = (
            loss_fn(torch.full((1, 1, 2, 2), -5.0), target).item()
            + loss_fn(torch.full((1, 1, 2, 2), 5.0), target).item()
        ) / 2
        assert result["loss"] == pytest.approx(expected_loss)


class TestFitMaxValBatches:
    def test_max_val_batches_is_threaded_through_to_evaluate(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):
        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=4, size=16)
        val_df = _patches_df(folder, n_rows=2, size=16)

        captured_max_batches = []

        def _recording_evaluate(model, loader, loss_fn, device, max_batches=None):
            captured_max_batches.append(max_batches)
            return {"loss": 0.0, "f1": 0.0, "degenerate": False, "positive_fraction": 0.0}

        monkeypatch.setattr("training.train.evaluate", _recording_evaluate)

        model = torch.nn.Conv2d(4, 1, 1)
        fit(
            model,
            train_df,
            val_df,
            dataset="starcop_mini",
            lr=1e-3,
            batch_size=2,
            max_epochs=1,
            patience=10,
            max_val_batches=3,
            log_to_mlflow=False,
        )

        assert captured_max_batches == [3]

    def test_defaults_to_no_cap(self, tmp_path, tiny_geotiff_factory, monkeypatch):
        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=4, size=16)
        val_df = _patches_df(folder, n_rows=2, size=16)

        captured_max_batches = []

        def _recording_evaluate(model, loader, loss_fn, device, max_batches=None):
            captured_max_batches.append(max_batches)
            return {"loss": 0.0, "f1": 0.0, "degenerate": False, "positive_fraction": 0.0}

        monkeypatch.setattr("training.train.evaluate", _recording_evaluate)

        model = torch.nn.Conv2d(4, 1, 1)
        fit(
            model,
            train_df,
            val_df,
            dataset="starcop_mini",
            lr=1e-3,
            batch_size=2,
            max_epochs=1,
            patience=10,
            log_to_mlflow=False,
        )

        assert captured_max_batches == [None]


class TestEpochMarker:
    def test_marks_a_new_best_epoch_with_a_positive_emoji(self):

        assert _epoch_marker(is_best=True) == "✅"

    def test_marks_a_non_improving_epoch_with_a_negative_emoji(self):
        assert _epoch_marker(is_best=False) == "❌"


class TestFitVerboseOutput:
    def test_prints_a_positive_marker_for_the_first_epoch(
        self, tmp_path, tiny_geotiff_factory, capsys
    ):

        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=4, size=16)
        val_df = _patches_df(folder, n_rows=2, size=16)

        model = torch.nn.Conv2d(4, 1, 1)
        fit(
            model,
            train_df,
            val_df,
            dataset="starcop_mini",
            lr=1e-3,
            batch_size=2,
            max_epochs=1,
            patience=10,
            log_to_mlflow=False,
            verbose=True,
        )

        captured = capsys.readouterr()
        assert "✅" in captured.out

    def test_prints_a_negative_marker_when_an_epoch_does_not_beat_the_best(
        self, tmp_path, tiny_geotiff_factory, capsys, monkeypatch
    ):
        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=4, size=16)
        val_df = _patches_df(folder, n_rows=2, size=16)

        canned_losses = iter([0.1, 0.5])

        def _fake_evaluate(model, loader, loss_fn, device, max_batches=None):
            return {
                "loss": next(canned_losses),
                "f1": 0.0,
                "degenerate": False,
                "positive_fraction": 0.0,
            }

        monkeypatch.setattr("training.train.evaluate", _fake_evaluate)

        model = torch.nn.Conv2d(4, 1, 1)
        fit(
            model,
            train_df,
            val_df,
            dataset="starcop_mini",
            lr=1e-3,
            batch_size=2,
            max_epochs=2,
            patience=100,
            log_to_mlflow=False,
            verbose=True,
        )

        lines = capsys.readouterr().out.strip().splitlines()
        epoch_lines = [line for line in lines if "epoch" in line and "step" in line]
        assert "✅" in epoch_lines[0]
        assert "❌" in epoch_lines[1]


class TestFitProgressHeartbeat:
    def test_prints_a_mid_epoch_heartbeat_every_progress_every_steps(
        self, tmp_path, tiny_geotiff_factory, capsys
    ):

        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=8, size=16)
        val_df = _patches_df(folder, n_rows=2, size=16)

        model = torch.nn.Conv2d(4, 1, 1)
        fit(
            model,
            train_df,
            val_df,
            dataset="starcop_mini",
            lr=1e-3,
            batch_size=2,
            max_epochs=1,
            patience=10,
            log_to_mlflow=False,
            verbose=True,
            progress_every=2,
        )

        captured = capsys.readouterr().out
        assert "step 2" in captured
        assert "step 4" in captured

    def test_defaults_to_no_heartbeat(self, tmp_path, tiny_geotiff_factory, capsys):

        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=5)
        train_df = _patches_df(folder, n_rows=4, size=16)
        val_df = _patches_df(folder, n_rows=2, size=16)

        model = torch.nn.Conv2d(4, 1, 1)
        fit(
            model,
            train_df,
            val_df,
            dataset="starcop_mini",
            lr=1e-3,
            batch_size=2,
            max_epochs=1,
            patience=10,
            log_to_mlflow=False,
            verbose=True,
        )

        captured = capsys.readouterr().out
        assert "..." not in captured


class TestEpochShuffleSampler:
    def test_yields_every_index_exactly_once(self):
        sampler = EpochShuffleSampler(10, seed=1)

        assert sorted(sampler) == list(range(10))
        assert len(sampler) == 10

    def test_the_order_is_a_pure_function_of_seed_and_epoch(self):
        first = EpochShuffleSampler(50, seed=1)
        second = EpochShuffleSampler(50, seed=1)
        first.set_epoch(4)
        second.set_epoch(4)

        assert list(first) == list(second)
        assert list(first) == list(first)

    def test_a_different_epoch_gives_a_different_order(self):
        sampler = EpochShuffleSampler(50, seed=1)
        sampler.set_epoch(1)
        epoch_one = list(sampler)
        sampler.set_epoch(2)

        assert list(sampler) != epoch_one

    def test_a_different_seed_gives_a_different_order(self):
        assert list(EpochShuffleSampler(50, seed=1)) != list(EpochShuffleSampler(50, seed=2))

    def test_shuffling_leaves_the_global_torch_rng_untouched(self):
        torch.manual_seed(3)
        untouched = torch.rand(3)
        torch.manual_seed(3)
        list(EpochShuffleSampler(50, seed=1))

        assert torch.equal(torch.rand(3), untouched)


def _varied_patches_df(folder, n_rows, window=8):
    """Rows cycling over four distinct windows of a 16x16 scene, so shuffle order matters."""
    offsets = [(0, 0), (0, 8), (8, 0), (8, 8)]
    rows = []
    for i in range(n_rows):
        row_off, col_off = offsets[i % 4]
        rows.append(
            {
                "id": f"patch_{i}",
                "name": "scene0",
                "folder": str(folder),
                "window_col_off": col_off,
                "window_row_off": row_off,
                "window_width": window,
                "window_height": window,
                "has_plume": i % 3 == 0,
                "frac_positives": 0.05 if i % 3 == 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _tiny_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Conv2d(4, 8, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(8, 1, 1)
    )


def _state_dicts_equal(first, second):
    return first.keys() == second.keys() and all(torch.equal(first[k], second[k]) for k in first)


class _ExactResumeFixture:
    """Two tiny scenes' worth of patches plus a `run()` that fits with per-name output paths."""

    def __init__(self, tmp_path, tiny_geotiff_factory):
        folder = tmp_path / "scene0"
        _make_scene(tiny_geotiff_factory, folder, size=16, positive_pixels=40)
        self.train_df = _varied_patches_df(folder, n_rows=12)
        self.val_df = _varied_patches_df(folder, n_rows=4)
        self.tmp_path = tmp_path

    def paths(self, name):
        return self.tmp_path / name / "ckpt.pt", self.tmp_path / name / "state.pt"

    def run(self, name, model=None, **overrides):
        checkpoint_path, state_path = self.paths(name)
        model = model if model is not None else _tiny_model()
        kwargs = {
            "dataset": "starcop_mini",
            "lr": 1e-3,
            "batch_size": 2,
            "max_epochs": 4,
            "patience": 10,
            "seed": 42,
            "augment": True,
            "log_to_mlflow": False,
            "checkpoint_path": checkpoint_path,
            "state_path": state_path,
            **overrides,
        }
        train_df = kwargs.pop("train_df", self.train_df)
        result = fit(model, train_df, self.val_df, **kwargs)
        return model, result


@pytest.fixture
def resume_fixture(tmp_path, tiny_geotiff_factory):
    return _ExactResumeFixture(tmp_path, tiny_geotiff_factory)


def _crash_on_evaluate_call(monkeypatch, call_number):
    real_evaluate = train.evaluate
    calls = {"n": 0}

    def _evaluate(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == call_number:
            raise RuntimeError("simulated GPU crash")
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr("training.train.evaluate", _evaluate)


class TestFitExactResume:
    @pytest.mark.parametrize("num_workers", [0, 2])
    def test_a_crashed_then_resumed_run_equals_an_uninterrupted_one(
        self, resume_fixture, monkeypatch, num_workers
    ):

        uninterrupted_model, uninterrupted = resume_fixture.run("a", num_workers=num_workers)

        with monkeypatch.context() as patched:
            _crash_on_evaluate_call(patched, call_number=3)
            with pytest.raises(RuntimeError, match="simulated GPU crash"):
                resume_fixture.run("b", num_workers=num_workers)
        resumed_model, resumed = resume_fixture.run("b", num_workers=num_workers, resume_state=True)

        assert _state_dicts_equal(resumed_model.state_dict(), uninterrupted_model.state_dict())
        for key in ("loss", "f1", "best_epoch", "epochs_run", "steps_run", "last_epoch_metrics"):
            assert resumed[key] == uninterrupted[key]
        checkpoint_a, _ = resume_fixture.paths("a")
        checkpoint_b, _ = resume_fixture.paths("b")
        assert _state_dicts_equal(torch.load(checkpoint_b), torch.load(checkpoint_a))

    def test_a_run_stopped_at_max_epochs_can_be_extended_by_resuming(self, resume_fixture):

        resume_fixture.run("a", max_epochs=2)
        extended_model, extended = resume_fixture.run("a", max_epochs=4, resume_state=True)

        straight_model, straight = resume_fixture.run("b", max_epochs=4)

        assert extended["epochs_run"] == 4
        assert _state_dicts_equal(extended_model.state_dict(), straight_model.state_dict())
        assert extended["last_epoch_metrics"] == straight["last_epoch_metrics"]

    def test_early_stopping_counters_survive_a_resume(self, resume_fixture, monkeypatch):

        monkeypatch.setattr("training.train.evaluate", _canned_evaluate([(0.5, 0.5), (0.6, 0.4)]))
        resume_fixture.run("a", max_epochs=2, patience=2, monitor="val_f1")

        monkeypatch.setattr("training.train.evaluate", _canned_evaluate([(0.7, 0.3)]))
        _, resumed = resume_fixture.run(
            "a", max_epochs=10, patience=2, monitor="val_f1", resume_state=True
        )

        assert resumed["epochs_run"] == 3
        assert resumed["stopped_early"] is True
        assert resumed["best_epoch"] == 1
        assert resumed["f1"] == 0.5

    def test_resuming_a_run_that_already_stopped_early_does_nothing(
        self, resume_fixture, monkeypatch
    ):
        monkeypatch.setattr("training.train.evaluate", _canned_evaluate([(0.5, 0.5), (0.6, 0.4)]))
        resume_fixture.run("a", max_epochs=10, patience=1, monitor="val_f1")
        monkeypatch.setattr("training.train.evaluate", _canned_evaluate([]))

        _, again = resume_fixture.run(
            "a", max_epochs=10, patience=1, monitor="val_f1", resume_state=True
        )

        assert again["epochs_run"] == 2
        assert again["stopped_early"] is True

    def test_the_resumed_wall_clock_includes_the_time_before_the_crash(
        self, resume_fixture, monkeypatch
    ):
        _, first = resume_fixture.run("a", max_epochs=2)
        _, resumed = resume_fixture.run("a", max_epochs=3, resume_state=True)

        assert resumed["wall_clock_seconds"] > first["wall_clock_seconds"]

    def test_the_state_file_records_the_last_completed_epoch(self, resume_fixture):
        _, result = resume_fixture.run("a", max_epochs=3)
        _, state_path = resume_fixture.paths("a")

        state = torch.load(state_path, weights_only=False)

        assert state["epoch"] == result["epochs_run"] == 3
        assert state["step_count"] == result["steps_run"]
        assert not list(state_path.parent.glob("*.tmp"))

    def test_resume_state_without_a_state_path_is_rejected(self, resume_fixture):
        with pytest.raises(ValueError, match="state_path"):
            resume_fixture.run("a", state_path=None, resume_state=True)

    def test_resume_state_with_a_missing_state_file_is_rejected(self, resume_fixture):
        with pytest.raises(FileNotFoundError):
            resume_fixture.run("never-ran", resume_state=True)

    def test_resume_state_together_with_weights_only_resume_from_is_rejected(
        self, resume_fixture, tmp_path
    ):
        resume_fixture.run("a", max_epochs=1)
        weights = tmp_path / "weights.pt"
        torch.save(_tiny_model().state_dict(), weights)

        with pytest.raises(ValueError, match="resume_from"):
            resume_fixture.run("a", resume_state=True, resume_from=weights)

    def test_a_fresh_run_refuses_to_overwrite_an_existing_state_file(self, resume_fixture):
        resume_fixture.run("a", max_epochs=1)

        with pytest.raises(FileExistsError):
            resume_fixture.run("a", max_epochs=1)

    def test_a_changed_hyperparameter_is_rejected_on_resume(self, resume_fixture):
        resume_fixture.run("a", max_epochs=1)

        with pytest.raises(ValueError, match="signature"):
            resume_fixture.run("a", resume_state=True, lr=5e-4)

    def test_a_changed_training_set_is_rejected_on_resume(self, resume_fixture):
        resume_fixture.run("a", max_epochs=1)

        with pytest.raises(ValueError, match="signature"):
            resume_fixture.run(
                "a", resume_state=True, train_df=resume_fixture.train_df.iloc[:-2].copy()
            )

    def test_a_different_architecture_is_rejected_on_resume(self, resume_fixture):
        resume_fixture.run("a", max_epochs=1)
        bigger = torch.nn.Sequential(
            torch.nn.Conv2d(4, 16, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(16, 1, 1)
        )

        with pytest.raises(ValueError, match="signature"):
            resume_fixture.run("a", model=bigger, resume_state=True)

    def test_changing_patience_or_max_epochs_is_allowed_on_resume(self, resume_fixture):
        resume_fixture.run("a", max_epochs=1, patience=3)

        _, resumed = resume_fixture.run("a", max_epochs=2, patience=7, resume_state=True)

        assert resumed["epochs_run"] == 2


class TestFitEpochWiring:
    def test_fit_advances_the_sampler_and_the_augmentation_epoch_every_epoch(
        self, resume_fixture, monkeypatch
    ):

        captured = []
        real_data_loader = train.DataLoader

        def _recording_data_loader(dataset, **kwargs):
            captured.append((dataset, kwargs))
            return real_data_loader(dataset, **kwargs)

        monkeypatch.setattr(train, "DataLoader", _recording_data_loader)

        resume_fixture.run("a", max_epochs=3)

        train_dataset, train_kwargs = captured[0]
        assert train_kwargs["sampler"].epoch == 3
        assert int(train_dataset._epoch[0]) == 3

    def test_resuming_restores_the_main_process_rng_state(self, resume_fixture):
        resume_fixture.run("a", max_epochs=2)
        _, state_path = resume_fixture.paths("a")
        saved_rng = torch.load(state_path, weights_only=False)["rng"]
        torch.manual_seed(999)

        resume_fixture.run("a", max_epochs=2, resume_state=True)

        assert torch.equal(torch.get_rng_state(), saved_rng["torch"])
        assert np.array_equal(np.random.get_state()[1], saved_rng["numpy"][1])
