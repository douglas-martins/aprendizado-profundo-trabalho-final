import numpy as np
import pandas as pd
import pytest
import torch

import visualization.qualitative_predictions as qualitative_predictions
from visualization.qualitative_predictions import (
    dataset_display_label,
    exclude_scenes,
    mini_scene_names,
    per_patch_f1_by_architecture,
    predict_probability_map,
    select_largest_gap_row,
)


def _patches_df(
    names: list[str], has_plume: list[bool], frac_positives: list[float]
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [f"patch_{i}" for i in range(len(names))],
            "name": names,
            "has_plume": has_plume,
            "frac_positives": frac_positives,
        }
    )


class _FixedLogitModel(torch.nn.Module):
    """Returns pre-baked logits regardless of input -- lets tests control predictions exactly.

    Duplicated rather than imported, matching this project's established
    per-file fixture convention (see `test_evaluate.py`'s own docstring).
    """

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.logits = logits
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.logits.expand(x.shape[0], *self.logits.shape[1:])


def _logit(probability: float) -> float:
    return float(np.log(probability / (1 - probability)))


def _make_scene(tmp_path, name: str, size: int = 4):
    """Return (folder, bands) for a tiny 4-input-band + label scene, all-zero to start.

    Callers mutate `bands["labelbinary"]` (and/or the input bands) before writing
    with `_write_scene`. Duplicated rather than imported, matching this project's
    established per-file fixture convention (see `test_dataset.py`'s own
    `_make_scene`/`_write_scene`).
    """
    folder = tmp_path / name
    bands = {
        "mag1c": np.zeros((size, size), dtype="float32"),
        "TOA_AVIRIS_640nm": np.zeros((size, size), dtype="float32"),
        "TOA_AVIRIS_550nm": np.zeros((size, size), dtype="float32"),
        "TOA_AVIRIS_460nm": np.zeros((size, size), dtype="float32"),
        "labelbinary": np.zeros((size, size), dtype="float32"),
    }
    return folder, bands


def _write_scene(tiny_geotiff_factory, folder, bands: dict) -> None:
    for band_name, array in bands.items():
        tiny_geotiff_factory(folder / f"{band_name}.tif", array)


def _patch_row(patch_id: str, folder, has_plume: bool, size: int = 4) -> dict:
    return {
        "id": patch_id,
        "name": folder.name,
        "folder": str(folder),
        "window_col_off": 0,
        "window_row_off": 0,
        "window_width": size,
        "window_height": size,
        "has_plume": has_plume,
    }


class TestDatasetDisplayLabel:
    def test_maps_starcop_mini_to_mini(self):
        assert dataset_display_label("starcop_mini") == "mini"

    def test_maps_starcop_raw_to_raw(self):
        assert dataset_display_label("starcop_raw") == "raw"


class TestMiniSceneNames:
    def test_unions_name_column_across_dataframes(self):
        train = _patches_df(["a", "b"], [True, False], [0.1, 0.0])
        val = _patches_df(["b", "c"], [True, False], [0.1, 0.0])
        assert mini_scene_names(train, val) == {"a", "b", "c"}

    def test_single_dataframe(self):
        df = _patches_df(["a", "a", "b"], [True, False, True], [0.1, 0.0, 0.2])
        assert mini_scene_names(df) == {"a", "b"}


class TestExcludeScenes:
    def test_drops_rows_whose_name_is_excluded(self):
        df = _patches_df(["a", "b", "c"], [True, True, True], [0.1, 0.2, 0.3])
        result = exclude_scenes(df, {"b"})
        assert sorted(result["name"]) == ["a", "c"]

    def test_keeps_all_rows_when_nothing_excluded(self):
        df = _patches_df(["a", "b"], [True, True], [0.1, 0.2])
        result = exclude_scenes(df, set())
        assert len(result) == 2

    def test_empty_result_when_everything_excluded(self):
        df = _patches_df(["a", "b"], [True, True], [0.1, 0.2])
        result = exclude_scenes(df, {"a", "b"})
        assert len(result) == 0


class TestPredictProbabilityMap:
    def test_applies_sigmoid_to_model_logits(self):
        logits = torch.full((1, 1, 2, 2), _logit(0.75))
        model = _FixedLogitModel(logits)
        input_tensor = torch.zeros(4, 2, 2)
        probs = predict_probability_map(model, input_tensor)
        assert probs.shape == (2, 2)
        assert torch.allclose(probs, torch.full((2, 2), 0.75), atol=1e-5)

    def test_runs_without_gradients_and_in_eval_mode(self):
        logits = torch.zeros(1, 1, 2, 2)
        model = _FixedLogitModel(logits)
        model.train()
        input_tensor = torch.zeros(4, 2, 2, requires_grad=True)
        probs = predict_probability_map(model, input_tensor)
        assert not probs.requires_grad
        assert model.training is False

    def test_adds_a_batch_dimension_rather_than_reshaping_the_channel_dimension(self):

        received_shapes = []

        class _ShapeRecordingModel(torch.nn.Module):
            def forward(self, x):
                received_shapes.append(tuple(x.shape))
                return torch.zeros(x.shape[0], 1, x.shape[-2], x.shape[-1])

        input_tensor = torch.zeros(4, 2, 2)
        predict_probability_map(_ShapeRecordingModel(), input_tensor)
        assert received_shapes == [(1, 4, 2, 2)]


class TestSelectLargestGapRow:
    def test_returns_the_row_with_the_largest_minuend_minus_subtrahend(self):
        df = pd.DataFrame({"id": ["a", "b", "c"], "E2": [0.5, 0.9, 0.3], "E3": [0.4, 0.2, 0.9]})
        result = select_largest_gap_row(df, minuend_col="E2", subtrahend_col="E3")
        assert result["id"] == "b"

    def test_a_negative_gap_can_still_be_the_largest(self):
        df = pd.DataFrame({"id": ["a", "b"], "E2": [0.1, 0.2], "E3": [0.9, 0.3]})
        result = select_largest_gap_row(df, minuend_col="E2", subtrahend_col="E3")
        assert result["id"] == "b"


class TestPerPatchF1ByArchitecture:
    def test_passes_augmentation_false_to_the_dataset(self, monkeypatch):
        calls = []

        class Dataset:
            def __init__(self, examples, **kwargs):
                calls.append(kwargs)

            def __getitem__(self, index):
                return {"input": torch.zeros(4, 4, 4), "output": torch.ones(1, 4, 4)}

        monkeypatch.setattr(qualitative_predictions, "PatchDataset", Dataset)
        examples = pd.DataFrame([{"id": "patch", "has_plume": True}])
        model = _FixedLogitModel(torch.zeros(1, 1, 4, 4))

        qualitative_predictions.per_patch_f1_by_architecture(
            examples, ["starcop_mini"], {"model": model}, threshold=0.5
        )

        assert calls == [{"dataset": "starcop_mini", "augment": False}]

    def test_returns_id_dataset_has_plume_and_per_architecture_pixel_f1(
        self, tmp_path, tiny_geotiff_factory
    ):
        positive_folder, positive_bands = _make_scene(tmp_path, "positive")
        positive_bands["labelbinary"][:] = 1.0
        _write_scene(tiny_geotiff_factory, positive_folder, positive_bands)

        negative_folder, negative_bands = _make_scene(tmp_path, "negative")

        _write_scene(tiny_geotiff_factory, negative_folder, negative_bands)

        examples = pd.DataFrame(
            [
                _patch_row("patch_pos", positive_folder, has_plume=True),
                _patch_row("patch_neg", negative_folder, has_plume=False),
            ]
        )

        datasets = ["starcop_mini", "starcop_raw"]
        good_model = _FixedLogitModel(torch.full((1, 1, 4, 4), _logit(0.9)))
        bad_model = _FixedLogitModel(torch.full((1, 1, 4, 4), _logit(0.1)))
        models = {"good": good_model, "bad": bad_model}

        result = per_patch_f1_by_architecture(examples, datasets, models, threshold=0.5)

        assert list(result["id"]) == ["patch_pos", "patch_neg"]
        assert list(result["dataset"]) == ["starcop_mini", "starcop_raw"]
        assert list(result["has_plume"]) == [True, False]

        assert result.loc[0, "good"] == pytest.approx(1.0)
        assert result.loc[1, "good"] == pytest.approx(0.0)

        assert result.loc[0, "bad"] == pytest.approx(0.0)
        assert result.loc[1, "bad"] == pytest.approx(0.0)

    def test_binarizes_with_a_strict_greater_than_not_greater_or_equal(
        self, tmp_path, tiny_geotiff_factory
    ):

        folder, bands = _make_scene(tmp_path, "boundary")
        bands["labelbinary"][:] = 1.0
        _write_scene(tiny_geotiff_factory, folder, bands)
        examples = pd.DataFrame([_patch_row("patch_boundary", folder, has_plume=True)])
        boundary_model = _FixedLogitModel(torch.full((1, 1, 4, 4), _logit(0.5)))

        result = per_patch_f1_by_architecture(
            examples, ["starcop_mini"], {"boundary": boundary_model}, threshold=0.5
        )

        assert result.loc[0, "boundary"] == pytest.approx(0.0)

    def test_disables_augmentation_during_evaluation(
        self, tmp_path, tiny_geotiff_factory, monkeypatch
    ):

        from data import dataset as dataset_module

        calls = []

        class _MarkerAugmenter:
            def __call__(self, input_batch, output_batch):
                calls.append(True)
                return input_batch, output_batch

        monkeypatch.setattr(dataset_module, "_build_augmenter", lambda: _MarkerAugmenter())

        folder, bands = _make_scene(tmp_path, "augment_check")
        bands["labelbinary"][:] = 1.0
        _write_scene(tiny_geotiff_factory, folder, bands)
        examples = pd.DataFrame([_patch_row("patch_augment", folder, has_plume=True)])
        model = _FixedLogitModel(torch.full((1, 1, 4, 4), _logit(0.9)))

        per_patch_f1_by_architecture(examples, ["starcop_mini"], {"m": model}, threshold=0.5)

        assert calls == []
