import numpy as np
import pandas as pd

from data.sampling import build_r2_manifest, sample_flightlines


def _patches_df(flightline_names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"id": [f"p{i}" for i in range(len(flightline_names))], "name": flightline_names}
    )


class TestSampleFlightlines:
    def test_returns_the_requested_count(self):
        df = _patches_df([f"fl{i}" for i in range(50) for _ in range(3)])
        selected = sample_flightlines(df, n_flightlines=10, seed=42)
        assert len(selected) == 10
        assert len(set(selected)) == 10

    def test_deterministic_given_same_seed(self):
        df = _patches_df([f"fl{i}" for i in range(50) for _ in range(3)])
        first = sample_flightlines(df, n_flightlines=10, seed=7)
        second = sample_flightlines(df, n_flightlines=10, seed=7)
        assert first == second

    def test_different_seeds_can_give_different_samples(self):
        df = _patches_df([f"fl{i}" for i in range(50) for _ in range(3)])
        first = sample_flightlines(df, n_flightlines=10, seed=1)
        second = sample_flightlines(df, n_flightlines=10, seed=2)
        assert first != second

    def test_raises_when_not_enough_flightlines(self):
        df = _patches_df([f"fl{i}" for i in range(3)])
        try:
            sample_flightlines(df, n_flightlines=10, seed=42)
            raise AssertionError("expected ValueError")
        except ValueError as error:
            assert "flightline" in str(error)

    def test_does_not_raise_when_exactly_enough_flightlines(self):
        df = _patches_df([f"fl{i}" for i in range(5)])
        selected = sample_flightlines(df, n_flightlines=5, seed=42)
        assert len(selected) == 5

    def test_samples_without_replacement(self, monkeypatch):
        calls = []

        class Generator:
            def choice(self, values, **kwargs):
                calls.append(kwargs)
                return values[: kwargs["size"]]

        monkeypatch.setattr(np.random, "default_rng", lambda seed: Generator())
        sample_flightlines(_patches_df(["a", "b", "c"]), n_flightlines=2, seed=42)

        assert calls == [{"size": 2, "replace": False}]


class TestBuildR2Manifest:
    def test_keeps_only_patches_from_sampled_flightlines(self):
        df = _patches_df([f"fl{i}" for i in range(20) for _ in range(5)])

        manifest = build_r2_manifest(df, n_flightlines=4, seed=42)

        assert manifest["name"].nunique() == 4
        assert len(manifest) == 4 * 5
        assert set(manifest["name"]) <= set(df["name"])

    def test_manifest_is_a_subset_of_rows_not_a_copy_with_new_data(self):
        df = _patches_df([f"fl{i}" for i in range(10) for _ in range(2)])

        manifest = build_r2_manifest(df, n_flightlines=3, seed=1)

        assert set(manifest["id"]) <= set(df["id"])

    def test_deterministic_given_same_seed(self):
        df = _patches_df([f"fl{i}" for i in range(20) for _ in range(5)])
        first = build_r2_manifest(df, n_flightlines=4, seed=99)
        second = build_r2_manifest(df, n_flightlines=4, seed=99)
        assert list(first["id"]) == list(second["id"])

    def test_index_is_reset_not_kept_as_a_column(self):
        df = _patches_df([f"fl{i}" for i in range(10) for _ in range(2)])
        manifest = build_r2_manifest(df, n_flightlines=3, seed=1)
        assert list(manifest.columns) == ["id", "name"]
        assert list(manifest.index) == list(range(len(manifest)))
