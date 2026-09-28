import numpy as np
import pandas as pd

from data import patch_cache


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _row(folder="scene", col=0, row=0, width=8, height=8) -> dict:
    return {
        "folder": folder,
        "window_col_off": col,
        "window_row_off": row,
        "window_width": width,
        "window_height": height,
        "id": f"{folder}-{col}-{row}",
    }


class TestFingerprint:
    def test_same_dataframe_gives_the_same_fingerprint(self):
        df = _df([_row(), _row(col=8)])

        assert patch_cache.fingerprint(df) == patch_cache.fingerprint(df.copy())

    def test_different_row_order_gives_a_different_fingerprint(self):
        df = _df([_row(), _row(col=8)])
        reordered = df.iloc[::-1].reset_index(drop=True)

        assert patch_cache.fingerprint(df) != patch_cache.fingerprint(reordered)

    def test_different_row_count_gives_a_different_fingerprint(self):
        df = _df([_row(), _row(col=8)])
        shorter = df.iloc[:1].reset_index(drop=True)

        assert patch_cache.fingerprint(df) != patch_cache.fingerprint(shorter)

    def test_different_window_content_gives_a_different_fingerprint(self):
        df = _df([_row(col=0)])
        moved = _df([_row(col=8)])

        assert patch_cache.fingerprint(df) != patch_cache.fingerprint(moved)

    def test_an_unrelated_extra_column_does_not_change_the_fingerprint(self):
        df = _df([_row()])
        with_extra = df.copy()
        with_extra["has_plume"] = [True]

        assert patch_cache.fingerprint(df) == patch_cache.fingerprint(with_extra)

    def test_non_default_index_labels_do_not_change_the_fingerprint(self):

        df = _df([_row(), _row(col=8)])
        relabeled = df.set_axis([100, 200])

        assert patch_cache.fingerprint(df) == patch_cache.fingerprint(relabeled)


class TestWriteAndIsValid:
    def test_a_freshly_written_cache_is_valid_for_the_same_dataframe(self, tmp_path):
        df = _df([_row(), _row(col=8)])
        inputs = np.zeros((2, 4, 8, 8), dtype=np.float16)
        outputs = np.zeros((2, 1, 8, 8), dtype=np.uint8)
        cache_dir = tmp_path / "cache"

        patch_cache.write(cache_dir, df, inputs, outputs)

        assert patch_cache.is_valid(cache_dir, df) is True

    def test_is_invalid_for_a_dataframe_with_different_content(self, tmp_path):
        df = _df([_row(), _row(col=8)])
        other_df = _df([_row(col=16), _row(col=24)])
        inputs = np.zeros((2, 4, 8, 8), dtype=np.float16)
        outputs = np.zeros((2, 1, 8, 8), dtype=np.uint8)
        cache_dir = tmp_path / "cache"

        patch_cache.write(cache_dir, df, inputs, outputs)

        assert patch_cache.is_valid(cache_dir, other_df) is False

    def test_is_invalid_when_the_directory_does_not_exist(self, tmp_path):
        df = _df([_row()])

        assert patch_cache.is_valid(tmp_path / "missing", df) is False

    def test_is_invalid_when_an_array_file_is_missing(self, tmp_path):
        df = _df([_row()])
        inputs = np.zeros((1, 4, 8, 8), dtype=np.float16)
        outputs = np.zeros((1, 1, 8, 8), dtype=np.uint8)
        cache_dir = tmp_path / "cache"
        patch_cache.write(cache_dir, df, inputs, outputs)
        (cache_dir / "outputs.npy").unlink()

        assert patch_cache.is_valid(cache_dir, df) is False

    def test_writing_twice_to_an_already_existing_cache_dir_does_not_raise(self, tmp_path):

        df = _df([_row(), _row(col=8)])
        inputs = np.zeros((2, 4, 8, 8), dtype=np.float16)
        outputs = np.zeros((2, 1, 8, 8), dtype=np.uint8)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)

        patch_cache.write(cache_dir, df, inputs, outputs)

        assert patch_cache.is_valid(cache_dir, df) is True


class TestResolveCacheDir:
    def test_returns_the_split_dir_when_it_matches(self, tmp_path):
        df = _df([_row(), _row(col=8)])
        inputs = np.zeros((2, 4, 8, 8), dtype=np.float16)
        outputs = np.zeros((2, 1, 8, 8), dtype=np.uint8)
        patch_cache.write(tmp_path / "starcop_raw" / "train", df, inputs, outputs)

        resolved = patch_cache.resolve_cache_dir(tmp_path, "starcop_raw", "train", df)

        assert resolved == tmp_path / "starcop_raw" / "train"

    def test_returns_none_when_no_cache_was_built(self, tmp_path):
        df = _df([_row()])

        assert patch_cache.resolve_cache_dir(tmp_path, "starcop_raw", "train", df) is None

    def test_returns_none_when_the_dataframe_does_not_match(self, tmp_path):

        built_for = _df([_row(), _row(col=8)])
        inputs = np.zeros((2, 4, 8, 8), dtype=np.float16)
        outputs = np.zeros((2, 1, 8, 8), dtype=np.uint8)
        patch_cache.write(tmp_path / "starcop_raw" / "val", built_for, inputs, outputs)

        sliced = built_for.iloc[:1].reset_index(drop=True)

        assert patch_cache.resolve_cache_dir(tmp_path, "starcop_raw", "val", sliced) is None


class TestLoad:
    def test_round_trips_values_through_the_expected_dtypes(self, tmp_path):
        df = _df([_row(), _row(col=8)])
        inputs = np.array([np.full((4, 8, 8), 0.5), np.full((4, 8, 8), 1.25)], dtype=np.float16)
        outputs = np.array([np.zeros((1, 8, 8)), np.ones((1, 8, 8))], dtype=np.uint8)
        cache_dir = tmp_path / "cache"
        patch_cache.write(cache_dir, df, inputs, outputs)

        loaded_inputs, loaded_outputs = patch_cache.load(cache_dir)

        assert loaded_inputs.dtype == np.float16
        assert loaded_outputs.dtype == np.uint8
        np.testing.assert_array_equal(loaded_inputs[1], inputs[1])
        np.testing.assert_array_equal(loaded_outputs[1], outputs[1])

    def test_returns_read_only_memmaps_not_fully_loaded_arrays(self, tmp_path):

        df = _df([_row(), _row(col=8)])
        inputs = np.zeros((2, 4, 8, 8), dtype=np.float16)
        outputs = np.zeros((2, 1, 8, 8), dtype=np.uint8)
        cache_dir = tmp_path / "cache"
        patch_cache.write(cache_dir, df, inputs, outputs)

        loaded_inputs, loaded_outputs = patch_cache.load(cache_dir)

        assert isinstance(loaded_inputs, np.memmap)
        assert isinstance(loaded_outputs, np.memmap)
