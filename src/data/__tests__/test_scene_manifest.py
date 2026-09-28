from pathlib import Path

import pandas as pd
import pytest

import data.scene_manifest as scene_manifest
from data.scene_manifest import (
    PATCHES_PER_SCENE,
    SCENE_SIZE,
    STRONG_QPLUME_THRESHOLD,
    assign_bucket,
    build_scene_manifest,
    load_scene_manifest,
)


def _labels(*rows):
    """Scene-level label rows: (id, qplume, has_plume)."""
    return pd.DataFrame(
        [{"id": i, "name": i.split("_")[0], "qplume": q, "has_plume": h} for i, q, h in rows]
    )


def _patches(scene_ids, n=PATCHES_PER_SCENE, patch_has_plume=False):
    """`n` patch rows per parent scene, each with a *patch-level* `has_plume` on purpose."""
    rows = []
    for scene_id in scene_ids:
        for k in range(n):
            rows.append(
                {
                    "id": f"{scene_id}_patch{k}",
                    "id_original": scene_id,
                    "folder": f"data/processed/x/selected/{scene_id}",
                    "window_col_off": 64 * (k % 7),
                    "window_row_off": 64 * (k // 7),
                    "window_width": 128,
                    "window_height": 128,
                    "has_plume": patch_has_plume,
                }
            )
    return pd.DataFrame(rows)


class TestAssignBucket:
    @pytest.mark.parametrize(
        ("has_plume", "qplume", "expected"),
        [
            (True, 1000.0, "strong"),
            (True, 5000.0, "strong"),
            (True, 999.99, "weak"),
            (True, 0.01, "weak"),
            (False, 0.0, "plume_free"),
        ],
    )
    def test_buckets_follow_the_paper_thresholds(self, has_plume, qplume, expected):
        assert assign_bucket(has_plume, qplume) == expected

    def test_the_strong_threshold_is_one_thousand_kg_per_hour(self):
        assert STRONG_QPLUME_THRESHOLD == 1000.0

    @pytest.mark.parametrize(
        ("has_plume", "qplume"),
        [(True, 0.0), (True, -1.0), (True, float("nan")), (False, 5.0), (False, float("nan"))],
    )
    def test_a_label_that_contradicts_itself_is_rejected(self, has_plume, qplume):
        with pytest.raises(ValueError, match="inconsistent"):
            assign_bucket(has_plume, qplume)


class TestBuildSceneManifest:
    def test_has_one_row_per_parent_scene(self):
        labels = _labels(("a_r0", 0.0, False), ("b_r0", 1500.0, True))

        manifest = build_scene_manifest(labels, _patches(["a_r0", "b_r0"]))

        assert list(manifest["scene_id"]) == ["a_r0", "b_r0"]
        assert list(manifest["n_patches"]) == [PATCHES_PER_SCENE, PATCHES_PER_SCENE]

    def test_takes_has_plume_and_qplume_from_the_labels_not_from_the_patch_column(self):

        labels = _labels(("a_r0", 0.0, False), ("b_r0", 300.0, True))
        patches = pd.concat([_patches(["a_r0"], patch_has_plume=True), _patches(["b_r0"])])

        manifest = build_scene_manifest(labels, patches).set_index("scene_id")

        assert not manifest.loc["a_r0", "has_plume"]
        assert manifest.loc["b_r0", "has_plume"]
        assert manifest.loc["b_r0", "qplume"] == 300.0

    def test_assigns_the_bucket_from_the_scene_label(self):
        labels = _labels(("a_r0", 0.0, False), ("b_r0", 1000.0, True), ("c_r0", 999.0, True))

        manifest = build_scene_manifest(labels, _patches(["a_r0", "b_r0", "c_r0"]))

        assert list(manifest["bucket"]) == ["plume_free", "strong", "weak"]

    def test_takes_the_processed_scene_folder_from_the_child_patches(self):
        labels = _labels(("a_r0", 0.0, False))

        manifest = build_scene_manifest(labels, _patches(["a_r0"]))

        assert manifest.loc[0, "folder"] == "data/processed/x/selected/a_r0"

    def test_each_scene_is_one_full_size_window_at_the_origin(self):
        labels = _labels(("a_r0", 0.0, False))

        row = build_scene_manifest(labels, _patches(["a_r0"])).iloc[0]

        assert SCENE_SIZE == 512
        assert (row["window_col_off"], row["window_row_off"]) == (0, 0)
        assert (row["window_width"], row["window_height"]) == (512, 512)

    def test_ignores_label_rows_that_have_no_patches(self):

        labels = _labels(("a_r0", 0.0, False), ("unused_r0", 800.0, True))

        manifest = build_scene_manifest(labels, _patches(["a_r0"]))

        assert list(manifest["scene_id"]) == ["a_r0"]

    def test_is_sorted_by_scene_id_with_a_fresh_index(self):
        labels = _labels(("b_r0", 0.0, False), ("a_r0", 0.0, False))

        manifest = build_scene_manifest(labels, _patches(["b_r0", "a_r0"]))

        assert list(manifest["scene_id"]) == ["a_r0", "b_r0"]
        assert list(manifest.index) == [0, 1]

    def test_raises_when_a_parent_scene_has_no_label(self):
        labels = _labels(("a_r0", 0.0, False))

        with pytest.raises(ValueError, match="no scene label.*b_r0"):
            build_scene_manifest(labels, _patches(["a_r0", "b_r0"]))

    def test_raises_when_a_scene_does_not_have_the_expected_number_of_patches(self):
        labels = _labels(("a_r0", 0.0, False))

        with pytest.raises(ValueError, match="a_r0.*48|48.*a_r0"):
            build_scene_manifest(labels, _patches(["a_r0"], n=48))

    def test_the_expected_patch_count_can_be_overridden(self):
        labels = _labels(("a_r0", 0.0, False))

        manifest = build_scene_manifest(labels, _patches(["a_r0"], n=4), patches_per_scene=4)

        assert manifest.loc[0, "n_patches"] == 4

    def test_raises_when_the_patches_of_one_scene_point_at_different_folders(self):
        labels = _labels(("a_r0", 0.0, False))
        patches = _patches(["a_r0"])
        patches.loc[0, "folder"] = "data/processed/x/selected/elsewhere"

        with pytest.raises(ValueError, match="more than one folder.*a_r0"):
            build_scene_manifest(labels, patches)

    def test_raises_on_a_contradictory_scene_label(self):
        labels = _labels(("a_r0", 0.0, True))

        with pytest.raises(ValueError, match="inconsistent"):
            build_scene_manifest(labels, _patches(["a_r0"]))

    def test_raises_when_the_same_scene_is_labelled_twice(self):
        labels = _labels(("a_r0", 0.0, False), ("a_r0", 0.0, False))

        with pytest.raises(ValueError, match="duplicate.*a_r0"):
            build_scene_manifest(labels, _patches(["a_r0"]))

    def test_the_missing_label_error_names_at_most_five_scenes(self):
        scene_ids = [f"s{i}_r0" for i in range(6)]

        with pytest.raises(ValueError, match="6 scene") as error:
            build_scene_manifest(_labels(("z_r0", 0.0, False)), _patches(scene_ids))

        assert "s4_r0" in str(error.value)
        assert "s5_r0" not in str(error.value)

    def test_the_duplicate_label_error_names_at_most_five_scenes(self):
        scene_ids = [f"d{i}_r0" for i in range(6)]
        labels = _labels(*[(scene_id, 0.0, False) for scene_id in scene_ids * 2])

        with pytest.raises(ValueError, match="duplicate") as error:
            build_scene_manifest(labels, _patches(["a_r0"]))

        assert "d4_r0" in str(error.value)
        assert "d5_r0" not in str(error.value)

    def test_the_split_folder_error_names_at_most_five_scenes(self):
        scene_ids = [f"f{i}_r0" for i in range(6)]
        labels = _labels(*[(scene_id, 0.0, False) for scene_id in scene_ids])
        patches = _patches(scene_ids)
        for scene_id in scene_ids:
            first_patch = patches.index[patches["id_original"] == scene_id][0]
            patches.loc[first_patch, "folder"] = "data/processed/x/selected/elsewhere"

        with pytest.raises(ValueError, match="more than one folder") as error:
            build_scene_manifest(labels, patches)

        assert "f4_r0" in str(error.value)
        assert "f5_r0" not in str(error.value)

    def test_does_not_modify_its_inputs(self):
        labels = _labels(("a_r0", 0.0, False))
        patches = _patches(["a_r0"])
        labels_before, patches_before = labels.copy(), patches.copy()

        build_scene_manifest(labels, patches)

        pd.testing.assert_frame_equal(labels, labels_before)
        pd.testing.assert_frame_equal(patches, patches_before)


class TestLoadSceneManifest:
    def test_reads_the_two_csv_files_from_disk(self, tmp_path):
        labels_csv, patches_csv = tmp_path / "labels.csv", tmp_path / "patches.csv"
        _labels(("a_r0", 0.0, False), ("b_r0", 2000.0, True)).to_csv(labels_csv, index=False)
        _patches(["a_r0", "b_r0"]).to_csv(patches_csv, index=False)

        manifest = load_scene_manifest(labels_csv, patches_csv)

        assert list(manifest["bucket"]) == ["plume_free", "strong"]
        assert manifest["has_plume"].dtype == bool

    def test_passes_keyword_arguments_through_to_the_builder(self, tmp_path):
        labels_csv, patches_csv = tmp_path / "labels.csv", tmp_path / "patches.csv"
        _labels(("a_r0", 0.0, False)).to_csv(labels_csv, index=False)
        _patches(["a_r0"], n=4).to_csv(patches_csv, index=False)

        manifest = load_scene_manifest(labels_csv, patches_csv, patches_per_scene=4)

        assert manifest.loc[0, "n_patches"] == 4

    def test_reads_only_the_columns_needed_from_each_csv(self, tmp_path, monkeypatch):
        labels_csv, patches_csv = tmp_path / "labels.csv", tmp_path / "patches.csv"
        labels_csv.write_text("id,qplume,has_plume,unused\\na_r0,0,False,ignored\\n")
        patches_csv.write_text("id_original,folder,unused\\na_r0,folder,ignored\\n")
        calls = []
        read_csv = pd.read_csv

        def recording_read_csv(path, **kwargs):
            calls.append(kwargs.get("usecols"))
            return read_csv(path, **kwargs)

        monkeypatch.setattr(scene_manifest.pd, "read_csv", recording_read_csv)

        load_scene_manifest(labels_csv, patches_csv, patches_per_scene=1)

        assert calls == [["id", "qplume", "has_plume"], ["id_original", "folder"]]


def _repo_root_with_data():
    for parent in Path(__file__).resolve().parents:
        if (parent / "data" / "starcop_raw" / "test.csv").is_file():
            return parent
    return None


REPO_ROOT = _repo_root_with_data()


@pytest.mark.skipif(REPO_ROOT is None, reason="real STARCOP data is not available here")
class TestRealData:
    def test_the_raw_test_split_reproduces_the_papers_342_scenes(self):
        manifest = load_scene_manifest(
            REPO_ROOT / "data/starcop_raw/test.csv",
            REPO_ROOT / "data/processed/starcop_raw/patches/test_tiled_128_128.csv",
        )

        assert len(manifest) == 342
        assert manifest["has_plume"].value_counts().to_dict() == {False: 176, True: 166}
        assert manifest["bucket"].value_counts().to_dict() == {
            "plume_free": 176,
            "weak": 109,
            "strong": 57,
        }

    def test_the_raw_validation_split_has_543_scenes(self):
        manifest = load_scene_manifest(
            REPO_ROOT / "data/starcop_raw/train.csv",
            REPO_ROOT / "data/processed/starcop_raw/patches/val_tiled_128_128.csv",
        )

        assert len(manifest) == 543
        assert manifest["n_patches"].eq(PATCHES_PER_SCENE).all()

    def test_the_mini_test_split_has_9_scenes(self):
        manifest = load_scene_manifest(
            REPO_ROOT / "data/processed/starcop_mini/splits/test.csv",
            REPO_ROOT / "data/processed/starcop_mini/patches/test_tiled_128_128.csv",
        )

        assert len(manifest) == 9

    def test_every_scene_folder_exists_and_holds_the_expected_bands(self):
        manifest = load_scene_manifest(
            REPO_ROOT / "data/starcop_raw/test.csv",
            REPO_ROOT / "data/processed/starcop_raw/patches/test_tiled_128_128.csv",
        )

        for folder in manifest["folder"]:
            scene_dir = REPO_ROOT / folder
            for band in ("mag1c", "TOA_AVIRIS_640nm", "labelbinary"):
                assert (scene_dir / f"{band}.tif").is_file(), scene_dir

    def test_a_scene_raster_really_is_512_by_512(self):
        import rasterio

        manifest = load_scene_manifest(
            REPO_ROOT / "data/starcop_raw/test.csv",
            REPO_ROOT / "data/processed/starcop_raw/patches/test_tiled_128_128.csv",
        )

        with rasterio.open(REPO_ROOT / manifest.loc[0, "folder"] / "labelbinary.tif") as raster:
            assert raster.shape == (SCENE_SIZE, SCENE_SIZE)
