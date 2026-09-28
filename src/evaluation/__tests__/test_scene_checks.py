import numpy as np
import pytest

from evaluation.scene_checks import (
    _axis_cover,
    compare_pooled_counts,
    expected_patch_positive_pixels,
    patch_cover_counts,
)


class TestPatchCoverCounts:
    def test_the_real_tiling_covers_the_centre_of_a_scene_four_by_four_times(self):
        cover = patch_cover_counts()

        assert cover.shape == (512, 512)
        assert cover[0, 0] == 1
        assert cover[64, 64] == 4
        assert cover[256, 256] == 4
        assert cover.max() == 4

    def test_every_pixel_is_covered_and_the_total_is_forty_nine_patches(self):
        cover = patch_cover_counts()

        assert cover.min() == 1
        assert cover.sum() == 49 * 128 * 128

    def test_a_small_tiling_by_hand(self):

        cover = patch_cover_counts(scene_size=8, patch_size=4, stride=2)

        assert cover[0].tolist() == [1, 1, 2, 2, 2, 2, 1, 1]
        assert cover[3, 3] == 4
        assert cover[0, 0] == 1

    def test_a_scene_the_windows_do_not_tile_evenly_is_rejected(self):
        with pytest.raises(ValueError, match="do not tile"):
            patch_cover_counts(scene_size=10, patch_size=4, stride=4)


class TestAxisCover:
    def test_counts_the_windows_containing_each_coordinate(self):
        cover = _axis_cover(8, 4, 2)

        assert cover.tolist() == [1, 1, 2, 2, 2, 2, 1, 1]
        assert cover.dtype == np.int64

    def test_a_stride_of_one_places_a_window_at_every_offset_and_no_further(self):

        assert _axis_cover(6, 3, 1).tolist() == [1, 2, 3, 3, 2, 1]

    def test_the_fit_is_checked_on_the_room_left_after_the_first_window(self):

        assert _axis_cover(10, 4, 3).tolist() == [1, 1, 1, 2, 1, 1, 2, 1, 1, 1]

    def test_a_scene_the_windows_do_not_tile_evenly_is_rejected(self):
        with pytest.raises(
            ValueError, match="windows of 4 with stride 4 do not tile a scene of 10"
        ):
            _axis_cover(10, 4, 4)


class TestExpectedPatchPositivePixels:
    def test_weights_each_positive_pixel_by_the_number_of_patches_covering_it(self):
        cover = patch_cover_counts(scene_size=8, patch_size=4, stride=2)
        label = np.zeros((8, 8))
        label[0, 0] = 1
        label[3, 3] = 1

        assert expected_patch_positive_pixels(label, cover) == 1 + 4

    def test_a_scene_without_positives_expects_none(self):
        cover = patch_cover_counts(scene_size=8, patch_size=4, stride=2)

        assert expected_patch_positive_pixels(np.zeros((8, 8)), cover) == 0

    def test_any_positive_value_counts_as_a_positive_pixel(self):
        cover = patch_cover_counts(scene_size=8, patch_size=4, stride=2)
        label = np.zeros((8, 8))
        label[3, 3] = 0.7

        assert expected_patch_positive_pixels(label, cover) == 4

    def test_the_label_must_match_the_cover_shape(self):
        with pytest.raises(ValueError, match="shape"):
            expected_patch_positive_pixels(np.zeros((4, 4)), patch_cover_counts(8, 4, 2))


class TestComparePooledCounts:
    def test_identical_counts_have_no_differences(self):
        report = compare_pooled_counts([5, 3, 2, 90], [5, 3, 2, 90])

        assert report == {"identical": True, "differences": {"tp": 0, "fp": 0, "fn": 0, "tn": 0}}

    def test_reports_each_count_that_differs_as_first_minus_second(self):
        report = compare_pooled_counts([5, 4, 2, 90], [5, 3, 3, 90])

        assert report["identical"] is False
        assert report["differences"] == {"tp": 0, "fp": 1, "fn": -1, "tn": 0}

    def test_needs_four_counts_each(self):
        with pytest.raises(
            ValueError, match="^both inputs need exactly four counts: tp, fp, fn, tn$"
        ):
            compare_pooled_counts([1, 2, 3], [1, 2, 3, 4])
