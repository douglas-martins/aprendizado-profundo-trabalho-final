import numpy as np

from data.preprocessing import BAND_NORMALIZATION, normalize_band


class TestBandNormalizationTable:
    def test_matches_the_documented_contract_exactly(self):
        assert BAND_NORMALIZATION == {
            "mag1c": {"offset": 0, "factor": 1750, "clip": (0, 2)},
            "TOA_AVIRIS_640nm": {"offset": 0, "factor": 60, "clip": (0, 2)},
            "TOA_AVIRIS_550nm": {"offset": 0, "factor": 60, "clip": (0, 2)},
            "TOA_AVIRIS_460nm": {"offset": 0, "factor": 60, "clip": (0, 2)},
        }


class TestNormalizeBand:
    def test_scales_by_offset_and_factor(self):
        array = np.array([[0.0, 875.0, 1750.0]], dtype="float32")
        result = normalize_band(array, offset=0, factor=1750, clip=(0, 2))
        np.testing.assert_allclose(result, [[0.0, 0.5, 1.0]])

    def test_subtracts_offset_before_scaling(self):
        array = np.array([[10.0, 70.0]], dtype="float32")
        result = normalize_band(array, offset=10, factor=60, clip=(0, 2))
        np.testing.assert_allclose(result, [[0.0, 1.0]])

    def test_clips_to_the_documented_range(self):
        array = np.array([[-100.0, 0.0, 60.0, 500.0]], dtype="float32")
        result = normalize_band(array, offset=0, factor=60, clip=(0, 2))
        np.testing.assert_allclose(result, [[0.0, 0.0, 1.0, 2.0]])

    def test_mag1c_sentinel_value_is_clamped_not_propagated(self):

        array = np.array([[100000.0]], dtype="float32")
        result = normalize_band(array, offset=0, factor=1750, clip=(0, 2))
        assert result[0, 0] == 2.0

    def test_output_dtype_is_float32(self):
        array = np.array([[0.0, 1.0]], dtype="float64")
        result = normalize_band(array, offset=0, factor=1, clip=(0, 2))
        assert result.dtype == np.float32

    def test_input_is_cast_to_float32_before_subtracting_offset(self):

        array = np.array([[16777217.0]], dtype="float64")
        result = normalize_band(array, offset=16777216, factor=1, clip=(0, 1_000_000))
        assert result[0, 0] == 0.0
