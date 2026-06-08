"""
Test suite for lgn_ibv.py
Run with:  python -m pytest test_lgn_ibv.py -v
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Import everything from the main module
from lgn_ibv import (
    LGN,
    calculate_optimal_p,
    double_convolve,
    estimate_depth,
    generate_ident_hash,
    linear_convolution,
    linear_disparity,
    normalize_disparity,
    open_norm,
    pixel_distance,
    resolve_existing_path,
    save_array_as_image,
    scale_disparity,
    unpack_filters,
)


# ===========================================================================
# Fixtures  (reusable objects shared across tests)
# ===========================================================================

@pytest.fixture
def small_lgn():
    """A small, fast LGN instance for structural tests (no wave generated)."""
    return LGN(width=16, p=0.5, r=1.0, t=1, trans=0.0, num_layers=2, make_wave=False)


@pytest.fixture
def wave_lgn():
    """A small LGN with a wave already grown, seeded for reproducibility."""
    np.random.seed(42)
    return LGN(width=16, p=0.5, r=1.0, t=1, trans=0.0, num_layers=2, make_wave=True)


@pytest.fixture
def tmp_dir():
    """Temporary directory that is cleaned up after each test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def dummy_image_path(tmp_dir):
    """Save a 32x32 random grayscale PNG and return its path."""
    arr = (np.random.rand(32, 32) * 255).astype(np.uint8)
    path = str(tmp_dir / "dummy.png")
    Image.fromarray(arr, mode="L").save(path)
    return path


# ===========================================================================
# 1. Utility functions
# ===========================================================================

class TestPixelDistance:
    def test_same_point_is_zero(self):
        assert pixel_distance(3, 4, 3, 4) == 0.0

    def test_known_3_4_5_triangle(self):
        assert pixel_distance(0, 0, 3, 4) == pytest.approx(5.0)

    def test_symmetry(self):
        assert pixel_distance(1, 2, 5, 6) == pytest.approx(pixel_distance(5, 6, 1, 2))

    def test_horizontal_distance(self):
        assert pixel_distance(0, 0, 10, 0) == pytest.approx(10.0)

    def test_returns_float(self):
        result = pixel_distance(0, 0, 1, 1)
        assert isinstance(result, (float, np.floating))


class TestCalculateOptimalP:
    def test_known_value(self):
        # p = t / ((pi * r^2 / 2) * (1 + a))
        # p = 1 / ((pi * 1 / 2) * (1 + 0)) = 2 / pi
        result = calculate_optimal_p(t=1, r=1.0, a=0.0)
        assert result == pytest.approx(2 / np.pi)

    def test_increases_with_t(self):
        """Higher threshold should require a higher recruitment probability."""
        p_low  = calculate_optimal_p(t=1, r=2.0, a=0.1)
        p_high = calculate_optimal_p(t=4, r=2.0, a=0.1)
        assert p_high > p_low

    def test_decreases_with_r(self):
        """Larger radius means more neighbours, so p can be lower."""
        p_small_r = calculate_optimal_p(t=2, r=1.0, a=0.1)
        p_large_r = calculate_optimal_p(t=2, r=3.0, a=0.1)
        assert p_large_r < p_small_r

    def test_decreases_with_a(self):
        """Higher inter-layer transfer reduces the p needed."""
        p_low_a  = calculate_optimal_p(t=2, r=2.0, a=0.1)
        p_high_a = calculate_optimal_p(t=2, r=2.0, a=0.9)
        assert p_high_a < p_low_a

    def test_returns_positive(self):
        assert calculate_optimal_p(t=3, r=2.0, a=0.3) > 0


class TestGenerateIdentHash:
    def test_returns_20_char_string(self):
        h = generate_ident_hash(100, 20, 10000, 9, 256, 0.5, 2.0, 3, 0.1, 1234.0)
        assert isinstance(h, str)
        assert len(h) == 20

    def test_same_inputs_same_hash(self):
        args = (100, 20, 10000, 9, 256, 0.5, 2.0, 3, 0.1, 1234.0)
        assert generate_ident_hash(*args) == generate_ident_hash(*args)

    def test_different_inputs_different_hash(self):
        h1 = generate_ident_hash(100, 20, 10000, 9, 256, 0.5, 2.0, 3, 0.1, 1234.0)
        h2 = generate_ident_hash(100, 20, 10000, 9, 256, 0.5, 2.0, 3, 0.2, 1234.0)
        assert h1 != h2

    def test_only_hex_characters(self):
        h = generate_ident_hash(1.0, 2.0, 3.0)
        assert all(c in "0123456789abcdef" for c in h)


class TestResolveExistingPath:
    def test_finds_first_match(self, tmp_dir):
        (tmp_dir / "b.png").touch()
        result = resolve_existing_path(tmp_dir, [Path("a.png"), Path("b.png")])
        assert result == tmp_dir / "b.png"

    def test_raises_when_none_exist(self, tmp_dir):
        with pytest.raises(FileNotFoundError):
            resolve_existing_path(tmp_dir, [Path("missing.png")])

    def test_prefers_earlier_candidate(self, tmp_dir):
        (tmp_dir / "first.png").touch()
        (tmp_dir / "second.png").touch()
        result = resolve_existing_path(tmp_dir, [Path("first.png"), Path("second.png")])
        assert result == tmp_dir / "first.png"


class TestOpenNorm:
    def test_output_mean_near_zero(self, dummy_image_path):
        norm = open_norm(dummy_image_path)
        assert abs(norm.mean()) < 1e-10

    def test_output_std_near_one(self, dummy_image_path):
        norm = open_norm(dummy_image_path)
        assert norm.std() == pytest.approx(1.0, abs=1e-6)

    def test_verbose_returns_tuple(self, dummy_image_path):
        result = open_norm(dummy_image_path, verbose=True)
        assert isinstance(result, tuple) and len(result) == 2

    def test_raw_shape_matches_norm_shape(self, dummy_image_path):
        raw, norm = open_norm(dummy_image_path, verbose=True)
        assert raw.shape == norm.shape

    def test_non_verbose_returns_array(self, dummy_image_path):
        result = open_norm(dummy_image_path, verbose=False)
        assert isinstance(result, np.ndarray)


class TestSaveArrayAsImage:
    def test_file_is_created(self, tmp_dir):
        arr = np.random.rand(16, 16)
        path = str(tmp_dir / "out.png")
        save_array_as_image(arr, path)
        assert os.path.exists(path)

    def test_saved_image_has_correct_size(self, tmp_dir):
        arr = np.random.rand(20, 30)
        path = str(tmp_dir / "out.png")
        save_array_as_image(arr, path)
        img = Image.open(path)
        assert img.size == (30, 20)  # PIL size is (width, height)

    def test_pixel_values_in_0_255(self, tmp_dir):
        arr = np.random.rand(16, 16) * 100 - 50  # negative values allowed
        path = str(tmp_dir / "out.png")
        save_array_as_image(arr, path)
        loaded = np.array(Image.open(path))
        assert loaded.min() >= 0 and loaded.max() <= 255


# ===========================================================================
# 2. LGN class
# ===========================================================================

class TestLGNInit:
    def test_parameters_stored_correctly(self, small_lgn):
        assert small_lgn.width == 16
        assert small_lgn.p == 0.5
        assert small_lgn.r == 1.0
        assert small_lgn.t == 1
        assert small_lgn.trans == 0.0
        assert small_lgn.num_layers == 2

    def test_make_wave_false_skips_reset(self):
        """With make_wave=False the active grid should not exist yet."""
        lgn = LGN(width=8, make_wave=False)
        assert not hasattr(lgn, "active")

    def test_single_layer(self):
        lgn = LGN(width=8, p=0.5, num_layers=1, make_wave=False)
        assert lgn.num_layers == 1


class TestLGNResetWave:
    def test_active_array_shape(self, wave_lgn):
        assert wave_lgn.active.shape == (2, 16, 16)

    def test_active_neighbors_shape(self, wave_lgn):
        assert wave_lgn.active_neighbors.shape == (2, 16, 16)

    def test_recruitable_shape(self, wave_lgn):
        assert wave_lgn.recruitable.shape == (2, 16, 16)

    def test_fraction_active_above_threshold(self, wave_lgn):
        """Wave should have grown past the 20 % threshold before stopping."""
        assert wave_lgn.fraction_active() >= 0.20

    def test_tot_active_positive(self, wave_lgn):
        assert wave_lgn.tot_active > 0

    def test_tot_recruitable_positive(self, wave_lgn):
        assert wave_lgn.tot_recruitable > 0


class TestLGNFractionActive:
    def test_zero_when_nothing_active(self, small_lgn):
        small_lgn.recruitable = np.ones((2, 16, 16), bool)
        small_lgn.tot_recruitable = int(small_lgn.recruitable.sum())
        small_lgn.tot_recruitable_active = 0
        assert small_lgn.fraction_active() == 0.0

    def test_nan_when_no_recruitable_cells(self, small_lgn):
        small_lgn.tot_recruitable = 0
        assert np.isnan(small_lgn.fraction_active())

    def test_one_when_fully_active(self, small_lgn):
        small_lgn.tot_recruitable = 100
        small_lgn.tot_recruitable_active = 100
        assert small_lgn.fraction_active() == 1.0


class TestLGNCorrelation:
    def test_returns_float(self, wave_lgn):
        result = wave_lgn.correlation()
        assert isinstance(result, (float, np.floating))

    def test_in_range_minus1_to_1(self, wave_lgn):
        c = wave_lgn.correlation()
        assert -1.0 <= c <= 1.0

    def test_monocular_returns_zero(self):
        lgn = LGN(width=8, p=0.5, num_layers=1, make_wave=False)
        assert lgn.correlation() == 0.0

    def test_identical_layers_gives_high_correlation(self):
        """Force both layers to be identical; correlation should be ~1."""
        lgn = LGN(width=8, p=0.5, num_layers=2, make_wave=False)
        lgn.active = np.zeros((2, 8, 8), bool)
        # Activate the same cells in both layers
        lgn.active[0, 1:5, 1:5] = True
        lgn.active[1, 1:5, 1:5] = True
        lgn.tot_recruitable = 32
        lgn.tot_recruitable_active = 16
        c = lgn.correlation()
        assert c == pytest.approx(1.0, abs=1e-6)


class TestLGNMakeImgMat:
    def test_output_shape(self, wave_lgn):
        mat = wave_lgn.make_img_mat()
        assert mat.shape == (2, 16, 16)

    def test_values_are_binary(self, wave_lgn):
        mat = wave_lgn.make_img_mat()
        unique = np.unique(mat)
        assert set(unique).issubset({0.0, 1.0})

    def test_active_cells_appear_in_matrix(self, wave_lgn):
        """Every active cell in layer 0 should be a 1 in the image matrix."""
        mat = wave_lgn.make_img_mat()
        active_positions = np.argwhere(wave_lgn.active[0, :-1, :-1])
        for x, y in active_positions:
            assert mat[0, x, y] == 1.0


# ===========================================================================
# 3. Disparity functions
# ===========================================================================

class TestLinearConvolution:
    def test_output_length(self):
        center = np.random.rand(4, 8)
        slide  = np.random.rand(4, 8)
        result = linear_convolution(center, slide)
        assert len(result) == 8 * 2

    def test_all_non_negative(self):
        """Function returns absolute values, so nothing should be negative."""
        center = np.random.rand(4, 8)
        slide  = np.random.rand(4, 8)
        result = linear_convolution(center, slide)
        assert np.all(result >= 0)

    def test_mismatched_shapes_returns_none(self):
        center = np.random.rand(4, 8)
        slide  = np.random.rand(3, 8)
        assert linear_convolution(center, slide) is None

    def test_zero_slide_gives_zero_output(self):
        center = np.random.rand(4, 8)
        slide  = np.zeros((4, 8))
        result = linear_convolution(center, slide)
        assert np.all(result == 0)


class TestLinearDisparity:
    def test_output_shape(self):
        n_filters, dim = 5, 4
        first_eye  = np.random.rand(n_filters, dim, dim)
        second_eye = np.random.rand(n_filters, dim, dim)
        result = linear_disparity(first_eye, second_eye)
        assert result.shape == (n_filters, dim * 2)

    def test_all_non_negative(self):
        first_eye  = np.random.rand(3, 4, 4)
        second_eye = np.random.rand(3, 4, 4)
        assert np.all(linear_disparity(first_eye, second_eye) >= 0)


class TestNormalizeDisparity:
    def test_output_shape_preserved(self):
        dm = np.random.rand(5, 10) + 0.1  # avoid zeros
        assert normalize_disparity(dm).shape == dm.shape

    def test_column_mean_near_one(self):
        """After normalisation each column mean should be ~1."""
        dm = np.random.rand(10, 8) + 0.1
        normed = normalize_disparity(dm)
        col_means = np.nanmean(normed, axis=0)
        np.testing.assert_allclose(col_means, np.ones(8), atol=1e-10)


# ===========================================================================
# 4. double_convolve
# ===========================================================================

class TestDoubleConvolve:
    def test_output_shape(self):
        img    = np.random.rand(20, 30)
        filt   = np.random.rand(5, 5)
        pd     = 3
        result = double_convolve(filt, filt, img, pd)
        # Output is cropped by pupillary_distance on each side
        assert result.shape[0] == img.shape[0]

    def test_no_negative_values(self):
        """Negatives are zeroed, so result must be >= 0."""
        img  = np.random.rand(20, 20)
        filt = np.random.rand(4, 4)
        result = double_convolve(filt, filt, img, 2)
        assert np.all(result >= 0)

    def test_zero_image_gives_zero_output(self):
        img  = np.zeros((20, 20))
        filt = np.random.rand(4, 4)
        result = double_convolve(filt, filt, img, 2)
        assert np.all(result == 0)


# ===========================================================================
# 5. scale_disparity
# ===========================================================================

class TestScaleDisparity:
    def test_output_shape(self):
        activity = np.random.rand(10, 10)
        disparity = np.random.rand(6)
        result = scale_disparity(activity, disparity)
        assert result.shape == (10, 10, 6)

    def test_zero_activity_gives_zero_output(self):
        activity  = np.zeros((5, 5))
        disparity = np.random.rand(4)
        result = scale_disparity(activity, disparity)
        assert np.all(result == 0)

    def test_unit_activity_preserves_disparity(self):
        """All-ones activity should leave the disparity vector unchanged."""
        activity  = np.ones((3, 3))
        disparity = np.array([1.0, 2.0, 3.0])
        result = scale_disparity(activity, disparity)
        for x in range(3):
            for y in range(3):
                np.testing.assert_allclose(result[x, y], disparity)


# ===========================================================================
# 6. estimate_depth
# ===========================================================================

class TestEstimateDepth:
    def test_output_shape(self):
        # activity shape: (height, width, n_disparities)
        activity = np.random.rand(8, 8, 10)
        depth = estimate_depth(activity)
        assert depth.shape == (8, 8)

    def test_all_non_negative(self):
        activity = np.random.rand(8, 8, 10)
        assert np.all(estimate_depth(activity) >= 0)

    def test_peak_at_centre_gives_zero_depth(self):
        """
        If the argmax is exactly at the midpoint of the disparity axis,
        the depth value should be 0.
        """
        n_disp = 10
        activity = np.zeros((4, 4, n_disp))
        centre = n_disp // 2
        activity[:, :, centre] = 1.0  # peak at centre
        depth = estimate_depth(activity)
        assert np.all(depth == 0)

    def test_peak_offset_gives_correct_depth(self):
        """Peak 3 positions away from centre → depth == 3."""
        n_disp = 10
        activity = np.zeros((4, 4, n_disp))
        centre = n_disp // 2          # 5
        activity[:, :, centre + 3] = 1.0   # peak at index 8
        depth = estimate_depth(activity)
        assert np.all(depth == 3)


# ===========================================================================
# 7. unpack_filters
# ===========================================================================

class TestUnpackFilters:
    def test_output_shapes(self, tmp_dir):
        n_filters, patch_size = 6, 4
        # Each filter is (patch_size^2 * 2) wide — left half + right half
        filters = np.random.rand(n_filters, patch_size ** 2 * 2)
        first, second = unpack_filters(filters, "testhash", str(tmp_dir))
        assert first.shape  == (n_filters, patch_size, patch_size)
        assert second.shape == (n_filters, patch_size, patch_size)

    def test_first_half_matches_first_eye(self, tmp_dir):
        """The left filter values should equal the first half of the raw filter."""
        filters = np.arange(32, dtype=float).reshape(2, 16)
        first, _ = unpack_filters(filters, "testhash2", str(tmp_dir))
        np.testing.assert_allclose(first[0].flatten(), filters[0, :8])

    def test_second_half_matches_second_eye(self, tmp_dir):
        filters = np.arange(32, dtype=float).reshape(2, 16)
        _, second = unpack_filters(filters, "testhash3", str(tmp_dir))
        np.testing.assert_allclose(second[0].flatten(), filters[0, 8:])

    def test_filter_images_saved_to_disk(self, tmp_dir):
        filters = np.random.rand(4, 18)   # 3x3 patches → 9*2 = 18
        unpack_filters(filters, "savehash", str(tmp_dir))
        r_folder = tmp_dir / "images" / "filters" / "savehash" / "r"
        assert r_folder.exists()