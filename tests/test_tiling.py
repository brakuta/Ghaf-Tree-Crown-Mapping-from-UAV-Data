"""Tests for the sliding-window geometry and overlap-add blending.

These need only NumPy, so they run anywhere -- including CI without a GPU,
mmcv, or the dataset.
"""

import numpy as np
import pytest

from ghaf.inference.tiling import Accumulator, Window, gaussian_weights, plan_windows

# --------------------------------------------------------------------------
# plan_windows
# --------------------------------------------------------------------------

@pytest.mark.parametrize('width,height', [
    (1024, 1024), (1025, 1024), (2048, 1536), (3000, 1700), (1, 1), (5000, 37),
])
def test_windows_cover_every_pixel(width, height):
    """No pixel may be missed -- that is the property the whole scheme rests on."""
    covered = np.zeros((height, width), bool)
    for w in plan_windows(width, height, tile=1024, overlap=512):
        covered[w.row_off:w.row_off + w.height, w.col_off:w.col_off + w.width] = True
    assert covered.all(), f'{(~covered).sum()} pixels uncovered'


@pytest.mark.parametrize('width,height', [(1024, 1024), (3000, 1700), (777, 4096)])
def test_windows_stay_inside_the_raster(width, height):
    for w in plan_windows(width, height, tile=1024, overlap=512):
        assert w.col_off >= 0 and w.col_off + w.width <= width
        assert w.row_off >= 0 and w.row_off + w.height <= height
        assert w.width > 0 and w.height > 0


def test_raster_smaller_than_one_tile_gives_one_window():
    assert plan_windows(300, 200, tile=1024, overlap=512) == [Window(0, 0, 300, 200)]


def test_exact_multiple_has_no_duplicate_final_window():
    # 1024 + 512 = 1536 is reached exactly by the stride, so clamping the last
    # window must not append a second copy of it.
    windows = plan_windows(1536, 1024, tile=1024, overlap=512)
    assert len(windows) == len(set(windows)), 'duplicate windows emitted'


def test_zero_overlap_is_allowed():
    windows = plan_windows(2048, 1024, tile=1024, overlap=0)
    assert len(windows) == 2
    assert [w.col_off for w in windows] == [0, 1024]


@pytest.mark.parametrize('tile,overlap', [(0, 0), (-1, 0), (1024, 1024), (1024, -1), (512, 600)])
def test_invalid_geometry_is_rejected(tile, overlap):
    with pytest.raises(ValueError):
        plan_windows(2048, 2048, tile=tile, overlap=overlap)


@pytest.mark.parametrize('width,height', [(0, 10), (10, 0), (-5, 10)])
def test_empty_raster_is_rejected(width, height):
    with pytest.raises(ValueError):
        plan_windows(width, height, tile=256, overlap=64)


# --------------------------------------------------------------------------
# gaussian_weights
# --------------------------------------------------------------------------

def test_weights_peak_at_centre_and_stay_positive():
    w = gaussian_weights(129, sigma=0.4)
    assert w.shape == (129, 129)
    assert w.dtype == np.float32
    assert (w > 0).all(), 'a zero weight would allow a zero denominator'
    assert w[64, 64] == pytest.approx(1.0)
    assert w.argmax() == 64 * 129 + 64


def test_weights_are_symmetric_and_decay_outward():
    w = gaussian_weights(64)
    np.testing.assert_allclose(w, w[::-1, :], rtol=1e-6)
    np.testing.assert_allclose(w, w[:, ::-1], rtol=1e-6)
    assert w[32, 0] < w[32, 16] < w[32, 31]


@pytest.mark.parametrize('tile,sigma', [(0, 0.4), (-8, 0.4), (64, 0.0), (64, -1.0)])
def test_invalid_weight_arguments_are_rejected(tile, sigma):
    with pytest.raises(ValueError):
        gaussian_weights(tile, sigma)


# --------------------------------------------------------------------------
# Accumulator
# --------------------------------------------------------------------------

def test_constant_field_is_reconstructed_exactly():
    """A constant prediction must survive blending unchanged, for any overlap.

    This is the sharpest check on the weighted mean: sum(w*c)/sum(w) == c.
    """
    tile, height, width = 64, 200, 300
    weights = gaussian_weights(tile)
    for overlap in (0, 16, 32, 48):
        acc = Accumulator(height, width)
        for win in plan_windows(width, height, tile, overlap):
            acc.add(win, np.full((tile, tile), 0.7, np.float32), weights)
        np.testing.assert_allclose(acc.result(), 0.7, rtol=1e-5)


def test_blending_matches_a_direct_weighted_mean():
    """Overlap-add must equal the definition it is an optimisation of."""
    rng = np.random.default_rng(0)
    tile, height, width, overlap = 32, 100, 120, 12
    weights = gaussian_weights(tile)
    windows = plan_windows(width, height, tile, overlap)
    tiles = [rng.random((tile, tile), np.float32) for _ in windows]

    acc = Accumulator(height, width)
    for win, t in zip(windows, tiles):
        acc.add(win, t, weights)

    num = np.zeros((height, width), np.float64)
    den = np.zeros((height, width), np.float64)
    for win, t in zip(windows, tiles):
        h, w = win.height, win.width
        num[win.row_off:win.row_off + h, win.col_off:win.col_off + w] += (
            t[:h, :w] * weights[:h, :w])
        den[win.row_off:win.row_off + h, win.col_off:win.col_off + w] += weights[:h, :w]
    np.testing.assert_allclose(acc.result(), num / den, rtol=1e-4)


def test_uncovered_pixels_raise_rather_than_silently_filling():
    acc = Accumulator(50, 50)
    acc.add(Window(0, 0, 10, 10), np.ones((10, 10), np.float32),
            np.ones((10, 10), np.float32))
    with pytest.raises(RuntimeError, match='no tile coverage'):
        acc.result()


def test_tile_smaller_than_its_window_is_rejected():
    acc = Accumulator(50, 50)
    with pytest.raises(ValueError):
        acc.add(Window(0, 0, 20, 20), np.ones((10, 10), np.float32),
                np.ones((10, 10), np.float32))


def test_accumulator_accepts_memmapped_arrays(tmp_path):
    """Large mosaics need out-of-core accumulators; the shapes must be honoured."""
    num = np.memmap(tmp_path / 'n.dat', np.float32, 'w+', shape=(40, 40))
    den = np.memmap(tmp_path / 'd.dat', np.float32, 'w+', shape=(40, 40))
    acc = Accumulator(40, 40, numerator=num, denominator=den)
    weights = gaussian_weights(20)
    for win in plan_windows(40, 40, tile=20, overlap=10):
        acc.add(win, np.full((20, 20), 0.25, np.float32), weights)
    np.testing.assert_allclose(acc.result(), 0.25, rtol=1e-5)


def test_mismatched_accumulator_shape_is_rejected():
    with pytest.raises(ValueError, match='expected'):
        Accumulator(10, 10, numerator=np.zeros((5, 5), np.float32))


# --------------------------------------------------------------------------
# streaming
# --------------------------------------------------------------------------

def test_blocks_tile_the_result_without_gaps_or_overlap():
    """Streaming must reconstruct exactly what result() returns, in order."""
    tile, height, width = 32, 205, 120
    acc = Accumulator(height, width)
    weights = gaussian_weights(tile)
    for win in plan_windows(width, height, tile, overlap=8):
        acc.add(win, np.full((tile, tile), 0.4, np.float32), weights)

    seen, rebuilt = 0, np.empty((height, width), np.float32)
    for rows, values in acc.blocks(64):
        assert rows.start == seen, 'stripes must be contiguous'
        assert values.shape == (rows.stop - rows.start, width)
        assert values.dtype == np.float32
        rebuilt[rows] = values
        seen = rows.stop
    assert seen == height, 'stripes must cover every row'
    np.testing.assert_allclose(rebuilt, acc.result(), rtol=1e-6)


def test_a_final_short_stripe_is_handled():
    acc = Accumulator(100, 10)
    acc.numerator[:] = 2.0
    acc.denominator[:] = 1.0
    heights = [rows.stop - rows.start for rows, _ in acc.blocks(30)]
    assert heights == [30, 30, 30, 10]


def test_streaming_reports_uncovered_pixels_rather_than_filling_them():
    acc = Accumulator(40, 40)
    acc.add(Window(0, 0, 10, 10), np.ones((10, 10), np.float32),
            np.ones((10, 10), np.float32))
    with pytest.raises(RuntimeError, match='no tile coverage'):
        list(acc.blocks(8))


@pytest.mark.parametrize('rows', [0, -1])
def test_invalid_stripe_height_is_rejected(rows):
    acc = Accumulator(10, 10)
    with pytest.raises(ValueError):
        list(acc.blocks(rows))
