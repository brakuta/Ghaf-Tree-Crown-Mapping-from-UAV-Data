"""Tests for cutting a small sample out of a large orthomosaic.

The window arithmetic is pure and tested directly; the clipping itself is run
against real GeoTIFFs, so georeferencing, band selection and the pixel grid
are checked against files on disk rather than asserted about.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip('rasterio')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_large_image import CRS, PIXEL_SIZE, write_raster  # noqa: E402
from tools import make_sample as M  # noqa: E402


def _grid():
    """The same pixel grid the other raster fixtures use."""
    from rasterio.transform import from_origin
    return from_origin(400000, 2700000, PIXEL_SIZE, PIXEL_SIZE)


# --------------------------------------------------------------------------
# window arithmetic
# --------------------------------------------------------------------------

def test_window_is_centred_by_default():
    clip = M.plan_clip(1000, 800, (100, 200))

    assert (clip.col_off, clip.row_off) == (450, 300)
    assert (clip.width, clip.height) == (100, 200)


def test_window_is_clamped_to_the_source():
    clip = M.plan_clip(50, 40, (100, 100))

    assert (clip.col_off, clip.row_off) == (0, 0)
    assert (clip.width, clip.height) == (50, 40)


def test_origin_places_the_window():
    clip = M.plan_clip(1000, 1000, (100, 100), origin=(700, 20))

    assert (clip.col_off, clip.row_off) == (700, 20)
    assert (clip.width, clip.height) == (100, 100)


def test_window_shrinks_rather_than_overrunning_the_edge():
    clip = M.plan_clip(1000, 1000, (400, 400), origin=(800, 950))

    assert (clip.width, clip.height) == (200, 50)
    assert clip.pixels == 10000


def test_origin_outside_the_raster_is_refused():
    with pytest.raises(ValueError, match='outside'):
        M.plan_clip(100, 100, (10, 10), origin=(100, 0))


@pytest.mark.parametrize('size', [(0, 10), (10, -1)])
def test_size_must_be_positive(size):
    with pytest.raises(ValueError, match='positive'):
        M.plan_clip(100, 100, size)


# --------------------------------------------------------------------------
# clipping real rasters
# --------------------------------------------------------------------------

def test_clip_keeps_the_source_pixel_grid(tmp_path):
    src = write_raster(tmp_path / 'mosaic.tif', 200, 160)
    out = tmp_path / 'sample.tif'

    M.make_sample(src, out, size=(40, 40))

    with rasterio.open(src) as a, rasterio.open(out) as b:
        assert b.width == b.height == 40
        assert b.crs.to_string() == CRS
        # The clip's top-left corner is the source's, offset by the window.
        assert b.transform * (0, 0) == a.transform * (80, 60)
        assert a.transform.a == b.transform.a


def test_alpha_band_is_dropped_by_default(tmp_path):
    src = write_raster(tmp_path / 'mosaic.tif', 100, 100, bands=4)
    out = tmp_path / 'sample.tif'

    M.make_sample(src, out, size=(30, 30))

    with rasterio.open(out) as dst:
        assert dst.count == 3


def test_requested_bands_are_kept_in_order(tmp_path):
    path = tmp_path / 'mosaic.tif'
    data = np.stack([np.full((20, 20), v, 'uint8') for v in (10, 20, 30)])
    with rasterio.open(path, 'w', driver='GTiff', width=20, height=20,
                       count=3, dtype='uint8', crs=CRS,
                       transform=_grid()) as dst:
        dst.write(data)
    out = tmp_path / 'sample.tif'

    M.make_sample(path, out, size=(10, 10), bands=(3, 1))

    with rasterio.open(out) as dst:
        assert dst.count == 2
        assert dst.read(1)[0, 0] == 30
        assert dst.read(2)[0, 0] == 10


def test_pixels_match_the_source_window(tmp_path):
    path = tmp_path / 'mosaic.tif'
    data = np.arange(60 * 60, dtype='uint8').reshape(1, 60, 60)
    with rasterio.open(path, 'w', driver='GTiff', width=60, height=60,
                       count=1, dtype='uint8', crs=CRS,
                       transform=_grid()) as dst:
        dst.write(data)
    out = tmp_path / 'sample.tif'

    M.make_sample(path, out, size=(8, 8), origin=(5, 7), bands=(1,))

    with rasterio.open(out) as dst:
        assert np.array_equal(dst.read(1), data[0, 7:15, 5:13])


def test_missing_source_is_named(tmp_path):
    with pytest.raises(FileNotFoundError):
        M.make_sample(tmp_path / 'absent.tif', tmp_path / 'out.tif')


def test_band_beyond_the_raster_is_refused(tmp_path):
    src = write_raster(tmp_path / 'mosaic.tif', 40, 40, bands=3)

    with pytest.raises(ValueError, match='band'):
        M.make_sample(src, tmp_path / 'out.tif', size=(10, 10), bands=(1, 2, 5))


def test_output_directory_is_created(tmp_path):
    src = write_raster(tmp_path / 'mosaic.tif', 40, 40)
    out = tmp_path / 'new' / 'nested' / 'sample.tif'

    M.make_sample(src, out, size=(10, 10))

    assert out.is_file()


def test_a_mostly_empty_clip_is_warned_about(tmp_path, caplog):
    src = write_raster(tmp_path / 'mosaic.tif', 40, 40,
                       nodata_box=(0, 40, 0, 30))

    with caplog.at_level('WARNING', logger='make_sample'):
        M.make_sample(src, tmp_path / 'out.tif', size=(40, 40))

    assert 'outside the imagery' in caplog.text


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------

def test_one_size_means_a_square():
    args = M.parse_args(['in.tif', '--output', 'out.tif', '--size', '256'])

    assert args.size == [256, 256]


def test_two_sizes_are_width_then_height():
    args = M.parse_args(['in.tif', '--output', 'out.tif',
                         '--size', '256', '128'])

    assert args.size == [256, 128]


def test_three_sizes_are_refused():
    with pytest.raises(SystemExit):
        M.parse_args(['in.tif', '--output', 'o.tif', '--size', '1', '2', '3'])


def test_main_writes_the_clip(tmp_path):
    src = write_raster(tmp_path / 'mosaic.tif', 60, 60)
    out = tmp_path / 'sample.tif'

    assert M.main([str(src), '--output', str(out), '--size', '16']) == 0

    with rasterio.open(out) as dst:
        assert dst.width == dst.height == 16
