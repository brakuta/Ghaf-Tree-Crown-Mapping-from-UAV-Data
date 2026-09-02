"""Tests for per-tile prediction over a dataset split.

The segmentor is stubbed; rasterio, the file layout and the written GeoTIFFs
are real, so georeferencing and encoding are checked against files on disk.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip('rasterio')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rasterio.transform import from_origin  # noqa: E402

from tests.test_large_image import (  # noqa: E402
    PIXEL_SIZE,
    _stub_inference,
    write_raster,
)
from tools import predict_split as P  # noqa: E402

CRS = 'EPSG:32640'


@pytest.fixture
def patched(monkeypatch):
    """Route the mmseg import to a stub returning a known probability."""
    def factory(probability, tile=64):
        real = P._import

        class _Api:
            inference_model = staticmethod(_stub_inference(probability, 2, tile))

        monkeypatch.setattr(
            P, '_import',
            lambda module, package: _Api if module == 'mmseg.apis'
            else real(module, package))
    return factory


def build_split(root: Path, split: str, count: int, size: int = 64) -> Path:
    directory = root / P.SPLIT_IMAGES[split]
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        write_raster(directory / f'tile_{i:03d}.tif', size, size)
    return directory


# --------------------------------------------------------------------------

def test_lists_tiles_in_a_stable_order(tmp_path):
    build_split(tmp_path, 'testing', 5)
    first = P.list_tiles(tmp_path, 'testing')
    assert [p.name for p in first] == [f'tile_{i:03d}.tif' for i in range(5)]
    assert P.list_tiles(tmp_path, 'testing') == first


def test_a_missing_split_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match='no such split'):
        P.list_tiles(tmp_path, 'validation')


def test_an_empty_split_is_reported(tmp_path):
    (tmp_path / P.SPLIT_IMAGES['testing']).mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match='no .* tiles'):
        P.list_tiles(tmp_path, 'testing')


def test_a_mask_is_written_for_every_tile(tmp_path, patched):
    patched(0.9)
    build_split(tmp_path, 'testing', 4)
    out = tmp_path / 'predictions'

    summary = P.predict_split(None, P.list_tiles(tmp_path, 'testing'), out,
                              progress=False)

    written = sorted((out / 'masks').glob('*.tif'))
    assert len(written) == 4
    assert summary['tiles'] == 4
    assert summary['canopy_fraction'] == 1.0


def test_masks_carry_the_source_georeferencing_and_encoding(tmp_path, patched):
    patched(0.9)
    build_split(tmp_path, 'testing', 1)
    out = tmp_path / 'predictions'
    tiles = P.list_tiles(tmp_path, 'testing')

    P.predict_split(None, tiles, out, progress=False)

    with rasterio.open(tiles[0]) as src, \
            rasterio.open(out / 'masks' / f'{tiles[0].stem}.tif') as dst:
        assert dst.crs == src.crs
        assert dst.transform == src.transform
        assert (dst.width, dst.height) == (src.width, src.height)
        assert dst.count == 1
        assert dst.dtypes[0] == 'uint8'
        assert set(np.unique(dst.read(1))) <= {0, 1}


def test_the_threshold_decides_the_prediction(tmp_path, patched):
    patched(0.6)
    build_split(tmp_path, 'testing', 2)
    tiles = P.list_tiles(tmp_path, 'testing')

    below = P.predict_split(None, tiles, tmp_path / 'a', threshold=0.5,
                            progress=False)
    above = P.predict_split(None, tiles, tmp_path / 'b', threshold=0.7,
                            progress=False)
    assert below['canopy_fraction'] == 1.0
    assert above['canopy_fraction'] == 0.0


def test_probability_maps_are_optional(tmp_path, patched):
    patched(0.75)
    build_split(tmp_path, 'testing', 2)
    tiles = P.list_tiles(tmp_path, 'testing')

    without = P.predict_split(None, tiles, tmp_path / 'a', progress=False)
    assert without['probability'] is None
    assert not (tmp_path / 'a' / 'probability').exists()

    with_probs = P.predict_split(None, tiles, tmp_path / 'b',
                                 save_probability=True, progress=False)
    maps = sorted((tmp_path / 'b' / 'probability').glob('*.tif'))
    assert len(maps) == 2
    assert with_probs['probability'] is not None
    with rasterio.open(maps[0]) as dst:
        assert dst.dtypes[0] == 'float32'
        np.testing.assert_allclose(dst.read(1), 0.75, atol=1e-4)


@pytest.mark.parametrize('batch_size', [1, 3, 16])
def test_batching_does_not_change_the_predictions(tmp_path, patched, batch_size):
    patched(0.8)
    build_split(tmp_path, 'testing', 5)
    tiles = P.list_tiles(tmp_path, 'testing')

    out = tmp_path / f'b{batch_size}'
    summary = P.predict_split(None, tiles, out, batch_size=batch_size,
                              progress=False)
    assert summary['tiles'] == 5
    assert len(sorted((out / 'masks').glob('*.tif'))) == 5
    assert summary['canopy_fraction'] == 1.0


@pytest.mark.parametrize('kwargs,match', [
    ({'threshold': 1.5}, 'threshold'),
    ({'threshold': -0.1}, 'threshold'),
    ({'batch_size': 0}, 'batch_size'),
])
def test_invalid_arguments_are_rejected(tmp_path, patched, kwargs, match):
    patched(0.9)
    build_split(tmp_path, 'testing', 1)
    with pytest.raises(ValueError, match=match):
        P.predict_split(None, P.list_tiles(tmp_path, 'testing'),
                        tmp_path / 'out', progress=False, **kwargs)


def test_a_16_bit_tile_is_rejected(tmp_path, patched):
    patched(0.9)
    directory = tmp_path / P.SPLIT_IMAGES['testing']
    directory.mkdir(parents=True, exist_ok=True)
    write_raster(directory / 'wide.tif', 64, 64, dtype='uint16')
    with pytest.raises(ValueError, match='8-bit'):
        P.predict_split(None, P.list_tiles(tmp_path, 'testing'),
                        tmp_path / 'out', progress=False)


def test_the_cli_defaults_to_the_testing_split(tmp_path):
    args = P.parse_args(['cfg.py', 'ckpt.pth'])
    assert args.split == 'testing'
    assert args.batch_size == 1
    assert args.threshold == 0.5
    assert args.save_probability is False


def test_summary_is_json_serialisable(tmp_path, patched):
    patched(0.9)
    build_split(tmp_path, 'testing', 2)
    summary = P.predict_split(None, P.list_tiles(tmp_path, 'testing'),
                              tmp_path / 'out', progress=False)
    assert json.loads(json.dumps(summary))['tiles'] == 2


# --------------------------------------------------------------------------
# PNG tiles, which is what the published dataset is cut as
# --------------------------------------------------------------------------

def write_png_tile(path: Path, size: int = 64, georeferenced: bool = False) -> Path:
    """Write an 8-bit RGB PNG tile, the way the published tiles were made.

    GDAL's PNG driver has no ``Create``, only ``CreateCopy``, so the tile is
    written as a GeoTIFF and converted. When the GeoTIFF carries a CRS and a
    transform, PNG cannot hold them and GDAL writes them into a
    ``<tile>.png.aux.xml`` beside it -- which is exactly how the test split's
    tiles carry their position, alongside a ``.pgw`` world file.
    """
    import rasterio.shutil

    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix('.staging.tif')
    values = np.random.default_rng(0).integers(0, 256, (3, size, size),
                                               dtype=np.uint8)
    placed = {}
    if georeferenced:
        placed = dict(crs=CRS, transform=from_origin(432652.06, 2770523.77,
                                                     PIXEL_SIZE, PIXEL_SIZE))
    with rasterio.open(staging, 'w', driver='GTiff', width=size, height=size,
                       count=3, dtype='uint8', **placed) as dst:
        dst.write(values)
    rasterio.shutil.copy(staging, path, driver='PNG')
    staging.unlink()
    return path


def test_png_tiles_are_listed(tmp_path):
    directory = tmp_path / P.SPLIT_IMAGES['testing']
    for i in range(3):
        write_png_tile(directory / f'tile_{i}.png')

    assert [p.name for p in P.list_tiles(tmp_path, 'testing')] == [
        'tile_0.png', 'tile_1.png', 'tile_2.png']


def test_a_png_tile_yields_a_readable_mask(tmp_path, patched):
    """Training and validation tiles have no sidecars; the run must still work."""
    patched(0.9)
    directory = tmp_path / P.SPLIT_IMAGES['testing']
    write_png_tile(directory / 'tile_0.png')

    P.predict_split(None, P.list_tiles(tmp_path, 'testing'), tmp_path / 'out',
                    progress=False)

    written = tmp_path / 'out' / 'masks' / 'tile_0.tif'
    assert written.is_file()
    with rasterio.open(written) as src:
        assert src.driver == 'GTiff'
        assert src.count == 1 and src.dtypes == ('uint8',)
        assert src.crs is None, 'a bare PNG has no CRS to carry over'
        assert (src.read(1) == 1).all()


def test_a_georeferenced_tile_places_its_prediction(tmp_path, patched):
    """The test split's tiles know where they are, so its masks must too.

    Without this a prediction looks right and lands in the wrong place -- or
    nowhere -- which no pixel comparison would reveal.
    """
    patched(0.9)
    directory = tmp_path / P.SPLIT_IMAGES['testing']
    write_png_tile(directory / 'tile_0.png', georeferenced=True)

    P.predict_split(None, P.list_tiles(tmp_path, 'testing'), tmp_path / 'out',
                    progress=False)

    with rasterio.open(tmp_path / 'out' / 'masks' / 'tile_0.tif') as src:
        assert src.crs.to_epsg() == 32640
        assert src.transform.a == pytest.approx(PIXEL_SIZE)
        assert src.transform.c == pytest.approx(432652.06)


def test_a_georeferenced_tile_still_carries_its_crs(tmp_path, patched):
    patched(0.9)
    build_split(tmp_path, 'testing', 1)

    P.predict_split(None, P.list_tiles(tmp_path, 'testing'), tmp_path / 'out',
                    progress=False)

    with rasterio.open(tmp_path / 'out' / 'masks' / 'tile_000.tif') as src:
        assert src.crs is not None
        assert str(src.crs) == CRS
