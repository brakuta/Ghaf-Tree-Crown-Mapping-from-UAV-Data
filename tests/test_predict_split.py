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

from tests.test_large_image import _stub_inference, write_raster  # noqa: E402
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
