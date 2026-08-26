"""End-to-end tests for area-wide inference, with a stubbed model.

Everything except the segmentor is real: rasterio reads the windows, the
accumulators blend, the results stream back into GeoTIFFs, and the outputs are
read from disk and checked. Only ``mmseg.apis.inference_model`` is replaced, by
a stub that returns known logits -- so these tests run without torch, mmcv or a
GPU while still exercising the code paths that touch real files.

Skipped when rasterio is unavailable.
"""

import numpy as np
import pytest

rasterio = pytest.importorskip('rasterio')
from rasterio.transform import from_origin  # noqa: E402

from ghaf.inference import large_image as L  # noqa: E402

CRS = 'EPSG:32640'          # UTM 40N, the UAE
PIXEL_SIZE = 0.05           # 5 cm ground sampling, as flown


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

class _Sample:
    """Stands in for an mmseg ``SegDataSample``.

    ``seg_logits.data`` is a genuine torch tensor, so
    ``_foreground_probability`` runs its real softmax rather than something
    imitating one.
    """

    def __init__(self, logits):
        self.seg_logits = type('_Logits', (), {'data': logits})()


def _stub_inference(probability, classes=2, tile=64):
    """An ``inference_model`` that always predicts ``probability`` for ghaf."""
    import torch

    p = float(probability)
    logits = torch.zeros(classes, tile, tile)
    if classes > L.GHAF_CLASS_INDEX:
        # softmax over [0, x] returns p when x = log(p / (1 - p))
        logits[L.GHAF_CLASS_INDEX] = (
            torch.logit(torch.tensor(p)) if 0 < p < 1 else (30.0 if p >= 1 else -30.0))

    def run(model, image):
        # Mirror mmseg: a sequence in, a list of the same length out.
        if isinstance(image, (list, tuple)):
            return [_Sample(logits) for _ in image]
        return _Sample(logits)
    return run


@pytest.fixture
def patched(monkeypatch):
    """Route ``_import('mmseg.apis', ...)`` to a stub, leave the rest real."""
    def factory(probability, tile=64, classes=2):
        real = L._import

        class _Api:
            inference_model = staticmethod(
                _stub_inference(probability, classes, tile))

        def fake(module, package):
            return _Api if module == 'mmseg.apis' else real(module, package)

        monkeypatch.setattr(L, '_import', fake)
    return factory


def write_raster(path, width, height, *, bands=3, nodata_box=None, dtype='uint8'):
    """A small georeferenced RGB raster, optionally with a nodata rectangle."""
    data = np.full((bands, height, width), 200, dtype)
    profile = dict(driver='GTiff', width=width, height=height, count=bands,
                   dtype=dtype, crs=CRS,
                   transform=from_origin(400000, 2700000, PIXEL_SIZE, PIXEL_SIZE))
    if nodata_box is not None:
        r0, r1, c0, c1 = nodata_box
        data[:, r0:r1, c0:c1] = 0
        profile['nodata'] = 0
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data)
    return path


# --------------------------------------------------------------------------
# geometry, georeferencing and content
# --------------------------------------------------------------------------

def test_outputs_are_written_and_georeferenced(tmp_path, patched):
    patched(0.9)
    src = write_raster(tmp_path / 'mosaic.tif', 150, 130)
    prob, mask = tmp_path / 'p.tif', tmp_path / 'm.tif'

    summary = L.predict_large_image(
        None, src, out_prob=prob, out_mask=mask,
        tile=64, overlap=16, progress=False)

    assert summary.width == 150 and summary.height == 130
    assert set(summary.outputs) == {prob, mask}

    with rasterio.open(src) as s, rasterio.open(prob) as p, rasterio.open(mask) as m:
        for out in (p, m):
            assert (out.width, out.height) == (s.width, s.height)
            assert out.crs == s.crs
            assert out.transform == s.transform
            assert out.count == 1
        assert p.dtypes[0] == 'float32'
        assert m.dtypes[0] == 'uint8'


def test_a_uniform_prediction_survives_blending(tmp_path, patched):
    """Gaussian weights must not distort a constant field anywhere, edges included."""
    patched(0.75)
    src = write_raster(tmp_path / 'mosaic.tif', 200, 90)
    prob = tmp_path / 'p.tif'
    L.predict_large_image(None, src, out_prob=prob, tile=64, overlap=32,
                          progress=False)

    with rasterio.open(prob) as dst:
        values = dst.read(1)
    np.testing.assert_allclose(values, 0.75, atol=1e-4)


@pytest.mark.parametrize('width,height', [(64, 64), (65, 64), (200, 37), (300, 300)])
def test_every_pixel_is_written_for_awkward_sizes(tmp_path, patched, width, height):
    patched(0.8)
    src = write_raster(tmp_path / 'm.tif', width, height)
    prob = tmp_path / 'p.tif'
    L.predict_large_image(None, src, out_prob=prob, tile=64, overlap=16,
                          progress=False)
    with rasterio.open(prob) as dst:
        values = dst.read(1)
    assert values.shape == (height, width)
    assert np.all(values > 0.5), 'unwritten pixels would read back as zero'


def test_threshold_decides_the_mask(tmp_path, patched):
    patched(0.6)
    src = write_raster(tmp_path / 'm.tif', 128, 128)

    below, above = tmp_path / 'b.tif', tmp_path / 'a.tif'
    s1 = L.predict_large_image(None, src, out_mask=below, tile=64, overlap=16,
                               threshold=0.5, progress=False)
    s2 = L.predict_large_image(None, src, out_mask=above, tile=64, overlap=16,
                               threshold=0.7, progress=False)

    assert s1.canopy_pixels == 128 * 128, '0.60 >= 0.50 is canopy'
    assert s2.canopy_pixels == 0, '0.60 < 0.70 is not'
    assert s1.canopy_fraction == 1.0 and s2.canopy_fraction == 0.0


def test_nodata_is_never_reported_as_canopy(tmp_path, patched):
    """Unsurveyed ground must stay out of the mask even at probability 1."""
    patched(0.99)
    src = write_raster(tmp_path / 'm.tif', 128, 128, nodata_box=(0, 40, 0, 128))
    mask, prob = tmp_path / 'k.tif', tmp_path / 'p.tif'

    summary = L.predict_large_image(None, src, out_mask=mask, out_prob=prob,
                                    tile=64, overlap=16, progress=False)

    with rasterio.open(mask) as m, rasterio.open(prob) as p:
        binary, values = m.read(1), p.read(1)
    assert binary[:40].sum() == 0, 'nodata rows classified as canopy'
    assert np.all(binary[40:] == 1), 'valid rows lost'
    assert np.all(values[:40] == 0.0), 'nodata rows carry a probability'
    assert summary.valid_pixels == 88 * 128
    assert summary.canopy_pixels == 88 * 128
    assert summary.canopy_fraction == 1.0


def test_polygons_are_written_in_the_source_crs(tmp_path, patched):
    pytest.importorskip('geopandas')
    patched(0.95)
    src = write_raster(tmp_path / 'm.tif', 128, 128)
    mask, gpkg = tmp_path / 'k.tif', tmp_path / 'crowns.gpkg'

    summary = L.predict_large_image(None, src, out_mask=mask, out_polygons=gpkg,
                                    tile=64, overlap=16, progress=False)

    import geopandas as gpd
    frame = gpd.read_file(gpkg)
    assert len(frame) >= 1
    assert frame.crs.to_string().endswith('32640')
    assert set(frame['class']) == {'ghaf'}
    assert gpkg in summary.outputs


# --------------------------------------------------------------------------
# failure modes
# --------------------------------------------------------------------------

def test_a_16_bit_raster_is_rejected_with_a_remedy(tmp_path, patched):
    patched(0.9)
    src = write_raster(tmp_path / 'm.tif', 128, 128, dtype='uint16')
    with pytest.raises(ValueError, match='gdal_translate'):
        L.predict_large_image(None, src, out_mask=tmp_path / 'k.tif',
                              tile=64, overlap=16, progress=False)


def test_requesting_a_band_the_raster_lacks_is_rejected(tmp_path, patched):
    patched(0.9)
    src = write_raster(tmp_path / 'm.tif', 64, 64, bands=2)
    with pytest.raises(ValueError, match='band'):
        L.predict_large_image(None, src, out_mask=tmp_path / 'k.tif',
                              tile=64, overlap=16, progress=False)


def test_a_single_class_model_is_rejected(tmp_path, patched):
    patched(0.9, classes=1)
    src = write_raster(tmp_path / 'm.tif', 64, 64)
    with pytest.raises(RuntimeError, match='at least 2'):
        L.predict_large_image(None, src, out_mask=tmp_path / 'k.tif',
                              tile=64, overlap=16, progress=False)


def test_a_tile_size_the_model_disagrees_with_is_rejected(tmp_path, patched):
    patched(0.9, tile=64)                     # stub always returns 64x64
    src = write_raster(tmp_path / 'm.tif', 256, 256)
    with pytest.raises(RuntimeError, match='must agree'):
        L.predict_large_image(None, src, out_mask=tmp_path / 'k.tif',
                              tile=128, overlap=32, progress=False)


@pytest.mark.parametrize('threshold', [-0.1, 1.5])
def test_an_out_of_range_threshold_is_rejected(tmp_path, patched, threshold):
    patched(0.9)
    src = write_raster(tmp_path / 'm.tif', 64, 64)
    with pytest.raises(ValueError, match='threshold'):
        L.predict_large_image(None, src, out_mask=tmp_path / 'k.tif',
                              threshold=threshold, tile=64, overlap=16,
                              progress=False)


def test_a_run_with_no_outputs_is_rejected(tmp_path, patched):
    patched(0.9)
    src = write_raster(tmp_path / 'm.tif', 64, 64)
    with pytest.raises(ValueError, match='nothing to write'):
        L.predict_large_image(None, src, tile=64, overlap=16, progress=False)


def test_a_missing_source_is_reported(tmp_path, patched):
    patched(0.9)
    with pytest.raises(FileNotFoundError):
        L.predict_large_image(None, tmp_path / 'absent.tif',
                              out_mask=tmp_path / 'k.tif', tile=64, overlap=16,
                              progress=False)


def test_scratch_is_removed_even_when_a_run_fails(tmp_path, patched):
    """A temporary directory left behind would fill the disk over a survey."""
    import tempfile
    from pathlib import Path

    def scratch_dirs():
        return set(Path(tempfile.gettempdir()).glob('ghaf-infer-*'))

    patched(0.9, classes=1)
    src = write_raster(tmp_path / 'm.tif', 64, 64)
    before = scratch_dirs()
    with pytest.raises(RuntimeError):
        L.predict_large_image(None, src, out_mask=tmp_path / 'k.tif',
                              tile=64, overlap=16, progress=False)
    assert scratch_dirs() == before, 'scratch directory left behind'


# --------------------------------------------------------------------------
# batching, scratch space and resource guards
# --------------------------------------------------------------------------

@pytest.mark.parametrize('batch_size', [1, 2, 5, 64])
def test_batching_does_not_change_the_result(tmp_path, patched, batch_size):
    """Batch size is a throughput knob; it must not move a single pixel."""
    patched(0.85)
    src = write_raster(tmp_path / 'm.tif', 200, 150)
    out = tmp_path / f'p{batch_size}.tif'
    L.predict_large_image(None, src, out_prob=out, tile=64, overlap=16,
                          batch_size=batch_size, progress=False)
    with rasterio.open(out) as dst:
        values = dst.read(1)
    np.testing.assert_allclose(values, 0.85, atol=1e-4)


def test_a_batch_larger_than_the_window_count_is_fine(tmp_path, patched):
    patched(0.9)
    src = write_raster(tmp_path / 'm.tif', 64, 64)       # exactly one window
    summary = L.predict_large_image(None, src, out_mask=tmp_path / 'k.tif',
                                    tile=64, overlap=16, batch_size=32,
                                    progress=False)
    assert summary.windows == 1


@pytest.mark.parametrize('batch_size', [0, -3])
def test_an_invalid_batch_size_is_rejected(tmp_path, patched, batch_size):
    patched(0.9)
    src = write_raster(tmp_path / 'm.tif', 64, 64)
    with pytest.raises(ValueError, match='batch_size'):
        L.predict_large_image(None, src, out_mask=tmp_path / 'k.tif', tile=64,
                              overlap=16, batch_size=batch_size, progress=False)


def test_a_model_returning_the_wrong_batch_length_is_caught(tmp_path, patched,
                                                            monkeypatch):
    """Silently dropping tiles would leave holes that blending would hide."""
    patched(0.9)
    src = write_raster(tmp_path / 'm.tif', 200, 200)

    real = L._import

    def truncating(module, package):
        api = real(module, package) if module != 'mmseg.apis' else None
        if module != 'mmseg.apis':
            return api

        class _Api:
            @staticmethod
            def inference_model(model, images):
                return _stub_inference(0.9)(model, list(images)[:1])
        return _Api

    monkeypatch.setattr(L, '_import', truncating)
    with pytest.raises(RuntimeError, match='result'):
        L.predict_large_image(None, src, out_mask=tmp_path / 'k.tif', tile=64,
                              overlap=16, batch_size=4, progress=False)


def test_scratch_dir_is_honoured_and_cleaned(tmp_path, patched):
    patched(0.9)
    src = write_raster(tmp_path / 'm.tif', 128, 128)
    scratch = tmp_path / 'scratch'

    L.predict_large_image(None, src, out_mask=tmp_path / 'k.tif', tile=64,
                          overlap=16, scratch_dir=scratch, progress=False)

    assert scratch.is_dir(), 'scratch directory should be created'
    assert list(scratch.iterdir()) == [], 'temporary files left behind'


def test_a_run_too_large_for_the_scratch_filesystem_fails_early(tmp_path, patched,
                                                                monkeypatch):
    """Better to refuse up front than to die hours in with ENOSPC."""
    patched(0.9)
    src = write_raster(tmp_path / 'm.tif', 128, 128)

    Usage = type('Usage', (), {'total': 0, 'used': 0, 'free': 1024})
    monkeypatch.setattr(L.shutil, 'disk_usage', lambda _p: Usage)

    with pytest.raises(OSError, match='scratch space'):
        L.predict_large_image(None, src, out_mask=tmp_path / 'k.tif', tile=64,
                              overlap=16, progress=False)


def test_the_space_guard_computes_nine_bytes_per_pixel(tmp_path, monkeypatch):
    seen = {}
    Usage = type('Usage', (), {'total': 0, 'used': 0, 'free': 10 ** 12})
    monkeypatch.setattr(L.shutil, 'disk_usage',
                        lambda p: seen.setdefault('path', p) and Usage or Usage)
    L.check_scratch_space(tmp_path, 1000, 2000, margin=1.0)     # 18 MB, passes

    Small = type('Small', (), {'total': 0, 'used': 0, 'free': 17_000_000})
    monkeypatch.setattr(L.shutil, 'disk_usage', lambda _p: Small)
    with pytest.raises(OSError, match='need 18.0 GB|need 0.0 GB|scratch space'):
        L.check_scratch_space(tmp_path, 10_000, 20_000, margin=1.0)
