"""Tests for batch prediction over a folder of images.

The listing, the output layout and the batch loop are exercised with a stub
engine, so they run without a model; one end-to-end test drives the real
windowed inference over real rasters with only the segmentor stubbed, so the
files on disk are the ones a run would actually produce.
"""

import sys
from pathlib import Path

import pytest

rasterio = pytest.importorskip('rasterio')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghaf.inference.large_image import PredictionSummary  # noqa: E402
from tools import predict_folder as F  # noqa: E402


def touch(path: Path, name: str) -> Path:
    """An empty file, parents created."""
    target = path / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b'')
    return target


def stub_predict(canopy=10, valid=100, recorder=None, masks=None, fails=()):
    """An engine that writes its outputs and reports a fixed canopy share."""
    def predict(model, image, out_prob=None, out_mask=None, out_polygons=None,
                **kwargs):
        if Path(image).name in fails:
            raise RuntimeError('boom')
        if recorder is not None:
            recorder.append(Path(image))
        if masks is not None:
            masks.append(Path(out_mask))
        written = []
        for path in (out_prob, out_mask, out_polygons):
            if path is not None:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_bytes(b'x')
                written.append(Path(path))
        return PredictionSummary(width=64, height=64, windows=1,
                                 canopy_pixels=canopy, valid_pixels=valid,
                                 outputs=tuple(written))
    return predict


# --------------------------------------------------------------------------
# listing
# --------------------------------------------------------------------------

def test_only_images_are_listed(tmp_path):
    for name in ('a.tif', 'b.PNG', 'c.jpg', 'notes.txt', 'index.json'):
        touch(tmp_path, name)
    assert [p.name for p in F.list_images(tmp_path)] == ['a.tif', 'b.PNG', 'c.jpg']


def test_subfolders_are_read_only_when_asked(tmp_path):
    touch(tmp_path, 'top.tif')
    touch(tmp_path, 'plot-3/north.tif')

    assert [p.name for p in F.list_images(tmp_path)] == ['top.tif']
    assert [p.name for p in F.list_images(tmp_path, recursive=True)] == [
        'north.tif', 'top.tif']


def test_a_pattern_selects_by_name(tmp_path):
    touch(tmp_path, 'plot1_rgb.tif')
    touch(tmp_path, 'plot1_dsm.tif')
    touch(tmp_path, 'PLOT2_RGB.tif')

    found = [p.name for p in F.list_images(tmp_path, pattern='*_rgb.tif')]
    assert found == ['PLOT2_RGB.tif', 'plot1_rgb.tif']


def test_the_output_folder_is_not_read_back_in(tmp_path):
    """A second run must not predict the first run's own rasters.

    Writing outputs inside the input folder is a natural thing to do, and with
    --save-mask a recursive listing would then pick up every mask written last
    time -- each one a valid single-band raster, so nothing would fail. The
    batch would simply double in size and fill with predictions of predictions.
    """
    touch(tmp_path, 'source.tif')
    out_dir = tmp_path / 'predictions'
    touch(out_dir, 'masks/source.tif')

    found = F.list_images(tmp_path, recursive=True, exclude=out_dir)
    assert [p.name for p in found] == ['source.tif']


def test_a_folder_with_no_images_is_an_error(tmp_path):
    touch(tmp_path, 'readme.txt')
    with pytest.raises(FileNotFoundError, match='no image'):
        F.list_images(tmp_path)


def test_a_missing_folder_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match='no such folder'):
        F.list_images(tmp_path / 'absent')


# --------------------------------------------------------------------------
# output layout
# --------------------------------------------------------------------------

def test_polygons_are_the_output_and_rasters_are_not(tmp_path):
    out = F.output_paths(tmp_path / 'a.tif', tmp_path, Path('out'))

    assert out.polygons == Path('out/polygons/a.gpkg')
    assert out.mask is None and out.probability is None
    assert out.paths() == [Path('out/polygons/a.gpkg')]


def test_outputs_mirror_the_input_tree(tmp_path):
    out = F.output_paths(tmp_path / 'plot-3' / 'north.tif', tmp_path,
                         Path('predictions'), save_mask=True,
                         save_probability=True)

    assert out.polygons == Path('predictions/polygons/plot-3/north.gpkg')
    assert out.mask == Path('predictions/masks/plot-3/north.tif')
    assert out.probability == Path('predictions/probability/plot-3/north.tif')


def test_same_name_in_two_folders_does_not_collide(tmp_path):
    first = F.output_paths(tmp_path / 'north' / 'plot.tif', tmp_path, Path('out'))
    second = F.output_paths(tmp_path / 'south' / 'plot.tif', tmp_path, Path('out'))
    assert first.polygons != second.polygons


def test_the_vector_format_follows_the_suffix(tmp_path):
    out = F.output_paths(tmp_path / 'a.tif', tmp_path, Path('out'),
                         polygon_suffix='.shp')
    assert out.polygons == Path('out/polygons/a.shp')


# --------------------------------------------------------------------------
# the batch loop
# --------------------------------------------------------------------------

def test_every_image_is_predicted_and_the_totals_add_up(tmp_path):
    images = [touch(tmp_path, name) for name in ('a.tif', 'b.tif', 'c.tif')]
    seen = []

    summary = F.predict_folder(None, images, tmp_path, tmp_path / 'out',
                               predict=stub_predict(canopy=10, valid=100,
                                                    recorder=seen))

    assert [p.name for p in seen] == ['a.tif', 'b.tif', 'c.tif']
    assert summary['images'] == summary['predicted'] == 3
    assert summary['canopy_pixels'] == 30 and summary['valid_pixels'] == 300
    assert summary['canopy_fraction'] == pytest.approx(0.1)
    assert (tmp_path / 'out' / 'polygons' / 'a.gpkg').exists()


def test_the_mask_is_temporary_unless_it_is_asked_for(tmp_path):
    """Polygons are traced from a mask, but the mask need not be kept.

    It goes to a temporary directory instead, so a folder of large images
    leaves crowns behind rather than the hundreds of megabytes of raster the
    crowns were traced from.
    """
    images = [touch(tmp_path, 'a.tif')]
    masks = []

    summary = F.predict_folder(None, images, tmp_path, tmp_path / 'out',
                               predict=stub_predict(masks=masks))

    used = masks[0]
    assert used.name == 'a.tif'
    assert not used.exists()                      # removed with its directory
    assert (tmp_path / 'out' / 'masks').exists() is False
    assert summary['results'][0]['outputs'] == [
        str(tmp_path / 'out' / 'polygons' / 'a.gpkg')]


def test_the_temporary_mask_can_be_placed_on_a_chosen_disk(tmp_path):
    """--scratch-dir points the accumulators and the mask at the same disk."""
    images = [touch(tmp_path, 'a.tif')]
    scratch = tmp_path / 'fast-disk'
    masks = []

    F.predict_folder(None, images, tmp_path, tmp_path / 'out',
                     scratch_dir=scratch, predict=stub_predict(masks=masks))

    assert scratch in masks[0].parents


def test_the_mask_is_kept_where_it_belongs_when_asked_for(tmp_path):
    images = [touch(tmp_path, 'a.tif')]
    masks = []

    summary = F.predict_folder(None, images, tmp_path, tmp_path / 'out',
                               save_mask=True, predict=stub_predict(masks=masks))

    kept = tmp_path / 'out' / 'masks' / 'a.tif'
    assert masks[0] == kept and kept.exists()
    assert str(kept) in summary['results'][0]['outputs']


def test_a_failed_image_does_not_stop_the_batch(tmp_path):
    """One bad file in a folder of hundreds must not cost the whole run."""
    images = [touch(tmp_path, name) for name in ('a.tif', 'bad.tif', 'c.tif')]
    seen = []

    summary = F.predict_folder(None, images, tmp_path, tmp_path / 'out',
                               predict=stub_predict(recorder=seen,
                                                    fails=('bad.tif',)))

    assert [p.name for p in seen] == ['a.tif', 'c.tif']
    assert summary['failed'] == 1 and summary['predicted'] == 2
    failure = next(r for r in summary['results'] if r['image'] == 'bad.tif')
    assert 'RuntimeError: boom' in failure['error']


def test_skip_existing_leaves_finished_images_alone(tmp_path):
    """An interrupted run is resumed, not repeated."""
    images = [touch(tmp_path, name) for name in ('a.tif', 'b.tif')]
    touch(tmp_path / 'out', 'polygons/a.gpkg')
    seen = []

    summary = F.predict_folder(None, images, tmp_path, tmp_path / 'out',
                               skip_existing=True,
                               predict=stub_predict(recorder=seen))

    assert [p.name for p in seen] == ['b.tif']
    assert summary['skipped'] == 1 and summary['predicted'] == 1


def test_a_missing_raster_is_not_treated_as_finished(tmp_path):
    """The crowns alone are not the whole job when --save-mask was asked for."""
    images = [touch(tmp_path, 'a.tif')]
    touch(tmp_path / 'out', 'polygons/a.gpkg')
    seen = []

    F.predict_folder(None, images, tmp_path, tmp_path / 'out',
                     save_mask=True, skip_existing=True,
                     predict=stub_predict(recorder=seen))

    assert [p.name for p in seen] == ['a.tif']


def test_the_canopy_share_is_weighted_by_valid_pixels(tmp_path):
    """Images differ in size, so the total is not the mean of the fractions."""
    images = [touch(tmp_path, 'a.tif'), touch(tmp_path, 'b.tif')]
    sizes = iter([(10, 100), (900, 900)])

    def predict(model, image, out_prob=None, out_mask=None, out_polygons=None,
                **kwargs):
        canopy, valid = next(sizes)
        Path(out_polygons).parent.mkdir(parents=True, exist_ok=True)
        Path(out_polygons).write_bytes(b'x')
        return PredictionSummary(width=1, height=1, windows=1,
                                 canopy_pixels=canopy, valid_pixels=valid,
                                 outputs=(Path(out_polygons),))

    summary = F.predict_folder(None, images, tmp_path, tmp_path / 'out',
                               predict=predict)
    assert summary['canopy_fraction'] == pytest.approx(910 / 1000)


def test_the_summary_names_images_the_way_the_user_typed_them(tmp_path):
    images = [touch(tmp_path, 'plot-3/north.tif')]
    summary = F.predict_folder(None, images, tmp_path, tmp_path / 'out',
                               predict=stub_predict())
    assert summary['results'][0]['image'] == 'plot-3/north.tif'


# --------------------------------------------------------------------------
# end to end, with only the segmentor stubbed
# --------------------------------------------------------------------------

def test_a_folder_of_rasters_becomes_a_folder_of_crowns(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    gpd = pytest.importorskip('geopandas')
    import ghaf.inference.large_image as L
    from tests.test_large_image import _stub_inference, write_raster

    real = L._import

    class _Api:
        inference_model = staticmethod(_stub_inference(0.9, 2, 64))

    monkeypatch.setattr(L, '_import',
                        lambda module, package: _Api if module == 'mmseg.apis'
                        else real(module, package))

    source = tmp_path / 'images'
    write_raster(source / 'plot-1.tif', 80, 70)
    write_raster(source / 'north' / 'plot-2.tif', 64, 64)

    images = F.list_images(source, recursive=True)
    summary = F.predict_folder(None, images, source, tmp_path / 'out',
                               tile=64, overlap=32, progress=False)

    assert summary['predicted'] == 2 and summary['failed'] == 0
    for relative in ('plot-1.gpkg', 'north/plot-2.gpkg'):
        crowns = gpd.read_file(tmp_path / 'out' / 'polygons' / relative)
        assert len(crowns) == 1                   # one blanket crown, all ghaf
        assert crowns.crs is not None
        assert crowns['area_m2'].iloc[0] > 0

    # The masks the crowns were traced from are not left behind.
    assert not (tmp_path / 'out' / 'masks').exists()
