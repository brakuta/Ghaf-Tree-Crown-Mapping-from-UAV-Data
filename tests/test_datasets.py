"""The dataset contract: two classes, PNG tiles, background supervised.

Building the dataset needs mmsegmentation, so these are skipped where the full
stack is absent -- including CI, which runs without mmcv. They run where it
matters: the machine that trains and evaluates.
"""

import pytest

pytest.importorskip('mmseg')

from ghaf.datasets import GhafDataset  # noqa: E402


def build(tmp_path, **kwargs) -> GhafDataset:
    """Construct the dataset without scanning a tile tree."""
    return GhafDataset(
        data_root=str(tmp_path),
        data_prefix=dict(img_path='images', seg_map_path='masks'),
        lazy_init=True, **kwargs)


def test_tiles_are_png_by_default(tmp_path):
    """The published tiles are PNG; a .tif default would find nothing."""
    dataset = build(tmp_path)
    assert dataset.img_suffix == '.png'
    assert dataset.seg_map_suffix == '.png'


def test_another_tile_format_can_be_asked_for(tmp_path):
    dataset = build(tmp_path, img_suffix='.tif', seg_map_suffix='.tif')
    assert dataset.img_suffix == '.tif'
    assert dataset.seg_map_suffix == '.tif'


def test_background_is_the_first_of_two_classes(tmp_path):
    assert GhafDataset.METAINFO['classes'] == ('background', 'ghaf')
    assert build(tmp_path).METAINFO['classes'][0] == 'background'


def test_dropping_the_background_class_is_refused(tmp_path):
    """reduce_zero_label would leave a one-class problem behind num_classes=2."""
    with pytest.raises(ValueError, match='reduce_zero_label'):
        build(tmp_path, reduce_zero_label=True)
