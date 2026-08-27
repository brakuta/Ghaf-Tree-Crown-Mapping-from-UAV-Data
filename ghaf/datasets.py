"""Dataset definition for Ghaf (*Prosopis cineraria*) crown segmentation."""

from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class GhafDataset(BaseSegDataset):
    """Binary Ghaf-crown segmentation over UAV orthomosaic tiles.

    Two classes, background first::

        0  background
        1  ghaf

    Images and masks share a stem, organised as::

        <data_root>/
          training/{images,masks}/*.png
          validation/{images,masks}/*.png
          testing/ghaf26/{images,masks}/*.png

    Masks are single-band, with pixel values equal to the class index.

    The tiles are PNG: they are cut from the orthomosaic at the model's input
    size, so each one's position is carried by its name rather than by a
    geotransform, and the georeferencing is reattached when predictions are
    written. ``img_suffix`` and ``seg_map_suffix`` take a tile tree in another
    format -- ``.tif`` for one cut with georeferencing intact.

    Note:
        ``reduce_zero_label`` must stay ``False``. Background is a supervised
        class here, not the ignore index it denotes in ADE20K-style datasets;
        setting it ``True`` would drop that class and leave a one-class problem
        behind a ``num_classes=2`` configuration.
    """

    METAINFO = dict(
        classes=('background', 'ghaf'),
        palette=[[120, 120, 120], [180, 120, 120]],
    )

    def __init__(self,
                 img_suffix: str = '.png',
                 seg_map_suffix: str = '.png',
                 reduce_zero_label: bool = False,
                 **kwargs) -> None:
        if reduce_zero_label:
            raise ValueError(
                'GhafDataset requires reduce_zero_label=False: label 0 is the '
                'background class, not an ignore index.')
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
