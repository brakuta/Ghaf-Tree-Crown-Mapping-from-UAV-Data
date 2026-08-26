"""Dataset definition for Ghaf (*Prosopis cineraria*) crown segmentation."""

from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class GhafDataset(BaseSegDataset):
    """Binary Ghaf-crown segmentation over UAV orthomosaic tiles.

    Two classes, background first::

        0  background
        1  ghaf

    Images and masks are GeoTIFFs sharing a stem, organised as::

        <data_root>/
          training/{images,masks}/*.tif
          validation/{images,masks}/*.tif
          testing/ghaf26/{images,masks}/*.tif

    Masks are single-band, with pixel values equal to the class index.

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
                 img_suffix: str = '.tif',
                 seg_map_suffix: str = '.tif',
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
