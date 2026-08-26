"""Dataset definition for the Ghaf (*Prosopis cineraria*) crown-mapping task."""

from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class GhafDataset(BaseSegDataset):
    """Binary Ghaf-crown segmentation over UAV orthomosaic tiles.

    Two classes, background first::

        0  background
        1  ghaf

    ``reduce_zero_label`` is ``False``: label 0 is a real, supervised class
    here, not the "ignore" index it denotes in ADE20K-style datasets. Setting
    it to ``True`` would silently drop the background class and leave a
    one-class problem behind a ``num_classes=2`` config.

    Images and masks are single-channel-indexed GeoTIFFs sharing a stem::

        <data_root>/
          training/{images,masks}/*.tif      7005 tiles
          validation/{images,masks}/*.tif     869 tiles
          testing/ghaf26/{images,masks}/*.tif 767 tiles

    Notes:
        The original code obtained this by editing ``mmseg/datasets/ade.py``
        in place, which is why every archived config and working directory is
        named ``*ade20k*`` despite ADE20K never being involved.
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
