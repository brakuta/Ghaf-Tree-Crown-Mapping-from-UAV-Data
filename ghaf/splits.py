"""Where each dataset split lives, relative to the dataset root.

One statement of the layout, imported by everything that has to agree with
it: the dataset class that trains on it, the checker that validates it, the
per-tile predictor, and the tool that copies it into a handover. A working
tile tree usually sits inside a larger working directory -- earlier surveys,
scratch output, notes -- so "the dataset" has to mean these directories
specifically, not whatever else shares the folder.

Dependency-free by design: the checker runs in an environment without
mmsegmentation, so nothing here may import it.
"""

from __future__ import annotations

from typing import Dict, Iterator, Tuple

#: split name -> (images directory, masks directory), relative to the root.
SPLITS: Dict[str, Tuple[str, str]] = {
    'training': ('training/images', 'training/masks'),
    'validation': ('validation/images', 'validation/masks'),
    'testing': ('testing/ghaf26/images', 'testing/ghaf26/masks'),
}


def directories() -> Iterator[str]:
    """Every directory the dataset occupies, in split order."""
    for images, masks in SPLITS.values():
        yield images
        yield masks
