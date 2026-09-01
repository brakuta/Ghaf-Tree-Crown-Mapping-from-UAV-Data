"""Config edits that a command line has to make correctly.

mmengine's ``--cfg-options`` sets keys by name, which is right for values the
runner reads at run time and quietly wrong for a value the config file used
while it was being parsed. ``data_root`` is one of the latter: it is a plain
module-level variable, copied into each dataloader as the file is read, so
setting it afterwards changes a key nothing looks at and leaves the
dataloaders pointing where they were.

This module makes the substitution the way it has to be made -- once per
dataset -- so ``--data-root`` moves every split together.
"""

from __future__ import annotations

import inspect
from typing import Iterable, List

#: The dataloaders a config may define, in the order they are reported.
DATALOADERS = ('train_dataloader', 'val_dataloader', 'test_dataloader')


def _point_at(dataset, root: str) -> None:
    """Set ``data_root`` on a dataset, or on the one a wrapper holds."""
    inner = dataset.get('dataset')
    if inner is not None:            # RepeatDataset and friends
        _point_at(inner, root)
    else:
        dataset['data_root'] = root


def set_data_root(cfg, root, loaders: Iterable[str] = DATALOADERS) -> List[str]:
    """Point every split at ``root``.

    Args:
        cfg: a parsed ``mmengine`` config.
        root: the dataset root, e.g. ``/data/ghaf``.
        loaders: which dataloaders to move; the default is all of them.

    Returns:
        The names of the dataloaders that were changed, so a caller can say
        what it did rather than assume.
    """
    root = str(root)
    cfg['data_root'] = root
    changed = []
    for name in loaders:
        loader = cfg.get(name)
        if loader is None or loader.get('dataset') is None:
            continue
        _point_at(loader['dataset'], root)
        changed.append(name)
    return changed


def skip_imagenet_weights(backbone: dict, cls) -> List[str]:
    """Build a backbone without fetching its ImageNet weights.

    Evaluating a checkpoint does not need them: every tensor comes from the
    checkpoint, which is loaded afterwards and overwrites whatever
    initialisation produced. Fetching them anyway costs a download of a few
    hundred megabytes and makes an offline machine fail at a step whose result
    is discarded.

    Which knob to turn depends on where the weights would come from, and the
    two conventions in play disagree about the type of ``pretrained``:

    * mmseg's backbones take ``init_cfg``, and their ``pretrained`` is a
      checkpoint path, so "no weights" is ``None`` -- ``False`` is rejected;
    * DPN-98 loads through timm, whose ``pretrained`` is a flag, so "no
      weights" is ``False``.

    The default the class declares settles which it is, and only arguments the
    backbone actually accepts are set.

    Args:
        backbone: the ``model.backbone`` section of a config, edited in place.
        cls: the backbone class, as the registry resolves its ``type``.

    Returns:
        The argument names that were set, so a caller can report what it did.
    """
    parameters = inspect.signature(cls.__init__).parameters
    catch_all = any(p.kind is inspect.Parameter.VAR_KEYWORD
                    for p in parameters.values())
    changed = []

    if 'init_cfg' in parameters or catch_all:
        backbone['init_cfg'] = None
        changed.append('init_cfg')

    declared = parameters.get('pretrained')
    if declared is not None or catch_all:
        default = getattr(declared, 'default', None)
        backbone['pretrained'] = False if isinstance(default, bool) else None
        changed.append('pretrained')

    return changed
