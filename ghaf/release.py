"""The published models, as a single source of truth.

Every number here was measured from the released checkpoint itself rather than
transcribed: parameter counts are exact sums over each ``state_dict``, and the
digests are SHA-256 over the ``.pth`` files as distributed.

Consumers:

* ``tools/smoke_test.py`` -- checks a freshly built model against
  :attr:`ReleasedModel.parameters`;
* ``tools/export_release.py`` -- assembles the shareable bundle and verifies
  each checkpoint against :attr:`ReleasedModel.sha256`;
* ``tests/test_release.py`` -- keeps this registry and ``configs/ghaf/`` in step.

Keeping them in one place means a model cannot be documented one way and
shipped another.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator, Optional

#: Bytes read per digest update. Large enough to keep I/O sequential, small
#: enough that a 1 GB checkpoint never lands in memory at once.
_DIGEST_CHUNK = 1 << 20

#: Environment variable that overrides config discovery, for installs that
#: place ``configs/`` somewhere this module cannot infer.
CONFIG_DIR_ENV = 'GHAF_CONFIG_DIR'


@lru_cache(maxsize=1)
def config_dir() -> Path:
    """Locate ``configs/ghaf/``.

    If ``GHAF_CONFIG_DIR`` is set it is used, and a bad value raises rather
    than falling back to somewhere the caller did not ask for. Otherwise the
    repository layout (an editable install or source checkout) is tried, then
    the current working directory.

    Returns:
        The directory holding the six model configs.

    Raises:
        FileNotFoundError: naming every location tried, because "config not
            found" is otherwise an unhelpful thing to debug.
    """
    override = os.environ.get(CONFIG_DIR_ENV)
    if override:
        # An explicit override is honoured or refused -- never silently
        # ignored in favour of a directory the caller did not ask for.
        chosen = Path(override)
        if not (chosen.is_dir() and any(chosen.glob('*.py'))):
            raise FileNotFoundError(
                f'{CONFIG_DIR_ENV} is set to {override!r}, which is not a '
                f'directory containing config files.')
        return chosen

    candidates = []
    package_parent = Path(__file__).resolve().parent.parent
    candidates += [
        package_parent / 'configs' / 'ghaf',
        package_parent / 'configs',
        Path.cwd() / 'configs' / 'ghaf',
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob('*.py')):
            return candidate
    raise FileNotFoundError(
        'could not locate configs/ghaf. Tried:\n  ' +
        '\n  '.join(str(c) for c in candidates) +
        f'\nInstall the project in editable mode (`pip install -e .`) or set '
        f'{CONFIG_DIR_ENV} to the directory holding the model configs.')


@dataclass(frozen=True)
class ReleasedModel:
    """One published model: its configuration, weights and measured results."""

    key: str
    """Config stem and release directory name, e.g. ``dpn98_fpn``."""

    backbone: str
    decode_head: str
    neck: Optional[str]

    checkpoint: str
    """Checkpoint file name as released."""

    sha256: str
    size_bytes: int

    parameters: int
    """Exact sum of ``numel()`` over the checkpoint's ``state_dict``."""

    miou: float
    """mIoU (%) on the held-out test split."""

    fscore: float
    """F1 (%) on the held-out test split."""

    @property
    def config_path(self) -> Path:
        """This model's config file.

        Raises:
            FileNotFoundError: if ``configs/ghaf`` cannot be located; see
                :func:`config_dir`.
        """
        return config_dir() / f'{self.key}.py'

    def verify(self, checkpoint_path: Path) -> None:
        """Check a checkpoint file against the released size and digest.

        Args:
            checkpoint_path: the ``.pth`` to check.

        Raises:
            FileNotFoundError: if the file does not exist.
            ValueError: if its size or digest differs from the release.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        actual_size = checkpoint_path.stat().st_size
        if actual_size != self.size_bytes:
            raise ValueError(
                f'{checkpoint_path.name}: expected {self.size_bytes:,} bytes, '
                f'found {actual_size:,}. The file is truncated or is not the '
                f'released checkpoint.')

        digest = sha256_of(checkpoint_path)
        if digest != self.sha256:
            raise ValueError(
                f'{checkpoint_path.name}: SHA-256 mismatch.\n'
                f'  expected {self.sha256}\n'
                f'  found    {digest}')


def sha256_of(path: Path, chunk: int = _DIGEST_CHUNK) -> str:
    """SHA-256 of a file, read incrementally so size does not matter."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


#: The six published models, best result first.
RELEASED_MODELS: Dict[str, ReleasedModel] = {
    model.key: model
    for model in (
        ReleasedModel(
            key='fastvit-ma36_mask2former',
            backbone='FastViT-MA36', decode_head='Mask2Former', neck=None,
            checkpoint='best_mIoU_iter_3500.pth',
            sha256='f26cd5257b55058f81c52349d51c888a382ca54a924da550bd0a711bcfafa84a',
            size_bytes=252_650_755, parameters=62_549_115,
            miou=79.32, fscore=87.22),
        ReleasedModel(
            key='poolformer-s36_fpn',
            backbone='PoolFormer-S36', decode_head='FPNHead', neck='FPN',
            checkpoint='iter_10200.pth',
            sha256='59683e3788548494f50206dffe7b3ce2a91610a7e1039e4e7f5999401b0156e0',
            size_bytes=416_437_419, parameters=34_600_137,
            miou=78.65, fscore=86.72),
        ReleasedModel(
            key='dpn98_fpn',
            backbone='DPN-98', decode_head='FPNHead', neck='FPN',
            checkpoint='best_mIoU_iter_14000.pth',
            sha256='e292f2262f132f57f0a81e9dc8be169b7a2696d397ab0a5b1a898dff28c964cf',
            size_bytes=263_385_401, parameters=65_346_639,
            miou=78.19, fscore=86.35),
        ReleasedModel(
            key='convnext-small_upernet',
            backbone='ConvNeXt-S', decode_head='UPerHead', neck=None,
            checkpoint='iter_14000.pth',
            sha256='8435410e2514054f53a68302b2b70c60b86dfdf1ab50ec385e6664f65d7b1036',
            size_bytes=983_511_196, parameters=81_776_049,
            miou=78.02, fscore=86.20),
        ReleasedModel(
            key='resnet-50_mask2former',
            backbone='ResNet-50', decode_head='Mask2Former', neck=None,
            checkpoint='best_mIoU_iter_38500.pth',
            sha256='5a79f618902be032d8a38c27b18c3e5ed7bd0605297b272626ae0cbd56fdf5b2',
            size_bytes=196_545_149, parameters=44_056_504,
            miou=77.69, fscore=85.98),
        ReleasedModel(
            key='efficientnet-b3_fpn',
            backbone='EfficientNet-B3', decode_head='FPNHead', neck='FPN',
            checkpoint='iter_6800.pth',
            sha256='9d6131e21a1ec5f38f36adf8c570d0c09fdd8d067e73ef57509459b970a5aa60',
            size_bytes=108_104_299, parameters=13_734_524,
            miou=70.77, fscore=80.29),
    )
}


def iter_models() -> Iterator[ReleasedModel]:
    """The published models in ranked order, best mIoU first."""
    return iter(RELEASED_MODELS.values())


def get(key: str) -> ReleasedModel:
    """Look up a published model by key.

    Raises:
        KeyError: with the available keys listed, rather than a bare miss.
    """
    try:
        return RELEASED_MODELS[key]
    except KeyError:
        raise KeyError(
            f'unknown model {key!r}; available: '
            f'{", ".join(sorted(RELEASED_MODELS))}') from None


__all__ = ['CONFIG_DIR_ENV', 'RELEASED_MODELS', 'ReleasedModel', 'config_dir',
           'get', 'iter_models', 'sha256_of']
