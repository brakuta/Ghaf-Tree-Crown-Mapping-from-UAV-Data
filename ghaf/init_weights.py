"""Keeping the ImageNet initialisation weights with the project.

Training starts every backbone from ImageNet weights, and the frameworks fetch
those from the internet the first time they are needed: mmengine and
torchvision through ``torch.hub``, timm through the Hugging Face hub. Five of
the six are named as URLs in the configs; DPN-98's are resolved by timm.

That is fine on a connected machine and a dead end everywhere else -- an
offline workstation, a machine whose certificate store Python cannot read, or
any time after one of those URLs stops resolving. So the weights are fetched
once, on a machine that can, and travel with the project.

Rather than rewrite each config to a local path -- six edits, one per
backbone, each in a different framework's convention -- this points every
backend's cache at one directory. What the frameworks fetch, they fetch into
the bundle; what they find there, they do not fetch again.

Layout::

    <directory>/
      torch/hub/checkpoints/     mmengine, torchvision
      huggingface/hub/           timm

Set the environment before importing timm: Hugging Face reads its variables
when the library is imported, whereas ``torch.hub`` reads them per call.
"""

from __future__ import annotations

import os
import ssl
from pathlib import Path
from typing import Dict, List, Optional

#: Environment variables that redirect each backend's download cache.
CACHE_VARIABLES = ('TORCH_HOME', 'HF_HOME', 'HUGGINGFACE_HUB_CACHE')

#: Extensions the caches store weights under.
WEIGHT_SUFFIXES = ('.pth', '.pt', '.tar', '.bin', '.safetensors', '.ckpt')


def cache_environment(directory) -> Dict[str, str]:
    """The environment that makes every backend read from ``directory``.

    Args:
        directory: the folder holding the initialisation weights.

    Returns:
        Variable name -> value, ready to put in ``os.environ``.
    """
    directory = Path(directory).expanduser().resolve()
    return {
        'TORCH_HOME': str(directory / 'torch'),
        'HF_HOME': str(directory / 'huggingface'),
        'HUGGINGFACE_HUB_CACHE': str(directory / 'huggingface' / 'hub'),
    }


def use(directory, create: bool = True) -> Dict[str, str]:
    """Point this process's downloads at ``directory``.

    Call before importing timm; ``torch.hub`` is read per call, but Hugging
    Face captures its variables at import.

    Args:
        directory: the folder holding the initialisation weights.
        create: make the cache directories if they are absent. Set ``False``
            to read from a bundle without writing to it.

    Returns:
        The variables that were set.
    """
    environment = cache_environment(directory)
    for name, value in environment.items():
        if create:
            Path(value).mkdir(parents=True, exist_ok=True)
        os.environ[name] = value
    return environment


def stored_weights(directory) -> List[Path]:
    """Every weight file under the caches in ``directory``, sorted.

    Used to report what a fetch collected and to check a bundle carries it.
    """
    directory = Path(directory)
    found = [
        path for path in directory.rglob('*')
        if path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES
    ]
    return sorted(found)


def use_certifi() -> Optional[str]:
    """Verify HTTPS against certifi's bundle rather than the system store.

    Python on Windows builds its default context from the Windows certificate
    store, and a store it cannot parse produces
    ``ssl.SSLError: [ASN1: NOT_ENOUGH_DATA]`` on every download. Setting
    ``SSL_CERT_FILE`` does not help: ``ssl.create_default_context`` reads the
    system store as well, and fails there before the variable is consulted.

    Passing a bundle explicitly skips the system store entirely, which is what
    pip does -- hence pip working while ``torch.hub`` does not.

    Returns:
        The bundle now in use, or ``None`` if certifi is not installed.
    """
    try:
        import certifi
    except ImportError:      # pragma: no cover - environment guard
        return None

    bundle = certifi.where()
    context = ssl.create_default_context(cafile=bundle)
    ssl._create_default_https_context = lambda *args, **kwargs: context
    os.environ.setdefault('SSL_CERT_FILE', bundle)
    os.environ.setdefault('REQUESTS_CA_BUNDLE', bundle)
    return bundle
