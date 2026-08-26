"""Ghaf tree-crown mapping from area-wide UAV imagery.

An overlay on a stock mmsegmentation install: importing the submodules
registers this project's dataset and custom backbones, and no file inside
``mmseg`` is patched.

Registration is explicit rather than a side effect of ``import ghaf``, so that
the dependency-free parts of the package (``ghaf.inference.tiling``) stay
importable without mmcv, torch, or a GPU.

The configs pull the registrations in the idiomatic mmengine way::

    custom_imports = dict(
        imports=['ghaf.datasets', 'ghaf.models'], allow_failed_imports=False)

In a script, call :func:`register_all` instead::

    import ghaf
    ghaf.register_all()
"""

__version__ = '1.0.0'

__all__ = ['__version__', 'register_all']


def register_all() -> None:
    """Import the submodules that register with mmseg's registries.

    Idempotent -- repeated calls are cheap, since Python caches the modules.

    Raises:
        ImportError: if mmsegmentation or its dependencies are missing. See
            ``docs/REPRODUCE.md`` for the supported versions.
    """
    from . import datasets, models  # noqa: F401  (imported for registration)
