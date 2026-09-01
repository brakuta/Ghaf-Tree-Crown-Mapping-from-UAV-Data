"""Checking that the project is running in the Python it was installed into.

The commonest way for any of this to fail is the simplest: several conda
environments on one machine, and the command typed in the wrong one. The
error that follows -- ``No module named 'mmseg'`` -- names a missing package
rather than the mistake, and repeats once per model if a tool is working
through a list.

So it is asked once, up front, and answered with the interpreter that is
actually running and the environment it belongs to.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Sequence

#: The frameworks a tool needs before it can build a model.
STACK = ('mmengine', 'mmcv', 'mmseg', 'mmdet', 'mmpretrain')


def missing_packages(packages: Sequence[str] = STACK) -> list:
    """Which of ``packages`` cannot be imported here."""
    absent = []
    for name in packages:
        try:
            importlib.import_module(name)
        except ImportError:
            absent.append(name)
    return absent


def describe_interpreter() -> str:
    """The running Python, and its conda environment when there is one."""
    environment = os.environ.get('CONDA_DEFAULT_ENV')
    named = f'conda environment "{environment}"' if environment else 'this Python'
    return f'{named}:\n  {sys.executable}'


def require_stack(packages: Sequence[str] = STACK) -> None:
    """Raise a message a reader can act on, if the stack is not installed.

    Raises:
        ModuleNotFoundError: naming what is absent, which interpreter was
            asked, and what to do about it.
    """
    absent = missing_packages(packages)
    if not absent:
        return
    verb = 'is' if len(absent) == 1 else 'are'
    raise ModuleNotFoundError(
        f'{", ".join(absent)} {verb} not installed in {describe_interpreter()}\n\n'
        f'Activate the environment the project was installed into -- '
        f'`conda activate ghaf` if you followed the guide -- and run the '
        f'command again. If this is a new machine, "Installation" in '
        f'README.md sets it up from scratch.')
