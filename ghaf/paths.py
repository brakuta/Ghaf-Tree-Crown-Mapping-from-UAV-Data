"""One ordering for files, on every operating system.

Sorting a list of :class:`~pathlib.Path` objects sorts them the way the
platform compares paths, and the platforms disagree: Windows compares without
regard to case, Linux and macOS compare by code point. So ``PLOT2_RGB.tif``
comes before ``plot1_rgb.tif`` on one machine and after it on another, from
the same folder and the same command.

That is not a cosmetic difference. The tools that read a folder also offer
``--limit``, which keeps the first *n* entries; a different order there is a
different set of images. The order also decides how every list in a
``summary.json`` reads, and two runs of one command that produce differently
ordered summaries look like two different runs.

So the order is stated here rather than inherited: case-insensitive, with the
case-sensitive form as the tie-breaker so that two names differing only in
case still have one fixed order. Case-insensitive because that is the order a
file browser shows and the order a reader checking a summary against a folder
expects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


def sort_key(path: Path) -> list:
    """The ordering key for one path. See the module docstring."""
    return [(part.lower(), part) for part in Path(path).parts]


def in_stable_order(paths: Iterable[Path]) -> List[Path]:
    """``paths`` sorted identically on Windows, Linux and macOS."""
    return sorted(paths, key=sort_key)
