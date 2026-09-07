"""One ordering for files, whichever operating system is reading them.

The order a folder is listed in decides which images ``--limit`` keeps and
how every list in a ``summary.json`` reads, so it has to be a decision the
project makes rather than one the platform makes for it.
"""

from pathlib import PurePosixPath, PureWindowsPath

from ghaf.paths import in_stable_order, sort_key


def order(names, kind=PurePosixPath):
    return [str(p) for p in in_stable_order(kind(n) for n in names)]


def test_names_differing_only_in_case_keep_a_fixed_order():
    assert order(['b.tif', 'A.tif', 'a.tif', 'B.tif']) == [
        'A.tif', 'a.tif', 'B.tif', 'b.tif']


def test_case_does_not_decide_the_order_of_different_names():
    """The failure that started this: PLOT2 sorted before plot1 on Linux."""
    assert order(['PLOT2_RGB.tif', 'plot1_rgb.tif']) == [
        'plot1_rgb.tif', 'PLOT2_RGB.tif']


def test_a_windows_path_is_ordered_the_same_way_as_a_posix_one():
    windows = order([r'D:\i\PLOT2.tif', r'D:\i\plot1.tif'], PureWindowsPath)
    assert [p.rsplit('\\', 1)[-1] for p in windows] == [
        'plot1.tif', 'PLOT2.tif']


def test_a_folder_sorts_by_folder_before_file_name():
    assert order(['top.tif', 'plot-3/north.tif']) == [
        'plot-3/north.tif', 'top.tif']


def test_the_key_is_comparable_across_depths():
    assert sort_key(PurePosixPath('a/b')) > sort_key(PurePosixPath('a'))
