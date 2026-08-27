"""Tests for the dataset validator.

The case that matters most is the one that looks like success: a tree with no
tiles in it passes every pairing and label check, because there is nothing to
pair and nothing to label. It must not be reported as usable.

Skipped when rasterio is unavailable.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip('rasterio')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import check_dataset as C  # noqa: E402


def write_tile(path: Path, values: np.ndarray) -> Path:
    """Write a single-band GeoTIFF of ``values``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = values.shape
    with rasterio.open(path, 'w', driver='GTiff', width=width, height=height,
                       count=1, dtype=values.dtype, crs='EPSG:32640',
                       transform=rasterio.transform.from_origin(0, 0, 0.05, 0.05)
                       ) as dst:
        dst.write(values, 1)
    return path


def build_split(root: Path, split: str, count: int, size: int = 8,
                suffix: str = '.tif') -> None:
    """Write ``count`` paired image/mask tiles into one split."""
    image_rel, mask_rel = C.SPLITS[split]
    for i in range(count):
        write_tile(root / image_rel / f'tile_{i}{suffix}',
                   np.full((size, size), 7, np.uint8))
        write_tile(root / mask_rel / f'tile_{i}{suffix}',
                   np.zeros((size, size), np.uint8))


def build_tree(root: Path, counts=(2, 2, 2), **kwargs) -> Path:
    for split, count in zip(C.SPLITS, counts):
        build_split(root, split, count, **kwargs)
    return root


# --------------------------------------------------------------------------
# a tree with nothing in it is not a healthy tree
# --------------------------------------------------------------------------

def test_directories_that_exist_but_hold_no_tiles_fail(tmp_path, capsys):
    """The false pass this test exists to prevent."""
    for image_rel, mask_rel in C.SPLITS.values():
        (tmp_path / image_rel).mkdir(parents=True)
        (tmp_path / mask_rel).mkdir(parents=True)

    assert C.main([str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert 'NO TILES' in out
    assert 'not usable' in out


def test_an_empty_split_names_what_it_found_instead(tmp_path, capsys):
    """An unexpected extension is the usual cause, so say so."""
    build_tree(tmp_path)
    image_rel, _ = C.SPLITS['validation']
    for tile in (tmp_path / image_rel).iterdir():
        tile.rename(tile.with_suffix('.jpg'))

    assert C.main([str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert '.jpg' in out, 'the extension actually present should be reported'


def test_one_empty_split_fails_the_whole_tree(tmp_path, capsys):
    """Two good splits do not make up for a third with nothing in it."""
    build_tree(tmp_path, counts=(2, 2, 0))
    for rel in C.SPLITS['testing']:              # exists, but holds no tiles
        (tmp_path / rel).mkdir(parents=True)

    assert C.main([str(tmp_path), '--full']) == 1
    out = capsys.readouterr().out
    assert 'NO TILES' in out and 'is empty' in out


def test_a_missing_directory_is_still_reported_as_missing(tmp_path, capsys):
    build_tree(tmp_path, counts=(2, 2, 0))       # testing/ never created
    assert C.main([str(tmp_path)]) == 1
    assert 'MISSING' in capsys.readouterr().out


# --------------------------------------------------------------------------
# a healthy tree
# --------------------------------------------------------------------------

def test_a_paired_tree_passes(tmp_path, capsys):
    build_tree(tmp_path)
    assert C.main([str(tmp_path), '--full']) == 0
    out = capsys.readouterr().out
    assert 'looks usable' in out
    assert '6 paired tile(s)' in out


def test_uppercase_extensions_are_recognised(tmp_path):
    """Windows tools hand back .TIF often enough to matter."""
    build_tree(tmp_path, suffix='.TIF')
    assert C.main([str(tmp_path), '--full']) == 0


# --------------------------------------------------------------------------
# the faults it exists to catch
# --------------------------------------------------------------------------

def test_an_image_without_a_mask_is_reported(tmp_path, capsys):
    build_tree(tmp_path)
    image_rel, _ = C.SPLITS['training']
    write_tile(tmp_path / image_rel / 'orphan.tif', np.zeros((8, 8), np.uint8))

    assert C.main([str(tmp_path)]) == 1
    assert 'images with no mask' in capsys.readouterr().out


def test_a_mask_of_the_wrong_size_is_reported(tmp_path, capsys):
    build_tree(tmp_path)
    _, mask_rel = C.SPLITS['training']
    write_tile(tmp_path / mask_rel / 'tile_0.tif', np.zeros((4, 4), np.uint8))

    assert C.main([str(tmp_path), '--full']) == 1
    assert 'size mismatch' in capsys.readouterr().out


def test_a_third_class_in_a_mask_is_reported(tmp_path, capsys):
    """The dataset is binary; a stray index would train a silently wrong model."""
    build_tree(tmp_path)
    _, mask_rel = C.SPLITS['training']
    labels = np.zeros((8, 8), np.uint8)
    labels[0, 0] = 3
    write_tile(tmp_path / mask_rel / 'tile_0.tif', labels)

    assert C.main([str(tmp_path), '--full']) == 1
    assert 'unexpected label values' in capsys.readouterr().out


def test_the_json_report_records_the_verdict(tmp_path):
    build_tree(tmp_path)
    report = tmp_path / 'out' / 'report.json'
    assert C.main([str(tmp_path), '--full', '--json', str(report)]) == 0

    written = json.loads(report.read_text())
    assert written['healthy'] is True
    assert {s['split'] for s in written['splits']} == set(C.SPLITS)
    assert all(s['usable'] for s in written['splits'])


def test_the_json_report_records_an_empty_tree_as_unusable(tmp_path):
    for image_rel, mask_rel in C.SPLITS.values():
        (tmp_path / image_rel).mkdir(parents=True)
        (tmp_path / mask_rel).mkdir(parents=True)
    report = tmp_path / 'report.json'

    assert C.main([str(tmp_path), '--json', str(report)]) == 1
    written = json.loads(report.read_text())
    assert written['healthy'] is False
    assert not any(s['usable'] for s in written['splits'])
