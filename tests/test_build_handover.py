"""Assembling the handover bundle.

The failure that matters is a bundle that looks complete and is not, so most
of these check that a missing or empty part is reported rather than passed
over.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import build_handover as B  # noqa: E402


def tree(root: Path, *names: str) -> Path:
    """Make a folder holding one small file per name."""
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'x' * 16)
    return root


def test_a_part_that_was_not_given_is_reported_as_such(tmp_path):
    part = B.add_part('samples', None, tmp_path / 'out', link=False,
                      dry_run=False)
    assert part.status == 'not given'
    assert part.ok, 'an omitted part is a choice, not a fault'


def test_a_source_that_does_not_exist_is_a_problem(tmp_path):
    part = B.add_part('data', tmp_path / 'absent', tmp_path / 'out',
                      link=False, dry_run=False)
    assert not part.ok
    assert 'does not exist' in ' '.join(part.notes)


def test_an_empty_source_is_a_problem(tmp_path):
    """An empty folder copies cleanly and helps nobody."""
    (tmp_path / 'empty').mkdir()
    part = B.add_part('data', tmp_path / 'empty', tmp_path / 'out',
                      link=False, dry_run=False)
    assert not part.ok
    assert 'no files' in ' '.join(part.notes)


def test_a_folder_is_copied_with_its_shape(tmp_path):
    source = tree(tmp_path / 'src', 'a.png', 'nested/b.png')
    part = B.add_part('data', source, tmp_path / 'out', link=False,
                      dry_run=False)

    assert part.ok and part.files == 2 and part.bytes == 32
    assert (tmp_path / 'out' / 'nested' / 'b.png').is_file()


def test_git_and_caches_are_left_behind(tmp_path):
    source = tree(tmp_path / 'code', 'tools/train.py', '.git/config',
                  '__pycache__/x.pyc', 'work_dirs/run/log.txt')
    B.add_part('code', source, tmp_path / 'out', link=False, dry_run=False)

    assert (tmp_path / 'out' / 'tools' / 'train.py').is_file()
    for unwanted in ('.git', '__pycache__', 'work_dirs'):
        assert not (tmp_path / 'out' / unwanted).exists(), unwanted


def test_a_dry_run_writes_nothing_but_still_measures(tmp_path):
    source = tree(tmp_path / 'src', 'a.png')
    part = B.add_part('data', source, tmp_path / 'out', link=False,
                      dry_run=True)

    assert part.files == 1 and part.bytes == 16
    assert not (tmp_path / 'out').exists()


def test_linking_leaves_the_same_content(tmp_path):
    source = tree(tmp_path / 'src', 'a.png')
    B.add_part('data', source, tmp_path / 'out', link=True, dry_run=False)
    assert (tmp_path / 'out' / 'a.png').read_bytes() == b'x' * 16


def test_a_run_reports_what_it_left_out(tmp_path, capsys):
    source = dataset(tmp_path / 'src')
    code = B.main(['--output', str(tmp_path / 'bundle'), '--data', str(source)])

    out = capsys.readouterr().out
    assert code == 0
    assert 'not included' in out
    assert 'samples' in out and 'models' in out


def test_the_bundle_gets_a_readme_and_a_manifest(tmp_path):
    source = dataset(tmp_path / 'src')
    B.main(['--output', str(tmp_path / 'bundle'), '--data', str(source)])

    readme = (tmp_path / 'bundle' / 'README.md').read_text(encoding='utf-8')
    assert 'GETTING_STARTED' in readme
    manifest = json.loads((tmp_path / 'bundle' / 'MANIFEST.json').read_text())
    assert {p['part'] for p in manifest['parts']} == {
        'code', 'models', 'init-weights', 'data', 'samples', 'predictions'}
    assert manifest['total_bytes'] == 6 * 16, 'six split directories'


def test_a_missing_checkpoint_fails_the_run(tmp_path, capsys):
    """Weights that did not arrive must not be reported as a finished bundle."""
    checkpoints = tree(tmp_path / 'ckpt', 'notes.txt')
    code = B.main(['--output', str(tmp_path / 'bundle'),
                   '--checkpoints', str(checkpoints)])

    assert code == 1
    assert 'not in the bundle' in capsys.readouterr().out


def test_an_empty_init_weights_folder_fails_the_run(tmp_path, capsys):
    weights = tree(tmp_path / 'init', 'MANIFEST.json')
    code = B.main(['--output', str(tmp_path / 'bundle'),
                   '--init-weights', str(weights)])

    assert code == 1
    assert 'was the fetch run' in capsys.readouterr().out


# --------------------------------------------------------------------------
# the dataset: named splits, not a folder
# --------------------------------------------------------------------------

def dataset(root: Path) -> Path:
    """A tile tree with one pair per split, laid out as the code expects."""
    from ghaf.splits import directories
    for relative in directories():
        tree(root / relative, 'tile_0001.png')
    return root


def test_every_split_is_copied(tmp_path):
    source = dataset(tmp_path / 'ghaf')
    out = tmp_path / 'bundle'

    part = B.add_dataset('data', source, out, link=False, dry_run=False)

    assert part.ok, part.notes
    assert (out / 'training/images/tile_0001.png').is_file()
    assert (out / 'validation/masks/tile_0001.png').is_file()
    assert (out / 'testing/ghaf26/images/tile_0001.png').is_file()
    assert part.files == 6


def test_what_sits_beside_the_splits_is_left_behind(tmp_path):
    source = dataset(tmp_path / 'ghaf')
    tree(source / 'New Training', 'a.tif', 'b.tif')
    tree(source / 'inference_errors', 'wrong.jpg')
    (source / 'notes.docx').write_bytes(b'x')
    out = tmp_path / 'bundle'

    part = B.add_dataset('data', source, out, link=False, dry_run=False)

    assert part.ok, part.notes
    assert not (out / 'New Training').exists()
    assert not (out / 'inference_errors').exists()
    assert not (out / 'notes.docx').exists()
    assert part.files == 6, 'only the split tiles should be counted'


def test_the_files_left_behind_are_reported(tmp_path):
    source = dataset(tmp_path / 'ghaf')
    tree(source / 'scratch', 'a.tif', 'b.tif', 'c.tif')
    out = tmp_path / 'bundle'

    part = B.add_dataset('data', source, out, link=False, dry_run=False)

    assert any('3 file(s) beside the splits' in note for note in part.notes)


def test_a_tidy_tree_is_reported_without_a_leftover_note(tmp_path):
    source = dataset(tmp_path / 'ghaf')

    part = B.add_dataset('data', source, tmp_path / 'bundle', link=False,
                         dry_run=False)

    assert not any('beside the splits' in note for note in part.notes)


def test_a_missing_split_is_a_problem(tmp_path):
    source = dataset(tmp_path / 'ghaf')
    import shutil
    shutil.rmtree(source / 'validation/masks')

    part = B.add_dataset('data', source, tmp_path / 'bundle', link=False,
                         dry_run=False)

    assert part.status == 'PROBLEM'
    assert any('validation/masks is missing' in note for note in part.notes)


def test_an_empty_split_is_a_problem(tmp_path):
    source = dataset(tmp_path / 'ghaf')
    for tile in (source / 'testing/ghaf26/images').iterdir():
        tile.unlink()

    part = B.add_dataset('data', source, tmp_path / 'bundle', link=False,
                         dry_run=False)

    assert part.status == 'PROBLEM'
    assert any('holds no tiles' in note for note in part.notes)


def test_a_dataset_that_was_not_given_is_reported_as_such(tmp_path):
    part = B.add_dataset('data', None, tmp_path / 'bundle', link=False,
                         dry_run=False)

    assert part.status == 'not given'


def test_a_missing_dataset_root_is_a_problem(tmp_path):
    part = B.add_dataset('data', tmp_path / 'absent', tmp_path / 'bundle',
                         link=False, dry_run=False)

    assert part.status == 'PROBLEM'


def test_a_dry_run_measures_without_writing(tmp_path):
    source = dataset(tmp_path / 'ghaf')
    out = tmp_path / 'bundle'

    part = B.add_dataset('data', source, out, link=False, dry_run=True)

    assert part.files == 6
    assert not out.exists()
