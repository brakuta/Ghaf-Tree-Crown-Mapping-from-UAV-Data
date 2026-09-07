"""The checkpoint folder is checked before ninety seconds of model building.

A wrong --checkpoints path is the commonest way to run this tool: an example
path copied out of the documentation, or a bundle unpacked somewhere else.
Reported per model, it produces six failures that read like six missing
checkpoints. Reported by the parser, it is one line, before anything is built.

A subprocess, because the message belongs to the command line rather than to
a function, and because this way the test needs none of the model stack.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / 'tools' / 'smoke_test.py'


def run(*args):
    return subprocess.run([sys.executable, str(TOOL), *args],
                          capture_output=True, text=True, cwd=ROOT)


def test_a_checkpoint_folder_that_is_not_there_is_refused(tmp_path):
    finished = run('--checkpoints', str(tmp_path / 'not-here'))

    assert finished.returncode == 2, finished.stderr
    assert 'not a directory' in finished.stderr
    assert 'not-here' in finished.stderr, 'the path itself must appear'


def test_a_file_is_not_a_folder_of_checkpoints(tmp_path):
    weights = tmp_path / 'best_mIoU_iter_3500.pth'
    weights.write_bytes(b'')

    finished = run('--checkpoints', str(weights))

    assert finished.returncode == 2, finished.stderr
    assert 'not a directory' in finished.stderr


def test_the_wrong_path_is_named_once_not_once_per_model(tmp_path):
    finished = run('--checkpoints', str(tmp_path / 'not-here'))

    assert finished.stderr.count('not a directory') == 1
    assert 'no best_mIoU_iter_3500.pth under' not in finished.stdout
