"""Every command-line tool must run from the checkout it lives in.

Two copies of this project on one machine is the normal state of a handover:
the repository the work was done in, and the copy inside the bundle. If a
tool reaches for an installed ``ghaf`` rather than the one beside it, the
command a reader copies out of the documentation runs code from somewhere
else -- silently, and only on the machine that has both.

So each tool is asked for its help text in a subprocess whose only route to
the package is the tool's own location.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOLS = sorted(p for p in (ROOT / 'tools').glob('*.py')
               if not p.name.startswith('_'))


@pytest.mark.parametrize('tool', TOOLS, ids=lambda p: p.name)
def test_a_tool_answers_for_itself(tool):
    finished = subprocess.run([sys.executable, str(tool), '--help'],
                              capture_output=True, text=True, cwd=ROOT.parent)
    assert finished.returncode == 0, finished.stderr
    assert 'usage:' in finished.stdout


def test_there_are_tools_to_check():
    assert len(TOOLS) >= 8, 'the glob stopped finding the tools'
