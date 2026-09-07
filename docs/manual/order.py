"""The order of the chapters, in one place.

The manual numbers its chapters by counting them as they are emitted, so
moving one renumbers everything after it -- and every sentence that says "the
remedy is in chapter 13" then points at the wrong chapter. There were
thirty-odd such sentences, and nothing checked them.

So a chapter is referred to by name here and the number is looked up:

    para(f'... {ch("errors")} has the remedy.')     -> "chapter 15"
    ['`train.py`', 'Trains a model', Ch('training')] -> "7"

``story()`` in each content module emits chapters in this order and asserts
that the number the typesetter reached is the number this file promises, so a
chapter added in the wrong place fails the build rather than the reader.

The order itself follows the order the work is done in: get the data right,
understand what the configuration will do with it, train, score, then predict
-- tiles first, then one image, then a folder of them.
"""

from __future__ import annotations

#: Chapter keys, in the order they appear.
ORDER = (
    'what',            # what the system does, and what was delivered
    'repository',      # the repository, folder by folder
    'installing',      # the environment
    'verifying',       # the three checks
    'data',            # the tile tree, and building one
    'configuration',   # configs/_base_/ghaf.py, field by field
    'training',        # train.py
    'evaluating',      # test.py, the published scores
    'tiles',           # predict_split.py, one mask per tile
    'orthomosaic',     # ghaf.inference.large_image, one image
    'folder',          # predict_folder.py, many images
    'reviewing',       # opening the outputs, and the figures to check
    'adapting',        # fine-tuning on a new site
    'handover',        # passing the system on
    'errors',          # the error catalogue
    'reference',       # tables, defaults, and what was never established
)


def n(key: str) -> int:
    """The number of the chapter called ``key``, counting from one."""
    try:
        return ORDER.index(key) + 1
    except ValueError:
        raise KeyError(
            f'no chapter named {key!r}; the chapters are {", ".join(ORDER)}'
        ) from None


def Ch(key: str) -> str:  # noqa: N802 - it reads as a chapter number
    """The bare number, for the narrow reference column of a table."""
    return str(n(key))


def ch(key: str) -> str:
    """``chapter 15``, for the middle of a sentence."""
    return f'chapter {n(key)}'


def chapters(*keys: str) -> str:
    """``chapters 3, 4 and 10``, in the order given."""
    numbers = [str(n(key)) for key in keys]
    if len(numbers) == 1:
        return f'chapter {numbers[0]}'
    return f'chapters {", ".join(numbers[:-1])} and {numbers[-1]}'


def emitted(typeset, key: str) -> None:
    """Fail the build if the chapter just emitted is not where it belongs.

    Called by ``story()`` after each chapter. Without it the promise this
    file makes to every cross-reference is unchecked, and a chapter inserted
    in the wrong place would silently make every later reference wrong -- the
    exact failure the file exists to prevent.
    """
    reached = typeset._counter['ch']
    if reached != n(key):
        raise AssertionError(
            f'chapter {key!r} was emitted as {reached}, but order.py places '
            f'it at {n(key)}: story() and ORDER disagree')
