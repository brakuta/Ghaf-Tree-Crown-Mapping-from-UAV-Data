"""Telling someone they are in the wrong environment, once and clearly.

The commonest failure in this project has nothing to do with the models: it
is a command typed in a conda environment that does not have the stack
installed. Pure import checking, so this runs anywhere.
"""

import pytest

from ghaf import environment


def test_packages_that_are_present_are_not_reported():
    assert environment.missing_packages(['sys', 'json']) == []


def test_packages_that_are_absent_are_named():
    absent = environment.missing_packages(['sys', 'definitely_not_installed'])
    assert absent == ['definitely_not_installed']


def test_nothing_is_raised_when_the_stack_is_there():
    environment.require_stack(['sys', 'json'])          # must not raise


def test_the_message_names_the_interpreter_that_was_asked():
    with pytest.raises(ModuleNotFoundError) as caught:
        environment.require_stack(['definitely_not_installed'])

    message = str(caught.value)
    assert 'definitely_not_installed' in message
    assert 'python' in message.lower(), 'the running interpreter should appear'
    assert 'conda activate ghaf' in message, 'say what to do about it'


def test_the_conda_environment_is_named_when_there_is_one(monkeypatch):
    monkeypatch.setenv('CONDA_DEFAULT_ENV', 'MMsegm2024')
    assert 'MMsegm2024' in environment.describe_interpreter()


def test_a_plain_interpreter_is_described_without_inventing_one(monkeypatch):
    monkeypatch.delenv('CONDA_DEFAULT_ENV', raising=False)
    description = environment.describe_interpreter()
    assert 'conda' not in description


def test_one_missing_package_reads_as_one(monkeypatch):
    monkeypatch.delenv('CONDA_DEFAULT_ENV', raising=False)
    with pytest.raises(ModuleNotFoundError, match='is not installed'):
        environment.require_stack(['definitely_not_installed'])


def test_several_missing_packages_read_as_several(monkeypatch):
    monkeypatch.delenv('CONDA_DEFAULT_ENV', raising=False)
    with pytest.raises(ModuleNotFoundError, match='are not installed'):
        environment.require_stack(['definitely_not_installed', 'nor_this_one'])


def test_the_default_is_the_frameworks_a_model_needs():
    assert set(environment.STACK) == {
        'mmengine', 'mmcv', 'mmseg', 'mmdet', 'mmpretrain'}
