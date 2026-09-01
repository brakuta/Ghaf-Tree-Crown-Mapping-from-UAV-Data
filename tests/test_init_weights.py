"""Redirecting the frameworks' download caches into the project.

Pure environment and path handling, so this runs anywhere -- no torch, no
timm, no network.
"""

import os
from pathlib import Path

from ghaf import init_weights


def test_every_backend_is_pointed_at_one_folder(tmp_path):
    environment = init_weights.cache_environment(tmp_path)

    assert set(environment) == set(init_weights.CACHE_VARIABLES)
    for value in environment.values():
        assert Path(value).is_relative_to(tmp_path.resolve()), value


def test_torch_and_hugging_face_get_separate_trees(tmp_path):
    environment = init_weights.cache_environment(tmp_path)
    assert Path(environment['TORCH_HOME']).name == 'torch'
    assert Path(environment['HF_HOME']).name == 'huggingface'
    assert Path(environment['HUGGINGFACE_HUB_CACHE']).parent.name == 'huggingface'


def test_a_relative_folder_is_resolved(tmp_path, monkeypatch):
    """A relative path would send the download somewhere unintended."""
    monkeypatch.chdir(tmp_path)
    environment = init_weights.cache_environment('weights')
    assert Path(environment['TORCH_HOME']).is_absolute()


def test_use_sets_the_environment_and_makes_the_folders(tmp_path, monkeypatch):
    for name in init_weights.CACHE_VARIABLES:
        monkeypatch.delenv(name, raising=False)

    environment = init_weights.use(tmp_path / 'init')

    for name, value in environment.items():
        assert os.environ[name] == value
        assert Path(value).is_dir()


def test_a_bundle_can_be_read_without_writing_to_it(tmp_path, monkeypatch):
    for name in init_weights.CACHE_VARIABLES:
        monkeypatch.delenv(name, raising=False)

    init_weights.use(tmp_path / 'read-only', create=False)

    assert not (tmp_path / 'read-only').exists()
    assert os.environ['TORCH_HOME'].endswith('torch')


def test_the_weights_in_a_bundle_are_listed(tmp_path):
    torch_cache = tmp_path / 'torch' / 'hub' / 'checkpoints'
    torch_cache.mkdir(parents=True)
    (torch_cache / 'resnet50.pth').write_bytes(b'weights')
    (torch_cache / 'fastvit_ma36.pth.tar').write_bytes(b'weights')
    hub = tmp_path / 'huggingface' / 'hub' / 'models--timm--dpn98'
    hub.mkdir(parents=True)
    (hub / 'model.safetensors').write_bytes(b'weights')
    (tmp_path / 'MANIFEST.json').write_text('{}')

    found = [p.name for p in init_weights.stored_weights(tmp_path)]

    assert set(found) == {'resnet50.pth', 'fastvit_ma36.pth.tar',
                          'model.safetensors'}
    assert 'MANIFEST.json' not in found, 'the manifest is not a weight file'


def test_an_empty_bundle_lists_nothing(tmp_path):
    assert init_weights.stored_weights(tmp_path) == []
