"""Keep the released-model registry and the configs in step.

``ghaf/release.py`` is the single source of truth for what was published. If it
drifts from ``configs/ghaf/``, the smoke test would check a model against the
wrong numbers and the exported bundle would document one model while shipping
another. These tests need only mmengine.
"""

import hashlib
import re
from pathlib import Path

import pytest
from mmengine.config import Config

from ghaf.release import RELEASED_MODELS, ReleasedModel, get, iter_models, sha256_of

MODELS = list(iter_models())
KEYS = [m.key for m in MODELS]


@pytest.mark.parametrize('model', MODELS, ids=KEYS)
def test_every_model_has_its_config(model):
    assert model.config_path.is_file(), f'missing {model.config_path}'


def test_registry_covers_exactly_the_configs():
    on_disk = {p.stem for p in (Path(__file__).parent.parent /
                                'configs' / 'ghaf').glob('*.py')}
    assert on_disk == set(RELEASED_MODELS), (
        f'registry and configs disagree: {on_disk ^ set(RELEASED_MODELS)}')


@pytest.mark.parametrize('model', MODELS, ids=KEYS)
def test_declared_architecture_matches_the_config(model):
    """The registry's prose must describe the model the config actually builds."""
    cfg = Config.fromfile(str(model.config_path), import_custom_modules=False)
    assert cfg.model.decode_head.type.replace('Head', '') in \
        model.decode_head.replace('Head', ''), model.key
    if model.neck is None:
        assert cfg.model.get('neck') is None, f'{model.key} declares no neck'
    else:
        assert cfg.model.neck.type == model.neck


@pytest.mark.parametrize('model', MODELS, ids=KEYS)
def test_digests_are_well_formed(model):
    assert re.fullmatch(r'[0-9a-f]{64}', model.sha256), model.key
    assert model.size_bytes > 0
    assert model.parameters > 0


def test_digests_are_unique():
    """Two models sharing a digest would mean the same file was released twice."""
    digests = [m.sha256 for m in MODELS]
    assert len(set(digests)) == len(digests)


def test_models_are_listed_best_first():
    scores = [m.miou for m in MODELS]
    assert scores == sorted(scores, reverse=True), (
        'RELEASED_MODELS should be ordered by descending mIoU')


@pytest.mark.parametrize('model', MODELS, ids=KEYS)
def test_scores_are_plausible_percentages(model):
    assert 0 < model.miou <= 100
    assert 0 < model.fscore <= 100
    assert model.fscore > model.miou, (
        'F1 exceeds IoU for any non-degenerate segmentation')


def test_lookup_of_an_unknown_key_lists_the_known_ones():
    with pytest.raises(KeyError, match='available'):
        get('no-such-model')


def test_sha256_of_reads_incrementally(tmp_path):
    blob = tmp_path / 'blob.bin'
    payload = b'ghaf' * 100_000
    blob.write_bytes(payload)
    assert sha256_of(blob) == hashlib.sha256(payload).hexdigest()
    assert sha256_of(blob, chunk=7) == hashlib.sha256(payload).hexdigest()


def _stub(tmp_path, model: ReleasedModel, payload: bytes) -> Path:
    path = tmp_path / model.checkpoint
    path.write_bytes(payload)
    return path


def _sparse(tmp_path, size: int) -> Path:
    """A file of the given size that costs no disk -- holes, not zeros."""
    path = tmp_path / 'sparse.pth'
    with open(path, 'wb') as handle:
        handle.truncate(size)
    return path


def test_verify_rejects_a_wrong_size(tmp_path):
    model = MODELS[0]
    with pytest.raises(ValueError, match='bytes'):
        model.verify(_stub(tmp_path, model, b'too short'))


def test_verify_rejects_a_correct_size_with_wrong_contents(tmp_path):
    """The size check alone would pass here; the digest must catch it."""
    model = MODELS[0]
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        model.verify(_sparse(tmp_path, model.size_bytes))


def test_verify_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        MODELS[0].verify(tmp_path / 'absent.pth')
