#!/usr/bin/env python
"""Fetch the ImageNet initialisation weights into a folder that travels.

Run this once, on a machine with internet access, before handing the project
on::

    python tools/fetch_init_weights.py --output D:\\ghaf-project\\init-weights

It builds each published model with its ImageNet initialisation enabled, which
makes every framework download what that backbone needs -- mmengine and
torchvision through torch.hub, timm through the Hugging Face hub -- into the
folder given, instead of into a cache under your home directory.

Afterwards, training on a machine with no internet access is::

    python tools/train.py <config> --data-root <data> \\
        --init-weights D:\\ghaf-project\\init-weights

Nothing else changes: the configs still name the same weights, they are simply
found locally.

Evaluating a released checkpoint needs none of this -- ``tools/test.py``
switches the initialisation off, because the checkpoint replaces it.
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghaf import init_weights  # noqa: E402
from ghaf.release import RELEASED_MODELS, get, iter_models, sha256_of  # noqa: E402


def trust_certifi() -> str:
    """Use certifi's certificate bundle if Python's default is unusable.

    Python on Windows reads the system certificate store, which can come back
    malformed -- ``ssl.SSLError: [ASN1: NOT_ENOUGH_DATA]`` -- and then no
    download works, though pip's do, because pip carries its own bundle. This
    borrows that bundle for the same effect.

    Returns:
        A line describing what was done, for the log.
    """
    for name in ('SSL_CERT_FILE', 'REQUESTS_CA_BUNDLE'):
        if os.environ.get(name):
            return f'{name} already set to {os.environ[name]}'
    try:
        ssl.create_default_context()
    except ssl.SSLError:
        pass
    else:
        return 'the default certificates load; leaving them alone'

    try:
        import certifi
    except ImportError:  # pragma: no cover - environment guard
        return ('the default certificates do not load and certifi is absent; '
                'install it with `pip install certifi` if downloads fail')
    bundle = certifi.where()
    os.environ['SSL_CERT_FILE'] = bundle
    os.environ['REQUESTS_CA_BUNDLE'] = bundle
    return f'the default certificates do not load; using certifi at {bundle}'


def fetch(key: str) -> None:
    """Build one model with its ImageNet initialisation, downloading it."""
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmseg.registry import MODELS

    import ghaf
    ghaf.register_all()
    init_default_scope('mmseg')

    cfg = Config.fromfile(str(get(key).config_path))
    MODELS.build(cfg.model)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output', type=Path, required=True,
                        help='folder to collect the weights into')
    parser.add_argument('--only', nargs='*', metavar='KEY',
                        choices=sorted(RELEASED_MODELS),
                        help='limit to these models')
    args = parser.parse_args(argv)

    print(trust_certifi())
    environment = init_weights.use(args.output)
    for name, value in environment.items():
        print(f'{name}={value}')
    print()

    # Import order matters: Hugging Face reads its variables when the library
    # is imported, so nothing may import timm before the lines above.
    models = [m for m in iter_models() if not args.only or m.key in set(args.only)]
    failures = []
    for model in models:
        print(f'{model.key:30s} {model.backbone:20s} ', end='', flush=True)
        try:
            fetch(model.key)
        except Exception as exc:                                # noqa: BLE001
            failures.append(f'{model.key}: {type(exc).__name__}: {exc}')
            print(f'FAILED: {type(exc).__name__}: {exc}')
        else:
            print('ok')

    collected = init_weights.stored_weights(args.output)
    total = sum(path.stat().st_size for path in collected)
    print(f'\n{len(collected)} file(s), {total / 1e9:.2f} GB in {args.output}')
    for path in collected:
        print(f'  {path.relative_to(args.output)}  '
              f'{path.stat().st_size / 1e6:.1f} MB')

    manifest = args.output / 'MANIFEST.json'
    manifest.write_text(json.dumps({
        'collected': date.today().isoformat(),
        'purpose': 'ImageNet initialisation weights for training and '
                   'fine-tuning, so no download is needed',
        'environment': {name: Path(value).name for name, value
                        in environment.items()},
        'files': [
            {
                'path': str(path.relative_to(args.output)).replace('\\', '/'),
                'size_bytes': path.stat().st_size,
                'sha256': sha256_of(path),
            }
            for path in collected
        ],
    }, indent=2) + '\n', encoding='utf-8')
    print(f'wrote {manifest}')

    if failures:
        print(f'\n{len(failures)} failure(s):')
        for failure in failures:
            print('  -', failure)
        return 1
    if not collected:
        print('\nno weights were collected, which cannot be right')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
