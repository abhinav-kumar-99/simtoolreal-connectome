#!/usr/bin/env python3
"""Prepare MaleCNS graph artifacts from one YAML contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from simtoolreal_shared.connectome_data import load_yaml, prepare_connectome


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    config_path = (
        args.config if args.config.is_absolute() else repository_root / args.config
    )
    kind = load_yaml(config_path).get('preparation_kind', 'original_csv')
    if kind == 'compact_paths':
        from simtoolreal_shared.compact_connectome import prepare_compact_connectome
        manifest = prepare_compact_connectome(config_path, repository_root)
    elif kind == 'original_csv':
        manifest = prepare_connectome(config_path, repository_root)
    else:
        raise ValueError(f'Unknown preparation_kind: {kind}')
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
