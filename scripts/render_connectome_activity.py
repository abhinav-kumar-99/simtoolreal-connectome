#!/usr/bin/env python3
"""Render a saved activity trace and its synchronized rollout using YAML."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from simtoolreal_shared.activity_trace import repository_path
from simtoolreal_shared.anatomical_activity import render_activity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(repository_path(args.config).read_text())
    print(json.dumps(render_activity(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
