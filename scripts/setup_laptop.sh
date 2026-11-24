#!/usr/bin/env bash
# Linux / macOS laptop setup. Run from the repo root:  bash scripts/setup_laptop.sh
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -e ".[dev]" -f https://k2-fsa.github.io/icefall/piper_phonemize.html
[ -f .env ] || cp .env.example .env
hri --profile laptop models download
hri --profile laptop info
