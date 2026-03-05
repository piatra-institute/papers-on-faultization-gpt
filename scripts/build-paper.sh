#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

pandoc docs/PAPER.md \
    -o docs/PAPER.pdf \
    --pdf-engine=xelatex \
    --resource-path=docs \
    -V mainfont="Palatino" \
    -V mathfont="Palatino"

echo "docs/PAPER.pdf"
