#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

pandoc docs/PAPER.md \
    -o docs/PAPER.pdf \
    --pdf-engine=xelatex \
    --resource-path=docs \
    -H scripts/preamble.tex \
    -V mainfont="Palatino" \
    -V mathfont="Palatino" \
    -V fontsize=11pt \
    -V geometry:"margin=1in" \
    -V linestretch=1.15

echo "docs/PAPER.pdf"
