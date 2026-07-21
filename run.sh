#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-.venv/bin/python}"
BRIR_PATH="${1:-BRIR/data/mit_kemar_anechoic_30deg_48k.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"

mkdir -p "$OUTPUT_DIR"
"$PYTHON" -m Model2CTC.cli design \
  --brir "$BRIR_PATH" \
  --config configs/model2ctc.json \
  --output "$OUTPUT_DIR/ctc_filter.json"
"$PYTHON" -m Model2CTC.cli export \
  --filter "$OUTPUT_DIR/ctc_filter.json" \
  --output "$OUTPUT_DIR/model2ctc_coefficients.h"

echo "Designed filter: $OUTPUT_DIR/ctc_filter.json"
echo "Edge header:    $OUTPUT_DIR/model2ctc_coefficients.h"
