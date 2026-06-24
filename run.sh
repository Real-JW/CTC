#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
EXAMPLES="${EXAMPLES:-192}"
DEMO_DURATION_S="${DEMO_DURATION_S:-4.0}"
DEMO_START_HZ="${DEMO_START_HZ:-20.0}"
DEMO_END_HZ="${DEMO_END_HZ:-20000.0}"

mkdir -p "$OUTPUT_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Creating virtual environment in $VENV_DIR"
  "$PYTHON" -m venv "$VENV_DIR"
fi

if [[ ! -x "$VENV_DIR/bin/ctc" ]]; then
  echo "Installing ctc into $VENV_DIR"
  "$VENV_DIR/bin/python" -m pip install -e . --no-use-pep517
fi

INPUT_WAV="${1:-$OUTPUT_DIR/demo_input.wav}"
MODEL_PATH="$OUTPUT_DIR/ml_filter_model.json"
STAGE1_OUT="$OUTPUT_DIR/preprocessed.wav"
STAGE2_OUT="$OUTPUT_DIR/simulated_binaural.wav"
METRICS_OUT="$OUTPUT_DIR/metrics.json"

if [[ $# -eq 0 ]]; then
  echo "Generating ${DEMO_DURATION_S}s demo chirp at $INPUT_WAV (${DEMO_START_HZ} Hz to ${DEMO_END_HZ} Hz)"
  "$VENV_DIR/bin/python" - "$INPUT_WAV" "$DEMO_DURATION_S" "$DEMO_START_HZ" "$DEMO_END_HZ" <<'PY'
import math
import sys

from ctc.wav import write_wav_pcm16

path = sys.argv[1]
duration_s = float(sys.argv[2])
start_hz = float(sys.argv[3])
end_hz = float(sys.argv[4])

sample_rate = 48_000
amplitude = 0.25
fade_s = 0.03
total = int(sample_rate * duration_s)
ratio = end_hz / start_hz
phase_scale = 2.0 * math.pi * start_hz * duration_s / math.log(ratio)
samples = []

for n in range(total):
    t = n / sample_rate
    sweep_phase = phase_scale * (ratio ** (t / duration_s) - 1.0)
    fade = min(1.0, t / fade_s, (duration_s - t) / fade_s)
    sample = amplitude * max(0.0, fade) * math.sin(sweep_phase)
    samples.append((sample, sample))

write_wav_pcm16(path, sample_rate, samples)
PY
else
  if [[ ! -f "$INPUT_WAV" ]]; then
    echo "Input WAV not found: $INPUT_WAV" >&2
    exit 1
  fi
fi

echo "Training ML filter model at $MODEL_PATH"
"$VENV_DIR/bin/ctc" train --output "$MODEL_PATH" --examples "$EXAMPLES"

echo "Running Stage 1 and Stage 2"
"$VENV_DIR/bin/ctc" run "$INPUT_WAV" \
  --model "$MODEL_PATH" \
  --stage1-output "$STAGE1_OUT" \
  --stage2-output "$STAGE2_OUT"

echo "Evaluating audio and CTC metrics at $METRICS_OUT"
"$VENV_DIR/bin/ctc" evaluate \
  --input "$INPUT_WAV" \
  --stage1 "$STAGE1_OUT" \
  --stage2 "$STAGE2_OUT" \
  --model "$MODEL_PATH" \
  --output "$METRICS_OUT"

echo
echo "Done."
echo "Input:  $INPUT_WAV"
echo "Model:  $MODEL_PATH"
echo "Stage1: $STAGE1_OUT"
echo "Stage2: $STAGE2_OUT"
echo "Metrics: $METRICS_OUT"
