# CTC

ML-guided low-latency crosstalk cancellation reference implementation.

This repo currently contains a dependency-free Python reference engine:

- Stage 1 reads a stereo 48 kHz WAV and writes ML-guided crosstalk-cancelled loudspeaker feeds to `preprocessed.wav`.
- Stage 2 simulates direct-path binaural playback and writes `simulated_binaural.wav`.
- `ctc train` creates a loadable `ml_filter_model.json` from analytic CTC teacher labels.

## Install

The system Python on this machine blocks global editable installs, so use the repo-local virtual environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e . --no-use-pep517
```

Then run:

```sh
.venv/bin/ctc --help
```

## Usage

Run the full demo pipeline:

```sh
./run.sh
```

With no input file, `run.sh` generates a 4-second stereo logarithmic chirp from 20 Hz to 20 kHz.

Or use your own 48 kHz stereo WAV:

```sh
./run.sh input.wav
```

Create default configs:

```sh
.venv/bin/ctc init-config --directory configs
```

Train a lightweight ML residual controller:

```sh
.venv/bin/ctc train --output ml_filter_model.json --examples 192
```

Run both stages:

```sh
.venv/bin/ctc run input.wav --model ml_filter_model.json
```

Outputs:

- `preprocessed.wav`
- `simulated_binaural.wav`
- `metrics.json`

## Evaluation Metrics

`ctc evaluate` writes a comprehensive JSON report. The most useful metrics are:

- Audio health: peak, RMS, crest factor, clipping fraction, DC offset, stereo correlation, mid/side ratio, zero-crossing rate.
- Output change: Stage 1/Stage 2 peak gain, RMS gain, and overlap error versus input.
- CTC behavior: overall crosstalk suppression, per-band suppression at 250 Hz through 12 kHz, desired-path flatness, desired-path latency, left/right latency mismatch, and filter gain.
- Runtime cost: render elapsed time, real-time factor from `ctc run`, estimated FIR multiply-accumulates per second from `ctc evaluate`.
- Acceptance checks: suppression thresholds, latency threshold, latency matching, and clipping checks.

Run evaluation directly:

```sh
.venv/bin/ctc evaluate \
  --input input.wav \
  --stage1 outputs/preprocessed.wav \
  --stage2 outputs/simulated_binaural.wav \
  --model outputs/ml_filter_model.json \
  --output outputs/metrics.json
```
