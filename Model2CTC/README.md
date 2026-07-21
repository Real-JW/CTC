# Model2CTC experiment

## Recommended method

Use a hybrid filter designer, not end-to-end neural audio:

1. Your existing model predicts the fixed setup's 2×2 BRIR: two speakers to two HATS ears.
2. A frequency-dependent Tikhonov inverse creates a safe 128-tap FIR baseline.
3. A tiny ridge model predicts a low-rank correction learned from prediction-versus-measurement errors.
4. Validate gain and crosstalk suppression, then export Q14 coefficients.
5. The edge device runs only four FIR paths. ML runs once during setup, not per audio block.

This architecture is stable because the analytic inverse is regularized and gain-limited. If the ML model is missing or rejected, the baseline filter still works.

## Data to collect

For every case, keep speaker/HATS placement fixed and save:

- `predicted_brir.json`: your model's `[2 ears, 2 speakers, taps]` BRIR.
- `measured_brir.json`: HATS measurement for the identical condition.
- Case metadata: room, speaker serials, microphone calibration, temperature, distance, speaker angle, HATS pose, sweep level, and timestamp.

Measure each speaker separately with a logarithmic sweep. Record both ears simultaneously, deconvolve to four impulse responses, align all responses to one timing reference, resample to 48 kHz, preserve absolute relative gain, and trim using the same rule. Repeat each measurement 3 times and reject cases with poor repeatability.

Suggested first dataset:

| Split | Conditions | Cases |
|---|---|---:|
| Train | 8-12 rooms/placements, 3 repeats, small nuisance variations | 80-150 |
| Validation | Held-out sessions in known rooms | 20-30 |
| Test | Entirely held-out rooms and days | 20-30 |

Although the nominal distance is 50 cm, include realistic deployment tolerances: ±2 cm lateral/forward displacement, ±5° HATS yaw, speaker gain ±1 dB, and temperature/session changes. Do not put repeats of one physical measurement in different splits.

The BRIR JSON format matches `BRIR/data/mit_kemar_anechoic_30deg_48k.json`:

```json
{
  "sample_rate": 48000,
  "shape": [2, 2, 1024],
  "filters": [[[0.0]]],
  "metadata": {"distance_m": 0.5, "session": "room01_day01"}
}
```

The path order is `[ear][speaker][sample]`: LL, LR, RL, RR.

## Experiment

Compare three systems on the **measured** held-out BRIR:

- No CTC: identity speaker feeds.
- Baseline: regularized inverse designed from predicted BRIR.
- Hybrid: baseline plus learned residual designed from predicted BRIR.

Primary metrics are median and 10th-percentile crosstalk suppression from 150 Hz-12 kHz, desired-path level error, peak filter gain, filter length, MAC/s, and measured edge latency. Also evaluate ±2 cm and ±5° robustness and listen for coloration/artifacts.

Success gate for the first experiment:

- Hybrid improves held-out median suppression by at least 3 dB over baseline.
- 10th-percentile suppression is at least 10 dB from 300 Hz-8 kHz.
- Desired response stays within ±6 dB for 95% of evaluated bins.
- Filter gain stays at or below +6 dB.
- The 128-tap Q14 runtime meets the target edge CPU and memory budget.

If hybrid does not beat the regularized baseline on held-out rooms, keep the baseline. More model capacity is not the first remedy; improve paired BRIR consistency and prediction-error coverage first.

## Commands

Create a stable baseline:

```sh
model2ctc design \
  --brir data/case001/predicted_brir.json \
  --output outputs/case001_filter.json
```

Train and evaluate a residual model:

```sh
model2ctc train \
  --manifest data/manifest.json \
  --output outputs/residual_model.json \
  --report outputs/training_report.json

model2ctc design \
  --brir data/test001/predicted_brir.json \
  --model outputs/residual_model.json \
  --output outputs/test001_filter.json

model2ctc evaluate \
  --brir data/test001/measured_brir.json \
  --filter outputs/test001_filter.json
```

Export Q14 coefficients for `edge/ctc_fir.c`:

```sh
model2ctc export \
  --filter outputs/test001_filter.json \
  --output edge/model2ctc_coefficients.h
```

## Manifest

Paths are relative to the manifest:

```json
{
  "cases": [
    {
      "id": "room01_day01_center",
      "split": "train",
      "predicted_brir": "room01/predicted_brir.json",
      "measured_brir": "room01/measured_brir.json"
    }
  ]
}
```
