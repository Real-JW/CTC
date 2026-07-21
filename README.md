# Model2CTC

Design stable, edge-ready stereo crosstalk-cancellation FIR filters from a predicted 2×2 BRIR.

The recommended hybrid method combines:

- a frequency-dependent regularized BRIR inverse;
- a tiny learned residual trained from paired predicted and HATS-measured BRIRs;
- a strict +6 dB filter-gain limit;
- Q14 coefficient export for a four-path, 128-tap C FIR runtime.

ML runs during setup when the BRIR changes. The edge audio loop runs only deterministic FIR convolution.

## Structure

```text
Model2CTC/   inverse design, residual model, training, evaluation, CLI
BRIR/        BRIR creation and JSON reference data
edge/        minimal Q15 stereo FIR implementation
configs/     design defaults
tests/       design, training, export, and BRIR tests
```

See [Model2CTC/README.md](Model2CTC/README.md) for the experiment, required data, success criteria, and commands.

## Install and verify

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests -v
model2ctc --help
```
