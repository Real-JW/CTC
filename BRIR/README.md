# BRIR FFT + FIR Design

This folder contains a small BRIR design helper for the acoustic chain:

```text
BRIR = HRTF convolved with RTF
```

In the frequency domain this is the same as:

```text
FFT(BRIR) = FFT(HRTF) * FFT(RTF)
```

The code prefers open-source NumPy/SciPy FFT routines when installed, and keeps a dependency-free direct-FIR fallback so the rest of this repo can still run in a clean Python environment.

## Is 1024 Samples Good?

At 48 kHz, 1024 samples is about 21.3 ms. That is a good low-latency choice for HRTF, direct sound, and early room reflections. It is not long enough to contain a full room decay such as RT60 = 250 ms, which would be about 12000 samples at 48 kHz.

Recommended split:

- 1024-tap BRIR FIR for HRTF plus direct/early RTF.
- Optional longer partitioned convolution or reverb path for the late room tail.

## Install Optional FFT Libraries

```sh
python3 -m pip install -e '.[brir]'
```

The optional libraries are:

- NumPy for FFT arrays and overlap-add filtering.
- SciPy for `scipy.signal.fftconvolve` and fast FFT sizing.

## Generate a Demo 1024-Tap BRIR

```sh
python3 BRIR/brir_fft_fir.py --output BRIR/demo_brir_1024.json --taps 1024
```

The demo generator is only a smoke-test BRIR. Replace its synthetic HRTF and RTF arrays with measured data for real rendering.

## Render an Input WAV

```sh
python3 BRIR/brir_fft_fir.py \
  --input-wav /Users/realjw/Project/CTC/input.wav \
  --output-wav /Users/realjw/Project/CTC/BRIR/output_brir.wav \
  --taps 1024
```

To render with a saved BRIR JSON:

```sh
python3 BRIR/brir_fft_fir.py \
  --filter-bank /Users/realjw/Project/CTC/BRIR/demo_brir_1024.json \
  --input-wav /Users/realjw/Project/CTC/input.wav \
  --output-wav /Users/realjw/Project/CTC/BRIR/output_brir.wav
```

## Python Usage

```python
from BRIR.brir_fft_fir import FftFirFilterBank, compose_brir_filter_bank

# Shapes:
# hrtf_bank[ear][speaker][tap]
# rtf_bank[ear][speaker][tap], or rtf_bank[speaker][tap]
brir = compose_brir_filter_bank(hrtf_bank, rtf_bank, taps=1024, normalize_peak=0.99)

renderer = FftFirFilterBank(brir, block_size=1024)
out = renderer.process_all(stereo_samples, include_tail=True)
print(renderer.engine)
```
