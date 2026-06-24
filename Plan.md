# Low-Latency Crosstalk Cancellation System

## Summary
Build a realtime-capable stereo crosstalk-cancellation app plus an offline simulator for 48 kHz stereo WAV input.

The acoustic problem is a 2x2 transaural system: each loudspeaker reaches both ears, so Stage 1 inverts the four transfer paths from speakers to ears and writes crosstalk-cancelled loudspeaker feeds to `preprocessed.wav`. Stage 2 simulates playback through near-field stereo speakers, a head/ear model, HRTF data, and room response, then writes the expected binaural ear signals to `simulated_binaural.wav`.

V1 is ML-guided while still using bounded DSP for sample processing. A neural filter controller predicts compact FIR corrections from geometry, HRTF, room, and optional head-pose context; the audio path applies the resulting 2x2 FIR filter bank. Analytic regularized inversion remains the teacher, bootstrap path, and safety fallback.

## Key Interfaces
- Input: arbitrary-length stereo WAV, 48 kHz, 16-bit PCM. Reject or resample other rates.
- Internal processing: float32 stereo blocks, default block size 128 samples.
- Stage 1 output: `preprocessed.wav`, stereo 32-bit float WAV by default, with optional 16-bit dithered export.
- Stage 2 output: `simulated_binaural.wav`, stereo 32-bit float WAV where left channel is left ear and right channel is right ear.
- Config files:
  - `geometry.json`: speaker spacing, speaker distance, head radius, ear positions, listener pose.
  - `room.json`: room dimensions, absorption, reflection order, late reverb settings.
  - `ctc_filter.npz` or equivalent: generated FIR filter bank plus metadata.
  - `ml_filter_model.onnx` or RTNeural weights: trained neural filter controller.
- Debug outputs:
  - `filters_ctc_48k_256tap.npz`
  - `physical_A_mag_phase.png`
  - `target_vs_simulated_error.png`
  - `ctc_metrics.json`

## Default Geometry And Runtime Targets
- Sample rate: 48,000 Hz.
- Realtime block size: 128 samples = 2.67 ms.
- Stage 1 audio path: 4 filters, 256 taps each, direct time-domain convolution.
- Stage 1 ML controller: predicts filter residuals, gains, and fractional delay trims once per 128-sample block.
- Stage 1 modeling delay: 128 samples = 2.67 ms.
- Total target app latency: about 5.33 ms before device driver latency.
- Speaker spacing: 20 cm total.
- Listener distance from head center to speaker midpoint: 50 cm.
- Speaker positions: left `(0.50, +0.10, 0)`, right `(0.50, -0.10, 0)` meters.
- Ear positions: left `(0, +0.0875, 0)`, right `(0, -0.0875, 0)` meters.
- Speaker azimuths: +/-11.31 degrees, a 22.62 degree stereo-dipole span.
- Direct path distances: ipsilateral 0.5002 m, contralateral 0.5340 m.
- Direct path delays: ipsilateral 70.0 samples, contralateral 74.7 samples at 48 kHz using 343 m/s.
- Default room for Stage 2: shoebox 4.0 m x 3.0 m x 2.5 m, listener at `(2.0, 1.5, 1.2)`, speakers at ear height, RT60 0.25 s, air absorption enabled.

## Libraries And Models
- Realtime app/plugin: C++17 + JUCE 8.
- Stage 1 short filters: custom 2x2 FIR in C++ instead of FFT convolution, because 4 x 256 taps at 48 kHz is small enough and avoids extra partition latency.
- Long simulator/reverb IRs: `juce::dsp::Convolution` is acceptable for offline or non-critical-latency paths. JUCE supports zero-latency and fixed-latency partitioned convolution. Source: [JUCE Convolution](https://docs.juce.com/master/classdsp_1_1Convolution.html).
- WAV I/O: libsndfile in C++ or `python-soundfile` in tooling. Source: [libsndfile](https://libsndfile.github.io/libsndfile/).
- Filter design tooling: Python 3.11+, NumPy, SciPy, SoundFile, pyfar, sofar. Sources: [sofar](https://github.com/pyfar/sofar), [pyfar](https://github.com/pyfar/pyfar).
- ML training: PyTorch for supervised training against analytic CTC filters.
- ML deployment: RTNeural for hard realtime C++ inference, or ONNX Runtime on a non-audio control thread.
- HRTF loader in C++: `libmysofa`, using `mysofa_open(..., 48000, ...)` and `mysofa_getfilter_float(...)` for interpolated HRIRs and delays. Source: [libmysofa](https://github.com/hoene/libmysofa).
- HRTF dataset: start with SOFA CIPIC `subject_003.sofa`. Source: [SOFA CIPIC database](https://sofacoustics.org/data/database/cipic/).
- BRIR simulator: SofaMyRoom for Stage 2, combining shoebox room simulation with SOFA HRTFs. Source: [SofaMyRoom](https://arxiv.org/abs/2106.12992).
- Optional room-only validation: pyroomacoustics for shoebox/image-source checks. Source: [pyroomacoustics](https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.room.html).

## Stage 1: ML-Guided Crosstalk Canceller
- Stage 1 is a hybrid system:
  - analytic DSP computes stable base filters and supervised labels;
  - ML predicts compact corrections/adaptations;
  - the realtime audio path applies bounded FIR filters.
- Build the physical acoustic matrix `A(f)` from actual-speaker BRIR/HRIR paths:

```text
ear_signal(f) = A(f) * speaker_feed(f)

A(f) = [[A_leftEar_leftSpeaker,  A_leftEar_rightSpeaker],
        [A_rightEar_leftSpeaker, A_rightEar_rightSpeaker]]
```

- Build target matrix `T(f)` from virtual speakers at +/-45 degrees:

```text
desired_ear_signal(f) = T(f) * input_stereo(f)
speaker_feed(f) = W(f) * input_stereo(f)
```

- Compute the analytic regularized inverse filter used for labels, initialization, and safety fallback:

```text
W(f) = inv(AH(f) * A(f) + beta(f) * I) * AH(f) * T(f)
```

- Use frequency grid `NFFT = 4096`; convert each of the four `W(f)` responses to a causal 256-tap FIR with 128-sample modeling delay.
- Use the neural controller to predict a residual over this analytic base filter:

```text
W_ml = W_analytic + PCA_basis * predicted_weights
```

- Clamp, smooth, and validate `W_ml` before it replaces the active filter bank.
- Use a modest frequency-aware widening target:
  - bass below 120 Hz: mostly mono, no aggressive cancellation;
  - 120-250 Hz: smooth cosine transition into widening;
  - 250 Hz-8 kHz: full CTC target;
  - 8-16 kHz: taper cancellation strength by 6 dB/octave;
  - avoid long decorrelation delays because low latency is the priority.
- Regularization:
  - `beta(f) = 0.01 * mean(diag(AH*A))` from 250 Hz to 8 kHz;
  - increase to `0.03` below 250 Hz and above 8 kHz;
  - increase to `0.10` below 120 Hz and above 16 kHz to avoid unstable bass/ultrasonic inversion.
- Safety:
  - apply -6 dB input trim before CTC;
  - cap per-filter boost to +6 dB by default, +12 dB absolute max;
  - normalize generated speaker-feed FIRs so full-scale pink noise does not exceed -1 dBFS after limiting;
  - use a lookahead-free soft clipper only as final protection.
- Realtime callback constraints:
  - no heap allocation;
  - no locks;
  - no file I/O;
  - no model loading;
  - no unbounded computation.

## Stage 2: Binaural Simulator
- Generate BRIRs by combining:
  - HRTF lookup/interpolation for direct sound from each speaker to each ear;
  - geometric delay and attenuation from the 20 cm speaker spacing and 50 cm listener distance;
  - image-source early reflections;
  - optional FDN or stochastic late reverb.
- Render `preprocessed.wav` through the BRIRs to produce `simulated_binaural.wav`.
- Use Stage 2 to compare the intended widened target against the modeled ear signals and to measure crosstalk suppression, magnitude response, ITD/ILD preservation, and clipping risk.

## Neural Filter Controller
ML is a first-class part of Stage 1. It predicts the compact parameters that adapt the canceller to geometry, HRTF, room, and small head-pose changes. It does not generate raw audio; all sample-level processing remains bounded FIR DSP.

The analytic inverse is retained as the training target, initialization path, and emergency fallback if the model output fails validation.

- Model shape:
  - 2-layer GRU, 64 hidden units per layer, about 50k parameters;
  - output every 128 samples;
  - 4 paths x 16 PCA filter weights = 64 values;
  - 4 path gains;
  - 4 fractional delay trims;
  - total: 72 outputs.
- FIR reconstruction:
  - `filter = analytic_base_filter + PCA_basis * predicted_weights`;
  - clamp boost to +6 dB default, +12 dB absolute max;
  - smooth coefficient changes over 64-128 samples to avoid zipper noise;
  - convert ML output into stable bounded filters before playback.
- Runtime inference options:
  - RTNeural for hard realtime C++ inference;
  - ONNX Runtime only on a non-audio control thread.
- Validation gate before coefficients become active:
  - reject NaN/Inf output;
  - reject filter boost above the +12 dB hard cap;
  - reject coefficient deltas that would create zipper noise;
  - fall back to the previous valid ML filter or the analytic base filter.
- Sources: [RNNoise](https://jmvalin.ca/demo/rnnoise/), [RTNeural paper](https://arxiv.org/abs/2106.03037), [ANIRA](https://arxiv.org/abs/2506.12665).

## Training And Data
- Generate supervised labels from analytic regularized CTC filters, not hand-labeled audio.
- Training cases:
  - speaker spacing: 16-30 cm;
  - listener distance: 35-80 cm;
  - default geometry: 20 cm spacing, 50 cm distance;
  - head radius: 7.5-9.5 cm;
  - yaw: -15 to +15 degrees;
  - room RT60: 0.15-0.60 s.
- Optional compact companion model for configuration-time prediction:
  - PyTorch MLP;
  - 6 numeric inputs;
  - 3 hidden layers of 256 units;
  - SiLU activations;
  - output 4 x 257 complex frequency bins for 512-point half-spectrum correction residuals;
  - export ONNX;
  - app loads ONNX Runtime only when geometry changes, then converts predicted responses to bounded 256-tap FIRs.

## Tests And Acceptance
- File handling:
  - arbitrary-length WAV;
  - 1-sample file;
  - non-multiple-of-128 length;
  - silence;
  - mono input handling or rejection;
  - full-scale sine;
  - clipped input.
- DSP equivalence:
  - offline block renderer must match realtime-style block renderer within -120 dB RMS error.
- Cancellation target in simulated sweet spot:
  - ML filter must stay within 3 dB of analytic CTC cancellation from 500 Hz-8 kHz;
  - crosstalk suppression at least 18 dB from 500 Hz-8 kHz;
  - at least 10 dB from 250-500 Hz and 8-12 kHz;
  - no hard cancellation requirement below 120 Hz.
- Tonal accuracy:
  - target-vs-simulated magnitude error within +/-3 dB from 250 Hz-8 kHz;
  - within +/-6 dB from 120-250 Hz and 8-12 kHz.
- Robustness:
  - with head offset +/-2 cm or yaw +/-5 degrees, suppression should remain at least 10 dB from 500 Hz-6 kHz.
- Performance:
  - Stage 1 CPU below 10% of one modern desktop core at 48 kHz, 128-sample blocks;
  - ML inference under 0.25 ms per 128-sample block on desktop CPU;
  - no allocation, locks, file I/O, ONNX calls, or other blocking work in the audio callback.
- Audio quality:
  - no unstable IIR filters in v1;
  - no direct neural waveform generation in v1;
  - final limiter only for safety, not loudness.
- End-to-end validation:
  - use impulses, sine sweeps, pink noise, and music samples;
  - compare `simulated_binaural.wav` against the intended widened target using error level, crosstalk suppression, magnitude response, ITD/ILD preservation, and listening checks.

## Assumptions
- "Near perfect" means near-perfect inside the modeled sweet spot, not across arbitrary real rooms or head movement.
- The first build prioritizes latency over maximum width.
- The plan assumes a C++ realtime DSP core with Python tooling for filter/BRIR design and analysis.
- The first HRTF is generic CIPIC subject 003; personalized measured BRIR/HRTF can replace it later.
- The analytic FIR path exists as a teacher and safety fallback, but the target product path is ML-guided.
- LSTM is allowed, but GRU is the default recurrent model because it is smaller and cheaper for realtime inference.
