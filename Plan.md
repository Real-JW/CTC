# Plan.md: Low-Latency Crosstalk Cancellation System

## Summary
Design a two-stage audio pipeline for 48 kHz stereo WAV input. Stage 1 produces a realtime-capable crosstalk-cancelled loudspeaker-feed WAV. Stage 2 simulates playback through near-field stereo speakers, a human head/ears model, HRTF, and room response, producing a binaural stereo WAV that should closely match the intended widened stereo image at the ears.

The first implementation should prioritize low latency over maximum widening. Use a parameterized acoustic model by default: speaker spacing 20 cm, listener distance 50 cm, ear-level speakers, human head/ear geometry, SOFA-compatible HRTF data, and a configurable room model.

## Key Interfaces
- Input: arbitrary-length stereo WAV, 48 kHz, 16-bit PCM.
- Internal processing: float32 stereo blocks, default block size 128 samples.
- Stage 1 output: `preprocessed.wav`, preferably 32-bit float WAV to preserve headroom, with optional 16-bit export.
- Stage 2 output: `simulated_binaural.wav`, stereo WAV representing left-ear/right-ear signals after speaker, head, HRTF, and room simulation.
- Config files:
  - `geometry.json`: speaker spacing, speaker distance, head radius, ear positions, listener pose.
  - `room.json`: room dimensions, absorption, reflection order, late reverb settings.
  - `ctc_filter.npz` or equivalent: generated FIR filter bank plus metadata.

## Implementation Plan
- Build a realtime DSP core that can also run offline over arbitrary-length files using the same block processor.
- Stage 1 should compute loudspeaker feeds using a regularized inverse of the 2x2 acoustic transfer matrix from speakers to ears:
  - direct path plus short early-response model for low latency;
  - causal FIR filters, initially 256-512 taps;
  - target latency goal: under 10 ms total processing latency;
  - bounded filter gain to avoid instability, clipping, and excessive coloration.
- Add a low-latency stereo widening target before inversion:
  - default to modest frequency-aware mid/side widening;
  - keep bass mostly mono below roughly 120 Hz;
  - avoid long decorrelation delays because low latency is the priority.
- Stage 2 should generate BRIRs by combining:
  - HRTF lookup/interpolation for direct sound from each speaker to each ear;
  - geometric delay and attenuation from the 20 cm speaker spacing and 50 cm listener distance;
  - image-source early reflections;
  - optional FDN or stochastic late reverb.
- Treat ML as optional filter assistance, not the realtime audio path:
  - baseline must work with analytic DSP;
  - later ML may predict FIR coefficients, regularization, or room/HRTF parameters from geometry;
  - ML output must still be converted into stable bounded filters before playback.

## Test Plan
- Verify WAV handling for arbitrary length, silence, mono, clipped input, and non-multiple block sizes.
- Confirm offline rendering and realtime-style block rendering produce matching output.
- Validate the crosstalk matrix: after Stage 1 plus Stage 2, off-diagonal leakage should be strongly reduced in the simulated sweet spot.
- Measure latency, CPU load, filter gain, peak levels, and clipping risk.
- Use impulses, sine sweeps, pink noise, and music samples for end-to-end tests.
- Compare `simulated_binaural.wav` against the intended widened target using error level, crosstalk suppression, magnitude response, ITD/ILD preservation, and listening checks.

## Assumptions
- “Near perfect” means near-perfect inside the simulated sweet spot, not guaranteed for all real rooms or head positions.
- Default first target is a realtime app architecture, with offline file rendering used for repeatable testing.
- The plan assumes a C++ realtime DSP core with Python or similar tooling for filter/BRIR design and analysis.
- Workspace inspection was unavailable, so this plan is technology-specific enough to implement but not tied to an existing repo structure.


# Plan.md: Concrete Realtime Crosstalk Cancellation Design

## Summary
Build a realtime-capable stereo crosstalk-cancellation app plus an offline simulator. The acoustic problem is a 2x2 transaural system: each loudspeaker reaches both ears, so the canceller must invert the four transfer paths L-speaker-to-L-ear, L-speaker-to-R-ear, R-speaker-to-L-ear, and R-speaker-to-R-ear. This is the standard transaural/crosstalk setup. ([en.wikipedia.org](https://en.wikipedia.org/wiki/Transaural?utm_source=openai))

Use the user geometry exactly: two speakers 20 cm apart, listener head center 50 cm away. This gives speaker azimuths of +/-11.31 degrees and a 22.62 degree stereo dipole span, which sits inside the common 10-30 degree stereo-dipole range. ([en.wikipedia.org](https://en.wikipedia.org/wiki/Stereo_dipole?utm_source=openai))

## Concrete Parameters
- Sample rate: 48,000 Hz only for v1. Reject/resample other rates.
- Input: stereo 16-bit PCM WAV, arbitrary length.
- Internal DSP: float32, interleaved or planar stereo blocks.
- Realtime block size: 128 samples = 2.67 ms.
- Stage 1 FIR: 4 filters, 256 taps each, direct time-domain convolution.
- Stage 1 modeling delay: 128 samples = 2.67 ms.
- Total target app latency: about 5.33 ms before device driver latency.
- Speaker positions: left `(0.50, +0.10, 0)`, right `(0.50, -0.10, 0)` meters.
- Ear positions: left `(0, +0.0875, 0)`, right `(0, -0.0875, 0)` meters.
- Direct path distances: ipsilateral 0.5002 m, contralateral 0.5340 m.
- Direct path delays: ipsilateral 70.0 samples, contralateral 74.7 samples at 48 kHz, using 343 m/s.
- Default virtual widening target: input stereo rendered as virtual speakers at +/-45 degrees, with bass summed mono below 120 Hz and widening faded in from 120-250 Hz.
- Headroom: apply -6 dB input trim before CTC; hard cap per-filter boost to +12 dB, default cap +6 dB.

## Libraries And Models
- Realtime app/plugin: C++17 + JUCE 8. Use JUCE for audio device/plugin shell and `juce::dsp::Convolution` only for long simulator/reverb IRs; JUCE documents zero-latency and fixed-latency partitioned convolution, with frequency-domain convolution efficient for IRs of 64 samples or greater. ([docs.juce.com](https://docs.juce.com/master/classdsp_1_1Convolution.html))
- Stage 1 short filters: custom 2x2 FIR in C++ instead of FFT convolution, because 4 x 256 taps at 48 kHz is small enough and avoids extra partition latency.
- WAV I/O: libsndfile in C++ or `python-soundfile` in tooling. libsndfile is a C library for sampled sound files including WAV/AIFF and supports format/type conversion. ([libsndfile.github.io](https://libsndfile.github.io/libsndfile/))
- Filter design tooling: Python 3.11+, NumPy, SciPy, SoundFile, pyfar, sofar. `sofar` reads/edits/writes SOFA files and verifies against AES69-2022; pyfar supports audio/filter/coordinate objects and audio processing. ([github.com](https://github.com/pyfar/sofar)) ([github.com](https://github.com/pyfar/pyfar))
- HRTF loader in C++: `libmysofa`, using `mysofa_open(..., 48000, ...)` and `mysofa_getfilter_float(...)` to retrieve interpolated left/right HRIRs and delays. ([github.com](https://github.com/hoene/libmysofa))
- HRTF dataset: start with SOFA CIPIC `subject_003.sofa`, because it is available in the SOFA acoustics database and used as an example by SofaMyRoom. ([sofacoustics.org](https://sofacoustics.org/data/database/cipic/)) ([arxiv.org](https://arxiv.org/abs/2106.12992))
- BRIR simulator: SofaMyRoom for Stage 2, because it combines shoebox room simulation with SOFA HRTFs and can save 2-channel WAV BRIRs. It uses image-source early reflections plus diffuse-rain late reflections. ([arxiv.org](https://arxiv.org/abs/2106.12992))
- Optional room-only validation: pyroomacoustics, whose docs describe shoebox rooms, image-source RIR generation, RT60 via Sabine inversion, and hybrid ISM/ray tracing. ([pyroomacoustics.readthedocs.io](https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.room.html))

## DSP Algorithm
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

- Compute the regularized inverse filter:

```text
W(f) = inv(AH(f) * A(f) + beta(f) * I) * AH(f) * T(f)
```

- Use frequency grid `NFFT = 4096`; convert each of the four `W(f)` responses to a causal 256-tap FIR with 128-sample modeling delay.
- Regularization:
  - `beta(f) = 0.01 * mean(diag(AH*A))` from 250 Hz to 8 kHz.
  - Increase to `0.03` below 250 Hz and above 8 kHz.
  - Increase to `0.10` below 120 Hz and above 16 kHz to avoid unstable bass/ultrasonic inversion.
- Equalization:
  - Bass below 120 Hz: no widening, no aggressive cancellation.
  - 120-250 Hz: smooth cosine transition.
  - 250 Hz-8 kHz: full CTC target.
  - 8-16 kHz: taper cancellation strength by 6 dB/octave.
- Safety:
  - Normalize generated speaker-feed FIRs so full-scale pink noise does not exceed -1 dBFS after limiting.
  - Add lookahead-free soft clipper only as final protection.

## Stage Outputs
- Stage 1 output: `preprocessed.wav`, stereo 32-bit float WAV by default, optional 16-bit dithered export.
- Stage 2 output: `simulated_binaural.wav`, stereo 32-bit float WAV where left channel is left ear and right channel is right ear.
- Debug outputs:
  - `filters_ctc_48k_256tap.npz`
  - `physical_A_mag_phase.png`
  - `target_vs_simulated_error.png`
  - `ctc_metrics.json`

## ML Filter Server
- Do not run ML in the audio callback.
- V1 works without ML using analytic regularized inversion.
- Add optional configuration-time ML later:
  - Generate training data with SofaMyRoom over speaker spacing 16-30 cm, listener distance 35-80 cm, yaw -10 to +10 degrees, head radius 7.5-9.5 cm, RT60 0.15-0.6 s.
  - Model: PyTorch MLP, 6 numeric inputs, 3 hidden layers of 256 units, SiLU activations, output 4 x 257 complex frequency bins for 512-point half-spectrum correction residuals.
  - Export ONNX; app loads ONNX Runtime only when geometry changes, then converts predicted responses to bounded 256-tap FIRs.
  - ML output must pass the same gain caps, latency, and crosstalk tests as analytic filters.

## Test Plan
- File tests: arbitrary-length WAV, 1-sample file, non-multiple-of-128 length, silence, full-scale sine, clipped input.
- DSP equivalence: offline block renderer must match realtime block renderer within -120 dB RMS error.
- Cancellation target in simulated sweet spot:
  - Crosstalk suppression at least 18 dB from 500 Hz-8 kHz.
  - At least 10 dB from 250-500 Hz and 8-12 kHz.
  - Do not require cancellation below 120 Hz.
- Tonal accuracy:
  - Target-vs-simulated magnitude error within +/-3 dB from 250 Hz-8 kHz.
  - Within +/-6 dB from 120-250 Hz and 8-12 kHz.
- Robustness:
  - With head offset +/-2 cm or yaw +/-5 degrees, suppression should remain at least 10 dB from 500 Hz-6 kHz.
- Performance:
  - Stage 1 CPU below 10% of one modern desktop core at 48 kHz, 128-sample blocks.
  - No heap allocation, locks, file I/O, or model inference in the audio callback.

## Assumptions
- “Near perfect” means near-perfect in the modeled sweet spot, not across arbitrary real rooms or head movement.
- The first build prioritizes latency over maximum width.
- The first HRTF is generic CIPIC subject 003; personalized measured BRIR/HRTF can replace it later.
- Room default for Stage 2: shoebox 4.0 m x 3.0 m x 2.5 m, listener at `(2.0, 1.5, 1.2)`, speakers at ear height, RT60 0.25 s, air absorption enabled.


# Concrete Plan: ML-Guided Realtime Crosstalk Cancellation

## Summary
Yes, a small RNN/LSTM can make sense for Stage 1, but not as a raw sample-to-sample audio generator. Use it as a **neural filter controller**: a tiny GRU predicts bounded FIR filter parameters, while classic DSP performs the actual crosstalk cancellation. This follows the hybrid DSP/deep-learning pattern used by RNNoise, where a small recurrent model controls DSP gains instead of replacing the whole signal path. Source: [RNNoise](https://jmvalin.ca/demo/rnnoise/).

## Stage 1: ML Filter Design
- Input: arbitrary-length stereo WAV, 48 kHz, 16-bit PCM.
- Output: `preprocessed.wav`, stereo 32-bit float WAV, optional 16-bit export.
- Realtime block size: 128 samples, 2.67 ms.
- Audio-thread DSP: 4 FIR filters, 256 taps each:
  - input L to speaker L
  - input L to speaker R
  - input R to speaker L
  - input R to speaker R
- ML model: 2-layer GRU, 64 hidden units per layer, about 50k parameters.
- ML output every 128 samples:
  - 4 paths x 16 PCA filter weights = 64 values
  - 4 path gains
  - 4 fractional delay trims
  - total: 72 outputs
- FIR reconstruction:
  - `filter = analytic_base_filter + PCA_basis * predicted_weights`
  - clamp boost to +6 dB default, +12 dB absolute max
  - smooth coefficient changes over 64-128 samples to avoid zipper noise
- Runtime inference:
  - use RTNeural for hard realtime C++ inference, or ONNX Runtime only on a non-audio control thread.
  - Source: [RTNeural paper](https://arxiv.org/abs/2106.03037). ANIRA also supports keeping inference away from the audio callback when engines cause realtime violations: [ANIRA](https://arxiv.org/abs/2506.12665).

## Training And Data
- Generate supervised labels from analytic regularized CTC filters, not hand-labeled audio.
- Training cases:
  - speaker spacing: 16-30 cm
  - listener distance: 35-80 cm
  - default geometry: 20 cm spacing, 50 cm distance
  - head radius: 7.5-9.5 cm
  - yaw: -15 to +15 degrees
  - room RT60: 0.15-0.60 s
- HRTF/BRIR sources:
  - SOFA CIPIC, default `subject_003.sofa`: [SOFA CIPIC database](https://sofacoustics.org/data/database/cipic/)
  - SOFA I/O: `sofar` / `pyfar`: [sofar](https://github.com/pyfar/sofar), [pyfar](https://github.com/pyfar/pyfar)
  - C++ HRTF lookup: `libmysofa`: [libmysofa](https://github.com/hoene/libmysofa)
  - BRIR simulation: SofaMyRoom, which combines SOFA HRTFs with shoebox room simulation: [SofaMyRoom](https://arxiv.org/abs/2106.12992)

## Stage 2: Simulation
- Generate simulated binaural output from `preprocessed.wav`.
- Default room: 4.0 m x 3.0 m x 2.5 m, RT60 0.25 s.
- Listener at room center, ear height 1.2 m.
- Speakers:
  - left: 20 cm total stereo spacing, +10 cm lateral offset
  - right: -10 cm lateral offset
  - distance from head center: 50 cm
  - resulting speaker angle: +/-11.31 degrees
- Output: `simulated_binaural.wav`, stereo float WAV, left channel = left ear, right channel = right ear.

## Tests And Acceptance
- ML filter must stay within 3 dB of analytic CTC cancellation from 500 Hz-8 kHz.
- Target crosstalk suppression:
  - at least 18 dB from 500 Hz-8 kHz in the sweet spot
  - at least 10 dB from 250-500 Hz and 8-12 kHz
  - no hard cancellation requirement below 120 Hz
- Realtime:
  - ML inference under 0.25 ms per 128-sample block on desktop CPU
  - no allocation, locks, file I/O, or ONNX calls in the audio callback
- Audio quality:
  - no unstable IIR filters in v1
  - no direct neural waveform generation in v1
  - final limiter only for safety, not loudness

## Assumptions
- The GRU is useful for adapting filters across geometry, HRTF, room, and small head-pose changes.
- For a fixed listener and fixed room, a static analytic FIR may outperform ML; the ML layer is mainly for compact adaptive filter prediction.
- LSTM is allowed, but GRU is the default because it is smaller and cheaper for realtime inference.
