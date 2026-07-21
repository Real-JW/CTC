from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from numbers import Real
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


Filter = List[float]
FilterBank = List[List[Filter]]
StereoSample = Tuple[float, float]
PathLike = Union[str, Path]
STEREO_CHANNELS = ("left", "right")
REFERENCE_BRIR_PATH = Path(__file__).resolve().parent / "data" / "mit_kemar_anechoic_30deg_48k.json"


try:  # Optional open-source acceleration path.
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - depends on local environment.
    _np = None

try:
    from scipy.fft import next_fast_len as _scipy_next_fast_len  # type: ignore
    from scipy.signal import fftconvolve as _scipy_fftconvolve  # type: ignore
except Exception:  # pragma: no cover - depends on local environment.
    _scipy_next_fast_len = None
    _scipy_fftconvolve = None


def fft_convolve(left: Sequence[float], right: Sequence[float]) -> Filter:
    """Convolve two FIRs, preferring SciPy/NumPy FFT routines when available."""
    a = _as_filter(left)
    b = _as_filter(right)
    if not a or not b:
        return []

    out_len = len(a) + len(b) - 1
    if _scipy_fftconvolve is not None:
        return _scipy_fftconvolve(a, b, mode="full").astype(float).tolist()

    if _np is not None:
        fft_size = _next_fft_len(out_len)
        a_fft = _np.fft.rfft(_np.asarray(a, dtype=float), n=fft_size)
        b_fft = _np.fft.rfft(_np.asarray(b, dtype=float), n=fft_size)
        return _np.fft.irfft(a_fft * b_fft, n=fft_size)[:out_len].astype(float).tolist()

    return _direct_convolve(a, b)


def compose_path_brir(
    hrtf: Sequence[float],
    rtf: Sequence[float],
    taps: int = 1024,
    tail_window: int = 64,
) -> Filter:
    """Create one BRIR path: BRIR(time) = HRTF(time) convolved with RTF(time)."""
    if taps <= 0:
        raise ValueError("taps must be positive")
    return _fit_fir_length(fft_convolve(hrtf, rtf), taps=taps, tail_window=tail_window)


def compose_brir_filter_bank(
    hrtf_bank: Sequence[Sequence[Sequence[float]]],
    rtf_bank: Sequence[Any],
    taps: int = 1024,
    tail_window: int = 64,
    normalize_peak: Optional[float] = None,
) -> FilterBank:
    """Compose a BRIR filter bank from HRTF and RTF FIRs.

    The expected output shape is [ear][speaker][tap].  The RTF input can either
    use the same [ear][speaker][tap] shape, or [speaker][tap] when the room
    response is shared across both ears before HRTF coloration.
    """
    _validate_filter_bank(hrtf_bank, "hrtf_bank")
    if taps <= 0:
        raise ValueError("taps must be positive")

    outputs = len(hrtf_bank)
    inputs = len(hrtf_bank[0])
    rtf_is_path_bank = _looks_like_filter_bank(rtf_bank)
    if rtf_is_path_bank:
        _validate_filter_bank(rtf_bank, "rtf_bank")
        if len(rtf_bank) != outputs or len(rtf_bank[0]) != inputs:
            raise ValueError("rtf_bank path shape must match hrtf_bank")
    elif not _looks_like_filter_list(rtf_bank):
        raise ValueError("rtf_bank must be [speaker][tap] or [ear][speaker][tap]")
    elif len(rtf_bank) != inputs:
        raise ValueError("speaker RTF count must match hrtf_bank input count")

    brir: FilterBank = []
    for ear in range(outputs):
        row: List[Filter] = []
        for speaker in range(inputs):
            rtf = rtf_bank[ear][speaker] if rtf_is_path_bank else rtf_bank[speaker]
            row.append(
                compose_path_brir(
                    hrtf_bank[ear][speaker],
                    rtf,
                    taps=taps,
                    tail_window=tail_window,
                )
            )
        brir.append(row)

    if normalize_peak is not None:
        return normalize_filter_bank(brir, peak=normalize_peak)
    return brir


def normalize_filter_bank(filters: FilterBank, peak: float = 0.99) -> FilterBank:
    if peak <= 0.0:
        raise ValueError("peak must be positive")
    current = max((abs(tap) for row in filters for path in row for tap in path), default=0.0)
    if current <= peak or current == 0.0:
        return [[list(path) for path in row] for row in filters]
    scale = peak / current
    return [[[tap * scale for tap in path] for path in row] for row in filters]


def stereo_path_metadata() -> List[Dict[str, Any]]:
    """Describe the 2x2 loudspeaker-to-ear acoustic matrix."""
    paths = []
    for ear_index, ear in enumerate(STEREO_CHANNELS):
        for speaker_index, speaker in enumerate(STEREO_CHANNELS):
            role = "desired" if ear_index == speaker_index else "crosstalk"
            paths.append(
                {
                    "ear_index": ear_index,
                    "speaker_index": speaker_index,
                    "ear": f"{ear}_ear",
                    "speaker": f"{speaker}_speaker",
                    "path": f"{speaker}_speaker_to_{ear}_ear",
                    "role": role,
                }
            )
    return paths


def stereo_path_report(filters: Sequence[Sequence[Sequence[float]]], sample_rate: int) -> List[Dict[str, Any]]:
    """Summarize desired and crosstalk paths in a stereo BRIR/HRTF bank."""
    _validate_filter_bank(filters, "filters")
    if len(filters) != 2 or len(filters[0]) != 2:
        raise ValueError("stereo path reports require a 2x2 filter bank")
    report = []
    for metadata in stereo_path_metadata():
        path = filters[metadata["ear_index"]][metadata["speaker_index"]]
        stats = _filter_path_stats(path, sample_rate)
        report.append({**metadata, **stats})
    return report


@dataclass(frozen=True)
class TapAssessment:
    taps: int
    sample_rate: int
    duration_ms: float
    fft_size_for_ola: int
    good_for_early_brir: bool
    captures_full_rt60: bool
    recommendation: str


def assess_brir_taps(taps: int = 1024, sample_rate: int = 48_000, rt60_s: float = 0.25) -> TapAssessment:
    """Assess whether a BRIR length is appropriate for low-latency rendering."""
    if taps <= 0:
        raise ValueError("taps must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if rt60_s < 0.0:
        raise ValueError("rt60_s must be non-negative")

    duration_ms = taps * 1000.0 / sample_rate
    captures_full_rt60 = taps >= int(round(rt60_s * sample_rate))
    good_for_early_brir = 15.0 <= duration_ms <= 40.0
    if captures_full_rt60:
        recommendation = "enough for the requested RT60 tail, with higher latency and CPU cost"
    elif good_for_early_brir:
        recommendation = "good low-latency early BRIR length; add a longer late-reverb path if room tail matters"
    else:
        recommendation = "usable, but tune by latency target and room-response energy decay"

    return TapAssessment(
        taps=taps,
        sample_rate=sample_rate,
        duration_ms=duration_ms,
        fft_size_for_ola=_next_fft_len((2 * taps) - 1),
        good_for_early_brir=good_for_early_brir,
        captures_full_rt60=captures_full_rt60,
        recommendation=recommendation,
    )


class FftFirFilterBank:
    """2-input/2-output FIR filter-bank renderer with optional FFT overlap-add."""

    def __init__(self, filters: Sequence[Sequence[Sequence[float]]], block_size: int = 1024):
        _validate_filter_bank(filters, "filters")
        if len(filters) != 2 or len(filters[0]) != 2:
            raise ValueError("FftFirFilterBank currently renders stereo 2x2 filter banks")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if _np is None:
            self._engine = _DirectFirFilterBank(filters)
            self.engine = "sparse-direct-fir-fallback"
        else:
            self._engine = _NumpyFftFirFilterBank(filters, block_size=block_size)
            self.engine = "numpy-fft-overlap-add"
        self.block_size = block_size
        self.output_count = self._engine.output_count
        self.input_count = self._engine.input_count
        self.taps = self._engine.taps

    def process_block(self, block: Sequence[Sequence[float]]) -> List[StereoSample]:
        return self._engine.process_block(block)

    def process_all(self, samples: Sequence[Sequence[float]], include_tail: bool = True) -> List[StereoSample]:
        rendered: List[StereoSample] = []
        for start in range(0, len(samples), self.block_size):
            rendered.extend(self.process_block(samples[start : start + self.block_size]))
        if include_tail:
            rendered.extend(self.flush())
        return rendered

    def flush(self, samples: Optional[int] = None) -> List[StereoSample]:
        return self._engine.flush(self.taps - 1 if samples is None else samples)


def design_demo_brir_filter_bank(
    taps: int = 1024,
    sample_rate: int = 48_000,
    rt60_s: float = 0.20,
) -> FilterBank:
    """Build a small synthetic 2x2 BRIR bank for smoke tests and demos.

    This is not a substitute for measured HRTF/RTF data.  It gives the renderer
    realistic delays, crosstalk, and early reflections while keeping the repo
    self-contained.
    """
    hrtf = design_demo_stereo_hrtf_filter_bank(sample_rate=sample_rate)
    rtf = design_demo_speaker_rtf_filter_bank(taps=max(taps // 2, 256), sample_rate=sample_rate, rt60_s=rt60_s)
    return compose_brir_filter_bank(hrtf, rtf, taps=taps, normalize_peak=0.99)


def load_reference_brir_filter_bank(
    taps: int = 1024,
    sample_rate: int = 48_000,
    path: PathLike = REFERENCE_BRIR_PATH,
) -> Tuple[FilterBank, Dict[str, Any]]:
    """Load the bundled measured MIT KEMAR anechoic stereo BRIR reference."""
    stored_rate, filters, metadata = read_filter_bank_json(path)
    if stored_rate != sample_rate:
        raise ValueError(
            f"reference BRIR is {stored_rate} Hz, got requested {sample_rate} Hz; "
            "regenerate or resample the reference data first"
        )
    return _fit_filter_bank_length(filters, taps=taps, tail_window=0), dict(metadata)


def design_demo_stereo_hrtf_filter_bank(sample_rate: int = 48_000) -> FilterBank:
    """Build a simple 2x2 HRTF bank with explicit stereo crosstalk paths.

    Shape is [ear][speaker][tap].  Diagonal paths are desired speaker-to-near-ear
    paths; off-diagonal paths are the HRTF crosstalk paths.
    """
    hrtf_taps = 96
    hrtf = _zero_filter_bank(outputs=2, inputs=2, taps=hrtf_taps)
    same_delay = int(round(0.18e-3 * sample_rate))
    cross_delay = int(round(0.72e-3 * sample_rate))
    hrtf[0][0][same_delay] = 1.0
    hrtf[1][1][same_delay] = 1.0
    hrtf[0][1][cross_delay] = 0.38
    hrtf[1][0][cross_delay] = 0.38
    _add_fractional_impulse(hrtf[0][1], cross_delay + 6.4, -0.08)
    _add_fractional_impulse(hrtf[1][0], cross_delay + 6.4, -0.08)
    return hrtf


def design_demo_speaker_rtf_filter_bank(
    taps: int = 512,
    sample_rate: int = 48_000,
    rt60_s: float = 0.20,
) -> List[Filter]:
    """Build per-speaker room transfer FIRs shared by both ears before HRTF."""
    rtf = []
    for speaker in range(2):
        path = [0.0 for _ in range(taps)]
        path[0] = 1.0
        sign = 1.0 if speaker == 0 else -1.0
        for delay_ms, gain in ((4.8, 0.32), (9.7, -0.22), (16.5, 0.14)):
            _add_fractional_impulse(path, delay_ms * 1e-3 * sample_rate, gain * sign)
        _add_sparse_late_tail(path, sample_rate=sample_rate, rt60_s=rt60_s, start_ms=18.0)
        rtf.append(path)
    return rtf


def write_filter_bank_json(
    path: PathLike,
    filters: FilterBank,
    sample_rate: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "sample_rate": sample_rate,
        "shape": [len(filters), len(filters[0]), len(filters[0][0])],
        "filters": filters,
        "metadata": metadata or {},
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def read_filter_bank_json(path: PathLike) -> Tuple[int, FilterBank, Dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    filters = payload["filters"]
    _validate_filter_bank(filters, "filters")
    return int(payload["sample_rate"]), filters, dict(payload.get("metadata", {}))


def render_wav_with_brir(
    input_path: PathLike,
    output_path: PathLike,
    filters: Sequence[Sequence[Sequence[float]]],
    sample_rate: int,
    block_size: int = 1024,
    pcm16: bool = False,
) -> Dict[str, Any]:
    """Render a stereo WAV through a 2x2 BRIR filter bank."""
    from .wav import peak_abs, read_wav, write_wav_float32, write_wav_pcm16

    input_rate, samples = read_wav(input_path)
    if input_rate != sample_rate:
        raise ValueError(f"expected {sample_rate} Hz input, got {input_rate} Hz")

    renderer = FftFirFilterBank(filters, block_size=block_size)
    rendered = renderer.process_all(samples, include_tail=True)
    if pcm16:
        write_wav_pcm16(output_path, sample_rate, rendered)
    else:
        write_wav_float32(output_path, sample_rate, rendered)

    return {
        "input": str(Path(input_path).resolve()),
        "output": str(Path(output_path).resolve()),
        "sample_rate": sample_rate,
        "input_samples": len(samples),
        "output_samples": len(rendered),
        "brir_taps": len(filters[0][0]),
        "block_size": block_size,
        "engine": renderer.engine,
        "peak_abs": peak_abs(rendered),
        "paths": stereo_path_report(filters, sample_rate),
    }


class _NumpyFftFirFilterBank:
    def __init__(self, filters: Sequence[Sequence[Sequence[float]]], block_size: int):
        assert _np is not None
        self.filters = _np.asarray(filters, dtype=float)
        self.output_count, self.input_count, self.taps = self.filters.shape
        self.block_size = block_size
        self.fft_size = _next_fft_len(block_size + self.taps - 1)
        self.spectra = _np.fft.rfft(self.filters, n=self.fft_size, axis=-1)
        self.overlap = _np.zeros((self.output_count, self.taps - 1), dtype=float)

    def process_block(self, block: Sequence[Sequence[float]]) -> List[StereoSample]:
        assert _np is not None
        if len(block) == 0:
            return []
        if len(block) > self.block_size:
            rendered: List[StereoSample] = []
            for start in range(0, len(block), self.block_size):
                rendered.extend(self.process_block(block[start : start + self.block_size]))
            return rendered

        x = _np.asarray(block, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.input_count:
            raise ValueError(f"expected block shape [samples][{self.input_count}]")

        count = x.shape[0]
        tail = self.taps - 1
        x_spectra = _np.fft.rfft(x.T, n=self.fft_size, axis=-1)
        rendered = _np.zeros((count, self.output_count), dtype=float)
        next_overlap = _np.zeros_like(self.overlap)
        for output in range(self.output_count):
            y_spectrum = _np.zeros_like(x_spectra[0])
            for input_channel in range(self.input_count):
                y_spectrum += x_spectra[input_channel] * self.spectra[output, input_channel]
            y = _np.fft.irfft(y_spectrum, n=self.fft_size)[: count + tail]
            if tail:
                y[:tail] += self.overlap[output]
                next_overlap[output] = y[count : count + tail]
            rendered[:, output] = y[:count]
        self.overlap = next_overlap
        return [(float(row[0]), float(row[1])) for row in rendered]

    def flush(self, samples: int) -> List[StereoSample]:
        return self.process_block([(0.0, 0.0)] * samples)


class _DirectFirFilterBank:
    def __init__(self, filters: Sequence[Sequence[Sequence[float]]]):
        self.filters = [[_as_filter(path) for path in row] for row in filters]
        self.sparse_filters = [
            [
                [(tap_index, coeff) for tap_index, coeff in enumerate(path) if abs(coeff) > 1e-12]
                for path in row
            ]
            for row in self.filters
        ]
        self.output_count = len(self.filters)
        self.input_count = len(self.filters[0])
        self.taps = len(self.filters[0][0])
        self.histories = [[0.0 for _ in range(self.taps)] for _ in range(self.input_count)]
        self.write_index = 0

    def process_block(self, block: Sequence[Sequence[float]]) -> List[StereoSample]:
        rendered_block: List[StereoSample] = []
        for sample in block:
            if len(sample) != self.input_count:
                raise ValueError(f"expected {self.input_count} input channels")
            for channel, value in enumerate(sample):
                self.histories[channel][self.write_index] = float(value)
            rendered = []
            for output in range(self.output_count):
                acc = 0.0
                for input_channel in range(self.input_count):
                    history = self.histories[input_channel]
                    coeffs = self.sparse_filters[output][input_channel]
                    for tap, coeff in coeffs:
                        acc += coeff * history[(self.write_index - tap) % self.taps]
                rendered.append(acc)
            rendered_block.append((rendered[0], rendered[1]))
            self.write_index = (self.write_index + 1) % self.taps
        return rendered_block

    def flush(self, samples: int) -> List[StereoSample]:
        return self.process_block([(0.0, 0.0)] * samples)


def _direct_convolve(left: Sequence[float], right: Sequence[float]) -> Filter:
    out = [0.0 for _ in range(len(left) + len(right) - 1)]
    for left_index, left_value in enumerate(left):
        if left_value == 0.0:
            continue
        for right_index, right_value in enumerate(right):
            out[left_index + right_index] += left_value * right_value
    return out


def _fit_fir_length(values: Sequence[float], taps: int, tail_window: int) -> Filter:
    fitted = [float(value) for value in values[:taps]]
    if len(fitted) < taps:
        fitted.extend([0.0] * (taps - len(fitted)))
        return fitted

    if len(values) > taps and tail_window > 0:
        fade = min(tail_window, taps)
        for index in range(fade):
            phase = (index + 1) / (fade + 1)
            fitted[taps - fade + index] *= 0.5 * (1.0 + math.cos(math.pi * phase))
    return fitted


def _fit_filter_bank_length(
    filters: Sequence[Sequence[Sequence[float]]],
    taps: int,
    tail_window: int,
) -> FilterBank:
    if taps <= 0:
        raise ValueError("taps must be positive")
    return [
        [_fit_fir_length(path, taps=taps, tail_window=tail_window) for path in row]
        for row in filters
    ]


def _next_fft_len(size: int) -> int:
    if _scipy_next_fast_len is not None:
        return int(_scipy_next_fast_len(size))
    return 1 << (max(1, size) - 1).bit_length()


def _zero_filter_bank(outputs: int, inputs: int, taps: int) -> FilterBank:
    return [[[0.0 for _ in range(taps)] for _ in range(inputs)] for _ in range(outputs)]


def _add_fractional_impulse(target: Filter, delay_samples: float, amplitude: float) -> None:
    if delay_samples < 0.0:
        return
    index = int(math.floor(delay_samples))
    frac = delay_samples - index
    if 0 <= index < len(target):
        target[index] += amplitude * (1.0 - frac)
    if 0 <= index + 1 < len(target):
        target[index + 1] += amplitude * frac


def _add_sparse_late_tail(target: Filter, sample_rate: int, rt60_s: float, start_ms: float) -> None:
    start = int(round(start_ms * 1e-3 * sample_rate))
    if start >= len(target) or rt60_s <= 0.0:
        return
    decay_per_sample = math.log(0.001) / max(rt60_s * sample_rate, 1.0)
    spacing = max(17, int(round(0.73e-3 * sample_rate)))
    for index in range(start, len(target), spacing):
        polarity = -1.0 if ((index // spacing) % 2) else 1.0
        envelope = math.exp(decay_per_sample * (index - start))
        target[index] += polarity * 0.045 * envelope


def _filter_path_stats(path: Sequence[float], sample_rate: int) -> Dict[str, Any]:
    values = _as_filter(path)
    peak_sample = max(range(len(values)), key=lambda index: abs(values[index]))
    peak = abs(values[peak_sample])
    first_nonzero = next((index for index, value in enumerate(values) if abs(value) > 1e-12), None)
    nonzero_taps = sum(1 for value in values if abs(value) > 1e-12)
    energy = sum(value * value for value in values)
    return {
        "peak": peak,
        "peak_sample": peak_sample,
        "peak_delay_ms": peak_sample * 1000.0 / sample_rate,
        "first_nonzero_sample": first_nonzero,
        "nonzero_taps": nonzero_taps,
        "energy": energy,
    }


def _validate_filter_bank(value: Sequence[Sequence[Sequence[float]]], name: str) -> None:
    if not _looks_like_filter_bank(value):
        raise ValueError(f"{name} must have shape [output][input][tap]")
    inputs = len(value[0])
    taps = len(value[0][0])
    if inputs == 0 or taps == 0:
        raise ValueError(f"{name} must not contain empty filters")
    for row in value:
        if len(row) != inputs:
            raise ValueError(f"{name} must have a rectangular output/input shape")
        for path in row:
            if len(path) != taps:
                raise ValueError(f"{name} paths must all have the same tap length")


def _looks_like_filter_bank(value: Sequence[Any]) -> bool:
    return (
        _is_sequence(value)
        and len(value) > 0
        and _is_sequence(value[0])
        and len(value[0]) > 0
        and _is_sequence(value[0][0])
        and _looks_like_filter(value[0][0])
    )


def _looks_like_filter_list(value: Sequence[Any]) -> bool:
    return _is_sequence(value) and len(value) > 0 and all(_looks_like_filter(path) for path in value)


def _looks_like_filter(value: Sequence[Any]) -> bool:
    return _is_sequence(value) and len(value) > 0 and all(isinstance(item, Real) for item in value)


def _is_sequence(value: Any) -> bool:
    return (
        hasattr(value, "__len__")
        and hasattr(value, "__getitem__")
        and not isinstance(value, (str, bytes, bytearray))
    )


def _as_filter(values: Sequence[float]) -> Filter:
    if not _looks_like_filter(values):
        raise ValueError("expected a non-empty numeric FIR sequence")
    return [float(value) for value in values]


def _main() -> int:
    parser = argparse.ArgumentParser(description="Design a 2x2 demo BRIR filter bank with FFT/FIR helpers")
    parser.add_argument("--output", default="BRIR/demo_brir_1024.json", help="BRIR JSON output path")
    parser.add_argument("--input-wav", help="stereo input WAV to render, for example /Users/realjw/Project/CTC/input.wav")
    parser.add_argument("--output-wav", help="rendered stereo WAV output path")
    parser.add_argument("--filter-bank", help="optional BRIR JSON file to render with instead of the demo BRIR")
    parser.add_argument("--synthetic-demo", action="store_true", help="use the old synthetic coefficients instead of the measured KEMAR reference")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--pcm16", action="store_true")
    parser.add_argument("--taps", type=int, default=1024)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--rt60", type=float, default=0.20)
    args = parser.parse_args()

    if bool(args.input_wav) != bool(args.output_wav):
        parser.error("--input-wav and --output-wav must be used together")

    metadata: Dict[str, Any] = {}
    sample_rate = args.sample_rate
    if args.filter_bank:
        sample_rate, filters, metadata = read_filter_bank_json(args.filter_bank)
    elif args.synthetic_demo:
        filters = design_demo_brir_filter_bank(taps=args.taps, sample_rate=sample_rate, rt60_s=args.rt60)
        metadata = {
            "scenario": "synthetic demo HRTF plus synthetic early room reflections",
            "source": "procedural placeholder coefficients",
        }
    else:
        filters, metadata = load_reference_brir_filter_bank(taps=args.taps, sample_rate=sample_rate)

    assessment = assess_brir_taps(taps=len(filters[0][0]), sample_rate=sample_rate, rt60_s=args.rt60)

    if args.input_wav and args.output_wav:
        metrics = render_wav_with_brir(
            args.input_wav,
            args.output_wav,
            filters,
            sample_rate=sample_rate,
            block_size=args.block_size,
            pcm16=args.pcm16,
        )
        print(json.dumps({**metrics, "tap_assessment": assessment.__dict__, "metadata": metadata}, indent=2))
        return 0

    write_filter_bank_json(
        args.output,
        filters,
        sample_rate=sample_rate,
        metadata={
            "engine_hint": "install ctc[brir] to use NumPy/SciPy FFT acceleration",
            "tap_assessment": assessment.__dict__,
            "paths": stereo_path_report(filters, sample_rate),
            **metadata,
        },
    )
    print(json.dumps({"output": str(Path(args.output).resolve()), **assessment.__dict__}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
