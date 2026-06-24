from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .config import GeometryConfig, RoomConfig, RuntimeConfig, load_geometry, load_room, load_runtime
from .filters import (
    FilterBank,
    build_direct_brir,
    db_to_gain,
    design_analytic_filter_bank,
    max_l1_gain,
)
from .neural import ControllerContext, apply_prediction, load_controller
from .wav import StereoSample, peak_abs, read_wav


PathLike = Union[str, Path]
Signal = List[float]

FREQUENCY_POINTS_HZ = [250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 12000.0]


def evaluate_pipeline(
    input_path: PathLike,
    stage1_path: PathLike,
    stage2_path: PathLike,
    geometry_path: Optional[PathLike] = None,
    room_path: Optional[PathLike] = None,
    runtime_path: Optional[PathLike] = None,
    model_path: Optional[PathLike] = None,
    output_path: Optional[PathLike] = None,
) -> Dict[str, object]:
    runtime = load_runtime(runtime_path)
    geometry = load_geometry(geometry_path)
    room = load_room(room_path)

    input_rate, input_samples = read_wav(input_path)
    stage1_rate, stage1_samples = read_wav(stage1_path)
    stage2_rate, stage2_samples = read_wav(stage2_path)

    _require_rate("input", input_rate, runtime)
    _require_rate("stage1", stage1_rate, runtime)
    _require_rate("stage2", stage2_rate, runtime)

    ctc_metrics = evaluate_ctc_model(input_samples, geometry, room, runtime, model_path)
    report: Dict[str, object] = {
        "files": {
            "input": summarize_audio(input_samples, input_rate),
            "stage1_preprocessed": summarize_audio(stage1_samples, stage1_rate),
            "stage2_binaural": summarize_audio(stage2_samples, stage2_rate),
        },
        "output_changes": {
            "stage1_vs_input": compare_audio(input_samples, stage1_samples),
            "stage2_vs_input": compare_audio(input_samples, stage2_samples),
        },
        "ctc_model": ctc_metrics,
        "acceptance": acceptance_checks(ctc_metrics, stage1_samples, stage2_samples),
        "metric_notes": {
            "crosstalk_suppression_db": "Higher is better; compares intended-ear energy to opposite-ear leakage.",
            "desired_flatness_db": "Lower is better; max-min desired path magnitude over the measured band.",
            "crest_factor_db": "Peak-to-RMS ratio; high values are transient-heavy or low average level.",
            "stereo_correlation": "-1 is opposite polarity, 0 is decorrelated, +1 is mono-like.",
            "estimated_stage1_mmac_s": "Millions of multiply-accumulates per second for the direct FIR audio path.",
        },
    }

    if output_path is not None:
        Path(output_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def summarize_audio(samples: Sequence[StereoSample], sample_rate: int) -> Dict[str, float]:
    duration = len(samples) / sample_rate if sample_rate else 0.0
    left = [sample[0] for sample in samples]
    right = [sample[1] for sample in samples]
    mid = [(sample[0] + sample[1]) * 0.5 for sample in samples]
    side = [(sample[0] - sample[1]) * 0.5 for sample in samples]
    rms_left = rms(left)
    rms_right = rms(right)
    rms_mid = rms(mid)
    rms_side = rms(side)
    peak = peak_abs(samples)
    sample_slots = max(len(samples) * 2, 1)
    clipping = sum(1 for left_value, right_value in samples if abs(left_value) >= 0.999) + sum(
        1 for left_value, right_value in samples if abs(right_value) >= 0.999
    )

    return {
        "sample_rate": float(sample_rate),
        "samples": float(len(samples)),
        "duration_s": duration,
        "peak_abs": peak,
        "rms_left": rms_left,
        "rms_right": rms_right,
        "rms_mid": rms_mid,
        "rms_side": rms_side,
        "rms_total": rms([value for sample in samples for value in sample]),
        "crest_factor_db": ratio_db(peak, max(rms_left, rms_right)),
        "dc_offset_left": mean(left),
        "dc_offset_right": mean(right),
        "clipping_fraction": clipping / sample_slots,
        "stereo_correlation": correlation(left, right),
        "mid_side_ratio_db": ratio_db(rms_side, rms_mid),
        "zero_crossing_rate_left": zero_crossing_rate(left),
        "zero_crossing_rate_right": zero_crossing_rate(right),
    }


def compare_audio(reference: Sequence[StereoSample], candidate: Sequence[StereoSample]) -> Dict[str, float]:
    ref_peak = peak_abs(reference)
    cand_peak = peak_abs(candidate)
    ref_rms = rms([value for sample in reference for value in sample])
    cand_rms = rms([value for sample in candidate for value in sample])
    overlap = min(len(reference), len(candidate))
    error_rms_value = 0.0
    if overlap:
        diffs = []
        for index in range(overlap):
            diffs.append(candidate[index][0] - reference[index][0])
            diffs.append(candidate[index][1] - reference[index][1])
        error_rms_value = rms(diffs)

    return {
        "sample_count_delta": float(len(candidate) - len(reference)),
        "peak_gain_db": ratio_db(cand_peak, ref_peak),
        "rms_gain_db": ratio_db(cand_rms, ref_rms),
        "overlap_error_rms": error_rms_value,
        "overlap_error_dbfs": amplitude_db(error_rms_value),
    }


def evaluate_ctc_model(
    input_samples: Sequence[StereoSample],
    geometry: GeometryConfig,
    room: RoomConfig,
    runtime: RuntimeConfig,
    model_path: Optional[PathLike],
) -> Dict[str, object]:
    base_filters = design_analytic_filter_bank(geometry, runtime)
    controller = load_controller(model_path)
    context_block = list(input_samples[: runtime.block_size])
    context = ControllerContext.from_block(context_block, geometry, room)
    prediction = controller.predict(context)
    filters = apply_prediction(base_filters, prediction, runtime, previous_filters=base_filters)
    brir = build_direct_brir(geometry, runtime)
    transfer = combine_filter_banks(brir, filters)

    left_desired = transfer[0][0]
    left_leakage = transfer[1][0]
    right_desired = transfer[1][1]
    right_leakage = transfer[0][1]

    band_rows = []
    for freq in FREQUENCY_POINTS_HZ:
        left_suppression = ratio_db(
            goertzel_magnitude(left_desired, runtime.sample_rate, freq),
            goertzel_magnitude(left_leakage, runtime.sample_rate, freq),
        )
        right_suppression = ratio_db(
            goertzel_magnitude(right_desired, runtime.sample_rate, freq),
            goertzel_magnitude(right_leakage, runtime.sample_rate, freq),
        )
        left_mag_db = amplitude_db(goertzel_magnitude(left_desired, runtime.sample_rate, freq))
        right_mag_db = amplitude_db(goertzel_magnitude(right_desired, runtime.sample_rate, freq))
        band_rows.append(
            {
                "frequency_hz": freq,
                "left_input_suppression_db": left_suppression,
                "right_input_suppression_db": right_suppression,
                "worst_suppression_db": min(left_suppression, right_suppression),
                "desired_left_mag_db": left_mag_db,
                "desired_right_mag_db": right_mag_db,
                "desired_balance_db": abs(left_mag_db - right_mag_db),
            }
        )

    core_bands = [row["worst_suppression_db"] for row in band_rows if 500.0 <= row["frequency_hz"] <= 8000.0]
    wide_bands = [row["worst_suppression_db"] for row in band_rows if 250.0 <= row["frequency_hz"] <= 12000.0]
    desired_mags = [
        row["desired_left_mag_db"]
        for row in band_rows
        if 250.0 <= row["frequency_hz"] <= 8000.0
    ] + [
        row["desired_right_mag_db"]
        for row in band_rows
        if 250.0 <= row["frequency_hz"] <= 8000.0
    ]

    left_latency = peak_index(left_desired)
    right_latency = peak_index(right_desired)
    taps = len(filters[0][0])
    macs_per_stereo_sample = 2.0 * 2.0 * taps

    return {
        "model_loaded": model_path is not None,
        "filter_taps": float(taps),
        "brir_taps": float(len(brir[0][0])),
        "combined_ir_samples": float(len(left_desired)),
        "filter_max_l1_gain_db": amplitude_db(max_l1_gain(filters)),
        "overall_suppression_left_input_db": ratio_db(rms(left_desired), rms(left_leakage)),
        "overall_suppression_right_input_db": ratio_db(rms(right_desired), rms(right_leakage)),
        "overall_worst_suppression_db": min(
            ratio_db(rms(left_desired), rms(left_leakage)),
            ratio_db(rms(right_desired), rms(right_leakage)),
        ),
        "band_suppression": band_rows,
        "core_500_8k_min_suppression_db": min(core_bands) if core_bands else 0.0,
        "wide_250_12k_min_suppression_db": min(wide_bands) if wide_bands else 0.0,
        "desired_flatness_250_8k_db": max(desired_mags) - min(desired_mags) if desired_mags else 0.0,
        "desired_latency_left_ms": samples_to_ms(left_latency, runtime.sample_rate),
        "desired_latency_right_ms": samples_to_ms(right_latency, runtime.sample_rate),
        "desired_latency_mean_ms": samples_to_ms((left_latency + right_latency) * 0.5, runtime.sample_rate),
        "desired_latency_mismatch_ms": samples_to_ms(abs(left_latency - right_latency), runtime.sample_rate),
        "estimated_stage1_macs_per_stereo_sample": macs_per_stereo_sample,
        "estimated_stage1_mmac_s": macs_per_stereo_sample * runtime.sample_rate / 1_000_000.0,
    }


def acceptance_checks(
    ctc_metrics: Dict[str, object],
    stage1_samples: Sequence[StereoSample],
    stage2_samples: Sequence[StereoSample],
) -> Dict[str, bool]:
    return {
        "core_suppression_500_8k_at_least_18db": float(ctc_metrics["core_500_8k_min_suppression_db"]) >= 18.0,
        "wide_suppression_250_12k_at_least_10db": float(ctc_metrics["wide_250_12k_min_suppression_db"]) >= 10.0,
        "latency_under_10ms": float(ctc_metrics["desired_latency_mean_ms"]) <= 10.0,
        "latency_match_under_0_25ms": float(ctc_metrics["desired_latency_mismatch_ms"]) <= 0.25,
        "stage1_no_full_scale_clipping": peak_abs(stage1_samples) < 0.999,
        "stage2_no_full_scale_clipping": peak_abs(stage2_samples) < 0.999,
    }


def combine_filter_banks(brir: FilterBank, filters: FilterBank) -> FilterBank:
    transfer: FilterBank = [[[] for _input in range(2)] for _ear in range(2)]
    for ear in range(2):
        for source in range(2):
            acc: Signal = []
            for speaker in range(2):
                path = convolve(brir[ear][speaker], filters[speaker][source])
                if not acc:
                    acc = [0.0 for _ in range(len(path))]
                for index, value in enumerate(path):
                    acc[index] += value
            transfer[ear][source] = acc
    return transfer


def convolve(left: Sequence[float], right: Sequence[float]) -> Signal:
    output = [0.0 for _ in range(len(left) + len(right) - 1)]
    for left_index, left_value in enumerate(left):
        if left_value == 0.0:
            continue
        for right_index, right_value in enumerate(right):
            if right_value != 0.0:
                output[left_index + right_index] += left_value * right_value
    return output


def goertzel_magnitude(signal: Sequence[float], sample_rate: int, frequency_hz: float) -> float:
    omega = 2.0 * math.pi * frequency_hz / sample_rate
    real = 0.0
    imag = 0.0
    for index, value in enumerate(signal):
        angle = omega * index
        real += value * math.cos(angle)
        imag -= value * math.sin(angle)
    return math.sqrt(real * real + imag * imag)


def rms(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    count = min(len(left), len(right))
    left_values = left[:count]
    right_values = right[:count]
    left_mean = mean(left_values)
    right_mean = mean(right_values)
    numerator = sum((left_values[i] - left_mean) * (right_values[i] - right_mean) for i in range(count))
    left_power = sum((value - left_mean) ** 2 for value in left_values)
    right_power = sum((value - right_mean) ** 2 for value in right_values)
    denominator = math.sqrt(left_power * right_power)
    return numerator / denominator if denominator > 0.0 else 0.0


def zero_crossing_rate(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    crossings = 0
    previous = values[0]
    for value in values[1:]:
        if (previous < 0.0 <= value) or (previous >= 0.0 > value):
            crossings += 1
        previous = value
    return crossings / (len(values) - 1)


def peak_index(values: Sequence[float]) -> int:
    if not values:
        return 0
    return max(range(len(values)), key=lambda index: abs(values[index]))


def samples_to_ms(samples: float, sample_rate: int) -> float:
    return samples / sample_rate * 1000.0 if sample_rate else 0.0


def amplitude_db(value: float) -> float:
    return 20.0 * math.log10(max(abs(value), 1e-12))


def ratio_db(numerator: float, denominator: float) -> float:
    return 20.0 * math.log10(max(abs(numerator), 1e-12) / max(abs(denominator), 1e-12))


def _require_rate(label: str, sample_rate: int, runtime: RuntimeConfig) -> None:
    if sample_rate != runtime.sample_rate:
        raise ValueError(f"{label} sample rate must be {runtime.sample_rate} Hz, got {sample_rate} Hz")
