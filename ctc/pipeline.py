from __future__ import annotations

from pathlib import Path
import time
from typing import Dict, List, Optional, Union

from .config import (
    GeometryConfig,
    RoomConfig,
    RuntimeConfig,
    load_geometry,
    load_room,
    load_runtime,
    write_default_configs,
)
from .filters import (
    FilterBankProcessor,
    build_direct_brir,
    db_to_gain,
    design_analytic_filter_bank,
    soft_clip_stereo,
)
from .neural import ControllerContext, apply_prediction, load_controller
from .wav import StereoSample, peak_abs, read_wav, write_wav_float32, write_wav_pcm16


PathLike = Union[str, Path]


def render_stage1(
    input_path: PathLike,
    output_path: PathLike = "preprocessed.wav",
    geometry_path: Optional[PathLike] = None,
    room_path: Optional[PathLike] = None,
    runtime_path: Optional[PathLike] = None,
    model_path: Optional[PathLike] = None,
    pcm16: bool = False,
) -> Dict[str, float]:
    start_time = time.perf_counter()
    runtime = load_runtime(runtime_path)
    geometry = load_geometry(geometry_path)
    room = load_room(room_path)
    sample_rate, samples = read_wav(input_path)
    _require_sample_rate(sample_rate, runtime)

    base_filters = design_analytic_filter_bank(geometry, runtime)
    controller = load_controller(model_path)
    processor = FilterBankProcessor(base_filters)
    input_trim = db_to_gain(runtime.input_trim_db)

    rendered: List[StereoSample] = []
    for block in _blocks(samples, runtime.block_size):
        context = ControllerContext.from_block(block, geometry, room)
        prediction = controller.predict(context)
        target_filters = apply_prediction(base_filters, prediction, runtime, processor.current_filters)
        trimmed = [(left * input_trim, right * input_trim) for left, right in block]
        rendered.extend(
            processor.process_block(
                trimmed,
                next_filters=target_filters,
                smooth_samples=runtime.smoothing_samples,
            )
        )

    rendered = soft_clip_stereo(rendered)
    if pcm16:
        write_wav_pcm16(output_path, runtime.sample_rate, rendered)
    else:
        write_wav_float32(output_path, runtime.sample_rate, rendered)

    elapsed_s = time.perf_counter() - start_time
    return {
        "sample_rate": float(runtime.sample_rate),
        "input_samples": float(len(samples)),
        "output_samples": float(len(rendered)),
        "peak_abs": peak_abs(rendered),
        "ml_model_loaded": 1.0 if model_path is not None else 0.0,
        "elapsed_s": elapsed_s,
        "real_time_factor": elapsed_s / max(len(samples) / runtime.sample_rate, 1e-12),
    }


def render_stage2(
    input_path: PathLike,
    output_path: PathLike = "simulated_binaural.wav",
    geometry_path: Optional[PathLike] = None,
    runtime_path: Optional[PathLike] = None,
    pcm16: bool = False,
) -> Dict[str, float]:
    start_time = time.perf_counter()
    runtime = load_runtime(runtime_path)
    geometry = load_geometry(geometry_path)
    sample_rate, speaker_feed = read_wav(input_path)
    _require_sample_rate(sample_rate, runtime)

    brir = build_direct_brir(geometry, runtime)
    processor = FilterBankProcessor(brir)
    rendered: List[StereoSample] = []
    for block in _blocks(speaker_feed, runtime.block_size):
        rendered.extend(processor.process_block(block))
    rendered.extend(processor.flush(runtime.brir_taps - 1))

    if pcm16:
        write_wav_pcm16(output_path, runtime.sample_rate, rendered)
    else:
        write_wav_float32(output_path, runtime.sample_rate, rendered)

    elapsed_s = time.perf_counter() - start_time
    return {
        "sample_rate": float(runtime.sample_rate),
        "input_samples": float(len(speaker_feed)),
        "output_samples": float(len(rendered)),
        "peak_abs": peak_abs(rendered),
        "elapsed_s": elapsed_s,
        "real_time_factor": elapsed_s / max(len(speaker_feed) / runtime.sample_rate, 1e-12),
    }


def run_pipeline(
    input_path: PathLike,
    stage1_output: PathLike = "preprocessed.wav",
    stage2_output: PathLike = "simulated_binaural.wav",
    geometry_path: Optional[PathLike] = None,
    room_path: Optional[PathLike] = None,
    runtime_path: Optional[PathLike] = None,
    model_path: Optional[PathLike] = None,
) -> Dict[str, Dict[str, float]]:
    stage1 = render_stage1(
        input_path=input_path,
        output_path=stage1_output,
        geometry_path=geometry_path,
        room_path=room_path,
        runtime_path=runtime_path,
        model_path=model_path,
    )
    stage2 = render_stage2(
        input_path=stage1_output,
        output_path=stage2_output,
        geometry_path=geometry_path,
        runtime_path=runtime_path,
    )
    return {"stage1": stage1, "stage2": stage2}


def init_configs(directory: PathLike) -> None:
    write_default_configs(directory)


def _blocks(samples: List[StereoSample], block_size: int):
    for start in range(0, len(samples), block_size):
        yield samples[start : start + block_size]


def _require_sample_rate(sample_rate: int, runtime: RuntimeConfig) -> None:
    if sample_rate != runtime.sample_rate:
        raise ValueError(
            f"expected {runtime.sample_rate} Hz input, got {sample_rate} Hz; "
            "resampling is intentionally not hidden in v1"
        )
