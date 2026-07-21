"""Stable regularized inversion and objective metrics for a 2x2 BRIR."""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any, Dict, Tuple

import numpy as np

from .config import DesignConfig


def design_regularized_ctc(
    brir: np.ndarray,
    config: DesignConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Design causal FIR W that preserves ipsilateral paths and cancels leakage.

    The frequency-dependent Tikhonov term prevents large gains near singular
    BRIR frequencies. A final global gain limiter makes the result safe to
    export even for a poor predicted BRIR.
    """
    config.validate()
    brir = np.asarray(brir, dtype=np.float64)
    if brir.ndim != 3 or brir.shape[:2] != (2, 2):
        raise ValueError("BRIR must have shape [ear=2, speaker=2, taps]")

    h = np.fft.rfft(brir, n=config.fft_size, axis=-1).transpose(2, 0, 1)
    frequencies = np.fft.rfftfreq(config.fft_size, 1.0 / config.sample_rate)
    delay = np.exp(-2j * np.pi * frequencies * config.design_delay / config.sample_rate)
    identity = np.eye(2, dtype=np.complex128)
    w = np.empty_like(h)
    condition_numbers = []

    for index, frequency in enumerate(frequencies):
        matrix = h[index]
        gram = matrix.conj().T @ matrix
        power = max(float(np.trace(gram).real / 2.0), 1e-10)
        relative_lambda = _regularization_at(float(frequency), config)
        regularized = gram + identity * (relative_lambda * power + 1e-10)
        target = np.diag(np.diag(matrix)) * delay[index]
        w[index] = np.linalg.solve(regularized, matrix.conj().T @ target)
        condition_numbers.append(float(np.linalg.cond(matrix)))

    time_filters = np.fft.irfft(w.transpose(1, 2, 0), n=config.fft_size, axis=-1)
    filters = time_filters[:, :, : config.filter_taps].copy()
    _taper_tail(filters)
    filters, limited_by_db = limit_filter_gain(filters, config)
    metrics = evaluate_ctc(brir, filters, config)
    metadata = {
        "method": "frequency-dependent Tikhonov inverse with ipsilateral target",
        "config": asdict(config),
        "gain_reduction_db": limited_by_db,
        "median_brir_condition_number": float(np.median(condition_numbers)),
        "metrics_on_design_brir": metrics,
    }
    return filters, metadata


def limit_filter_gain(filters: np.ndarray, config: DesignConfig) -> Tuple[np.ndarray, float]:
    candidate = np.asarray(filters, dtype=np.float64).copy()
    response = np.fft.rfft(candidate, n=config.fft_size, axis=-1)
    peak = float(np.max(np.linalg.svd(response.transpose(2, 0, 1), compute_uv=False)))
    allowed = 10.0 ** (config.max_filter_gain_db / 20.0)
    if peak <= allowed:
        return candidate, 0.0
    scale = allowed / peak
    return candidate * scale, 20.0 * math.log10(scale)


def evaluate_ctc(brir: np.ndarray, filters: np.ndarray, config: DesignConfig) -> Dict[str, float]:
    h = np.fft.rfft(brir, n=config.fft_size, axis=-1).transpose(2, 0, 1)
    w = np.fft.rfft(filters, n=config.fft_size, axis=-1).transpose(2, 0, 1)
    transfer = h @ w
    frequencies = np.fft.rfftfreq(config.fft_size, 1.0 / config.sample_rate)
    band = (frequencies >= config.low_hz) & (frequencies <= config.high_hz)
    selected = transfer[band]
    desired = np.concatenate((np.abs(selected[:, 0, 0]), np.abs(selected[:, 1, 1])))
    leakage = np.concatenate((np.abs(selected[:, 0, 1]), np.abs(selected[:, 1, 0])))
    direct = np.concatenate((np.abs(h[band, 0, 0]), np.abs(h[band, 1, 1])))
    eps = 1e-12
    suppression = 20.0 * np.log10((desired + eps) / (leakage + eps))
    desired_error_db = 20.0 * np.log10((desired + eps) / (direct + eps))
    return {
        "median_crosstalk_suppression_db": float(np.median(suppression)),
        "p10_crosstalk_suppression_db": float(np.percentile(suppression, 10)),
        "median_desired_error_db": float(np.median(desired_error_db)),
        "p95_desired_error_db": float(np.percentile(np.abs(desired_error_db), 95)),
        "peak_filter_gain_db": _peak_gain_db(w),
    }


def _regularization_at(frequency: float, config: DesignConfig) -> float:
    if frequency < config.low_hz:
        return config.low_regularization
    if frequency > config.high_hz:
        return config.high_regularization
    return config.regularization


def _peak_gain_db(response: np.ndarray) -> float:
    peak = float(np.max(np.linalg.svd(response, compute_uv=False)))
    return 20.0 * math.log10(max(peak, 1e-12))


def _taper_tail(filters: np.ndarray, length: int = 16) -> None:
    length = min(length, filters.shape[-1])
    taper = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, length)))
    filters[:, :, -length:] *= taper
