"""Tiny setup-time model that corrects predicted-BRIR CTC filters."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np

from .config import DesignConfig
from .design import design_regularized_ctc, evaluate_ctc, limit_filter_gain


PathLike = Union[str, Path]


@dataclass
class ResidualModel:
    """Ridge regressor plus a shared low-rank FIR residual basis."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    basis: np.ndarray
    metadata: Dict[str, Any]

    def predict_residual(self, brir: np.ndarray, config: DesignConfig) -> np.ndarray:
        features = brir_features(brir, config)
        if features.shape != self.feature_mean.shape:
            raise ValueError("model/config feature size mismatch")
        normalized = (features - self.feature_mean) / self.feature_scale
        coefficients = np.concatenate(([1.0], normalized)) @ self.weights
        return (coefficients.reshape(4, self.basis.shape[0]) @ self.basis).reshape(
            2, 2, config.filter_taps
        )

    def save(self, path: PathLike) -> None:
        payload = {
            "model_type": "model2ctc_ridge_residual_v1",
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "weights": self.weights.tolist(),
            "basis": self.basis.tolist(),
            "metadata": self.metadata,
        }
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, path: PathLike) -> "ResidualModel":
        data = json.loads(Path(path).read_text())
        if data.get("model_type") != "model2ctc_ridge_residual_v1":
            raise ValueError("unsupported Model2CTC model")
        return cls(
            feature_mean=np.asarray(data["feature_mean"], dtype=np.float64),
            feature_scale=np.asarray(data["feature_scale"], dtype=np.float64),
            weights=np.asarray(data["weights"], dtype=np.float64),
            basis=np.asarray(data["basis"], dtype=np.float64),
            metadata=dict(data.get("metadata", {})),
        )


def brir_features(brir: np.ndarray, config: DesignConfig) -> np.ndarray:
    """Compact scale-aware complex spectral features from all four BRIR paths."""
    response = np.fft.rfft(brir, n=config.fft_size, axis=-1)
    frequencies = np.fft.rfftfreq(config.fft_size, 1.0 / config.sample_rate)
    targets = np.geomspace(config.low_hz, config.high_hz, config.feature_bins)
    indices = np.unique(np.searchsorted(frequencies, targets).clip(0, len(frequencies) - 1))
    values = []
    for index in indices:
        matrix = response[:, :, index]
        scale = max(float(np.linalg.norm(matrix)), 1e-9)
        normalized = matrix / scale
        values.append(np.log(scale))
        values.extend(normalized.real.ravel())
        values.extend(normalized.imag.ravel())
    return np.asarray(values, dtype=np.float64)


def design_hybrid_ctc(
    predicted_brir: np.ndarray,
    model: ResidualModel,
    config: DesignConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    base, base_metadata = design_regularized_ctc(predicted_brir, config)
    candidate = base + model.predict_residual(predicted_brir, config)
    candidate, reduction_db = limit_filter_gain(candidate, config)
    return candidate, {
        "method": "regularized inverse plus learned low-rank residual",
        "base": base_metadata,
        "ml_gain_reduction_db": reduction_db,
        "metrics_on_predicted_brir": evaluate_ctc(predicted_brir, candidate, config),
    }
