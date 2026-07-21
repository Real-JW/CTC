"""Configuration for stable model-to-CTC filter design."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Union


PathLike = Union[str, Path]


@dataclass(frozen=True)
class DesignConfig:
    sample_rate: int = 48_000
    filter_taps: int = 128
    design_delay: int = 64
    fft_size: int = 2048
    low_hz: float = 150.0
    high_hz: float = 12_000.0
    regularization: float = 0.02
    low_regularization: float = 0.10
    high_regularization: float = 0.08
    max_filter_gain_db: float = 6.0
    feature_bins: int = 24
    residual_components: int = 8

    def validate(self) -> None:
        if self.filter_taps <= 0 or self.fft_size < self.filter_taps:
            raise ValueError("fft_size must be at least filter_taps > 0")
        if not 0 <= self.design_delay < self.filter_taps:
            raise ValueError("design_delay must be inside the FIR")
        if not 0 < self.low_hz < self.high_hz < self.sample_rate / 2:
            raise ValueError("invalid CTC frequency band")
        if min(self.regularization, self.low_regularization, self.high_regularization) <= 0:
            raise ValueError("regularization must be positive")

    @classmethod
    def load(cls, path: PathLike) -> "DesignConfig":
        data = json.loads(Path(path).read_text())
        allowed = {field.name for field in fields(cls)}
        config = cls(**{key: value for key, value in data.items() if key in allowed})
        config.validate()
        return config

    def save(self, path: PathLike) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")
