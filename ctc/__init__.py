"""Reference implementation for ML-guided crosstalk cancellation."""

from .config import GeometryConfig, RoomConfig, RuntimeConfig
from .pipeline import render_stage1, render_stage2, run_pipeline

__all__ = [
    "GeometryConfig",
    "RoomConfig",
    "RuntimeConfig",
    "render_stage1",
    "render_stage2",
    "run_pipeline",
]
