"""Stable hybrid ML/FIR crosstalk-cancellation filter design."""

from .config import DesignConfig
from .design import design_regularized_ctc, evaluate_ctc
from .model import ResidualModel, design_hybrid_ctc

__all__ = [
    "DesignConfig",
    "ResidualModel",
    "design_hybrid_ctc",
    "design_regularized_ctc",
    "evaluate_ctc",
]
