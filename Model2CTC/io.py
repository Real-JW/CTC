"""JSON interchange for predicted/measured BRIRs and designed CTC filters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np


PathLike = Union[str, Path]


def _require_bank(values: Any, name: str) -> np.ndarray:
    bank = np.asarray(values, dtype=np.float64)
    if bank.ndim != 3 or bank.shape[:2] != (2, 2) or bank.shape[2] < 2:
        raise ValueError(f"{name} must have shape [2, 2, taps], got {bank.shape}")
    if not np.isfinite(bank).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return bank


def load_brir(path: PathLike) -> Tuple[int, np.ndarray, Dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    sample_rate = int(data["sample_rate"])
    brir = _require_bank(data["filters"], "BRIR")
    return sample_rate, brir, dict(data.get("metadata", {}))


def load_filter(path: PathLike) -> Tuple[int, np.ndarray, Dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    sample_rate = int(data["sample_rate"])
    filters = _require_bank(data["filters"], "CTC filter")
    return sample_rate, filters, dict(data.get("metadata", {}))


def save_filter(
    path: PathLike,
    sample_rate: int,
    filters: np.ndarray,
    metadata: Dict[str, Any],
) -> None:
    bank = _require_bank(filters, "CTC filter")
    payload = {
        "sample_rate": sample_rate,
        "shape": list(bank.shape),
        "filters": bank.tolist(),
        "metadata": metadata,
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")
