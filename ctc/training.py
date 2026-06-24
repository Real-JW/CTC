from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .config import GeometryConfig, RoomConfig, RuntimeConfig
from .filters import FilterBank, design_analytic_filter_bank
from .neural import (
    ControllerContext,
    build_residual_basis,
    prediction_from_vector,
    training_features,
)


PathLike = Union[str, Path]


def train_linear_residual_model(
    output_path: PathLike,
    examples: int = 192,
    ridge: float = 1e-3,
    runtime: Optional[RuntimeConfig] = None,
) -> Dict[str, float]:
    runtime = RuntimeConfig() if runtime is None else runtime
    base_geometry = GeometryConfig()
    base_filters = design_analytic_filter_bank(base_geometry, runtime)

    rows: List[List[float]] = []
    targets: List[List[float]] = []
    for index in range(examples):
        geometry, room = sample_case(index)
        filters = design_analytic_filter_bank(geometry, runtime)
        context = ControllerContext(
            speaker_spacing_m=geometry.speaker_spacing_m,
            listener_distance_m=geometry.listener_distance_m,
            head_radius_m=geometry.head_radius_m,
            yaw_degrees=geometry.yaw_degrees,
            rt60_s=room.rt60_s,
            rms_left=0.0,
            rms_right=0.0,
            side_ratio=0.0,
        )
        rows.append(training_features(context))
        targets.append(target_vector(base_filters, filters, runtime))

    weights = fit_multi_output_ridge(rows, targets, ridge)
    model = {
        "model_type": "linear_residual_v1",
        "description": "Dependency-free supervised residual controller trained from analytic CTC labels.",
        "examples": examples,
        "ridge": ridge,
        "residual_components": 16,
        "feature_count": len(rows[0]),
        "output_count": len(weights),
        "weights": weights,
    }
    Path(output_path).write_text(json.dumps(model, indent=2) + "\n")

    mse = mean_squared_error(rows, targets, weights)
    return {
        "examples": float(examples),
        "feature_count": float(len(rows[0])),
        "output_count": float(len(weights)),
        "mse": mse,
    }


def sample_case(index: int) -> Tuple[GeometryConfig, RoomConfig]:
    spacing = _lerp(0.16, 0.30, _unit(index, 1.0))
    distance = _lerp(0.35, 0.80, _unit(index, 2.0))
    head_radius = _lerp(0.075, 0.095, _unit(index, 3.0))
    yaw = _lerp(-15.0, 15.0, _unit(index, 4.0))
    rt60 = _lerp(0.15, 0.60, _unit(index, 5.0))
    geometry = GeometryConfig(
        speaker_spacing_m=spacing,
        listener_distance_m=distance,
        head_radius_m=head_radius,
        yaw_degrees=yaw,
    )
    room = RoomConfig(rt60_s=rt60)
    return geometry, room


def target_vector(base_filters: FilterBank, target_filters: FilterBank, runtime: RuntimeConfig) -> List[float]:
    basis = build_residual_basis(runtime.filter_taps, 16)
    values: List[float] = []
    for output in range(2):
        for input_channel in range(2):
            base = base_filters[output][input_channel]
            target = target_filters[output][input_channel]
            residual = [target[index] - base[index] for index in range(runtime.filter_taps)]
            for component in basis:
                denom = sum(value * value for value in component) or 1.0
                coeff = sum(residual[index] * component[index] for index in range(runtime.filter_taps)) / denom
                values.append(max(-0.05, min(0.05, coeff)))

    for output in range(2):
        for input_channel in range(2):
            base_gain = _l1(base_filters[output][input_channel])
            target_gain = _l1(target_filters[output][input_channel])
            gain_db = 20.0 * math.log10(max(target_gain, 1e-9) / max(base_gain, 1e-9))
            values.append(max(-0.75, min(0.75, gain_db)))

    for output in range(2):
        for input_channel in range(2):
            base_center = _center_of_energy(base_filters[output][input_channel])
            target_center = _center_of_energy(target_filters[output][input_channel])
            values.append(max(-0.25, min(0.25, target_center - base_center)))

    prediction_from_vector(values, residual_components=16)
    return values


def fit_multi_output_ridge(rows: Sequence[Sequence[float]], targets: Sequence[Sequence[float]], ridge: float) -> List[List[float]]:
    if not rows:
        raise ValueError("at least one training example is required")
    feature_count = len(rows[0])
    output_count = len(targets[0])

    xtx = [[0.0 for _ in range(feature_count)] for _ in range(feature_count)]
    xty = [[0.0 for _ in range(feature_count)] for _ in range(output_count)]

    for row, target in zip(rows, targets):
        for i in range(feature_count):
            for j in range(feature_count):
                xtx[i][j] += row[i] * row[j]
        for output in range(output_count):
            for i in range(feature_count):
                xty[output][i] += target[output] * row[i]

    for i in range(feature_count):
        xtx[i][i] += ridge

    return [solve_linear_system(xtx, xty_row) for xty_row in xty]


def solve_linear_system(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> List[float]:
    size = len(rhs)
    augmented = [list(matrix[row]) + [rhs[row]] for row in range(size)]

    for pivot in range(size):
        pivot_row = max(range(pivot, size), key=lambda row: abs(augmented[row][pivot]))
        if abs(augmented[pivot_row][pivot]) < 1e-12:
            raise ValueError("singular training matrix")
        augmented[pivot], augmented[pivot_row] = augmented[pivot_row], augmented[pivot]

        scale = augmented[pivot][pivot]
        augmented[pivot] = [value / scale for value in augmented[pivot]]
        for row in range(size):
            if row == pivot:
                continue
            factor = augmented[row][pivot]
            if factor == 0.0:
                continue
            augmented[row] = [
                augmented[row][col] - factor * augmented[pivot][col]
                for col in range(size + 1)
            ]

    return [augmented[row][-1] for row in range(size)]


def mean_squared_error(rows: Sequence[Sequence[float]], targets: Sequence[Sequence[float]], weights: Sequence[Sequence[float]]) -> float:
    error = 0.0
    count = 0
    for row, target in zip(rows, targets):
        for output, expected in enumerate(target):
            predicted = sum(weights[output][index] * row[index] for index in range(len(row)))
            error += (predicted - expected) ** 2
            count += 1
    return error / max(count, 1)


def _unit(index: int, salt: float) -> float:
    raw = math.sin((index + 1) * 12.9898 + salt * 78.233) * 43758.5453
    return raw - math.floor(raw)


def _lerp(low: float, high: float, value: float) -> float:
    return low + (high - low) * value


def _l1(values: Sequence[float]) -> float:
    return sum(abs(value) for value in values)


def _center_of_energy(values: Sequence[float]) -> float:
    weight = sum(abs(value) for value in values)
    if weight <= 0.0:
        return 0.0
    return sum(index * abs(value) for index, value in enumerate(values)) / weight
