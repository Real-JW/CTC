from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import List, Optional, Protocol, Sequence, Union

from .config import GeometryConfig, RoomConfig, RuntimeConfig
from .filters import FilterBank, clone_filter_bank, db_to_gain, max_l1_gain
from .wav import StereoSample


def _sigmoid(value: float) -> float:
    if value < -60.0:
        return 0.0
    if value > 60.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-value))


def _weight(seed: float, row: int, col: int, scale: float) -> float:
    raw = math.sin((row + 1) * 12.9898 + (col + 1) * 78.233 + seed * 37.719)
    frac = (raw * 43758.5453) % 1.0
    return (frac * 2.0 - 1.0) * scale


def _mat_vec(matrix: List[List[float]], vector: Sequence[float], bias: Sequence[float]) -> List[float]:
    out = []
    for row_index, row in enumerate(matrix):
        acc = bias[row_index]
        for col_index, value in enumerate(vector):
            acc += row[col_index] * value
        out.append(acc)
    return out


class GRULayer:
    def __init__(self, input_size: int, hidden_size: int, seed: float):
        input_scale = 0.20 / math.sqrt(max(input_size, 1))
        hidden_scale = 0.15 / math.sqrt(max(hidden_size, 1))
        self.hidden_size = hidden_size
        self.w_z = self._matrix(hidden_size, input_size, seed + 1.0, input_scale)
        self.u_z = self._matrix(hidden_size, hidden_size, seed + 2.0, hidden_scale)
        self.b_z = [0.0 for _ in range(hidden_size)]
        self.w_r = self._matrix(hidden_size, input_size, seed + 3.0, input_scale)
        self.u_r = self._matrix(hidden_size, hidden_size, seed + 4.0, hidden_scale)
        self.b_r = [0.0 for _ in range(hidden_size)]
        self.w_n = self._matrix(hidden_size, input_size, seed + 5.0, input_scale)
        self.u_n = self._matrix(hidden_size, hidden_size, seed + 6.0, hidden_scale)
        self.b_n = [0.0 for _ in range(hidden_size)]
        self.hidden = [0.0 for _ in range(hidden_size)]

    @staticmethod
    def _matrix(rows: int, cols: int, seed: float, scale: float) -> List[List[float]]:
        return [[_weight(seed, row, col, scale) for col in range(cols)] for row in range(rows)]

    def step(self, inputs: Sequence[float]) -> List[float]:
        z_in = _mat_vec(self.w_z, inputs, self.b_z)
        r_in = _mat_vec(self.w_r, inputs, self.b_r)
        n_in = _mat_vec(self.w_n, inputs, self.b_n)
        z_hidden = _mat_vec(self.u_z, self.hidden, [0.0 for _ in range(self.hidden_size)])
        r_hidden = _mat_vec(self.u_r, self.hidden, [0.0 for _ in range(self.hidden_size)])

        z = [_sigmoid(z_in[i] + z_hidden[i]) for i in range(self.hidden_size)]
        r = [_sigmoid(r_in[i] + r_hidden[i]) for i in range(self.hidden_size)]
        reset_hidden = [r[i] * self.hidden[i] for i in range(self.hidden_size)]
        n_hidden = _mat_vec(self.u_n, reset_hidden, [0.0 for _ in range(self.hidden_size)])
        candidate = [math.tanh(n_in[i] + n_hidden[i]) for i in range(self.hidden_size)]
        self.hidden = [
            (1.0 - z[i]) * candidate[i] + z[i] * self.hidden[i]
            for i in range(self.hidden_size)
        ]
        return list(self.hidden)


@dataclass(frozen=True)
class ControllerContext:
    speaker_spacing_m: float
    listener_distance_m: float
    head_radius_m: float
    yaw_degrees: float
    rt60_s: float
    rms_left: float
    rms_right: float
    side_ratio: float

    @classmethod
    def from_block(
        cls,
        block: Sequence[StereoSample],
        geometry: GeometryConfig,
        room: RoomConfig,
    ) -> "ControllerContext":
        if not block:
            rms_left = 0.0
            rms_right = 0.0
            side_ratio = 0.0
        else:
            sum_left = sum(left * left for left, _right in block)
            sum_right = sum(right * right for _left, right in block)
            sum_side = sum(((left - right) * 0.5) ** 2 for left, right in block)
            sum_mid = sum(((left + right) * 0.5) ** 2 for left, right in block)
            rms_left = math.sqrt(sum_left / len(block))
            rms_right = math.sqrt(sum_right / len(block))
            side_ratio = math.sqrt(sum_side / max(sum_mid + sum_side, 1e-12))

        return cls(
            speaker_spacing_m=geometry.speaker_spacing_m,
            listener_distance_m=geometry.listener_distance_m,
            head_radius_m=geometry.head_radius_m,
            yaw_degrees=geometry.yaw_degrees,
            rt60_s=room.rt60_s,
            rms_left=rms_left,
            rms_right=rms_right,
            side_ratio=side_ratio,
        )

    def features(self) -> List[float]:
        return [
            (self.speaker_spacing_m - 0.20) / 0.10,
            (self.listener_distance_m - 0.50) / 0.30,
            (self.head_radius_m - 0.0875) / 0.02,
            self.yaw_degrees / 15.0,
            (self.rt60_s - 0.25) / 0.35,
            min(2.0, self.rms_left * 2.0),
            min(2.0, self.rms_right * 2.0),
            max(0.0, min(1.0, self.side_ratio)),
        ]


@dataclass(frozen=True)
class MLFilterPrediction:
    residual_weights: List[List[float]]
    gains_db: List[float]
    delay_trims: List[float]


class NeuralFilterController:
    """Small recurrent controller that predicts bounded FIR filter corrections.

    The default weights are deterministic placeholders. They make the full ML
    inference path executable before a supervised training/export pipeline exists.
    """

    def __init__(self, hidden_size: int = 64, residual_components: int = 16):
        self.residual_components = residual_components
        self.layer1 = GRULayer(input_size=8, hidden_size=hidden_size, seed=1.0)
        self.layer2 = GRULayer(input_size=hidden_size, hidden_size=hidden_size, seed=2.0)
        self.output_size = 4 * residual_components + 8
        scale = 0.10 / math.sqrt(hidden_size)
        self.w_out = [
            [_weight(10.0, row, col, scale) for col in range(hidden_size)]
            for row in range(self.output_size)
        ]
        self.b_out = [0.0 for _ in range(self.output_size)]

    def predict(self, context: ControllerContext) -> MLFilterPrediction:
        hidden1 = self.layer1.step(context.features())
        hidden2 = self.layer2.step(hidden1)
        raw = _mat_vec(self.w_out, hidden2, self.b_out)

        residual_weights: List[List[float]] = []
        cursor = 0
        for _path in range(4):
            path_weights = []
            for _component in range(self.residual_components):
                path_weights.append(math.tanh(raw[cursor]) * 0.025)
                cursor += 1
            residual_weights.append(path_weights)

        gains_db = [math.tanh(raw[cursor + i]) * 0.75 for i in range(4)]
        cursor += 4
        delay_trims = [math.tanh(raw[cursor + i]) * 0.25 for i in range(4)]
        return MLFilterPrediction(residual_weights, gains_db, delay_trims)


class FilterController(Protocol):
    def predict(self, context: ControllerContext) -> MLFilterPrediction:
        ...


class LinearResidualController:
    def __init__(self, weights: List[List[float]], residual_components: int = 16):
        self.weights = weights
        self.residual_components = residual_components
        self.output_size = 4 * residual_components + 8
        if len(weights) != self.output_size:
            raise ValueError(f"expected {self.output_size} output rows, got {len(weights)}")

    def predict(self, context: ControllerContext) -> MLFilterPrediction:
        features = training_features(context)
        raw = []
        for row in self.weights:
            if len(row) != len(features):
                raise ValueError(f"model feature count mismatch: expected {len(row)}, got {len(features)}")
            raw.append(sum(row[index] * features[index] for index in range(len(features))))
        return prediction_from_vector(raw, self.residual_components)


def training_features(context: ControllerContext) -> List[float]:
    features = context.features()
    geometry_terms = features[:5]
    return [1.0] + features + [term * term for term in geometry_terms]


def prediction_from_vector(values: Sequence[float], residual_components: int = 16) -> MLFilterPrediction:
    expected = 4 * residual_components + 8
    if len(values) != expected:
        raise ValueError(f"expected {expected} prediction values, got {len(values)}")

    residual_weights: List[List[float]] = []
    cursor = 0
    for _path in range(4):
        residual_weights.append([
            max(-0.05, min(0.05, values[cursor + component]))
            for component in range(residual_components)
        ])
        cursor += residual_components

    gains_db = [max(-0.75, min(0.75, values[cursor + index])) for index in range(4)]
    cursor += 4
    delay_trims = [max(-0.25, min(0.25, values[cursor + index])) for index in range(4)]
    return MLFilterPrediction(residual_weights, gains_db, delay_trims)


PathLike = Union[str, Path]


def load_controller(path: Optional[PathLike]) -> FilterController:
    if path is None:
        return NeuralFilterController()

    data = json.loads(Path(path).read_text())
    model_type = data.get("model_type")
    if model_type != "linear_residual_v1":
        raise ValueError(f"unsupported ML model type: {model_type!r}")
    return LinearResidualController(
        weights=data["weights"],
        residual_components=int(data.get("residual_components", 16)),
    )


def build_residual_basis(taps: int, components: int) -> List[List[float]]:
    basis: List[List[float]] = []
    width = max(4, taps // (components * 2))
    for component in range(components):
        center = int(round((component + 1) * (taps - 1) / (components + 1)))
        path = []
        for tap in range(taps):
            distance = abs(tap - center)
            value = max(0.0, 1.0 - distance / width)
            if component % 2:
                value = -value
            path.append(value)
        norm = sum(abs(value) for value in path) or 1.0
        basis.append([value / norm for value in path])
    return basis


def apply_prediction(
    base_filters: FilterBank,
    prediction: MLFilterPrediction,
    runtime: RuntimeConfig,
    previous_filters: Optional[FilterBank] = None,
) -> FilterBank:
    taps = len(base_filters[0][0])
    basis = build_residual_basis(taps, len(prediction.residual_weights[0]))
    candidate = clone_filter_bank(base_filters)

    for output in range(2):
        for input_channel in range(2):
            path_index = output * 2 + input_channel
            path = candidate[output][input_channel]
            for component, weight in enumerate(prediction.residual_weights[path_index]):
                for tap in range(taps):
                    path[tap] += weight * basis[component][tap]
            gain = db_to_gain(prediction.gains_db[path_index])
            path = [tap * gain for tap in path]
            path = fractional_shift(path, prediction.delay_trims[path_index])
            candidate[output][input_channel] = path

    if not validate_filter_bank(candidate, runtime, previous_filters):
        return clone_filter_bank(previous_filters if previous_filters is not None else base_filters)
    return candidate


def fractional_shift(path: Sequence[float], shift_samples: float) -> List[float]:
    shifted = []
    for out_index in range(len(path)):
        source = out_index - shift_samples
        lower = math.floor(source)
        frac = source - lower
        value = 0.0
        if 0 <= lower < len(path):
            value += path[lower] * (1.0 - frac)
        upper = lower + 1
        if 0 <= upper < len(path):
            value += path[upper] * frac
        shifted.append(value)
    return shifted


def validate_filter_bank(
    filters: FilterBank,
    runtime: RuntimeConfig,
    previous_filters: Optional[FilterBank] = None,
) -> bool:
    for output in filters:
        for path in output:
            for tap in path:
                if not math.isfinite(tap):
                    return False

    if max_l1_gain(filters) > db_to_gain(runtime.hard_filter_boost_db):
        return False

    if previous_filters is not None:
        max_delta = 0.0
        for output in range(2):
            for input_channel in range(2):
                current = filters[output][input_channel]
                previous = previous_filters[output][input_channel]
                for index in range(len(current)):
                    max_delta = max(max_delta, abs(current[index] - previous[index]))
        if max_delta > 0.75:
            return False

    return True
