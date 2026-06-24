from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence, Tuple

from .config import GeometryConfig, RuntimeConfig
from .wav import StereoSample


Filter = List[float]
FilterBank = List[List[Filter]]
SparsePath = List[Tuple[int, float]]
SparseFilterBank = List[List[SparsePath]]
Point3 = Tuple[float, float, float]


def db_to_gain(db: float) -> float:
    return 10.0 ** (db / 20.0)


def speaker_positions(geometry: GeometryConfig) -> Tuple[Point3, Point3]:
    half_spacing = geometry.speaker_spacing_m / 2.0
    return (
        (geometry.listener_distance_m, half_spacing, 0.0),
        (geometry.listener_distance_m, -half_spacing, 0.0),
    )


def ear_positions(geometry: GeometryConfig) -> Tuple[Point3, Point3]:
    yaw = math.radians(geometry.yaw_degrees)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    def rotate(point: Point3) -> Point3:
        x, y, z = point
        return (x * cos_yaw - y * sin_yaw, x * sin_yaw + y * cos_yaw, z)

    return (
        rotate((0.0, geometry.head_radius_m, 0.0)),
        rotate((0.0, -geometry.head_radius_m, 0.0)),
    )


def distance(a: Point3, b: Point3) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def add_fractional_impulse(target: Filter, delay_samples: float, amplitude: float) -> None:
    if delay_samples < 0.0:
        return
    index = int(math.floor(delay_samples))
    frac = delay_samples - index
    if 0 <= index < len(target):
        target[index] += amplitude * (1.0 - frac)
    if 0 <= index + 1 < len(target):
        target[index + 1] += amplitude * frac


def zero_filter_bank(outputs: int, inputs: int, taps: int) -> FilterBank:
    return [[[0.0 for _ in range(taps)] for _ in range(inputs)] for _ in range(outputs)]


def clone_filter_bank(filters: FilterBank) -> FilterBank:
    return [[list(path) for path in output] for output in filters]


@dataclass(frozen=True)
class DirectPath:
    distance_m: float
    delay_samples: float
    amplitude: float


def direct_path_matrix(geometry: GeometryConfig, runtime: RuntimeConfig) -> List[List[DirectPath]]:
    speakers = speaker_positions(geometry)
    ears = ear_positions(geometry)
    matrix: List[List[DirectPath]] = []
    for ear in ears:
        row = []
        for speaker in speakers:
            dist = distance(ear, speaker)
            delay = dist / runtime.speed_of_sound_m_s * runtime.sample_rate
            amplitude = geometry.listener_distance_m / max(dist, 1e-9)
            row.append(DirectPath(dist, delay, amplitude))
        matrix.append(row)
    return matrix


def build_direct_brir(
    geometry: GeometryConfig,
    runtime: RuntimeConfig,
    taps: Optional[int] = None,
) -> FilterBank:
    taps = runtime.brir_taps if taps is None else taps
    paths = direct_path_matrix(geometry, runtime)
    brir = zero_filter_bank(outputs=2, inputs=2, taps=taps)
    for ear in range(2):
        for speaker in range(2):
            path = paths[ear][speaker]
            add_fractional_impulse(brir[ear][speaker], path.delay_samples, path.amplitude)
    return brir


def design_analytic_filter_bank(geometry: GeometryConfig, runtime: RuntimeConfig) -> FilterBank:
    filters = zero_filter_bank(outputs=2, inputs=2, taps=runtime.filter_taps)
    paths = direct_path_matrix(geometry, runtime)
    delay = float(runtime.modeling_delay_samples)

    add_fractional_impulse(filters[0][0], delay, 1.0)
    add_fractional_impulse(filters[1][1], delay, 1.0)

    left_to_right_ear = paths[1][0]
    right_to_right_ear = paths[1][1]
    cancel_right_delay = delay + left_to_right_ear.delay_samples - right_to_right_ear.delay_samples
    cancel_right_gain = -left_to_right_ear.amplitude / max(right_to_right_ear.amplitude, 1e-9)
    add_fractional_impulse(filters[1][0], cancel_right_delay, cancel_right_gain)

    right_to_left_ear = paths[0][1]
    left_to_left_ear = paths[0][0]
    cancel_left_delay = delay + right_to_left_ear.delay_samples - left_to_left_ear.delay_samples
    cancel_left_gain = -right_to_left_ear.amplitude / max(left_to_left_ear.amplitude, 1e-9)
    add_fractional_impulse(filters[0][1], cancel_left_delay, cancel_left_gain)

    return enforce_l1_limit(filters, db_to_gain(runtime.default_filter_boost_db))


def enforce_l1_limit(filters: FilterBank, max_gain: float) -> FilterBank:
    limited = clone_filter_bank(filters)
    for output in range(len(limited)):
        for input_channel in range(len(limited[output])):
            l1 = sum(abs(tap) for tap in limited[output][input_channel])
            if l1 > max_gain and l1 > 0.0:
                scale = max_gain / l1
                limited[output][input_channel] = [tap * scale for tap in limited[output][input_channel]]
    return limited


def max_l1_gain(filters: FilterBank) -> float:
    return max(
        sum(abs(tap) for tap in path)
        for output in filters
        for path in output
    )


def soft_clip(value: float) -> float:
    return math.tanh(value)


def soft_clip_stereo(samples: Sequence[StereoSample]) -> List[StereoSample]:
    return [(soft_clip(left), soft_clip(right)) for left, right in samples]


class FilterBankProcessor:
    def __init__(self, filters: FilterBank):
        self.current_filters = clone_filter_bank(filters)
        self.current_sparse = sparse_filter_bank(self.current_filters)
        self.taps = len(filters[0][0])
        self.input_count = len(filters[0])
        self.output_count = len(filters)
        self.histories = [[0.0 for _ in range(self.taps)] for _ in range(self.input_count)]
        self.write_index = 0

    def process_block(
        self,
        block: Sequence[StereoSample],
        next_filters: Optional[FilterBank] = None,
        smooth_samples: int = 0,
    ) -> List[StereoSample]:
        target = self.current_filters if next_filters is None else next_filters
        target_sparse = self.current_sparse if next_filters is None else sparse_filter_bank(target)
        smooth_samples = max(1, smooth_samples)
        output: List[StereoSample] = []

        for sample_index, (left, right) in enumerate(block):
            inputs = (left, right)
            for channel in range(self.input_count):
                self.histories[channel][self.write_index] = inputs[channel]

            if next_filters is not None:
                current_rendered = self._render_sparse(self.current_sparse)
                target_rendered = self._render_sparse(target_sparse)
                alpha = min(1.0, (sample_index + 1) / smooth_samples)
                rendered = [
                    current_rendered[channel] + alpha * (target_rendered[channel] - current_rendered[channel])
                    for channel in range(self.output_count)
                ]
            else:
                rendered = self._render_sparse(self.current_sparse)

            output.append((rendered[0], rendered[1]))
            self.write_index = (self.write_index + 1) % self.taps

        if next_filters is not None:
            self.current_filters = clone_filter_bank(target)
            self.current_sparse = target_sparse
        return output

    def flush(self, samples: int) -> List[StereoSample]:
        return self.process_block([(0.0, 0.0)] * samples)

    def _render_sparse(self, filters: SparseFilterBank) -> List[float]:
        rendered = []
        for output_channel in range(self.output_count):
            acc = 0.0
            for input_channel in range(self.input_count):
                history = self.histories[input_channel]
                for tap_index, coeff in filters[output_channel][input_channel]:
                    acc += coeff * history[(self.write_index - tap_index) % self.taps]
            rendered.append(acc)
        return rendered


def sparse_filter_bank(filters: FilterBank, threshold: float = 1e-12) -> SparseFilterBank:
    return [
        [
            [(tap_index, coeff) for tap_index, coeff in enumerate(path) if abs(coeff) > threshold]
            for path in output
        ]
        for output in filters
    ]


def process_static(
    samples: Sequence[StereoSample],
    filters: FilterBank,
    block_size: int,
    include_tail: bool = False,
) -> List[StereoSample]:
    processor = FilterBankProcessor(filters)
    output: List[StereoSample] = []
    for start in range(0, len(samples), block_size):
        output.extend(processor.process_block(samples[start : start + block_size]))
    if include_tail:
        output.extend(processor.flush(processor.taps - 1))
    return output
