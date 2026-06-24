from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union


T = TypeVar("T")


@dataclass(frozen=True)
class RuntimeConfig:
    sample_rate: int = 48_000
    block_size: int = 128
    filter_taps: int = 256
    brir_taps: int = 512
    modeling_delay_samples: int = 128
    speed_of_sound_m_s: float = 343.0
    input_trim_db: float = -6.0
    default_filter_boost_db: float = 6.0
    hard_filter_boost_db: float = 12.0
    smoothing_samples: int = 128


@dataclass(frozen=True)
class GeometryConfig:
    speaker_spacing_m: float = 0.20
    listener_distance_m: float = 0.50
    head_radius_m: float = 0.0875
    yaw_degrees: float = 0.0


@dataclass(frozen=True)
class RoomConfig:
    width_m: float = 4.0
    depth_m: float = 3.0
    height_m: float = 2.5
    rt60_s: float = 0.25
    listener_x_m: float = 2.0
    listener_y_m: float = 1.5
    listener_z_m: float = 1.2
    air_absorption: bool = True


PathLike = Union[str, Path]


def _load_dataclass(cls: Type[T], path: PathLike) -> T:
    data = json.loads(Path(path).read_text())
    allowed = {field.name for field in fields(cls)}
    filtered: Dict[str, Any] = {key: value for key, value in data.items() if key in allowed}
    return cls(**filtered)


def load_geometry(path: Optional[PathLike]) -> GeometryConfig:
    return GeometryConfig() if path is None else _load_dataclass(GeometryConfig, path)


def load_room(path: Optional[PathLike]) -> RoomConfig:
    return RoomConfig() if path is None else _load_dataclass(RoomConfig, path)


def load_runtime(path: Optional[PathLike]) -> RuntimeConfig:
    return RuntimeConfig() if path is None else _load_dataclass(RuntimeConfig, path)


def write_json(path: PathLike, value: Any) -> None:
    Path(path).write_text(json.dumps(asdict(value), indent=2) + "\n")


def write_default_configs(directory: PathLike) -> None:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "geometry.json", GeometryConfig())
    write_json(root / "room.json", RoomConfig())
    write_json(root / "runtime.json", RuntimeConfig())
