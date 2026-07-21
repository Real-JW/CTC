"""Small dependency-free stereo WAV reader/writer used by BRIR demos."""

from __future__ import annotations

import math
from pathlib import Path
import struct
from typing import Iterable, List, Sequence, Tuple, Union


StereoSample = Tuple[float, float]
PathLike = Union[str, Path]


def read_wav(path: PathLike) -> Tuple[int, List[StereoSample]]:
    data = Path(path).read_bytes()
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("expected a RIFF/WAVE file")
    fmt = payload = None
    offset = 12
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        size = struct.unpack_from("<I", data, offset + 4)[0]
        chunk = data[offset + 8 : offset + 8 + size]
        if chunk_id == b"fmt ":
            fmt = struct.unpack_from("<HHIIHH", chunk, 0)
        elif chunk_id == b"data":
            payload = chunk
        offset += 8 + size + size % 2
    if fmt is None or payload is None:
        raise ValueError("WAV must contain fmt and data chunks")

    audio_format, channels, sample_rate, _rate, block_align, bits = fmt
    samples = []
    for frame in range(len(payload) // block_align):
        values = [
            _decode(payload, frame * block_align + channel * (bits // 8), audio_format, bits)
            for channel in range(channels)
        ]
        samples.append((values[0], values[0] if channels == 1 else values[1]))
    return sample_rate, samples


def _decode(data: bytes, offset: int, audio_format: int, bits: int) -> float:
    if audio_format == 1 and bits == 16:
        return struct.unpack_from("<h", data, offset)[0] / 32768.0
    if audio_format == 1 and bits == 32:
        return struct.unpack_from("<i", data, offset)[0] / 2147483648.0
    if audio_format == 3 and bits == 32:
        value = struct.unpack_from("<f", data, offset)[0]
        return value if math.isfinite(value) else 0.0
    raise ValueError(f"unsupported WAV encoding: format={audio_format}, bits={bits}")


def write_wav_float32(path: PathLike, sample_rate: int, samples: Sequence[StereoSample]) -> None:
    payload = b"".join(struct.pack("<ff", _finite(left), _finite(right)) for left, right in samples)
    _write_wav(path, sample_rate, 3, 32, payload)


def write_wav_pcm16(path: PathLike, sample_rate: int, samples: Sequence[StereoSample]) -> None:
    payload = b"".join(struct.pack("<hh", _q15(left), _q15(right)) for left, right in samples)
    _write_wav(path, sample_rate, 1, 16, payload)


def _write_wav(path: PathLike, sample_rate: int, audio_format: int, bits: int, payload: bytes) -> None:
    channels = 2
    block_align = channels * bits // 8
    fmt = struct.pack("<HHIIHH", audio_format, channels, sample_rate, sample_rate * block_align, block_align, bits)
    riff_size = 4 + 8 + len(fmt) + 8 + len(payload)
    Path(path).write_bytes(
        b"RIFF" + struct.pack("<I", riff_size) + b"WAVE"
        + b"fmt " + struct.pack("<I", len(fmt)) + fmt
        + b"data" + struct.pack("<I", len(payload)) + payload
    )


def _q15(value: float) -> int:
    return int(round(max(-1.0, min(1.0, _finite(value))) * 32767.0))


def _finite(value: float) -> float:
    return value if math.isfinite(value) else 0.0


def peak_abs(samples: Iterable[StereoSample]) -> float:
    return max((max(abs(left), abs(right)) for left, right in samples), default=0.0)
