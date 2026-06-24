from __future__ import annotations

import math
from pathlib import Path
import struct
from typing import Iterable, List, Sequence, Tuple, Union


StereoSample = Tuple[float, float]
PathLike = Union[str, Path]


class WavError(ValueError):
    pass


def _chunks(data: bytes):
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise WavError("expected a RIFF/WAVE file")

    offset = 12
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", data, offset + 4)[0]
        start = offset + 8
        end = start + chunk_size
        if end > len(data):
            raise WavError(f"truncated WAV chunk {chunk_id!r}")
        yield chunk_id, data[start:end]
        offset = end + (chunk_size % 2)


def read_wav(path: PathLike) -> Tuple[int, List[StereoSample]]:
    data = Path(path).read_bytes()
    fmt = None
    payload = None
    for chunk_id, chunk_data in _chunks(data):
        if chunk_id == b"fmt ":
            if len(chunk_data) < 16:
                raise WavError("WAV fmt chunk is too small")
            fmt = struct.unpack_from("<HHIIHH", chunk_data, 0)
        elif chunk_id == b"data":
            payload = chunk_data

    if fmt is None or payload is None:
        raise WavError("WAV file must contain fmt and data chunks")

    audio_format, channels, sample_rate, _byte_rate, block_align, bits_per_sample = fmt
    if channels < 1:
        raise WavError("WAV file must have at least one channel")
    if block_align <= 0:
        raise WavError("invalid WAV block alignment")

    frame_count = len(payload) // block_align
    samples: List[StereoSample] = []
    for frame in range(frame_count):
        base = frame * block_align
        channel_values = []
        for channel in range(channels):
            channel_base = base + channel * (bits_per_sample // 8)
            channel_values.append(_decode_sample(payload, channel_base, audio_format, bits_per_sample))
        if channels == 1:
            samples.append((channel_values[0], channel_values[0]))
        else:
            samples.append((channel_values[0], channel_values[1]))
    return sample_rate, samples


def _decode_sample(data: bytes, offset: int, audio_format: int, bits_per_sample: int) -> float:
    if audio_format == 1 and bits_per_sample == 16:
        return struct.unpack_from("<h", data, offset)[0] / 32768.0
    if audio_format == 1 and bits_per_sample == 32:
        return struct.unpack_from("<i", data, offset)[0] / 2147483648.0
    if audio_format == 3 and bits_per_sample == 32:
        value = struct.unpack_from("<f", data, offset)[0]
        return 0.0 if not math.isfinite(value) else value
    raise WavError(
        f"unsupported WAV encoding: format={audio_format}, bits={bits_per_sample}; "
        "supported encodings are PCM16, PCM32, and IEEE float32"
    )


def write_wav_float32(path: PathLike, sample_rate: int, samples: Sequence[StereoSample]) -> None:
    payload = bytearray()
    for left, right in samples:
        payload.extend(struct.pack("<ff", _finite(left), _finite(right)))
    _write_wav(path, sample_rate, channels=2, audio_format=3, bits_per_sample=32, payload=bytes(payload))


def write_wav_pcm16(path: PathLike, sample_rate: int, samples: Sequence[StereoSample]) -> None:
    payload = bytearray()
    for left, right in samples:
        payload.extend(struct.pack("<hh", _to_i16(left), _to_i16(right)))
    _write_wav(path, sample_rate, channels=2, audio_format=1, bits_per_sample=16, payload=bytes(payload))


def _write_wav(
    path: PathLike,
    sample_rate: int,
    channels: int,
    audio_format: int,
    bits_per_sample: int,
    payload: bytes,
) -> None:
    block_align = channels * bits_per_sample // 8
    byte_rate = sample_rate * block_align
    fmt_chunk = struct.pack(
        "<HHIIHH",
        audio_format,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    )
    riff_size = 4 + (8 + len(fmt_chunk)) + (8 + len(payload))
    out = bytearray()
    out.extend(b"RIFF")
    out.extend(struct.pack("<I", riff_size))
    out.extend(b"WAVE")
    out.extend(b"fmt ")
    out.extend(struct.pack("<I", len(fmt_chunk)))
    out.extend(fmt_chunk)
    out.extend(b"data")
    out.extend(struct.pack("<I", len(payload)))
    out.extend(payload)
    Path(path).write_bytes(bytes(out))


def _to_i16(value: float) -> int:
    value = max(-1.0, min(1.0, _finite(value)))
    return int(round(value * 32767.0))


def _finite(value: float) -> float:
    return value if math.isfinite(value) else 0.0


def peak_abs(samples: Iterable[StereoSample]) -> float:
    peak = 0.0
    for left, right in samples:
        peak = max(peak, abs(left), abs(right))
    return peak
