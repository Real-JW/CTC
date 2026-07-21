"""Export designed FIR coefficients for a small fixed-point edge runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np


PathLike = Union[str, Path]


def export_fixed_header(
    filters: np.ndarray,
    path: PathLike,
    symbol: str = "model2ctc_coeffs",
    coefficient_q: int = 14,
) -> None:
    bank = np.asarray(filters, dtype=np.float64)
    if bank.ndim != 3 or bank.shape[:2] != (2, 2):
        raise ValueError("CTC filters must have shape [2, 2, taps]")
    scale = 1 << coefficient_q
    quantized = np.clip(np.rint(bank * scale), -32768, 32767).astype(np.int16)
    taps = bank.shape[-1]
    rows = []
    for output in range(2):
        paths = []
        for input_channel in range(2):
            values = ", ".join(str(int(value)) for value in quantized[output, input_channel])
            paths.append("        {" + values + "}")
        rows.append("    {\n" + ",\n".join(paths) + "\n    }")
    content = f"""#ifndef MODEL2CTC_COEFFICIENTS_H
#define MODEL2CTC_COEFFICIENTS_H

#include <stdint.h>

#define MODEL2CTC_TAPS {taps}
#define MODEL2CTC_COEFF_Q {coefficient_q}
static const int16_t {symbol}[2][2][MODEL2CTC_TAPS] = {{
{','.join(chr(10) + row for row in rows)}
}};

#endif
"""
    Path(path).write_text(content)
