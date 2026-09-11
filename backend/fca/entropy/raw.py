"""Raw / uncompressed pass-through symbol coder."""

from __future__ import annotations

import numpy as np
from fca.entropy.base import EntropyCoder


class RawEntropyCoder(EntropyCoder):
    """Raw byte-packing for integers (fallback when entropy coding does not compress)."""

    CODER_ID = 0
    NAME = "raw"

    def encode(self, symbols: np.ndarray, **kwargs) -> tuple[bytes, dict]:
        """Pack symbols as signed 32-bit integers little-endian."""
        arr = np.ascontiguousarray(symbols, dtype="<i4")
        return arr.tobytes(), {"coder_id": self.CODER_ID}

    def decode(self, payload: bytes, count: int, **kwargs) -> np.ndarray:
        """Unpack signed 32-bit integers."""
        return np.frombuffer(payload[: count * 4], dtype="<i4").copy()
