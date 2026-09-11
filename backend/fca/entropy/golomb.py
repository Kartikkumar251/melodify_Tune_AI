"""Golomb entropy coder for arbitrary positive modulus M."""

from __future__ import annotations

import math
import numpy as np
from fca.entropy.base import EntropyCoder
from fca.entropy.bitstream import BitWriter, BitReader


class GolombCoder(EntropyCoder):
    """Golomb entropy coder supporting arbitrary modulus M >= 1."""

    CODER_ID = 3
    NAME = "golomb"

    @classmethod
    def find_optimal_m(cls, u_symbols: np.ndarray) -> int:
        """Heuristic optimal M estimation: M = ceil(-ln(2) / ln(p)) ~ ceil(0.693 * mean)."""
        if len(u_symbols) == 0:
            return 1
        mean_val = float(np.mean(u_symbols))
        if mean_val <= 0.5:
            return 1
        # Golomb parameter approximation
        m = max(1, int(round(0.693147 * mean_val)))
        return min(m, 65535)

    def encode(self, symbols: np.ndarray, m: int | None = None, **kwargs) -> tuple[bytes, dict]:
        """Encode signed symbols using Golomb coding."""
        if len(symbols) == 0:
            return b"\x00\x01", {"coder_id": self.CODER_ID, "m": 1}

        # Zigzag
        s = symbols.astype(np.int64)
        u = np.where(s >= 0, s << 1, (-s << 1) - 1)

        if m is None or m < 1:
            m = self.find_optimal_m(u)

        writer = BitWriter()
        # Store M in first 16 bits of payload
        writer.write_bits(m, 16)

        b = math.ceil(math.log2(m)) if m > 1 else 0
        cutoff = (1 << b) - m if b > 0 else 0

        for val in u:
            q = int(val // m)
            r = int(val % m)
            writer.write_unary_zeros(q)
            if b > 0:
                if r < cutoff:
                    writer.write_bits(r, b - 1)
                else:
                    writer.write_bits(r + cutoff, b)

        payload = writer.to_bytes()
        return payload, {"coder_id": self.CODER_ID, "m": m}

    def decode(self, payload: bytes, count: int, m: int | None = None, **kwargs) -> np.ndarray:
        """Decode payload into `count` signed integers."""
        if count == 0:
            return np.zeros(0, dtype=np.int64)

        reader = BitReader(payload)
        if m is None:
            m = reader.read_bits(16)
        if m < 1:
            m = 1

        b = math.ceil(math.log2(m)) if m > 1 else 0
        cutoff = (1 << b) - m if b > 0 else 0

        decoded = np.empty(count, dtype=np.int64)
        for i in range(count):
            q = reader.read_unary_zeros()
            if b == 0:
                r = 0
            else:
                first_bits = reader.read_bits(b - 1)
                if first_bits < cutoff:
                    r = first_bits
                else:
                    extra_bit = reader.read_bit()
                    full_val = (first_bits << 1) | extra_bit
                    r = full_val - cutoff

            u = q * m + r
            s = (u >> 1) if (u & 1) == 0 else -((u + 1) >> 1)
            decoded[i] = s

        return decoded
