"""Adaptive Rice entropy coder for geometric / Laplacian distributed residuals."""

from __future__ import annotations

import numpy as np
from fca.entropy.base import EntropyCoder
from fca.entropy.bitstream import BitWriter, BitReader, zigzag_encode, zigzag_decode


class RiceCoder(EntropyCoder):
    """Adaptive Rice entropy coder."""

    CODER_ID = 1
    NAME = "rice"
    MAX_K = 16

    @classmethod
    def find_optimal_k(cls, u_symbols: np.ndarray) -> int:
        """Find parameter k in 0..16 that minimizes the encoded bit length."""
        if len(u_symbols) == 0:
            return 0

        # Mean-based initial estimate
        mean_val = float(np.mean(u_symbols))
        if mean_val <= 0.5:
            return 0

        best_k = 0
        min_bits = float("inf")

        u = u_symbols.astype(np.int64)
        n = len(u)

        # Test k from 0 up to 16
        for k in range(cls.MAX_K + 1):
            # Each symbol takes: quotient (q) + 1 (stop bit) + k (remainder bits)
            q_sum = np.sum(u >> k)
            total_bits = q_sum + n * (1 + k)
            if total_bits < min_bits:
                min_bits = total_bits
                best_k = k

        return best_k

    def encode(self, symbols: np.ndarray, k: int | None = None, **kwargs) -> tuple[bytes, dict]:
        """Encode signed symbols using zigzag mapping and Rice coding.

        If k is not provided, the optimal k is automatically computed.
        The parameter k is packed in the first byte of payload.
        """
        n = len(symbols)
        if n == 0:
            return b"\x00", {"coder_id": self.CODER_ID, "k": 0}

        # Convert to unsigned zigzag
        s = symbols.astype(np.int64)
        u = np.where(s >= 0, s << 1, (-s << 1) - 1)

        if k is None or k < 0:
            k = self.find_optimal_k(u)
        k = min(k, self.MAX_K)

        writer = BitWriter()
        # Store k in the first 8 bits of payload
        writer.write_bits(k, 8)

        mask = (1 << k) - 1
        for val in u:
            q = int(val >> k)
            r = int(val & mask)
            writer.write_unary_zeros(q)
            if k > 0:
                writer.write_bits(r, k)

        payload = writer.to_bytes()
        return payload, {"coder_id": self.CODER_ID, "k": k}

    def decode(self, payload: bytes, count: int, k: int | None = None, **kwargs) -> np.ndarray:
        """Decode payload into `count` signed integers."""
        if count == 0:
            return np.zeros(0, dtype=np.int64)

        reader = BitReader(payload)
        if k is None:
            # Read stored k from header byte
            k = reader.read_bits(8)

        mask = (1 << k) - 1 if k > 0 else 0
        decoded = np.empty(count, dtype=np.int64)

        for i in range(count):
            q = reader.read_unary_zeros()
            r = reader.read_bits(k) if k > 0 else 0
            u = (q << k) | r
            # Inverse zigzag
            s = (u >> 1) if (u & 1) == 0 else -((u + 1) >> 1)
            decoded[i] = s

        return decoded
