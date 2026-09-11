"""Range / Asymmetric Numeral Systems (rANS) entropy coder.

Implements precision integer rANS (ANS/FSE family) for exact probability distribution coding.
"""

from __future__ import annotations

from collections import Counter
import numpy as np
from fca.entropy.base import EntropyCoder
from fca.entropy.bitstream import BitWriter, BitReader


class RangeCoder(EntropyCoder):
    """Range / Asymmetric Numeral Systems (rANS) entropy coder."""

    CODER_ID = 4
    NAME = "range_ans"

    M_BITS = 12
    M = 1 << M_BITS
    L = 1 << 16

    def encode(self, symbols: np.ndarray, **kwargs) -> tuple[bytes, dict]:
        """Encode 1D integer symbols using rANS."""
        n = len(symbols)
        if n == 0:
            return b"", {"coder_id": self.CODER_ID}

        # Map to zigzag
        s = symbols.astype(np.int64)
        u = np.where(s >= 0, s << 1, (-s << 1) - 1).tolist()

        counts = Counter(u)
        sym_list = sorted(counts.keys())
        num_symbols = len(sym_list)

        # Single unique symbol optimization
        if num_symbols == 1:
            writer = BitWriter()
            writer.write_bits(1, 16)
            writer.write_bits(sym_list[0], 32)
            return writer.to_bytes(), {"coder_id": self.CODER_ID}

        # Normalize frequencies so their sum == M
        freq: dict[int, int] = {}
        cum: dict[int, int] = {}
        running = 0
        remaining = self.M

        for i, sym in enumerate(sym_list):
            if i == num_symbols - 1:
                f = remaining
            else:
                f = max(1, int(round(counts[sym] * self.M / n)))
                f = min(f, remaining - (num_symbols - 1 - i))
            freq[sym] = f
            cum[sym] = running
            running += f
            remaining -= f

        # rANS encode in reverse order
        out_words: list[int] = []
        x = self.L
        shift_bits = 16 - self.M_BITS

        for sym in reversed(u):
            f = freq[sym]
            c = cum[sym]
            max_x = ((self.L >> self.M_BITS) << 16) * f
            while x >= max_x:
                out_words.append(x & 0xFFFF)
                x >>= 16
            x = (x // f) * self.M + c + (x % f)

        out_words.append(x & 0xFFFF)
        out_words.append((x >> 16) & 0xFFFF)

        # Serialize header: num_symbols (uint16) + entries (symbol: uint32, freq: uint16) + words (uint16...)
        writer = BitWriter()
        writer.write_bits(num_symbols, 16)
        for sym in sym_list:
            writer.write_bits(sym, 32)
            writer.write_bits(freq[sym], 16)

        # Append words count (uint32)
        writer.write_bits(len(out_words), 32)
        for w in out_words:
            writer.write_bits(w, 16)

        return writer.to_bytes(), {"coder_id": self.CODER_ID}

    def decode(self, payload: bytes, count: int, **kwargs) -> np.ndarray:
        """Decode payload back into `count` signed integer symbols."""
        if count == 0 or len(payload) == 0:
            return np.zeros(0, dtype=np.int64)

        reader = BitReader(payload)
        num_symbols = reader.read_bits(16)
        if num_symbols == 0:
            return np.zeros(count, dtype=np.int64)

        if num_symbols == 1:
            sym = reader.read_bits(32)
            s = (sym >> 1) if (sym & 1) == 0 else -((sym + 1) >> 1)
            return np.full(count, s, dtype=np.int64)

        # Read frequency table
        freq: dict[int, int] = {}
        for _ in range(num_symbols):
            sym = reader.read_bits(32)
            f = reader.read_bits(16)
            freq[sym] = f

        num_words = reader.read_bits(32)
        words = [reader.read_bits(16) for _ in range(num_words)]

        # Build slot lookup table for O(1) symbol decode
        lookup = [0] * self.M
        slot_f = [0] * self.M
        slot_c = [0] * self.M
        running = 0
        for sym, f in sorted(freq.items()):
            for slot in range(running, running + f):
                lookup[slot] = sym
                slot_f[slot] = f
                slot_c[slot] = running
            running += f

        x = (words.pop() << 16) | words.pop()
        decoded = np.empty(count, dtype=np.int64)

        for i in range(count):
            slot = x & (self.M - 1)
            u = lookup[slot]
            f = slot_f[slot]
            c = slot_c[slot]

            s = (u >> 1) if (u & 1) == 0 else -((u + 1) >> 1)
            decoded[i] = s

            x = f * (x >> self.M_BITS) + (slot - c)
            while x < self.L and words:
                x = (x << 16) | words.pop()

        return decoded
