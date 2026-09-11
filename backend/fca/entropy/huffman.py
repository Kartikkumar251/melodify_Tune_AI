"""Canonical Huffman entropy coder with self-describing codebook."""

from __future__ import annotations

import heapq
from collections import Counter
import numpy as np
from fca.entropy.base import EntropyCoder
from fca.entropy.bitstream import BitWriter, BitReader


class HuffmanNode:
    __slots__ = ("symbol", "freq", "left", "right")

    def __init__(
        self,
        symbol: int | None = None,
        freq: int = 0,
        left: HuffmanNode | None = None,
        right: HuffmanNode | None = None,
    ) -> None:
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other: HuffmanNode) -> bool:
        return self.freq < other.freq


class HuffmanCoder(EntropyCoder):
    """Canonical Huffman entropy coder."""

    CODER_ID = 2
    NAME = "huffman"

    def encode(self, symbols: np.ndarray, **kwargs) -> tuple[bytes, dict]:
        """Encode 1D symbols using Huffman coding."""
        if len(symbols) == 0:
            return b"", {"coder_id": self.CODER_ID}

        # Map to zigzag
        s = symbols.astype(np.int64)
        u = np.where(s >= 0, s << 1, (-s << 1) - 1).tolist()

        counts = Counter(u)
        unique_symbols = list(counts.keys())

        # If only 1 unique symbol
        if len(unique_symbols) == 1:
            sym = unique_symbols[0]
            writer = BitWriter()
            writer.write_bits(1, 16)  # Table size = 1
            writer.write_bits(sym, 32)
            writer.write_bits(0, 8)  # Length = 0
            # All symbols are identical, no payload bits needed
            return writer.to_bytes(), {"coder_id": self.CODER_ID}

        # Build Huffman tree
        heap: list[HuffmanNode] = [HuffmanNode(sym, freq) for sym, freq in counts.items()]
        heapq.heapify(heap)

        while len(heap) > 1:
            n1 = heapq.heappop(heap)
            n2 = heapq.heappop(heap)
            merged = HuffmanNode(symbol=None, freq=n1.freq + n2.freq, left=n1, right=n2)
            heapq.heappush(heap, merged)

        root = heap[0]

        # Generate prefix codes
        code_table: dict[int, tuple[int, int]] = {}  # symbol -> (code_val, bit_len)

        def traverse(node: HuffmanNode | None, curr_code: int, bit_len: int) -> None:
            if node is None:
                return
            if node.symbol is not None:
                code_table[node.symbol] = (curr_code, max(1, bit_len))
                return
            traverse(node.left, (curr_code << 1), bit_len + 1)
            traverse(node.right, (curr_code << 1) | 1, bit_len + 1)

        traverse(root, 0, 0)

        # Serialize header: table size (uint16) + entries (symbol: uint32, length: uint8, code: uint32)
        writer = BitWriter()
        writer.write_bits(len(code_table), 16)
        for sym, (code, length) in code_table.items():
            writer.write_bits(sym, 32)
            writer.write_bits(length, 8)
            writer.write_bits(code, 32)

        # Write symbol bitstream
        for sym in u:
            code, length = code_table[sym]
            writer.write_bits(code, length)

        return writer.to_bytes(), {"coder_id": self.CODER_ID}

    def decode(self, payload: bytes, count: int, **kwargs) -> np.ndarray:
        """Decode payload back into `count` symbols."""
        if count == 0 or len(payload) == 0:
            return np.zeros(0, dtype=np.int64)

        reader = BitReader(payload)
        table_size = reader.read_bits(16)
        if table_size == 0:
            return np.zeros(count, dtype=np.int64)

        if table_size == 1:
            sym = reader.read_bits(32)
            _ = reader.read_bits(8)
            s = (sym >> 1) if (sym & 1) == 0 else -((sym + 1) >> 1)
            return np.full(count, s, dtype=np.int64)

        # Read code table: (code, length) -> symbol
        lookup: dict[tuple[int, int], int] = {}
        max_len = 0
        for _ in range(table_size):
            sym = reader.read_bits(32)
            length = reader.read_bits(8)
            code = reader.read_bits(32)
            lookup[(code, length)] = sym
            if length > max_len:
                max_len = length

        decoded = np.empty(count, dtype=np.int64)
        for i in range(count):
            curr_code = 0
            bit_len = 0
            found = False
            while bit_len <= max_len:
                b = reader.read_bit()
                curr_code = (curr_code << 1) | b
                bit_len += 1
                if (curr_code, bit_len) in lookup:
                    u = lookup[(curr_code, bit_len)]
                    s = (u >> 1) if (u & 1) == 0 else -((u + 1) >> 1)
                    decoded[i] = s
                    found = True
                    break
            if not found:
                decoded[i] = 0

        return decoded
