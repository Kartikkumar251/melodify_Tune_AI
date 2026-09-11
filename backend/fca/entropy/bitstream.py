"""Bit-level stream reader and writer for entropy coding."""

from __future__ import annotations


def zigzag_encode(s: int) -> int:
    """Map signed integer to non-negative integer: 0->0, -1->1, 1->2, -2->3, ..."""
    return (s << 1) if s >= 0 else (-s << 1) - 1


def zigzag_decode(u: int) -> int:
    """Map non-negative integer back to signed integer."""
    return (u >> 1) if (u & 1) == 0 else -((u + 1) >> 1)


class BitWriter:
    """High-performance bit-level writer."""

    __slots__ = ("_buffer", "_bit_count", "_byte_array")

    def __init__(self) -> None:
        self._buffer: int = 0
        self._bit_count: int = 0
        self._byte_array = bytearray()

    def write_bit(self, bit: int) -> None:
        """Write a single bit (0 or 1)."""
        self._buffer = (self._buffer << 1) | (bit & 1)
        self._bit_count += 1
        if self._bit_count == 8:
            self._byte_array.append(self._buffer)
            self._buffer = 0
            self._bit_count = 0

    def write_bits(self, value: int, num_bits: int) -> None:
        """Write `num_bits` of `value` (most significant bit first)."""
        if num_bits <= 0:
            return
        # Mask value to num_bits
        value &= (1 << num_bits) - 1
        self._buffer = (self._buffer << num_bits) | value
        self._bit_count += num_bits
        while self._bit_count >= 8:
            self._bit_count -= 8
            byte_val = (self._buffer >> self._bit_count) & 0xFF
            self._byte_array.append(byte_val)
        self._buffer &= (1 << self._bit_count) - 1

    def write_unary_zeros(self, count: int) -> None:
        """Write `count` zeros followed by a terminating 1."""
        # Fast write chunks of zeros
        zeros_left = count
        while zeros_left >= 8:
            self.write_bits(0, 8)
            zeros_left -= 8
        if zeros_left > 0:
            self.write_bits(0, zeros_left)
        self.write_bit(1)

    def to_bytes(self) -> bytes:
        """Flush remaining bits padded with zeros to byte boundary and return bytes."""
        arr = bytearray(self._byte_array)
        if self._bit_count > 0:
            padded = (self._buffer << (8 - self._bit_count)) & 0xFF
            arr.append(padded)
        return bytes(arr)


class BitReader:
    """High-performance bit-level reader."""

    __slots__ = ("_data", "_byte_pos", "_bit_pos", "_data_len")

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._byte_pos: int = 0
        self._bit_pos: int = 0  # 0 to 7, where 0 is MSB of current byte
        self._data_len: int = len(data)

    def read_bit(self) -> int:
        """Read a single bit (0 or 1). Returns 0 on EOF."""
        if self._byte_pos >= self._data_len:
            return 0
        byte_val = self._data[self._byte_pos]
        bit = (byte_val >> (7 - self._bit_pos)) & 1
        self._bit_pos += 1
        if self._bit_pos == 8:
            self._bit_pos = 0
            self._byte_pos += 1
        return bit

    def read_bits(self, num_bits: int) -> int:
        """Read `num_bits` (most significant bit first)."""
        if num_bits <= 0:
            return 0
        val = 0
        for _ in range(num_bits):
            val = (val << 1) | self.read_bit()
        return val

    def read_unary_zeros(self) -> int:
        """Read zeros until a 1 bit is encountered, returning the count of zeros."""
        count = 0
        while self.read_bit() == 0:
            count += 1
            if self._byte_pos >= self._data_len and count > 1000000:
                break
        return count
