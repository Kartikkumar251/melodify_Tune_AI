"""CRC32 checksum calculations for block integrity verification."""

from __future__ import annotations

import zlib


def calculate_crc32(data: bytes) -> int:
    """Calculate 32-bit unsigned CRC32 checksum of bytes."""
    return zlib.crc32(data) & 0xFFFFFFFF


def verify_crc32(data: bytes, expected_crc: int) -> bool:
    """Verify that calculated CRC32 equals expected CRC32."""
    return calculate_crc32(data) == (expected_crc & 0xFFFFFFFF)
