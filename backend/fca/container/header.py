"""FCA container header definition and serialization."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from fca.verify.crc import calculate_crc32


MAGIC_V1 = b"FCA\x01"
# Header format without CRC: Magic(4s), HeaderSize(H), Major(B), Minor(B), Mode(B), Flags(B),
# Channels(B), BitDepth(B), SampleRate(I), TotalSamples(Q), BlockCount(I), MetadataLen(H)
HEADER_PRE_CRC_STRUCT = struct.Struct("<4sHBBBBBBQIHI")
HEADER_SIZE = HEADER_PRE_CRC_STRUCT.size + 4  # +4 for uint32 CRC


class ContainerHeaderError(Exception):
    """Raised when header validation fails."""


@dataclass
class FCAHeader:
    """FCA file container header."""

    sample_rate: int
    channels: int
    bit_depth: int
    total_samples: int
    block_count: int
    mode: int = 0  # 0: FCA-L (Lossless), 1: FCA-P (Perceptual)
    version_major: int = 1
    version_minor: int = 0
    flags: int = 0
    metadata: dict = field(default_factory=dict)

    def pack(self) -> bytes:
        """Serialize header and metadata to bytes with CRC32 protection."""
        metadata_bytes = json.dumps(self.metadata).encode("utf-8") if self.metadata else b""
        meta_len = len(metadata_bytes)
        if meta_len > 65535:
            raise ValueError(f"Metadata JSON too large ({meta_len} bytes, max 65535)")

        pre_crc = HEADER_PRE_CRC_STRUCT.pack(
            MAGIC_V1,
            HEADER_SIZE,
            self.version_major,
            self.version_minor,
            self.mode,
            self.flags,
            self.channels,
            self.bit_depth,
            self.sample_rate,
            self.total_samples,
            self.block_count,
            meta_len,
        )
        crc = calculate_crc32(pre_crc)
        header_bytes = pre_crc + struct.pack("<I", crc)

        if metadata_bytes:
            header_bytes += metadata_bytes

        return header_bytes

    @classmethod
    def unpack(cls, data: bytes) -> tuple[FCAHeader, int]:
        """Deserialize and validate header from bytes.

        Returns
        -------
        tuple[FCAHeader, int]
            The parsed header and total bytes consumed (header + metadata).
        """
        if len(data) < HEADER_SIZE:
            raise ContainerHeaderError(
                f"Data too short for FCA header (got {len(data)} bytes, expected at least {HEADER_SIZE})"
            )

        pre_crc_bytes = data[: HEADER_PRE_CRC_STRUCT.size]
        stored_crc = struct.unpack("<I", data[HEADER_PRE_CRC_STRUCT.size : HEADER_SIZE])[0]

        calc_crc = calculate_crc32(pre_crc_bytes)
        if calc_crc != stored_crc:
            raise ContainerHeaderError(
                f"FCA header checksum failure: stored 0x{stored_crc:08X} != calculated 0x{calc_crc:08X}"
            )

        (
            magic,
            hdr_size,
            v_major,
            v_minor,
            mode,
            flags,
            channels,
            bit_depth,
            sample_rate,
            total_samples,
            block_count,
            meta_len,
        ) = HEADER_PRE_CRC_STRUCT.unpack(pre_crc_bytes)

        if magic != MAGIC_V1:
            raise ContainerHeaderError(f"Invalid magic identifier {magic!r}. Expected {MAGIC_V1!r}")

        if v_major != 1:
            raise ContainerHeaderError(f"Unsupported FCA major version {v_major}. Only v1 is supported.")

        if channels <= 0 or channels > 64:
            raise ContainerHeaderError(f"Invalid channel count {channels}")

        if bit_depth not in (8, 16, 24, 32):
            raise ContainerHeaderError(f"Invalid bit depth {bit_depth}")

        if sample_rate <= 0 or sample_rate > 384000:
            raise ContainerHeaderError(f"Invalid sample rate {sample_rate}")

        offset = HEADER_SIZE
        metadata = {}
        if meta_len > 0:
            if len(data) < offset + meta_len:
                raise ContainerHeaderError("Truncated metadata in FCA header")
            meta_bytes = data[offset : offset + meta_len]
            try:
                metadata = json.loads(meta_bytes.decode("utf-8"))
            except Exception as e:
                raise ContainerHeaderError(f"Failed to parse metadata JSON: {e}") from e
            offset += meta_len

        header = cls(
            sample_rate=sample_rate,
            channels=channels,
            bit_depth=bit_depth,
            total_samples=total_samples,
            block_count=block_count,
            mode=mode,
            version_major=v_major,
            version_minor=v_minor,
            flags=flags,
            metadata=metadata,
        )
        return header, offset
