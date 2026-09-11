"""FCA block record definition, serialization, and CRC validation."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from fca.verify.crc import calculate_crc32


BLOCK_SYNC_MARKER = b"\xFB\xCA"
# Sync(2s), BlockIndex(I), SampleCount(I), TransformID(B), ChannelMode(B),
# EntropyCoderID(B), PredictionOrder(B), AuxParam(i), PayloadLen(I)
BLOCK_PRE_CRC_STRUCT = struct.Struct("<2sIIBBBBii")
BLOCK_HEADER_SIZE = BLOCK_PRE_CRC_STRUCT.size + 4  # +4 for uint32 CRC


class BlockCorruptError(Exception):
    """Raised when block integrity verification or decoding fails."""


@dataclass
class EncodedBlockRecord:
    """Represents a serialized and self-describing FCA audio block."""

    block_index: int
    sample_count: int
    transform_id: int
    channel_mode: int
    entropy_coder_id: int
    prediction_order: int
    aux_param: int
    payload: bytes

    def pack(self) -> bytes:
        """Serialize block header and payload with CRC32."""
        payload_len = len(self.payload)
        pre_crc = BLOCK_PRE_CRC_STRUCT.pack(
            BLOCK_SYNC_MARKER,
            self.block_index,
            self.sample_count,
            self.transform_id,
            self.channel_mode,
            self.entropy_coder_id,
            self.prediction_order,
            self.aux_param,
            payload_len,
        )
        crc = calculate_crc32(pre_crc + self.payload)
        return pre_crc + struct.pack("<I", crc) + self.payload

    @classmethod
    def unpack(cls, data: bytes, offset: int = 0) -> tuple[EncodedBlockRecord, int]:
        """Deserialize block record from byte buffer.

        Returns
        -------
        tuple[EncodedBlockRecord, int]
            The parsed block record and new byte offset in data.
        """
        remaining = len(data) - offset
        if remaining < BLOCK_HEADER_SIZE:
            raise BlockCorruptError(
                f"Unexpected EOF reading block header (needed {BLOCK_HEADER_SIZE} bytes, got {remaining})"
            )

        pre_crc_bytes = data[offset : offset + BLOCK_PRE_CRC_STRUCT.size]
        stored_crc = struct.unpack(
            "<I", data[offset + BLOCK_PRE_CRC_STRUCT.size : offset + BLOCK_HEADER_SIZE]
        )[0]

        (
            sync,
            block_idx,
            sample_count,
            transform_id,
            channel_mode,
            entropy_coder_id,
            prediction_order,
            aux_param,
            payload_len,
        ) = BLOCK_PRE_CRC_STRUCT.unpack(pre_crc_bytes)

        if sync != BLOCK_SYNC_MARKER:
            raise BlockCorruptError(
                f"Invalid block sync marker at offset {offset}: {sync!r} != {BLOCK_SYNC_MARKER!r}"
            )

        # Protect against excessive allocation from corrupt length
        if payload_len > 64 * 1024 * 1024:  # 64 MB max per block
            raise BlockCorruptError(f"Block payload length suspiciously large: {payload_len} bytes")

        payload_start = offset + BLOCK_HEADER_SIZE
        payload_end = payload_start + payload_len

        if len(data) < payload_end:
            raise BlockCorruptError(
                f"Truncated block payload at index {block_idx} (expected {payload_len} bytes, got {len(data) - payload_start})"
            )

        payload = data[payload_start:payload_end]
        calc_crc = calculate_crc32(pre_crc_bytes + payload)

        if calc_crc != stored_crc:
            raise BlockCorruptError(
                f"Block {block_idx} CRC32 mismatch: stored 0x{stored_crc:08X} != calculated 0x{calc_crc:08X}"
            )

        record = cls(
            block_index=block_idx,
            sample_count=sample_count,
            transform_id=transform_id,
            channel_mode=channel_mode,
            entropy_coder_id=entropy_coder_id,
            prediction_order=prediction_order,
            aux_param=aux_param,
            payload=payload,
        )
        return record, payload_end
