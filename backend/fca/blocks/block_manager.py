"""Block splitting and reassembly for deterministic block-based audio processing."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from fca.pcm.samples import AudioData


@dataclass
class BlockData:
    """Represents a discrete temporal slice (block) of audio."""

    block_index: int
    sample_start: int
    sample_count: int
    samples: np.ndarray  # Shape: (sample_count, channels)


class BlockManager:
    """Manages dividing audio into deterministic blocks and reassembling them."""

    DEFAULT_BLOCK_SIZE = 4096

    @staticmethod
    def split(audio: AudioData, block_size: int = DEFAULT_BLOCK_SIZE) -> list[BlockData]:
        """Split AudioData into deterministic blocks.

        Handles exact block divisions as well as partial final blocks cleanly.
        """
        if block_size <= 0:
            raise ValueError(f"Block size must be positive, got {block_size}")

        total_samples = audio.num_samples
        blocks: list[BlockData] = []
        block_idx = 0
        offset = 0

        while offset < total_samples:
            count = min(block_size, total_samples - offset)
            slice_samples = audio.samples[offset : offset + count].copy()
            blocks.append(
                BlockData(
                    block_index=block_idx,
                    sample_start=offset,
                    sample_count=count,
                    samples=slice_samples,
                )
            )
            offset += count
            block_idx += 1

        return blocks

    @staticmethod
    def assemble(
        blocks: list[BlockData],
        sample_rate: int,
        channels: int,
        bit_depth: int,
    ) -> AudioData:
        """Reassemble a sequence of blocks into an AudioData object."""
        if not blocks:
            # Empty audio
            dtype = np.int16 if bit_depth in (8, 16) else np.int32
            return AudioData(
                sample_rate=sample_rate,
                channels=channels,
                bit_depth=bit_depth,
                samples=np.zeros((0, channels), dtype=dtype),
            )

        # Sort blocks by block_index to guarantee order
        sorted_blocks = sorted(blocks, key=lambda b: b.block_index)
        concatenated = np.vstack([b.samples for b in sorted_blocks])

        return AudioData(
            sample_rate=sample_rate,
            channels=channels,
            bit_depth=bit_depth,
            samples=concatenated,
        )
