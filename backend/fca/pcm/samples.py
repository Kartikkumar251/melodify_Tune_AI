"""Audio PCM sample abstraction and representations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
import numpy as np


@dataclass
class AudioData:
    """Represents multi-channel integer PCM audio data.

    Samples are stored in a 2D NumPy array of shape (num_samples, channels)
    with an appropriate integer dtype (np.int16 or np.int32).
    """

    sample_rate: int
    channels: int
    bit_depth: int
    samples: np.ndarray  # Shape: (N, C), dtype: integer

    def __post_init__(self) -> None:
        if self.channels <= 0:
            raise ValueError(f"Channel count must be positive, got {self.channels}")
        if self.sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {self.sample_rate}")
        if self.bit_depth not in (8, 16, 24, 32):
            raise ValueError(f"Unsupported bit depth: {self.bit_depth}. Must be 8, 16, 24, or 32.")

        # Ensure 2D shape (samples, channels)
        if self.samples.ndim == 1:
            self.samples = self.samples.reshape(-1, 1)

        if self.samples.shape[1] != self.channels:
            raise ValueError(
                f"Sample array channels ({self.samples.shape[1]}) != audio channels ({self.channels})"
            )

        # Standardize dtype
        if self.bit_depth in (8, 16):
            if self.samples.dtype != np.int16:
                self.samples = self.samples.astype(np.int16)
        elif self.bit_depth in (24, 32):
            if self.samples.dtype != np.int32:
                self.samples = self.samples.astype(np.int32)

    @property
    def num_samples(self) -> int:
        """Total number of sample frames (samples per channel)."""
        return int(self.samples.shape[0])

    @property
    def duration_seconds(self) -> float:
        """Audio duration in seconds."""
        return self.num_samples / float(self.sample_rate)

    def to_pcm_bytes(self) -> bytes:
        """Serialize audio samples to standard raw PCM little-endian byte stream."""
        if self.bit_depth == 8:
            # 8-bit PCM is traditionally unsigned 0..255 or signed -128..127
            # In standard RIFF WAV, 8-bit is unsigned uint8.
            u8 = (np.clip(self.samples, -128, 127) + 128).astype(np.uint8)
            return u8.tobytes()
        elif self.bit_depth == 16:
            return self.samples.astype("<i2").tobytes()
        elif self.bit_depth == 24:
            # 24-bit 3 bytes per sample little-endian
            i32 = np.clip(self.samples, -8388608, 8388607).astype("<i4")
            raw_4 = i32.tobytes()
            # Pack 3 bytes out of every 4
            arr = np.frombuffer(raw_4, dtype=np.uint8).reshape(-1, 4)[:, :3]
            return arr.tobytes()
        elif self.bit_depth == 32:
            return self.samples.astype("<i4").tobytes()
        else:
            raise ValueError(f"Unsupported bit depth: {self.bit_depth}")

    @classmethod
    def from_pcm_bytes(
        cls,
        pcm_bytes: bytes,
        sample_rate: int,
        channels: int,
        bit_depth: int,
    ) -> AudioData:
        """Deserialize audio from standard raw PCM little-endian byte stream."""
        bytes_per_sample = bit_depth // 8
        frame_size = bytes_per_sample * channels
        if len(pcm_bytes) % frame_size != 0:
            raise ValueError(
                f"PCM byte length ({len(pcm_bytes)}) is not a multiple of frame size ({frame_size})"
            )
        num_frames = len(pcm_bytes) // frame_size

        if bit_depth == 8:
            u8 = np.frombuffer(pcm_bytes, dtype=np.uint8).reshape(num_frames, channels)
            samples = u8.astype(np.int16) - 128
        elif bit_depth == 16:
            samples = np.frombuffer(pcm_bytes, dtype="<i2").reshape(num_frames, channels)
        elif bit_depth == 24:
            u8 = np.frombuffer(pcm_bytes, dtype=np.uint8).reshape(-1, 3)
            # Pad 4th byte with sign extension
            u8_padded = np.zeros((u8.shape[0], 4), dtype=np.uint8)
            u8_padded[:, :3] = u8
            # Sign extension for negative numbers
            sign_mask = (u8[:, 2] & 0x80) != 0
            u8_padded[sign_mask, 3] = 0xFF
            samples = np.frombuffer(u8_padded.tobytes(), dtype="<i4").reshape(num_frames, channels)
        elif bit_depth == 32:
            samples = np.frombuffer(pcm_bytes, dtype="<i4").reshape(num_frames, channels)
        else:
            raise ValueError(f"Unsupported bit depth: {bit_depth}")

        return cls(
            sample_rate=sample_rate,
            channels=channels,
            bit_depth=bit_depth,
            samples=samples,
        )

    def compute_sha256(self) -> str:
        """Compute SHA-256 cryptographic checksum of normalized PCM bytes."""
        hasher = hashlib.sha256()
        hasher.update(self.to_pcm_bytes())
        return hasher.hexdigest()

    def equals(self, other: AudioData) -> bool:
        """Check exact sample-level equality with another AudioData instance."""
        if (
            self.sample_rate != other.sample_rate
            or self.channels != other.channels
            or self.bit_depth != other.bit_depth
            or self.num_samples != other.num_samples
        ):
            return False
        return bool(np.array_equal(self.samples, other.samples))
