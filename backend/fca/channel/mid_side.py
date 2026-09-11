"""Reversible integer Mid/Side channel decorrelation."""

from __future__ import annotations

import numpy as np


class MidSideChannelCoder:
    """Bit-exact reversible integer Mid/Side stereo decorrelation.

    Uses integer lifting steps:
        difference (side): d = L - R
        mid:               m = R + (d >> 1)

    Reconstruction:
        R = m - (d >> 1)
        L = R + d
    """

    MODE_ID = 1

    @staticmethod
    def encode(samples: np.ndarray) -> list[np.ndarray]:
        """Encode 2-channel samples into Mid and Side channels."""
        if samples.shape[1] != 2:
            raise ValueError(f"Mid/Side decorrelation requires 2 channels, got {samples.shape[1]}")

        L = samples[:, 0].astype(np.int64)
        R = samples[:, 1].astype(np.int64)

        d = L - R
        m = R + (d >> 1)

        return [m, d]

    @staticmethod
    def decode(channel_data: list[np.ndarray], target_dtype: np.dtype = np.int16) -> np.ndarray:
        """Decode Mid and Side channels back to Left and Right channels."""
        if len(channel_data) != 2:
            raise ValueError(f"Mid/Side decorrelation requires 2 channels, got {len(channel_data)}")

        m = channel_data[0].astype(np.int64)
        d = channel_data[1].astype(np.int64)

        R = m - (d >> 1)
        L = R + d

        interleaved = np.column_stack([L, R])
        return interleaved.astype(target_dtype)
