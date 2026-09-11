"""Independent channel processing (no cross-channel decorrelation)."""

from __future__ import annotations

import numpy as np


class IndependentChannelCoder:
    """Pass-through independent channel coder."""

    MODE_ID = 0

    @staticmethod
    def encode(samples: np.ndarray) -> list[np.ndarray]:
        """Convert (N, C) array into list of C 1D arrays."""
        if samples.ndim == 1:
            return [samples.copy()]
        return [samples[:, c].copy() for c in range(samples.shape[1])]

    @staticmethod
    def decode(channel_data: list[np.ndarray]) -> np.ndarray:
        """Combine list of C 1D arrays into (N, C) array."""
        if len(channel_data) == 1:
            return channel_data[0].reshape(-1, 1)
        return np.column_stack(channel_data)
