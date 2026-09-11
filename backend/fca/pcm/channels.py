"""PCM channel management utilities."""

from __future__ import annotations

import numpy as np


def split_channels(samples: np.ndarray) -> list[np.ndarray]:
    """Split 2D array of shape (N, C) into a list of 1D arrays, one per channel."""
    if samples.ndim == 1:
        return [samples.copy()]
    return [samples[:, c].copy() for c in range(samples.shape[1])]


def interleave_channels(channel_list: list[np.ndarray]) -> np.ndarray:
    """Interleave list of 1D arrays into 2D array of shape (N, C)."""
    if not channel_list:
        raise ValueError("Cannot interleave empty channel list")
    num_samples = len(channel_list[0])
    for c, ch in enumerate(channel_list):
        if len(ch) != num_samples:
            raise ValueError(f"Channel {c} length {len(ch)} mismatch with channel 0 length {num_samples}")
    return np.column_stack(channel_list)
