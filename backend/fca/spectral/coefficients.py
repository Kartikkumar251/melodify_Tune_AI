"""Spectral coefficient storage and utility structures."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SpectralRepresentation:
    """Encapsulates magnitude and phase time-frequency data for a block."""

    magnitude: np.ndarray  # Shape: (freq_bins, frames)
    phase: np.ndarray      # Shape: (freq_bins, frames)
    n_fft: int
    hop_length: int
    window: str

    @property
    def shape(self) -> tuple[int, int]:
        return self.magnitude.shape
