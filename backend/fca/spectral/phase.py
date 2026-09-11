"""Spectral phase extraction and complex signal reconstruction."""

from __future__ import annotations

import numpy as np


def extract_phase(complex_spec: np.ndarray) -> np.ndarray:
    """Extract phase angle in radians (-pi to +pi) from complex spectrogram."""
    return np.angle(complex_spec)


def synthesize_complex(magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Reconstruct complex STFT spectrogram from magnitude and phase arrays."""
    return magnitude * np.exp(1j * phase)
