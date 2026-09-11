"""Spectral magnitude representation and manipulation."""

from __future__ import annotations

import numpy as np


def extract_magnitude(complex_spec: np.ndarray) -> np.ndarray:
    """Extract magnitude from complex STFT spectrogram."""
    return np.abs(complex_spec)


def quantize_magnitude(magnitude: np.ndarray, step_size: float) -> np.ndarray:
    """Uniform scalar quantization of spectral magnitude."""
    if step_size <= 0:
        return magnitude
    return np.round(magnitude / step_size) * step_size
