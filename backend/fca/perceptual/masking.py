"""Psychoacoustic masking model and absolute threshold of hearing (ATH)."""

from __future__ import annotations

import numpy as np


def absolute_threshold_of_hearing(freq_hz: np.ndarray) -> np.ndarray:
    """Terhardt's formula for the Absolute Threshold of Hearing (ATH) in dB SPL.

    ATH(f) = 3.64*(f/1000)^-0.8 - 6.5*exp(-0.6*(f/1000 - 3.3)^2) + 10^-3*(f/1000)^4
    """
    f_khz = np.maximum(freq_hz / 1000.0, 0.02)  # avoid division by zero below 20 Hz
    ath = (
        3.64 * (f_khz ** -0.8)
        - 6.5 * np.exp(-0.6 * ((f_khz - 3.3) ** 2))
        + 0.001 * (f_khz ** 4)
    )
    return ath


def apply_masking_threshold(magnitude_spec: np.ndarray, sample_rate: int) -> np.ndarray:
    """Zero out spectral bins below psychoacoustic audibility threshold."""
    num_bins, num_frames = magnitude_spec.shape
    freqs = np.linspace(0, sample_rate / 2.0, num_bins)
    ath_db = absolute_threshold_of_hearing(freqs)

    # Convert ATH dB to linear scale relative to peak
    ath_linear = 10.0 ** ((ath_db - 96.0) / 20.0)
    ath_linear = ath_linear[:, np.newaxis]

    masked = np.where(magnitude_spec < ath_linear, 0.0, magnitude_spec)
    return masked
