"""Spectral quantization and coefficient reduction for perceptual mode."""

from __future__ import annotations

import numpy as np
from fca.perceptual.quality import QualityProfile


class PerceptualQuantizer:
    """Quantizes time-frequency representations according to quality profiles."""

    @staticmethod
    def quantize_stft(
        magnitude: np.ndarray,
        profile: QualityProfile,
    ) -> np.ndarray:
        """Apply band-limiting cutoff and dead-zone uniform quantization."""
        quantized = magnitude.copy()
        num_bins = quantized.shape[0]
        cutoff_bin = int(num_bins * profile.high_freq_cutoff_ratio)

        # Zero out frequencies above cutoff
        if cutoff_bin < num_bins:
            quantized[cutoff_bin:, :] = 0.0

        # Uniform scalar quantization
        step = profile.step_size
        if step > 0:
            quantized = np.round(quantized / step) * step

        return quantized
