"""Perceptual quality configuration and mapping."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QualityProfile:
    """Settings corresponding to a perceptual quality level (1 to 10)."""

    quality_level: int
    step_size: float
    high_freq_cutoff_ratio: float
    n_fft: int = 512
    hop_length: int = 256


def get_quality_profile(quality: int) -> QualityProfile:
    """Retrieve quality parameters for level 1 (lowest bitrate) to 10 (highest fidelity)."""
    q = max(1, min(10, quality))
    # Step size: level 10 -> 0.05, level 1 -> 5.0
    step_size = round(0.05 * (2.0 ** ((10 - q) * 0.7)), 3)
    # High frequency preservation: level 10 -> 1.0 (all bins), level 1 -> 0.4 (40% spectrum)
    cutoff = round(0.4 + (q - 1) * (0.6 / 9.0), 3)

    return QualityProfile(
        quality_level=q,
        step_size=step_size,
        high_freq_cutoff_ratio=cutoff,
    )
