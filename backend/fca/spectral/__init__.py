"""Spectral processing package."""

from fca.spectral.magnitude import extract_magnitude, quantize_magnitude
from fca.spectral.phase import extract_phase, synthesize_complex
from fca.spectral.coefficients import SpectralRepresentation

__all__ = [
    "extract_magnitude",
    "quantize_magnitude",
    "extract_phase",
    "synthesize_complex",
    "SpectralRepresentation",
]
