"""Prediction and decorrelation package."""

from fca.prediction.fixed import FixedPredictor
from fca.prediction.lpc import LPCPredictor

__all__ = ["FixedPredictor", "LPCPredictor"]
