"""Abstract base class and registry for entropy coders."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class EntropyCoder(ABC):
    """Abstract interface for entropy coding engines."""

    CODER_ID: int = 0
    NAME: str = "base"

    @abstractmethod
    def encode(self, symbols: np.ndarray, **kwargs) -> tuple[bytes, dict]:
        """Encode a 1D array of integer symbols into bytes.

        Returns
        -------
        tuple[bytes, dict]
            The encoded bytes and a dictionary of parameters needed for decoding.
        """

    @abstractmethod
    def decode(self, payload: bytes, count: int, **kwargs) -> np.ndarray:
        """Decode payload bytes back into a 1D array of `count` integer symbols."""
