"""Reversible transforms for bit-exact time-frequency signal decomposition."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class ReversibleTransform(ABC):
    """Abstract interface for bit-exact reversible transforms."""

    TRANSFORM_ID: int = 0
    NAME: str = "identity"

    @abstractmethod
    def forward(self, samples: np.ndarray) -> np.ndarray:
        """Transform 1D integer samples into 1D integer coefficients."""

    @abstractmethod
    def inverse(self, coefficients: np.ndarray) -> np.ndarray:
        """Invert 1D integer coefficients back to original 1D integer samples."""


class IdentityTransform(ReversibleTransform):
    """Direct PCM pass-through transform (ID = 0)."""

    TRANSFORM_ID = 0
    NAME = "identity"

    def forward(self, samples: np.ndarray) -> np.ndarray:
        return samples.astype(np.int64).copy()

    def inverse(self, coefficients: np.ndarray) -> np.ndarray:
        return coefficients.astype(np.int64).copy()


class IntegerLiftingDWT(ReversibleTransform):
    """Bit-exact integer-to-integer Haar wavelet transform using lifting (ID = 1).

    Decomposes signal into low-frequency approximation and high-frequency details.
    """

    TRANSFORM_ID = 1
    NAME = "integer_haar"

    def forward(self, samples: np.ndarray) -> np.ndarray:
        n = len(samples)
        if n <= 1:
            return samples.astype(np.int64).copy()

        x = samples.astype(np.int64)
        # Pad with last sample if odd length
        is_odd = (n % 2) != 0
        if is_odd:
            x = np.append(x, x[-1])

        even = x[0::2]
        odd = x[1::2]

        # Integer lifting
        d = odd - even  # Detail (high-frequency)
        s = even + (d >> 1)  # Smooth (low-frequency)

        # Concatenate [s, d]
        out = np.empty_like(x)
        half = len(s)
        out[:half] = s
        out[half:] = d
        return out if not is_odd else np.append(out, 1)  # Tag odd length with 1

    def inverse(self, coefficients: np.ndarray) -> np.ndarray:
        n = len(coefficients)
        if n <= 1:
            return coefficients.astype(np.int64).copy()

        is_odd = False
        coeffs = coefficients.astype(np.int64)
        if len(coeffs) % 2 != 0:
            is_odd = True
            coeffs = coeffs[:-1]

        half = len(coeffs) // 2
        s = coeffs[:half]
        d = coeffs[half:]

        # Invert lifting
        even = s - (d >> 1)
        odd = even + d

        reconstructed = np.empty(len(coeffs), dtype=np.int64)
        reconstructed[0::2] = even
        reconstructed[1::2] = odd

        if is_odd:
            reconstructed = reconstructed[:-1]

        return reconstructed


_TRANSFORMS: dict[int, type[ReversibleTransform]] = {
    IdentityTransform.TRANSFORM_ID: IdentityTransform,
    IntegerLiftingDWT.TRANSFORM_ID: IntegerLiftingDWT,
}


def get_transform(transform_id: int) -> ReversibleTransform:
    """Retrieve an instantiated transform by ID."""
    if transform_id not in _TRANSFORMS:
        raise ValueError(f"Unknown transform ID: {transform_id}")
    return _TRANSFORMS[transform_id]()
