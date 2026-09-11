"""Fixed linear predictors for audio signal decorrelation (orders 0 through 4).

Uses exact integer successive differences and cumulative sums,
achieving C-speed execution via NumPy while ensuring bit-exact reversibility.
"""

from __future__ import annotations

import numpy as np


class FixedPredictor:
    """Predictor supporting orders 0, 1, 2, 3, and 4."""

    MAX_ORDER = 4

    @staticmethod
    def compute_residuals(samples: np.ndarray, order: int) -> np.ndarray:
        """Compute prediction residuals for a 1D array of integer samples.

        Parameters
        ----------
        samples : np.ndarray
            1D integer sample array.
        order : int
            Prediction order (0, 1, 2, 3, or 4).

        Returns
        -------
        np.ndarray
            1D int64 residual array of same length.
        """
        if order < 0 or order > FixedPredictor.MAX_ORDER:
            raise ValueError(f"Invalid prediction order {order}. Must be 0..{FixedPredictor.MAX_ORDER}")

        if len(samples) <= order or order == 0:
            return samples.astype(np.int64).copy()

        res = samples.astype(np.int64).copy()
        for _ in range(order):
            # Difference from index 1 onward
            res[1:] = res[1:] - res[:-1]

        return res

    @staticmethod
    def restore_samples(residuals: np.ndarray, order: int) -> np.ndarray:
        """Restore original integer samples from residuals.

        Parameters
        ----------
        residuals : np.ndarray
            1D integer residual array.
        order : int
            Prediction order used during encoding.

        Returns
        -------
        np.ndarray
            1D int64 restored samples.
        """
        if order < 0 or order > FixedPredictor.MAX_ORDER:
            raise ValueError(f"Invalid prediction order {order}. Must be 0..{FixedPredictor.MAX_ORDER}")

        if len(residuals) <= order or order == 0:
            return residuals.astype(np.int64).copy()

        x = residuals.astype(np.int64).copy()
        for _ in range(order):
            x = np.cumsum(x)

        return x

    @classmethod
    def find_best_order(cls, samples: np.ndarray) -> tuple[int, np.ndarray]:
        """Evaluate orders 0..4 and select the order that produces the minimum residual magnitude."""
        if len(samples) <= 4:
            return 0, samples.astype(np.int64).copy()

        best_order = 0
        best_residuals = samples.astype(np.int64).copy()
        best_metric = np.sum(np.abs(best_residuals))

        for order in range(1, cls.MAX_ORDER + 1):
            cand_res = cls.compute_residuals(samples, order)
            metric = np.sum(np.abs(cand_res))
            if metric < best_metric:
                best_metric = metric
                best_order = order
                best_residuals = cand_res

        return best_order, best_residuals
