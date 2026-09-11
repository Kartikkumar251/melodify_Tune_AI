"""Short-Time Fourier Transform (STFT) research and perceptual analysis layer."""

from __future__ import annotations

import numpy as np
from scipy import signal


class STFTProcessor:
    """Configurable STFT analysis and synthesis processor."""

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 256,
        window: str = "hann",
    ) -> None:
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window

    def forward(self, samples: np.ndarray) -> np.ndarray:
        """Compute one-sided STFT on 1D sample array.

        Returns
        -------
        np.ndarray
            Complex spectrogram of shape (num_freq_bins, num_frames).
        """
        _, _, Zxx = signal.stft(
            samples.astype(np.float64),
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            window=self.window,
            boundary="zeros",
            padded=True,
        )
        return Zxx

    def inverse(self, Zxx: np.ndarray, target_length: int | None = None) -> np.ndarray:
        """Compute Inverse STFT using overlap-add reconstruction.

        Returns
        -------
        np.ndarray
            Reconstructed 1D signal.
        """
        _, x_rec = signal.istft(
            Zxx,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            window=self.window,
            boundary=True,
        )
        if target_length is not None:
            if len(x_rec) >= target_length:
                x_rec = x_rec[:target_length]
            else:
                x_rec = np.pad(x_rec, (0, target_length - len(x_rec)))
        return x_rec
