"""WAV file reader supporting 8, 16, 24, and 32-bit PCM."""

from __future__ import annotations

import os
import wave
from fca.pcm.samples import AudioData


class WavReadError(Exception):
    """Raised when WAV reading fails or format is unsupported."""


def read_wav(file_path: str | os.PathLike) -> AudioData:
    """Read a WAV audio file and return an AudioData object.

    Supports 8-bit, 16-bit, 24-bit, and 32-bit integer PCM WAV files.
    Rejects unsupported formats with WavReadError.
    """
    file_path = str(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        with wave.open(file_path, "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sampwidth = wf.getsampwidth()
            num_frames = wf.getnframes()
            comptype = wf.getcomptype()

            if comptype != "NONE":
                raise WavReadError(f"Compressed WAV formats ({comptype}) are not supported.")

            bit_depth = sampwidth * 8
            if bit_depth not in (8, 16, 24, 32):
                raise WavReadError(f"Unsupported sample bit depth: {bit_depth} bits.")

            raw_bytes = wf.readframes(num_frames)
            return AudioData.from_pcm_bytes(
                pcm_bytes=raw_bytes,
                sample_rate=sample_rate,
                channels=channels,
                bit_depth=bit_depth,
            )
    except Exception as e:
        if isinstance(e, WavReadError):
            raise
        raise WavReadError(f"Failed to read WAV file '{file_path}': {e}") from e
