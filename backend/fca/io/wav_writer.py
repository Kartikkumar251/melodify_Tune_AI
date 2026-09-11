"""WAV file writer supporting 8, 16, 24, and 32-bit PCM."""

from __future__ import annotations

import os
import wave
from fca.pcm.samples import AudioData


class WavWriteError(Exception):
    """Raised when WAV writing fails."""


def write_wav(audio: AudioData, file_path: str | os.PathLike) -> None:
    """Write an AudioData object to a standard PCM WAV file.

    Guarantees bit-exact integer preservation for 8, 16, 24, and 32-bit PCM.
    """
    file_path = str(file_path)
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    try:
        raw_bytes = audio.to_pcm_bytes()
        with wave.open(file_path, "wb") as wf:
            wf.setnchannels(audio.channels)
            wf.setsampwidth(audio.bit_depth // 8)
            wf.setframerate(audio.sample_rate)
            wf.writeframes(raw_bytes)
    except Exception as e:
        raise WavWriteError(f"Failed to write WAV file '{file_path}': {e}") from e
