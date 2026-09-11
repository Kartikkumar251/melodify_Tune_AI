"""Audio I/O package."""

from fca.io.wav_reader import read_wav, WavReadError
from fca.io.wav_writer import write_wav, WavWriteError

__all__ = ["read_wav", "write_wav", "WavReadError", "WavWriteError"]
