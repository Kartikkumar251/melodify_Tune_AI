"""Cryptographic SHA-256 hash calculation for bit-exact verification."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fca.pcm.samples import AudioData


def calculate_sha256(data: bytes) -> str:
    """Compute hex-encoded SHA-256 digest of data bytes."""
    return hashlib.sha256(data).hexdigest()


def calculate_pcm_sha256(audio: AudioData) -> str:
    """Compute SHA-256 hash of normalized PCM samples from an AudioData object."""
    return audio.compute_sha256()


def verify_pcm_equality(audio1: AudioData, audio2: AudioData) -> tuple[bool, str]:
    """Check exact equality and hash match between two AudioData instances.

    Returns (is_equal, status_message).
    """
    if audio1.sample_rate != audio2.sample_rate:
        return False, f"Sample rate mismatch: {audio1.sample_rate} vs {audio2.sample_rate}"
    if audio1.channels != audio2.channels:
        return False, f"Channel mismatch: {audio1.channels} vs {audio2.channels}"
    if audio1.bit_depth != audio2.bit_depth:
        return False, f"Bit depth mismatch: {audio1.bit_depth} vs {audio2.bit_depth}"
    if audio1.num_samples != audio2.num_samples:
        return False, f"Sample count mismatch: {audio1.num_samples} vs {audio2.num_samples}"

    h1 = calculate_pcm_sha256(audio1)
    h2 = calculate_pcm_sha256(audio2)

    if h1 != h2:
        return False, f"SHA-256 mismatch: {h1} vs {h2}"

    if not audio1.equals(audio2):
        return False, "Sample array comparison failed despite identical hash"

    return True, f"Exact match verified! SHA-256: {h1}"
