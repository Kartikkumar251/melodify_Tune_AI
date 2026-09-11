"""
fca_optimizer.py — Frequency Coded Audio (FCA) Optimizer Service for Melodify
Integrates the FCA audio compression research codec (FCA-L Lossless & FCA-P Perceptual)
into Melodify for efficient audio archiving, artifact storage, and transport.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Ensure backend and root paths are available
_BACKEND_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _BACKEND_DIR.parent
for _p in [str(_BACKEND_DIR), str(_ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import soundfile as sf
import librosa

# Import FCA core package
from fca.pcm.samples import AudioData
from fca.io.wav_reader import read_wav
from fca.encoder import FCAEncoder
from fca.decoder import FCADecoder
from fca.verify.hash import verify_pcm_equality, calculate_pcm_sha256

FCA_OUTPUTS_DIR = _ROOT_DIR / "fca_outputs"
FCA_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_or_convert_to_audiodata(input_path: str | Path) -> Tuple[AudioData, Path, int]:
    """
    Loads an audio file into an FCA AudioData instance.
    If the file is a standard 16/24-bit PCM WAV, reads directly.
    Otherwise (e.g. float32 WAV, MP3, FLAC), converts to clean 16-bit PCM AudioData.
    """
    path = Path(input_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Source audio file not found: {path}")

    orig_size_bytes = os.path.getsize(path)

    # Attempt direct FCA WAV read
    try:
        audio = read_wav(str(path))
        return audio, path, orig_size_bytes
    except Exception:
        pass

    # Fallback: Load via soundfile / librosa and construct normalized AudioData
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    channels = data.shape[1]
    num_samples = data.shape[0]

    # Convert float32 to int16 PCM range
    int16_samples = np.clip(data * 32767.0, -32768.0, 32767.0).astype(np.int16)

    audio = AudioData(
        sample_rate=int(sr),
        channels=int(channels),
        bit_depth=16,
        num_samples=int(num_samples),
        samples=int16_samples,
    )
    return audio, path, orig_size_bytes


def optimize_audio_to_fca(
    audio_path: str | Path,
    mode: str = "lossless",
    quality: int = 5,
    custom_output_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compress an audio track into an FCA container file (.fca).

    Parameters
    ----------
    audio_path : str | Path
        Path to the source audio file.
    mode : str
        'lossless' (FCA-L) for bit-exact reconstruction, or 'perceptual' (FCA-P) for quality-controlled compression.
    quality : int
        Quality level 1..10 (used for perceptual mode, default 5).
    custom_output_name : Optional[str]
        Optional custom base name for the output .fca file.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing all measured compression metrics, SHA-256 hash,
        verification status, and generated artifact URLs.
    """
    t_start = time.perf_counter()
    mode_normalized = "lossless" if mode.lower() in ("lossless", "fca-l", "fcal", "0") else "perceptual"
    quality_clamped = max(1, min(10, int(quality)))

    # 1. Load source audio
    audio, source_path, orig_file_size = _load_or_convert_to_audiodata(audio_path)

    # 2. Determine output .fca filename
    ts = time.strftime("%H%M%S")
    stem_name = custom_output_name or source_path.stem
    mode_tag = "fca_l" if mode_normalized == "lossless" else f"fca_p_q{quality_clamped}"
    fca_filename = f"{stem_name}_{ts}_{mode_tag}.fca"
    fca_file_path = FCA_OUTPUTS_DIR / fca_filename

    # 3. Execute FCA Encoding
    encoder = FCAEncoder(block_size=4096)
    encode_result = encoder.encode_file(
        audio=audio,
        output_path=fca_file_path,
        mode=mode_normalized,
        quality=quality_clamped,
    )

    fca_file_size = os.path.getsize(fca_file_path)
    total_elapsed = round(time.perf_counter() - t_start, 4)

    # 4. If Lossless (FCA-L), execute bit-exact decode verification
    is_verified_lossless = False
    verification_msg = ""
    decoded_sha256 = ""

    if mode_normalized == "lossless":
        try:
            decoder = FCADecoder()
            decode_result = decoder.decode_file(fca_file_path)
            is_verified_lossless, verification_msg = verify_pcm_equality(audio, decode_result.audio)
            decoded_sha256 = decode_result.sha256_pcm
        except Exception as e:
            is_verified_lossless = False
            verification_msg = f"Verification error: {str(e)}"
    else:
        verification_msg = f"Perceptual encoding completed (Quality {quality_clamped}/10)"

    # 5. Compute real metrics
    raw_pcm_bytes = encode_result.input_bytes
    saved_bytes = max(0, orig_file_size - fca_file_size)
    reduction_pct = round(max(0.0, ((orig_file_size - fca_file_size) / max(orig_file_size, 1)) * 100.0), 2)
    comp_ratio = round(orig_file_size / max(fca_file_size, 1), 2)

    return {
        "success": True,
        "original_filename": source_path.name,
        "fca_filename": fca_filename,
        "fca_url": f"/fca/{fca_filename}",
        "mode": "FCA-L" if mode_normalized == "lossless" else "FCA-P",
        "mode_raw": mode_normalized,
        "quality": quality_clamped if mode_normalized == "perceptual" else None,
        "sample_rate": audio.sample_rate,
        "channels": audio.channels,
        "bit_depth": audio.bit_depth,
        "duration_seconds": round(audio.duration_seconds, 2),
        "original_size_bytes": orig_file_size,
        "compressed_size_bytes": fca_file_size,
        "saved_bytes": saved_bytes,
        "reduction_percent": reduction_pct,
        "compression_ratio": comp_ratio,
        "verified_lossless": is_verified_lossless,
        "is_lossless_verified": is_verified_lossless,
        "verification_message": verification_msg,
        "sha256_pcm": encode_result.sha256_pcm,
        "checksum_sha256": encode_result.sha256_pcm,
        "decoded_sha256": decoded_sha256 if decoded_sha256 else None,
        "space_saved_percent": reduction_pct,
        "space_saved_bytes": saved_bytes,
        "encode_time_seconds": encode_result.encode_time_seconds,
        "total_time_seconds": total_elapsed,
        "metrics": {
            "compression_ratio": comp_ratio,
            "space_saving_pct": reduction_pct,
            "saved_bytes": saved_bytes,
            "original_size": orig_file_size,
            "compressed_size": fca_file_size,
            "encode_time": encode_result.encode_time_seconds,
        },
    }

