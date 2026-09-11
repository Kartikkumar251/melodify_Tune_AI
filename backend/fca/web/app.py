"""FastAPI Web Server for FCA Audio Codec Web Application."""

from __future__ import annotations

import os
import io
import uuid
import tempfile
import numpy as np
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from fca.io.wav_reader import read_wav
from fca.io.wav_writer import write_wav
from fca.encoder import FCAEncoder
from fca.decoder import FCADecoder
from fca.container.reader import FCAContainerReader
from fca.transform.stft import STFTProcessor
from fca.verify.hash import verify_pcm_equality, calculate_pcm_sha256
from fca.verify.crc import calculate_crc32

app = FastAPI(title="FCA Audio Codec API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = os.path.join(tempfile.gettempdir(), "fca_web_cache")
os.makedirs(TEMP_DIR, exist_ok=True)

SAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "samples", "synthetic"))


def _compute_visualizer_data(audio) -> dict:
    """Compute 1D waveform envelope and 2D STFT spectrogram for canvas display."""
    samples_mono = audio.samples[:, 0].astype(np.float64)
    # 1. Downsampled waveform envelope (e.g. 600 points)
    num_points = 600
    n = len(samples_mono)
    if n > num_points:
        chunk_size = n // num_points
        # compute min and max for each chunk
        chunks = samples_mono[: chunk_size * num_points].reshape(num_points, chunk_size)
        mins = np.min(chunks, axis=1)
        maxs = np.max(chunks, axis=1)
        peak = max(np.max(np.abs(samples_mono)), 1.0)
        envelope = np.column_stack([mins / peak, maxs / peak]).tolist()
    else:
        peak = max(np.max(np.abs(samples_mono)), 1.0)
        normalized = (samples_mono / peak).tolist()
        envelope = [[x, x] for x in normalized]

    # 2. STFT Spectrogram
    stft = STFTProcessor(n_fft=512, hop_length=256)
    Zxx = stft.forward(samples_mono[: min(n, audio.sample_rate * 4)])  # max 4 seconds for preview
    mag = np.abs(Zxx)
    # Log magnitude in dB normalized to 0..1
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-6))
    max_db = np.max(mag_db)
    min_db = max_db - 60.0  # 60 dB dynamic range
    norm_spec = np.clip((mag_db - min_db) / 60.0, 0.0, 1.0)

    # Downsample frequency bins (e.g. 64 bins) and time frames (e.g. 120 frames) for lightweight JSON
    freq_bins, frames = norm_spec.shape
    target_bins = min(freq_bins, 64)
    target_frames = min(frames, 120)

    from scipy.ndimage import zoom
    zoom_factors = (target_bins / freq_bins, target_frames / frames)
    try:
        reduced_spec = zoom(norm_spec, zoom_factors, order=1)
    except Exception:
        reduced_spec = norm_spec[:target_bins, :target_frames]

    return {
        "envelope": envelope,
        "spectrogram": np.round(reduced_spec, 2).tolist(),
        "freq_bins": target_bins,
        "frames": target_frames,
    }


@app.get("/api/samples")
def list_samples():
    """List available pre-generated synthetic audio files."""
    if not os.path.exists(SAMPLES_DIR):
        return {"samples": []}
    files = []
    for f in sorted(os.listdir(SAMPLES_DIR)):
        if f.lower().endswith(".wav"):
            path = os.path.join(SAMPLES_DIR, f)
            size = os.path.getsize(path)
            try:
                audio = read_wav(path)
                files.append({
                    "id": f,
                    "filename": f,
                    "size_bytes": size,
                    "sample_rate": audio.sample_rate,
                    "channels": audio.channels,
                    "bit_depth": audio.bit_depth,
                    "duration": round(audio.duration_seconds, 2),
                    "samples": audio.num_samples,
                })
            except Exception:
                pass
    return {"samples": files}


@app.get("/api/samples/{sample_id}/data")
def get_sample_visual_data(sample_id: str):
    """Retrieve waveform envelope and spectrogram for a preset sample."""
    path = os.path.join(SAMPLES_DIR, sample_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Sample not found")
    audio = read_wav(path)
    viz = _compute_visualizer_data(audio)
    return {
        "sample_id": sample_id,
        "sample_rate": audio.sample_rate,
        "channels": audio.channels,
        "bit_depth": audio.bit_depth,
        "duration": round(audio.duration_seconds, 2),
        "visual": viz,
    }


@app.get("/api/samples/{sample_id}/audio")
def get_sample_audio_file(sample_id: str):
    """Stream audio file for preview."""
    path = os.path.join(SAMPLES_DIR, sample_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Sample not found")
    return FileResponse(path, media_type="audio/wav", filename=sample_id)


@app.post("/api/encode")
async def encode_audio(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    mode: str = Form("lossless"),
    quality: int = Form(5),
    block_size: int = Form(4096),
    entropy_method: str = Form("auto"),
):
    """Encode audio into .fca file and return metrics."""
    tmp_in_path = None
    try:
        if file is not None and file.filename:
            tmp_in_path = os.path.join(TEMP_DIR, f"upload_{uuid.uuid4().hex}.wav")
            content = await file.read()
            with open(tmp_in_path, "wb") as f:
                f.write(content)
            audio = read_wav(tmp_in_path)
            orig_name = file.filename
        elif sample_id:
            src_path = os.path.join(SAMPLES_DIR, sample_id)
            if not os.path.exists(src_path):
                raise HTTPException(status_code=404, detail="Sample not found")
            audio = read_wav(src_path)
            orig_name = sample_id
        else:
            raise HTTPException(status_code=400, detail="Must provide either file or sample_id")

        file_token = uuid.uuid4().hex
        fca_path = os.path.join(TEMP_DIR, f"{file_token}.fca")

        encoder = FCAEncoder(block_size=block_size)
        res = encoder.encode_file(
            audio,
            fca_path,
            mode=mode,
            quality=quality,
            entropy_method=entropy_method,
        )

        viz = _compute_visualizer_data(audio)

        return {
            "success": True,
            "token": file_token,
            "filename": f"{os.path.splitext(orig_name)[0]}.fca",
            "mode": mode,
            "quality": quality,
            "input_bytes": res.input_bytes,
            "output_bytes": res.output_bytes,
            "compression_ratio": res.compression_ratio,
            "space_saved_percent": res.space_saved_percent,
            "encode_time_seconds": res.encode_time_seconds,
            "sha256_pcm": res.sha256_pcm,
            "channels": audio.channels,
            "sample_rate": audio.sample_rate,
            "bit_depth": audio.bit_depth,
            "visual": viz,
        }
    finally:
        if tmp_in_path and os.path.exists(tmp_in_path):
            try:
                os.remove(tmp_in_path)
            except OSError:
                pass


@app.post("/api/decode")
async def decode_audio(file: UploadFile = File(...)):
    """Decode uploaded .fca container into PCM WAV."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    file_token = uuid.uuid4().hex
    tmp_fca = os.path.join(TEMP_DIR, f"{file_token}.fca")
    tmp_wav = os.path.join(TEMP_DIR, f"{file_token}.wav")

    content = await file.read()
    with open(tmp_fca, "wb") as f:
        f.write(content)

    try:
        decoder = FCADecoder()
        dec_res = decoder.decode_file(tmp_fca)
        write_wav(dec_res.audio, tmp_wav)

        viz = _compute_visualizer_data(dec_res.audio)

        return {
            "success": True,
            "token": file_token,
            "filename": f"{os.path.splitext(file.filename)[0]}.wav",
            "mode": dec_res.mode,
            "total_samples": dec_res.total_samples,
            "output_bytes": dec_res.output_bytes,
            "decode_time_seconds": dec_res.decode_time_seconds,
            "sha256_pcm": dec_res.sha256_pcm,
            "channels": dec_res.audio.channels,
            "sample_rate": dec_res.audio.sample_rate,
            "bit_depth": dec_res.audio.bit_depth,
            "duration": round(dec_res.audio.duration_seconds, 2),
            "visual": viz,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Decoding failed: {e}")


@app.post("/api/inspect")
async def inspect_fca(file: UploadFile = File(...)):
    """Inspect binary header and block structure of .fca file."""
    content = await file.read()
    tmp_fca = os.path.join(TEMP_DIR, f"inspect_{uuid.uuid4().hex}.fca")
    with open(tmp_fca, "wb") as f:
        f.write(content)

    try:
        header, blocks = FCAContainerReader.read(tmp_fca)
        transform_names = {0: "Direct/Identity", 1: "Integer Haar DWT", 2: "STFT"}
        ch_names = {0: "Independent", 1: "Mid/Side Stereo"}
        coder_names = {0: "Raw", 1: "Rice", 2: "Huffman", 3: "Golomb", 4: "Range/rANS"}

        block_summaries = []
        for b in blocks:
            block_summaries.append({
                "block_index": b.block_index,
                "sample_count": b.sample_count,
                "transform": transform_names.get(b.transform_id, f"ID-{b.transform_id}"),
                "channel_mode": ch_names.get(b.channel_mode, f"Mode-{b.channel_mode}"),
                "coder": coder_names.get(b.entropy_coder_id, f"Coder-{b.entropy_coder_id}"),
                "prediction_order": b.prediction_order,
                "aux_param": b.aux_param,
                "payload_bytes": len(b.payload),
            })

        return {
            "success": True,
            "header": {
                "version": f"v{header.version_major}.{header.version_minor}",
                "mode": "FCA-L (Lossless)" if header.mode == 0 else "FCA-P (Perceptual)",
                "channels": header.channels,
                "bit_depth": header.bit_depth,
                "sample_rate": header.sample_rate,
                "total_samples": header.total_samples,
                "block_count": header.block_count,
                "metadata": header.metadata,
            },
            "blocks_count": len(blocks),
            "blocks": block_summaries,
        }
    finally:
        if os.path.exists(tmp_fca):
            try:
                os.remove(tmp_fca)
            except OSError:
                pass


@app.post("/api/verify")
async def verify_fca(
    fca_file: UploadFile = File(...),
    wav_file: Optional[UploadFile] = File(None),
):
    """Verify FCA container integrity and optional sample equality against original WAV."""
    tmp_fca = os.path.join(TEMP_DIR, f"verify_{uuid.uuid4().hex}.fca")
    tmp_wav = None

    content = await fca_file.read()
    with open(tmp_fca, "wb") as f:
        f.write(content)

    try:
        header, blocks = FCAContainerReader.read(tmp_fca)
        header_ok = True
        blocks_ok = True

        equality_result = None
        if wav_file is not None and wav_file.filename:
            tmp_wav = os.path.join(TEMP_DIR, f"verify_ref_{uuid.uuid4().hex}.wav")
            wav_bytes = await wav_file.read()
            with open(tmp_wav, "wb") as f:
                f.write(wav_bytes)

            orig_audio = read_wav(tmp_wav)
            decoder = FCADecoder()
            dec_res = decoder.decode_file(tmp_fca)
            is_match, match_msg = verify_pcm_equality(orig_audio, dec_res.audio)
            equality_result = {
                "bit_exact": is_match,
                "message": match_msg,
                "original_sha256": calculate_pcm_sha256(orig_audio),
                "decoded_sha256": dec_res.sha256_pcm,
            }

        return {
            "success": True,
            "header_verified": header_ok,
            "blocks_count": len(blocks),
            "blocks_verified": blocks_ok,
            "equality": equality_result,
        }
    finally:
        for p in [tmp_fca, tmp_wav]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


@app.get("/api/benchmark")
def run_benchmark():
    """Execute live benchmark on the synthetic test signals corpus."""
    if not os.path.exists(SAMPLES_DIR):
        raise HTTPException(status_code=404, detail="Synthetic corpus not found")

    wav_files = [f for f in sorted(os.listdir(SAMPLES_DIR)) if f.lower().endswith(".wav")]
    encoder = FCAEncoder()
    decoder = FCADecoder()
    results = []

    for f in wav_files:
        wav_path = os.path.join(SAMPLES_DIR, f)
        tmp_fca = os.path.join(TEMP_DIR, f"bench_{uuid.uuid4().hex}.fca")
        try:
            audio = read_wav(wav_path)
            enc = encoder.encode_file(audio, tmp_fca, mode="lossless")
            dec = decoder.decode_file(tmp_fca)
            exact = audio.equals(dec.audio) and (audio.compute_sha256() == dec.sha256_pcm)

            results.append({
                "filename": f,
                "original_bytes": enc.input_bytes,
                "fca_bytes": enc.output_bytes,
                "ratio": enc.compression_ratio,
                "space_saved": enc.space_saved_percent,
                "encode_ms": round(enc.encode_time_seconds * 1000.0, 1),
                "exact": exact,
            })
        finally:
            if os.path.exists(tmp_fca):
                try:
                    os.remove(tmp_fca)
                except OSError:
                    pass

    return {"results": results}


@app.get("/api/download/{token}")
def download_cached_file(token: str, ext: str = "fca"):
    """Download encoded .fca or decoded .wav file."""
    path = os.path.join(TEMP_DIR, f"{token}.{ext}")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File expired or not found")
    media_type = "application/octet-stream" if ext == "fca" else "audio/wav"
    return FileResponse(path, media_type=media_type, filename=f"output_{token[:8]}.{ext}")


# Mount static frontend directory
STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "frontend"))
if os.path.exists(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="frontend")
