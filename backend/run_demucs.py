"""
Demucs wrapper that patches torchaudio.load with soundfile so demucs works
even when torchcodec is unavailable (FFmpeg 4 vs required FFmpeg 5).
Usage: python run_demucs.py <input_file> <out_dir>
"""
import sys, os

# ── Patch torchaudio.load BEFORE demucs loads it ────────────────────────────
import soundfile as sf
import numpy as np
import torch
import torchaudio as _ta

def _sf_load(uri, frame_offset=0, num_frames=-1, normalize=True,
             channels_first=True, format=None, buffer_size=4096, backend=None):
    data, sr = sf.read(str(uri), always_2d=True, dtype="float32")
    # soundfile returns (samples, channels) → we need (channels, samples)
    tensor = torch.from_numpy(data.T)
    if num_frames > 0:
        tensor = tensor[:, frame_offset:frame_offset + num_frames]
    elif frame_offset > 0:
        tensor = tensor[:, frame_offset:]
    return tensor, sr

_ta.load = _sf_load

def _sf_save(uri, src, sample_rate, channels_first=True, format=None,
             encoding=None, bits_per_sample=None, buffer_size=4096,
             backend=None, compression=None):
    # src is (channels, samples) if channels_first else (samples, channels)
    if isinstance(src, torch.Tensor):
        data = src.detach().cpu().numpy()
    else:
        data = np.array(src)
    if channels_first:
        data = data.T   # → (samples, channels) for soundfile
    # Pick bit depth
    subtype = "PCM_24" if bits_per_sample == 24 else "PCM_16"
    sf.write(str(uri), data, sample_rate, subtype=subtype)

_ta.save = _sf_save

# ── Now run demucs normally ──────────────────────────────────────────────────
if len(sys.argv) < 3:
    print("Usage: run_demucs.py <input_file> <out_dir>", file=sys.stderr)
    sys.exit(1)

input_file = sys.argv[1]
out_dir    = sys.argv[2]

# Build demucs argv and call main
sys.argv = ["demucs", "--out", out_dir, "-n", "htdemucs", input_file]

from demucs.separate import main
main()
