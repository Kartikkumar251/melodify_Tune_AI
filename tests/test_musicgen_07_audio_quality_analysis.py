import torch
import torchaudio
import librosa
import numpy as np
from audiocraft.models import MusicGen
from pathlib import Path

print("="*60)
print("TEST 7: Audio Quality & Analysis")
print("="*60)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = Path('musicgen_test_outputs')
output_dir.mkdir(exist_ok=True)

model = MusicGen.get_model('medium', device=device)
model.set_generation_params(duration=8.0)

# Generate sample
prompt = "upbeat electronic dance music with synthesizers"
print(f"\n[GENERATING] {prompt}")
wav = model.generate([prompt])

# Save
filename = output_dir / 'quality_test.wav'
torchaudio.save(str(filename), wav, model.sample_rate)

# Load with librosa for analysis
audio, sr = librosa.load(str(filename), sr=None)

print(f"\n[AUDIO INFORMATION]")
print(f"  Sample rate: {sr} Hz")
print(f"  Duration: {len(audio) / sr:.2f} seconds")
print(f"  Channels: {wav.shape[0]}")
print(f"  Bit depth: 16-bit (float32 in PyTorch)")

# Audio statistics
print(f"\n[AUDIO STATISTICS]")
print(f"  RMS Energy: {np.sqrt(np.mean(audio**2)):.4f}")
print(f"  Peak: {np.max(np.abs(audio)):.4f}")
print(f"  Mean: {np.mean(audio):.4f}")
print(f"  Std Dev: {np.std(audio):.4f}")

# Frequency analysis
stft = librosa.stft(audio)
magnitude = np.abs(stft)
frequencies = librosa.fft_frequencies(sr=sr)

print(f"\n[FREQUENCY ANALYSIS]")
print(f"  Frequency range: 0 - {sr/2} Hz")
print(f"  Spectral centroid: {librosa.feature.spectral_centroid(y=audio, sr=sr)[0, 0]:.2f} Hz")
print(f"  Spectral rolloff: {librosa.feature.spectral_rolloff(y=audio, sr=sr)[0, 0]:.2f} Hz")

# Tempo/BPM estimation
onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
tempogram = librosa.feature.tempogram(onset_strength=onset_env, sr=sr)
estimated_tempo = librosa.beat.tempo(onset_strength=onset_env, sr=sr)[0]

print(f"\n[TEMPO ANALYSIS]")
print(f"  Estimated tempo: {estimated_tempo:.1f} BPM")

# Zero crossing rate
zcr = librosa.feature.zero_crossing_rate(audio)
print(f"\n[PERCEPTUAL FEATURES]")
print(f"  Zero Crossing Rate (avg): {zcr.mean():.4f}")

# MFCC
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
print(f"  MFCC coefficients: {mfcc.shape[0]} features")

print(f"\n[OK] Audio quality analysis complete!")
print(f"[OK] Audio saved to: {filename}")
