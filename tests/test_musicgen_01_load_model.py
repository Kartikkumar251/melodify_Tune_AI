import torch
import torchaudio
from audiocraft.models import MusicGen
import time
from pathlib import Path

print("="*60)
print("TEST 1: MusicGen Model Loading & Initialization")
print("="*60)

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n[DEVICE] Using: {device}")
if device == 'cuda':
    print(f"[GPU] {torch.cuda.get_device_name(0)}")
    print(f"[VRAM] Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Load model
print(f"\n[LOADING] Starting MusicGen-medium model loading...")
start_time = time.time()
model = MusicGen.get_model('medium', device=device)
load_time = time.time() - start_time

print(f"[SUCCESS] Model loaded in {load_time:.2f} seconds")
print(f"\n[MODEL INFO]")
print(f"  - Model: {model.__class__.__name__}")
print(f"  - Sample rate: {model.sample_rate} Hz")
print(f"  - Device: {device}")

# Test model properties
print(f"\n[GENERATION PARAMS]")
print(f"  - Available params: {list(model.generation_params.keys())}")

print("\n[OK] Model successfully loaded and ready for generation!")
