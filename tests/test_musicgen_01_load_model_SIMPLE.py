import torch
import torchaudio
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import time
from pathlib import Path

print("="*60)
print("TEST 1 (SIMPLIFIED): MusicGen Model Loading")
print("="*60)

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n[DEVICE] Using: {device}")
if device == 'cuda':
    print(f"[GPU] {torch.cuda.get_device_name(0)}")
    print(f"[VRAM] Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print(f"[INFO] Running on CPU - this will be slow!")

# Load model
print(f"\n[LOADING] Starting MusicGen model loading...")
print(f"[NOTE] First run will download model (~5-10 GB)...")

start_time = time.time()

model_name = "facebook/musicgen-small"
print(f"[LOADING] Loading processor...")
processor = AutoProcessor.from_pretrained(model_name)

print(f"[LOADING] Loading model...")
model = MusicgenForConditionalGeneration.from_pretrained(model_name)
model.to(device)

load_time = time.time() - start_time

print(f"\n[SUCCESS] Model loaded in {load_time:.2f} seconds")
print(f"\n[MODEL INFO]")
print(f"  - Model: {model.__class__.__name__}")
print(f"  - Device: {device}")
print(f"  - Model size: {model.num_parameters() / 1e6:.1f}M parameters")

print("\n[OK] Model successfully loaded and ready for generation!")
