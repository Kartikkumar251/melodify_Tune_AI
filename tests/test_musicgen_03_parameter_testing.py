import torch
import torchaudio
from audiocraft.models import MusicGen
import time
from pathlib import Path

print("="*60)
print("TEST 3: Parameter Testing (Duration, Temperature, Top-P)")
print("="*60)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = Path('musicgen_test_outputs/parameter_tests')
output_dir.mkdir(parents=True, exist_ok=True)

model = MusicGen.get_model('medium', device=device)

# Test 1: Different Durations
print("\n[TEST 3A] DURATION TESTING")
print("-" * 40)
durations = [4.0, 8.0, 16.0, 30.0]
prompt = "upbeat electronic dance music"

for duration in durations:
    print(f"Duration: {duration}s", end=" ... ")
    model.set_generation_params(duration=duration)
    
    start_time = time.time()
    wav = model.generate([prompt])
    gen_time = time.time() - start_time
    
    filename = f"duration_{duration}s.wav"
    torchaudio.save(str(output_dir / filename), wav, model.sample_rate)
    
    print(f"[OK] {gen_time:.2f}s | Size: {wav.shape[-1] / model.sample_rate:.2f}s")

# Test 2: Temperature (Creativity)
print("\n[TEST 3B] TEMPERATURE TESTING (Creativity)")
print("-" * 40)
temperatures = [0.5, 0.7, 1.0, 1.5]
model.set_generation_params(duration=8.0)

for temp in temperatures:
    print(f"Temperature: {temp}", end=" ... ")
    model.generation_params['temperature'] = temp
    
    start_time = time.time()
    wav = model.generate([prompt])
    gen_time = time.time() - start_time
    
    filename = f"temperature_{temp}.wav"
    torchaudio.save(str(output_dir / filename), wav, model.sample_rate)
    
    print(f"[OK] {gen_time:.2f}s")

# Test 3: Top-P (Nucleus Sampling)
print("\n[TEST 3C] TOP-P TESTING (Sampling Diversity)")
print("-" * 40)
top_ps = [0.7, 0.8, 0.9, 1.0]

for top_p in top_ps:
    print(f"Top-P: {top_p}", end=" ... ")
    model.generation_params['top_p'] = top_p
    
    start_time = time.time()
    wav = model.generate([prompt])
    gen_time = time.time() - start_time
    
    filename = f"top_p_{top_p}.wav"
    torchaudio.save(str(output_dir / filename), wav, model.sample_rate)
    
    print(f"[OK] {gen_time:.2f}s")

print(f"\n[OK] Parameter tests saved to: {output_dir}")
