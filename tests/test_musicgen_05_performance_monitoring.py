import torch
import torchaudio
from audiocraft.models import MusicGen
import time
from pathlib import Path
import psutil
import tracemalloc

print("="*60)
print("TEST 5: Performance Monitoring & Resource Usage")
print("="*60)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = Path('musicgen_test_outputs')
output_dir.mkdir(exist_ok=True)

# System info
print(f"\n[SYSTEM INFO]")
print(f"  CPU Cores: {psutil.cpu_count()}")
print(f"  RAM: {psutil.virtual_memory().total / 1e9:.2f} GB")
print(f"  Device: {device.upper()}")

if device == 'cuda':
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU: {props.name}")
    print(f"  VRAM: {props.total_memory / 1e9:.2f} GB")

# Load model
print(f"\n[LOADING MODEL]")
tracemalloc.start()
start_time = time.time()

model = MusicGen.get_model('medium', device=device)
model.set_generation_params(duration=16.0)

load_time = time.time() - start_time
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"  Load time: {load_time:.2f}s")
print(f"  Memory used: {current / 1e6:.2f} MB")
print(f"  Peak memory: {peak / 1e6:.2f} MB")

# Generation performance test
print(f"\n[GENERATION PERFORMANCE]")
prompts = [
    "upbeat electronic dance music with synthesizers and strong beat",
    "calm ambient piano music with strings and pad sounds",
]

results = []

for prompt in prompts:
    print(f"\n  Prompt: '{prompt}'")
    
    # CPU usage
    process = psutil.Process()
    process.cpu_num()
    
    # GPU memory reset
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    
    # Track CPU usage
    cpu_percent_before = process.cpu_percent()
    
    # Timing
    start_time = time.time()
    wav = model.generate([prompt])
    gen_time = time.time() - start_time
    
    # CPU usage after
    cpu_percent_after = process.cpu_percent(interval=0.1)
    
    # Memory
    gpu_memory = 0
    if device == 'cuda':
        gpu_memory = torch.cuda.max_memory_allocated() / 1e9
    
    results.append({
        'prompt': prompt,
        'gen_time': gen_time,
        'cpu_percent': cpu_percent_after,
        'gpu_memory': gpu_memory,
        'audio_shape': wav.shape,
        'duration': wav.shape[-1] / model.sample_rate
    })
    
    print(f"    Generation time: {gen_time:.2f}s")
    print(f"    CPU usage: {cpu_percent_after:.1f}%")
    if device == 'cuda':
        print(f"    GPU memory peak: {gpu_memory:.2f} GB")
    print(f"    Output duration: {wav.shape[-1] / model.sample_rate:.2f}s")
    
    # Save
    filename = f"perf_test_{len(results):02d}.wav"
    torchaudio.save(str(output_dir / filename), wav, model.sample_rate)

# Summary
print(f"\n[SUMMARY]")
print("-" * 40)
for i, result in enumerate(results, 1):
    print(f"Sample {i}:")
    print(f"  Generation time: {result['gen_time']:.2f}s")
    print(f"  CPU usage: {result['cpu_percent']:.1f}%")
    if result['gpu_memory'] > 0:
        print(f"  GPU memory: {result['gpu_memory']:.2f} GB")

print(f"\n[OK] Performance test complete!")
