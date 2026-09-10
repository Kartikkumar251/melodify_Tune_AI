import torch
import torchaudio
from audiocraft.models import MusicGen
import time
from pathlib import Path

print("="*60)
print("TEST 6: Model Variants Comparison")
print("="*60)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = Path('musicgen_test_outputs/variants')
output_dir.mkdir(parents=True, exist_ok=True)

# Available models - small is faster, large is better quality
models_available = ['small', 'medium', 'large']
prompt = "upbeat electronic dance music with synthesizers"

print(f"\n[DEVICE] {device.upper()}")
print(f"[PROMPT] {prompt}\n")

results = []

for model_name in models_available:
    print(f"[Testing] {model_name.upper()}")
    print("-" * 40)
    
    try:
        # Load model
        load_start = time.time()
        model = MusicGen.get_model(model_name, device=device)
        load_time = time.time() - load_start
        
        # Set duration
        model.set_generation_params(duration=8.0)
        
        # GPU memory reset
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        
        # Generate
        gen_start = time.time()
        wav = model.generate([prompt])
        gen_time = time.time() - gen_start
        
        # Memory
        gpu_memory = 0
        if device == 'cuda':
            gpu_memory = torch.cuda.max_memory_allocated() / 1e9
        
        # Save
        filename = f"variant_{model_name}.wav"
        torchaudio.save(str(output_dir / filename), wav, model.sample_rate)
        
        print(f"  Load time: {load_time:.2f}s")
        print(f"  Generation time: {gen_time:.2f}s")
        print(f"  GPU memory: {gpu_memory:.2f} GB")
        print(f"  Sample rate: {model.sample_rate} Hz")
        print(f"  [OK] Saved: {filename}\n")
        
        results.append({
            'model': model_name,
            'load_time': load_time,
            'gen_time': gen_time,
            'gpu_memory': gpu_memory
        })
        
        # Cleanup
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"  [X] Error: {str(e)}\n")

# Comparison summary
print("[COMPARISON SUMMARY]")
print("-" * 60)
print(f"{'Model':<10} {'Load (s)':<12} {'Gen (s)':<12} {'GPU (GB)':<10}")
print("-" * 60)
for result in results:
    print(f"{result['model']:<10} {result['load_time']:<12.2f} {result['gen_time']:<12.2f} {result['gpu_memory']:<10.2f}")

print(f"\n[OK] All variants tested and saved to: {output_dir}")
