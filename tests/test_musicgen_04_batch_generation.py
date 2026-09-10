import torch
import torchaudio
from audiocraft.models import MusicGen
import time
from pathlib import Path

print("="*60)
print("TEST 4: Batch Generation (Multiple Prompts at Once)")
print("="*60)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = Path('musicgen_test_outputs/batch_tests')
output_dir.mkdir(parents=True, exist_ok=True)

model = MusicGen.get_model('medium', device=device)
model.set_generation_params(duration=8.0)

# Different batch sizes
batch_sizes = [1, 2, 4]

prompts = {
    1: ["upbeat electronic dance music"],
    2: ["upbeat electronic dance music", "calm ambient piano"],
    4: [
        "upbeat electronic dance music",
        "calm ambient piano",
        "energetic rock song",
        "smooth jazz"
    ]
}

print(f"\n[DEVICE] {device.upper()}")
if device == 'cuda':
    print(f"[VRAM] {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB available")

for batch_size in batch_sizes:
    print(f"\n[Test] Batch size: {batch_size}")
    print("-" * 40)
    
    batch_prompts = prompts[batch_size]
    
    try:
        # Memory before
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            pre_mem = torch.cuda.memory_allocated() / 1e9
        
        # Generate
        start_time = time.time()
        wav = model.generate(batch_prompts)
        gen_time = time.time() - start_time
        
        # Memory after
        if device == 'cuda':
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"  Memory used: {peak_mem:.2f} GB")
        
        print(f"  Generation time: {gen_time:.2f}s")
        print(f"  Per-sample time: {gen_time/batch_size:.2f}s")
        print(f"  Output shape: {wav.shape}")
        
        # Save all outputs
        for i, (audio, prompt) in enumerate(zip(wav, batch_prompts)):
            filename = f"batch_{batch_size:02d}_sample_{i+1}.wav"
            torchaudio.save(str(output_dir / filename), audio.unsqueeze(0), model.sample_rate)
        
        print(f"  [OK] Saved {batch_size} audio files")
        
    except Exception as e:
        print(f"  [X] Error with batch size {batch_size}: {str(e)}")

print(f"\n[OK] Batch tests saved to: {output_dir}")
