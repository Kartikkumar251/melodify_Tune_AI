import torch
import torchaudio
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import time
import numpy as np
from pathlib import Path

print("="*60)
print("TEST 2 (SIMPLIFIED): Simple Music Generation")
print("="*60)

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = Path('musicgen_test_outputs')
output_dir.mkdir(exist_ok=True)

print(f"\n[DEVICE] Using: {device}")
print(f"[NOTE] CPU generation is SLOW (~3-5 minutes per 8 seconds)")
print(f"[NOTE] Consider GPU for practical testing\n")

# Load model and processor
print(f"[LOADING] Initializing MusicGen...")
model_name = "facebook/musicgen-small"

try:
    processor = AutoProcessor.from_pretrained(model_name)
    model = MusicgenForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    
    # Use smaller model for testing on CPU
    model.generation_config.max_length = 256  # ~8 seconds
    
    print(f"[SUCCESS] Model loaded\n")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    exit(1)

# Test prompts
prompts = [
    "upbeat electronic dance music with synthesizers",
    "calm ambient piano music",
    "upbeat pop song",
    "smooth jazz with saxophone",
    "lo-fi hip hop beat"
]

print(f"[GENERATING] {len(prompts)} music samples...\n")

# Generate music for each prompt
for idx, prompt in enumerate(prompts, 1):
    print(f"[{idx}/{len(prompts)}] Prompt: '{prompt}'")
    
    try:
        # Prepare inputs
        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        # Generate
        print(f"  [..] Generating (this may take 3-5 minutes on CPU)...")
        start_time = time.time()
        
        with torch.no_grad():
            audio_values = model.generate(**inputs, max_new_tokens=256)
        
        gen_time = time.time() - start_time
        
        # Save output
        sampling_rate = model.config.audio_encoder.sampling_rate
        filename = f"test_gen_{idx:02d}.wav"
        filepath = output_dir / filename
        
        # Convert to numpy and save using soundfile (more reliable)
        audio_np = audio_values.cpu().numpy()
        
        # Handle shape: [1, channels, samples] -> [samples, channels] or [samples]
        if len(audio_np.shape) == 3:
            audio_np = audio_np[0]  # Remove batch dimension
        if len(audio_np.shape) == 2:
            audio_np = audio_np.T  # Convert to [samples, channels]
        else:
            audio_np = audio_np.squeeze()  # Ensure 1D
        
        # Use soundfile for save if available, else torchaudio
        if HAS_SOUNDFILE:
            sf.write(str(filepath), audio_np, sampling_rate)
        else:
            # Fallback to torchaudio (may require torchcodec)
            if len(audio_np.shape) == 1:
                audio_np = audio_np.unsqueeze(0)
            torchaudio.save(str(filepath), torch.from_numpy(audio_np), sampling_rate)
        
        print(f"  [OK] Generated in {gen_time:.1f} seconds")
        print(f"  [OK] Saved: {filename}")
        print(f"  [OK] Duration: {audio_values.shape[-1] / sampling_rate:.1f} seconds")
        print()
        
    except Exception as e:
        print(f"  [X] Error: {str(e)}\n")

print(f"[OK] All outputs saved to: {output_dir}")
print(f"\n[TIP] Listen to the generated files:")
print(f"  - Open: {output_dir}/test_gen_01.wav")
print(f"  - Check: Are they different? Do they sound like music?")
