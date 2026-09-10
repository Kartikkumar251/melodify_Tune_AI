# MusicGen Model - Complete Testing Guide

## Overview

This guide covers testing and validating the **facebook/musicgen-small** model for the BeatFlow AI project. We'll test:
- Model loading and initialization
- Music generation with various prompts
- Performance metrics and inference time
- GPU/CPU resource usage
- Output quality and format
- Batch processing capabilities

---

## Installation & Setup

### Step 1: Install Required Packages

```bash
# Core audio generation
pip install transformers torch torchaudio librosa

# MusicGen specific
pip install audiocraft

# Audio processing
pip install soundfile scipy numpy

# Performance monitoring
pip install psutil GPUtil

# Progress tracking
pip install tqdm
```

### Step 2: Verify Installations

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import audiocraft; print(f'AudioCraft: {audiocraft.__version__}')"
python -c "import librosa; print(f'Librosa loaded successfully')"
python -c "import torchaudio; print(f'TorchAudio: {torchaudio.__version__}')"
```

### Step 3: GPU Setup (Optional but Recommended)

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

---

## Test Scripts

### Test 1: Basic Model Loading & Information

**File: `test_musicgen_01_load_model.py`**

```python
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
print(f"\n[LOADING] Starting MusicGen-small model loading...")
start_time = time.time()
model = MusicGen.get_model('medium', device=device)
load_time = time.time() - start_time

print(f"[SUCCESS] Model loaded in {load_time:.2f} seconds")
print(f"\n[MODEL INFO]")
print(f"  - Model: {model.__class__.__name__}")
print(f"  - Sample rate: {model.sample_rate} Hz")
print(f"  - Device: {device}")
print(f"  - Top-K: {model.generation_params.get('top_k', 'N/A')}")
print(f"  - Top-P: {model.generation_params.get('top_p', 'N/A')}")

# Test model properties
print(f"\n[GENERATION PARAMS]")
print(f"  - Default duration: {model.generation_params}")

print("\n✓ Model successfully loaded and ready for generation!")
```

**Run:** `python test_musicgen_01_load_model.py`

---

### Test 2: Simple Music Generation

**File: `test_musicgen_02_generate_simple.py`**

```python
import torch
import torchaudio
from audiocraft.models import MusicGen
import time
from pathlib import Path

print("="*60)
print("TEST 2: Simple Music Generation")
print("="*60)

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = Path('musicgen_test_outputs')
output_dir.mkdir(exist_ok=True)

# Load model
print(f"\n[LOADING] Initializing MusicGen...")
model = MusicGen.get_model('medium', device=device)
model.set_generation_params(duration=8.0)  # 8 seconds

# Test prompts
prompts = [
    "upbeat electronic dance music with synthesizers",
    "calm ambient piano music",
    "energetic rock song with drums and guitar",
    "smooth jazz with saxophone",
    "lo-fi hip hop beat"
]

# Generate music for each prompt
for idx, prompt in enumerate(prompts, 1):
    print(f"\n[{idx}/{len(prompts)}] Generating: '{prompt}'")
    
    try:
        start_time = time.time()
        wav = model.generate([prompt])
        gen_time = time.time() - start_time
        
        # Save output
        filename = f"test_gen_{idx:02d}.wav"
        filepath = output_dir / filename
        torchaudio.save(str(filepath), wav, model.sample_rate)
        
        print(f"  ✓ Generated in {gen_time:.2f} seconds")
        print(f"  ✓ Saved to: {filepath}")
        print(f"  ✓ Audio shape: {wav.shape} (batch, channels, samples)")
        print(f"  ✓ Duration: {wav.shape[-1] / model.sample_rate:.2f} seconds")
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")

print(f"\n✓ All outputs saved to: {output_dir}")
```

**Run:** `python test_musicgen_02_generate_simple.py`

---

### Test 3: Parameter Testing

**File: `test_musicgen_03_parameter_testing.py`**

```python
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
    
    print(f"✓ {gen_time:.2f}s | Size: {wav.shape[-1] / model.sample_rate:.2f}s")

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
    
    print(f"✓ {gen_time:.2f}s")

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
    
    print(f"✓ {gen_time:.2f}s")

print(f"\n✓ Parameter tests saved to: {output_dir}")
```

**Run:** `python test_musicgen_03_parameter_testing.py`

---

### Test 4: Batch Generation

**File: `test_musicgen_04_batch_generation.py`**

```python
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
batch_sizes = [1, 2, 4, 8]

prompts = {
    1: ["upbeat electronic dance music"],
    2: ["upbeat electronic dance music", "calm ambient piano"],
    4: [
        "upbeat electronic dance music",
        "calm ambient piano",
        "energetic rock song",
        "smooth jazz"
    ],
    8: [
        "upbeat electronic dance music",
        "calm ambient piano",
        "energetic rock song",
        "smooth jazz",
        "lo-fi hip hop beat",
        "classical orchestral music",
        "country folk acoustic",
        "metal heavy guitars"
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
        
        print(f"  ✓ Saved {batch_size} audio files")
        
    except Exception as e:
        print(f"  ✗ Error with batch size {batch_size}: {str(e)}")

print(f"\n✓ Batch tests saved to: {output_dir}")
```

**Run:** `python test_musicgen_04_batch_generation.py`

---

### Test 5: Performance & Resource Monitoring

**File: `test_musicgen_05_performance_monitoring.py`**

```python
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
    if device == 'cuda':
        gpu_memory = torch.cuda.max_memory_allocated() / 1e9
    
    results.append({
        'prompt': prompt,
        'gen_time': gen_time,
        'cpu_percent': cpu_percent_after,
        'gpu_memory': gpu_memory if device == 'cuda' else None,
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
    if result['gpu_memory']:
        print(f"  GPU memory: {result['gpu_memory']:.2f} GB")

print(f"\n✓ Performance test complete!")
```

**Run:** `python test_musicgen_05_performance_monitoring.py`

---

### Test 6: Model Variants Comparison

**File: `test_musicgen_06_model_variants.py`**

```python
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

# Available models
models_available = ['small', 'medium', 'large', 'melody']
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
        if device == 'cuda':
            gpu_memory = torch.cuda.max_memory_allocated() / 1e9
        else:
            gpu_memory = 0
        
        # Save
        filename = f"variant_{model_name}.wav"
        torchaudio.save(str(output_dir / filename), wav, model.sample_rate)
        
        print(f"  Load time: {load_time:.2f}s")
        print(f"  Generation time: {gen_time:.2f}s")
        print(f"  GPU memory: {gpu_memory:.2f} GB")
        print(f"  Sample rate: {model.sample_rate} Hz")
        print(f"  ✓ Saved: {filename}\n")
        
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
        print(f"  ✗ Error: {str(e)}\n")

# Comparison summary
print("[COMPARISON SUMMARY]")
print("-" * 60)
print(f"{'Model':<10} {'Load (s)':<12} {'Gen (s)':<12} {'GPU (GB)':<10}")
print("-" * 60)
for result in results:
    print(f"{result['model']:<10} {result['load_time']:<12.2f} {result['gen_time']:<12.2f} {result['gpu_memory']:<10.2f}")

print(f"\n✓ All variants tested and saved to: {output_dir}")
```

**Run:** `python test_musicgen_06_model_variants.py`

---

### Test 7: Quality & Audio Analysis

**File: `test_musicgen_07_audio_quality_analysis.py`**

```python
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

print(f"\n✓ Audio quality analysis complete!")
print(f"✓ Audio saved to: {filename}")
```

**Run:** `python test_musicgen_07_audio_quality_analysis.py`

---

## Running All Tests in Sequence

**File: `run_all_musicgen_tests.py`**

```python
import subprocess
import sys
from pathlib import Path

test_files = [
    'test_musicgen_01_load_model.py',
    'test_musicgen_02_generate_simple.py',
    'test_musicgen_03_parameter_testing.py',
    'test_musicgen_04_batch_generation.py',
    'test_musicgen_05_performance_monitoring.py',
    'test_musicgen_06_model_variants.py',
    'test_musicgen_07_audio_quality_analysis.py',
]

print("="*60)
print("Running All MusicGen Tests")
print("="*60)

for idx, test_file in enumerate(test_files, 1):
    print(f"\n\n{'='*60}")
    print(f"[{idx}/{len(test_files)}] Running {test_file}")
    print("="*60)
    
    result = subprocess.run([sys.executable, test_file], cwd=Path(__file__).parent)
    
    if result.returncode != 0:
        print(f"\n✗ Test failed: {test_file}")
        sys.exit(1)
    
    print(f"\n✓ Test passed: {test_file}")

print(f"\n\n{'='*60}")
print("✓ ALL TESTS COMPLETED SUCCESSFULLY!")
print("="*60)
```

**Run all:** `python run_all_musicgen_tests.py`

---

## Expected Outputs

All tests will create files in `musicgen_test_outputs/`:
```
musicgen_test_outputs/
├── test_gen_01.wav
├── test_gen_02.wav
├── test_gen_03.wav
├── parameter_tests/
│   ├── duration_4.0s.wav
│   ├── temperature_0.5.wav
│   └── top_p_0.7.wav
├── batch_tests/
│   ├── batch_01_sample_1.wav
│   └── batch_08_sample_8.wav
├── variants/
│   ├── variant_small.wav
│   ├── variant_medium.wav
│   └── variant_large.wav
└── quality_test.wav
```

---

## Troubleshooting

### Issue: CUDA Out of Memory
**Solution:**
```python
# Reduce batch size or duration
model.set_generation_params(duration=4.0)
# Or use CPU
device = 'cpu'
```

### Issue: Model download fails
**Solution:**
```bash
# Set cache directory
export AUDIOCRAFT_MODEL_ROOT=/path/to/models
# Then run tests
```

### Issue: Audio quality is poor
**Solution:**
- Use 'large' model instead of 'small'
- Try 'melody' model for better musicality
- Adjust temperature (lower = more conservative, higher = more creative)

---

## Next Steps

After testing MusicGen:
1. Move to Demucs (Stem Separation)
2. Test Librosa (Audio Analysis)
3. Create an integrated test pipeline
4. Build FastAPI endpoints
5. Connect to frontend
