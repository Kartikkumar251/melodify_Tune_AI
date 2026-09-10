# MusicGen Testing - Quick Start Guide

## Quick Setup

```bash
# 1. Install dependencies
pip install -r musicgen_requirements.txt

# 2. Run individual tests
python test_musicgen_01_load_model.py
python test_musicgen_02_generate_simple.py
python test_musicgen_03_parameter_testing.py
python test_musicgen_04_batch_generation.py
python test_musicgen_05_performance_monitoring.py
python test_musicgen_06_model_variants.py
python test_musicgen_07_audio_quality_analysis.py

# 3. Or run all tests at once
python run_all_musicgen_tests.py
```

## Test Descriptions

### Test 1: Load Model ⏱️ ~2-5 min
- Checks CUDA availability
- Loads MusicGen-medium model
- Verifies model properties
- **Output:** Console output with device info

### Test 2: Simple Generation ⏱️ ~5-10 min
- Generates 5 music samples with different prompts
- Tests basic functionality
- **Output:** `musicgen_test_outputs/test_gen_*.wav`

### Test 3: Parameter Testing ⏱️ ~10-15 min
- Tests different durations (4s, 8s, 16s, 30s)
- Tests temperature/creativity levels (0.5, 0.7, 1.0, 1.5)
- Tests sampling diversity (top-p)
- **Output:** `musicgen_test_outputs/parameter_tests/`

### Test 4: Batch Generation ⏱️ ~10 min
- Generates 1, 2, 4 samples in batch
- Compares performance and memory usage
- **Output:** `musicgen_test_outputs/batch_tests/`

### Test 5: Performance Monitoring ⏱️ ~5-10 min
- Monitors CPU usage
- Monitors GPU memory allocation
- Tracks generation time
- **Output:** Console metrics + audio files

### Test 6: Model Variants ⏱️ ~15-30 min
- Compares small, medium, large model variants
- Shows quality vs speed tradeoff
- **Output:** `musicgen_test_outputs/variants/`

### Test 7: Audio Quality ⏱️ ~5 min
- Analyzes generated audio
- Extracts BPM, frequency, spectral features
- **Output:** Console analysis + quality_test.wav

## Expected VRAM Usage

- **Small model:** ~2 GB VRAM
- **Medium model:** ~4 GB VRAM (recommended for RTX 5050)
- **Large model:** ~8 GB VRAM
- **CPU fallback:** Works but ~5-10x slower

## Troubleshooting

### CUDA Out of Memory
```bash
# Use smaller model or reduce batch size
# Edit test file and change:
# model = MusicGen.get_model('small', device=device)  # instead of 'medium'
```

### Model Download Issues
```bash
# Models are downloaded to ~/.cache/huggingface/
# Set custom cache:
export HF_HOME=/path/to/cache
export AUDIOCRAFT_MODEL_ROOT=/path/to/models
```

### Import Errors
```bash
# Reinstall audiocraft:
pip uninstall audiocraft
pip install audiocraft --upgrade
```

## Next Steps After Testing

1. **Review Test Results**
   - Compare quality vs speed
   - Check memory usage
   - Identify best model variant for your hardware

2. **Move to Demucs Testing**
   - Stem separation model
   - Test audio analysis

3. **Move to Librosa Testing**
   - BPM detection
   - Key detection
   - Audio analysis

4. **Integration Testing**
   - Combined MusicGen + Demucs + Librosa
   - Test full pipeline

5. **FastAPI Integration**
   - Create API endpoints
   - Add async task processing with Celery

## Output Structure

```
hack/
├── musicgen_test_outputs/
│   ├── test_gen_01.wav
│   ├── test_gen_02.wav
│   ├── parameter_tests/
│   │   ├── duration_4.0s.wav
│   │   ├── temperature_0.5.wav
│   │   └── ...
│   ├── batch_tests/
│   │   ├── batch_01_sample_1.wav
│   │   └── ...
│   ├── variants/
│   │   ├── variant_small.wav
│   │   ├── variant_medium.wav
│   │   └── variant_large.wav
│   └── quality_test.wav
├── test_musicgen_*.py (7 test files)
├── run_all_musicgen_tests.py
├── musicgen_requirements.txt
└── MUSICGEN_TESTING_GUIDE.md
```

## Performance Tips

1. **For faster testing:**
   - Use 'small' model instead of 'medium'
   - Reduce prompt lengths
   - Use shorter durations (4-8s)

2. **For better quality:**
   - Use 'large' model if you have 8+ GB VRAM
   - Longer durations (16-30s)
   - Lower temperature (0.7-0.9)

3. **For batch processing:**
   - Batch size 1-4 for VRAM consistency
   - Avoid batch size > 8 on RTX 5050
   - Use smaller model for overlapping batches

## Expected Output Format

MusicGen outputs:
- **Mono audio:** 1 channel
- **Stereo audio:** 2 channels
- **Sample rate:** 16 kHz (standard)
- **Bit depth:** 16-bit
- **Format:** WAV (uncompressed)

## Success Indicators

✓ Test 1: Model loads < 10 seconds
✓ Test 2: Can generate 5 samples
✓ Test 3: Parameters change output predictably
✓ Test 4: Batch generation faster per-sample than sequential
✓ Test 5: Memory usage reasonable for your hardware
✓ Test 6: Medium is good balance of quality/speed
✓ Test 7: BPM detection works (usually 90-140 BPM)

## Questions?

Check test output or refer to:
- AudioCraft docs: https://github.com/facebookresearch/audiocraft
- MusicGen paper: https://arxiv.org/abs/2306.05284
