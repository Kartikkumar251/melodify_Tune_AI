# MusicGen ML Testing Suite - Complete Setup

## 📋 Files Created

### Test Scripts (7 tests)
1. **test_musicgen_01_load_model.py** - Model initialization and device checking
2. **test_musicgen_02_generate_simple.py** - Basic music generation with 5 prompts
3. **test_musicgen_03_parameter_testing.py** - Duration, temperature, top-P variations
4. **test_musicgen_04_batch_generation.py** - Batch processing performance
5. **test_musicgen_05_performance_monitoring.py** - CPU/GPU resource tracking
6. **test_musicgen_06_model_variants.py** - Compare small/medium/large models
7. **test_musicgen_07_audio_quality_analysis.py** - Audio analysis with librosa

### Utility Files
- **run_all_musicgen_tests.py** - Master script to run all 7 tests
- **setup_musicgen.bat** - Windows setup script (automated installation)
- **musicgen_requirements.txt** - All Python dependencies
- **MUSICGEN_TESTING_GUIDE.md** - Comprehensive testing documentation
- **MUSICGEN_QUICK_START.md** - Quick reference guide
- **MUSICGEN_ML_SUMMARY.md** - This file

---

## 🚀 How to Get Started

### Option 1: Automatic Setup (Windows)
```bash
cd path/to/hack
setup_musicgen.bat
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r musicgen_requirements.txt

# Run individual test
python test_musicgen_01_load_model.py

# Or run all tests
python run_all_musicgen_tests.py
```

---

## 📚 Test Suite Overview

| Test | Purpose | Duration | Output |
|------|---------|----------|--------|
| Test 1 | Load model & check device | 2-5 min | Console info |
| Test 2 | Generate 5 music samples | 5-10 min | WAV files |
| Test 3 | Duration/temperature/top-P | 10-15 min | Parameter variations |
| Test 4 | Batch generation performance | 10 min | Batch comparisons |
| Test 5 | Performance monitoring | 5-10 min | CPU/GPU metrics |
| Test 6 | Model variants comparison | 15-30 min | Quality vs speed |
| Test 7 | Audio quality analysis | 5 min | Audio metrics |

**Total Time:** ~60-85 minutes for full test suite

---

## 💾 File Structure

```
hack/
├── Test Scripts (7 files)
│   ├── test_musicgen_01_load_model.py
│   ├── test_musicgen_02_generate_simple.py
│   ├── test_musicgen_03_parameter_testing.py
│   ├── test_musicgen_04_batch_generation.py
│   ├── test_musicgen_05_performance_monitoring.py
│   ├── test_musicgen_06_model_variants.py
│   └── test_musicgen_07_audio_quality_analysis.py
│
├── Utilities
│   ├── run_all_musicgen_tests.py
│   ├── setup_musicgen.bat
│   ├── musicgen_requirements.txt
│   │
│   └── Documentation
│       ├── MUSICGEN_TESTING_GUIDE.md (detailed guide)
│       ├── MUSICGEN_QUICK_START.md (quick reference)
│       └── MUSICGEN_ML_SUMMARY.md (this file)
│
└── musicgen_test_outputs/ (created after running tests)
    ├── test_gen_*.wav
    ├── parameter_tests/
    ├── batch_tests/
    ├── variants/
    └── quality_test.wav
```

---

## 🔧 System Requirements

### Minimum (CPU Only)
- Python 3.10+
- 8 GB RAM
- 5 GB disk space (for models)
- ~5-15 minutes per generation

### Recommended (GPU)
- **GPU:** RTX 5050 (2560 MB VRAM) or better
- **CUDA:** 11.8+
- **PyTorch:** 2.0+
- **Models:** ~3-4 GB VRAM for medium model
- **Time:** ~30-60 seconds per generation

---

## 📊 What Each Test Does

### Test 1: Model Loading
```
Input:  Device info
Output: Model loaded, ready state
Checks: CUDA availability, model initialization
```

### Test 2: Simple Generation
```
Input:  5 prompts (8 seconds each)
Output: 5 WAV files
Examples: Electronic, Ambient, Rock, Jazz, Lo-fi
```

### Test 3: Parameter Testing
```
Variations tested:
- Duration: 4s, 8s, 16s, 30s
- Temperature: 0.5, 0.7, 1.0, 1.5 (creativity)
- Top-P: 0.7, 0.8, 0.9, 1.0 (diversity)
Output: 12 WAV files showing parameter effects
```

### Test 4: Batch Generation
```
Batch sizes: 1, 2, 4
Shows: Performance scaling, memory usage, per-sample time
Useful for: Understanding production capacity
```

### Test 5: Performance Monitoring
```
Tracks: CPU %, GPU memory, generation time
2 samples with detailed metrics
Useful for: System optimization
```

### Test 6: Model Variants
```
Models: small, medium, large
Compares: Load time, generation time, memory, quality
Useful for: Choosing optimal model for RTX 5050
```

### Test 7: Audio Quality
```
Analysis:
- Sample rate, duration, channels
- RMS energy, peak levels
- Spectral centroid, rolloff
- Estimated BPM (90-180 typical)
- MFCC features (13 coefficients)
```

---

## 🎯 Expected Results

### Successful Test Run
```
✓ Test 1: Model loads in 3-5 seconds
✓ Test 2: 5 unique audio files created
✓ Test 3: Parameter variations are perceptible
✓ Test 4: Batch processing is faster per-sample
✓ Test 5: GPU memory < 4 GB for medium model
✓ Test 6: Medium model balances quality/speed
✓ Test 7: BPM detected in range 80-160
```

### GPU Usage Examples (RTX 5050 - 2.5 GB VRAM)
- **Small model:** 1.2 GB
- **Medium model:** 3.0 GB (slight overflow but OK)
- **Large model:** 6.0 GB (not recommended)

### Performance (RTX 5050)
- **8-second generation:** 30-60 seconds
- **16-second generation:** 60-120 seconds
- **Batch 4 @ 8s each:** ~90-120 seconds (faster than 4x sequential)

---

## 🔍 Understanding Model Variants

### Small Model
- **Pros:** Fast (30-45s for 8s), low VRAM (1.2 GB)
- **Cons:** Lower quality
- **Use for:** Testing, development, streaming

### Medium Model (RECOMMENDED)
- **Pros:** Good balance (50-90s for 8s), 3 GB VRAM
- **Cons:** Moderate quality
- **Use for:** Production, BeatFlow AI

### Large Model
- **Pros:** High quality
- **Cons:** Requires 6+ GB VRAM, very slow
- **Use for:** High-quality offline rendering

---

## 📈 Performance Metrics to Track

After running tests, compare these:

```
1. Generation Time
   Target: < 2 minutes for 8-second audio

2. Memory Usage
   Target: < 4 GB VRAM for medium model

3. CPU Efficiency
   Target: < 50% CPU during generation

4. Audio Quality
   Target: Coherent, listenable music

5. BPM Accuracy
   Target: Detected within 5 BPM of prompt intent
```

---

## 🛠️ Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| CUDA not found | Install CUDA 11.8+, add to PATH |
| Out of memory | Use small model, reduce batch size |
| Model download fails | Check internet, set HF_HOME cache |
| Slow generation | OK on CPU, use GPU for speedup |
| Audio quality poor | Use large model or adjust temperature |

---

## 📝 Next Steps (After Testing)

### Phase 1: Continue with Other Models
```
1. DEMUCS Testing (stem separation)
   - test_demucs_*.py scripts
   
2. LIBROSA Testing (audio analysis)
   - test_librosa_*.py scripts
```

### Phase 2: Integration Testing
```
1. Combined MusicGen + Demucs + Librosa
2. Full pipeline end-to-end test
3. Performance optimization
```

### Phase 3: FastAPI Integration
```
1. Create API endpoints
2. Add Celery async tasks
3. Connect to frontend
4. Deploy to production
```

---

## ⚡ Quick Commands

```bash
# Install everything
setup_musicgen.bat

# Test model loads
python test_musicgen_01_load_model.py

# Generate 5 samples
python test_musicgen_02_generate_simple.py

# Run all tests
python run_all_musicgen_tests.py

# Listen to results
# Open: musicgen_test_outputs/test_gen_01.wav
```

---

## 📞 Support

### Common Questions

**Q: My GPU isn't being used**
- A: Check CUDA with: `python -c "import torch; print(torch.cuda.is_available())"`

**Q: Models are very slow to generate**
- A: Normal on CPU. GPU gives 10-20x speedup. Install GPU drivers.

**Q: Memory is very high**
- A: Install just one model variant. Delete cache between tests.

**Q: How do I use these for the backend?**
- A: These are validation tests. Backend will use same code but in FastAPI endpoints + Celery tasks.

---

## 🎓 Learning Resources

- **MusicGen Paper:** https://arxiv.org/abs/2306.05284
- **AudioCraft Repo:** https://github.com/facebookresearch/audiocraft
- **Librosa Docs:** https://librosa.org/
- **PyTorch Guide:** https://pytorch.org/tutorials/

---

## ✨ Success Criteria

You've successfully tested MusicGen when:

- [ ] All 7 tests run without errors
- [ ] Audio files are generated and playable
- [ ] Performance metrics are reasonable
- [ ] Model variants show quality/speed tradeoff
- [ ] Audio analysis extracts meaningful features
- [ ] You understand which model works best for RTX 5050

**After meeting these criteria, proceed to DEMUCS testing.**

---

## 📅 Execution Plan

```
Week 1: MusicGen Testing (YOU ARE HERE)
├── Day 1: Setup & Test 1-2
├── Day 2: Test 3-5
└── Day 3: Test 6-7 + Analysis

Week 2: DEMUCS Testing
├── Setup scripts
├── Stem separation tests
└── Quality analysis

Week 3: LIBROSA + Integration
├── Audio analysis testing
├── Combined pipeline tests
└── Performance optimization

Week 4: FastAPI + Frontend
├── Create API endpoints
├── Connect to Celery
└── Frontend integration
```

---

**Ready? Start with:**
```bash
python test_musicgen_01_load_model.py
```

Then report results back for next phase!
