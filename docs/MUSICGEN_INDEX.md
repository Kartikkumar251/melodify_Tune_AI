# MusicGen ML Testing Suite - Index & Navigation

## 📂 File Inventory

### 🧪 Test Scripts (7 Tests - 60-85 min total)
- **test_musicgen_01_load_model.py** (2-5 min)
  - Loads model, checks CUDA, displays device info
  - **Start here first to verify setup**

- **test_musicgen_02_generate_simple.py** (5-10 min)
  - Generates 5 music samples with different prompts
  - **Good for: Initial validation, listening to output**

- **test_musicgen_03_parameter_testing.py** (10-15 min)
  - Tests: Duration (4-30s), Temperature (0.5-1.5), Top-P (0.7-1.0)
  - **Good for: Understanding parameters impact**

- **test_musicgen_04_batch_generation.py** (10 min)
  - Batch sizes: 1, 2, 4 samples
  - **Good for: Performance scaling analysis**

- **test_musicgen_05_performance_monitoring.py** (5-10 min)
  - Tracks CPU%, GPU memory, generation time
  - **Good for: System optimization insights**

- **test_musicgen_06_model_variants.py** (15-30 min)
  - Compares small, medium, large models
  - **Good for: Choosing optimal model for RTX 5050**

- **test_musicgen_07_audio_quality_analysis.py** (5 min)
  - Audio analysis: BPM, spectral features, MFCC
  - **Good for: Quality metrics verification**

### 🔧 Utility Scripts
- **run_all_musicgen_tests.py**
  - Master script to run all 7 tests sequentially
  - Usage: `python run_all_musicgen_tests.py`

- **setup_musicgen.bat** (Windows only)
  - Automated installation of all dependencies
  - Usage: `setup_musicgen.bat`

### 📚 Documentation Files

#### Quick Start Guides
| File | Purpose | Read Time |
|------|---------|-----------|
| **MUSICGEN_QUICK_START.md** | Step-by-step setup | 5 min |
| **MUSICGEN_QUICK_REFERENCE.txt** | Command cheatsheet | 3 min |
| **MUSICGEN_ML_SUMMARY.md** | Complete overview | 10 min |
| **This File (INDEX.md)** | Navigation guide | 5 min |

#### Detailed Guides
- **MUSICGEN_TESTING_GUIDE.md** (Comprehensive)
  - Detailed explanation of each test
  - Installation instructions
  - Troubleshooting guide
  - Expected outputs

### 📋 Configuration Files
- **musicgen_requirements.txt**
  - All Python dependencies
  - Install: `pip install -r musicgen_requirements.txt`

---

## 🚀 Getting Started Paths

### Path 1: Quick Test (15 minutes)
```
1. setup_musicgen.bat (or pip install -r musicgen_requirements.txt)
2. python test_musicgen_01_load_model.py
3. python test_musicgen_02_generate_simple.py
4. Listen to: musicgen_test_outputs/test_gen_01.wav
```

### Path 2: Full Validation (60-85 minutes)
```
1. setup_musicgen.bat
2. python run_all_musicgen_tests.py
3. Review all outputs in musicgen_test_outputs/
4. Check MUSICGEN_ML_SUMMARY.md for analysis
```

### Path 3: Focused Testing (30 minutes)
```
1. Test 1: Load model
2. Test 2: Generate samples
3. Test 5: Performance monitoring
4. Review results
```

---

## 📊 Test Execution Matrix

```
Test # | File                      | Time  | VRAM | CPU | GPU | Output Files
-------|---------------------------|-------|------|-----|-----|------------------
1      | Load model               | 2-5   | 3 GB | 5%  | ✓   | Console only
2      | Simple generation        | 5-10  | 3 GB | 20% | 40% | 5 WAV files
3      | Parameter testing        | 10-15 | 3 GB | 20% | 40% | 12 WAV files
4      | Batch generation         | 10    | 3 GB | 20% | 50% | 7 WAV files
5      | Performance monitoring   | 5-10  | 3 GB | 30% | 50% | 2 WAV + metrics
6      | Model variants           | 15-30 | 1-6  | 20% | 50% | 3 WAV files
7      | Audio quality analysis   | 5     | 3 GB | 20% | 40% | 1 WAV + metrics
-------|---------------------------|-------|------|-----|-----|------------------
TOTAL  | run_all_musicgen_tests   | 60-85 | 6 GB | ~20%| ~45%| 30+ WAV files
```

---

## 🎯 What to Do First

### Step 1: Setup (5 minutes)
```bash
# Option A (Windows - Automated)
setup_musicgen.bat

# Option B (Manual - All platforms)
pip install -r musicgen_requirements.txt
```

### Step 2: Verify (2-5 minutes)
```bash
python test_musicgen_01_load_model.py
```

Expected output:
```
Device: cuda (or cpu)
Model: MusicGen
Status: Ready for generation
```

### Step 3: Generate (5-10 minutes)
```bash
python test_musicgen_02_generate_simple.py
```

Expected output:
```
musicgen_test_outputs/
├── test_gen_01.wav ✓
├── test_gen_02.wav ✓
├── test_gen_03.wav ✓
├── test_gen_04.wav ✓
└── test_gen_05.wav ✓
```

### Step 4: Listen & Verify
```bash
# Open any WAV file to listen
# All 5 should be unique, listenable music
```

---

## 📈 Test Results Checklist

After running each test, mark completion:

- [ ] **Test 1** - Model loads successfully
- [ ] **Test 2** - 5 unique audio files generated
- [ ] **Test 3** - Parameter variations work
- [ ] **Test 4** - Batch processing shows speedup
- [ ] **Test 5** - Performance metrics reasonable
- [ ] **Test 6** - Model variants compared
- [ ] **Test 7** - BPM and audio metrics extracted

---

## 🖥️ Hardware Requirements

### Minimum
```
CPU:  4+ cores
RAM:  8 GB
GPU:  Not required (but slow)
Disk: 5 GB
```

### Recommended (for RTX 5050)
```
CPU:  Intel i7/AMD R7
GPU:  RTX 5050 (2560 MB)
RAM:  16 GB
VRAM: 4+ GB
Disk: 10 GB
```

---

## 📂 Output Directory Structure

After running tests:

```
musicgen_test_outputs/
├── test_gen_01.wav
├── test_gen_02.wav
├── test_gen_03.wav
├── test_gen_04.wav
├── test_gen_05.wav
├── perf_test_01.wav
├── perf_test_02.wav
├── quality_test.wav
├── parameter_tests/
│   ├── duration_4.0s.wav
│   ├── duration_8.0s.wav
│   ├── duration_16.0s.wav
│   ├── duration_30.0s.wav
│   ├── temperature_0.5.wav
│   ├── temperature_0.7.wav
│   ├── temperature_1.0.wav
│   ├── temperature_1.5.wav
│   ├── top_p_0.7.wav
│   ├── top_p_0.8.wav
│   ├── top_p_0.9.wav
│   └── top_p_1.0.wav
├── batch_tests/
│   ├── batch_01_sample_1.wav
│   ├── batch_02_sample_1.wav
│   ├── batch_02_sample_2.wav
│   ├── batch_04_sample_1.wav
│   ├── batch_04_sample_2.wav
│   ├── batch_04_sample_3.wav
│   └── batch_04_sample_4.wav
└── variants/
    ├── variant_small.wav
    ├── variant_medium.wav
    └── variant_large.wav
```

---

## 🎓 Documentation Reading Order

1. **MUSICGEN_QUICK_REFERENCE.txt** (3 min) - Read first for quick start
2. **MUSICGEN_QUICK_START.md** (5 min) - Installation instructions
3. **MUSICGEN_ML_SUMMARY.md** (10 min) - Overview of full suite
4. **MUSICGEN_TESTING_GUIDE.md** (20 min) - Detailed test explanations
5. **This INDEX.md** - For navigation

---

## 💡 Pro Tips

### Tip 1: Run tests in this order
1. Test 1 (verify setup works)
2. Test 2 (quick validation)
3. Test 5 (check performance)
4. Test 6 (compare models)
5. All others (detailed analysis)

### Tip 2: Save outputs
```bash
# Create backup of results
mkdir musicgen_results_backup
xcopy musicgen_test_outputs musicgen_results_backup /E
```

### Tip 3: Monitor resources
- Open Task Manager (Windows) during tests
- Watch VRAM usage for model variants
- Note CPU vs GPU usage patterns

### Tip 4: Test different prompts
- Edit prompt strings in test files
- Experiment with different genres
- Note what works well

---

## 🔗 Related Files in Project

These complementary files already exist:

- **PHASE_1_PROJECT_SETUP.md** - Backend environment setup
- **PHASE_2_DATABASE_SCHEMA.md** - Database models
- **PHASE_3_CELERY_AI_MODELS.md** - Async task processing
- **PHASE_4_FASTAPI_ENDPOINTS.md** - API endpoints
- **FRONTEND_PAGES_BLUEPRINT.md** - Frontend architecture

**Integration:** These tests validate the ML models before FastAPI integration.

---

## 🚀 Next Steps After Testing

### After Tests Complete
1. Review MUSICGEN_ML_SUMMARY.md for analysis
2. Choose optimal model variant (usually 'medium')
3. Document performance baseline
4. Prepare for DEMUCS testing

### Create DEMUCS Test Suite (Same structure)
```
test_demucs_01_load_model.py
test_demucs_02_separate_stems.py
test_demucs_03_stem_quality.py
... (similar structure)
run_all_demucs_tests.py
DEMUCS_TESTING_GUIDE.md
```

### Then LIBROSA Testing
```
test_librosa_01_bpm_detection.py
test_librosa_02_key_detection.py
test_librosa_03_feature_extraction.py
... (similar structure)
```

---

## 📞 Quick Help

### Installation Issues?
→ Read: MUSICGEN_QUICK_START.md

### Test Fails?
→ Read: MUSICGEN_TESTING_GUIDE.md (Troubleshooting section)

### Want Details?
→ Read: MUSICGEN_ML_SUMMARY.md

### Just Need Commands?
→ Read: MUSICGEN_QUICK_REFERENCE.txt

### Where's Everything?
→ You're reading it! (INDEX.md)

---

## ✅ Success Indicators

You've completed MusicGen testing successfully when:

- ✓ All 7 tests run without errors
- ✓ 30+ WAV files generated successfully
- ✓ Performance metrics captured
- ✓ Model variants compared
- ✓ Audio quality verified
- ✓ BPM/analysis working
- ✓ Understand parameter effects
- ✓ Know which model variant to use

---

## 🎬 Ready to Start?

### Option 1: Quick Test (15 min)
```bash
setup_musicgen.bat
python test_musicgen_01_load_model.py
python test_musicgen_02_generate_simple.py
```

### Option 2: Full Suite (60-85 min)
```bash
setup_musicgen.bat
python run_all_musicgen_tests.py
```

### Option 3: Custom Test
```bash
setup_musicgen.bat
python test_musicgen_06_model_variants.py  # Start here if you want
```

---

**Status:** Ready to test MusicGen ML models ✓

**Start with:** `test_musicgen_01_load_model.py`

**Estimated Time:** 60-85 minutes for full suite

**Output:** 30+ WAV files + performance metrics

Let's go! 🎵🚀
