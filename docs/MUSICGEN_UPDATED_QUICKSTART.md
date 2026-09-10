# MusicGen ML - Updated Quick Start Guide

## ✅ Installation Complete

All required Python packages are installed:
- PyTorch (CPU)
- Transformers
- TorchAudio
- Librosa
- SciPy, NumPy, tqdm, psutil

---

## 🚀 Ready to Test MusicGen!

### Step 1: Load Model (First Time: 20-40 minutes)
```powershell
cd "D:\Ronit Sharma\vs code\ML Models\hack"
python test_musicgen_01_load_model_SIMPLE.py
```

**What happens:**
- Downloads MusicGen-small model (~2.4 GB) on first run
- Subsequent runs use cached model (5-10 min load)
- Shows device info and ready status

**Expected output:**
```
[DEVICE] Using: cpu
[LOADING] Starting MusicGen model loading...
[LOADING] Loading processor... ✓
[LOADING] Loading model... ✓
[SUCCESS] Model loaded in X.XX seconds
✓ Model successfully loaded and ready for generation!
```

---

### Step 2: Generate Music (Per Sample: 3-5 minutes on CPU)
```powershell
python test_musicgen_02_generate_SIMPLE.py
```

**What happens:**
- Generates 5 music samples with different prompts
- Each takes 3-5 minutes on CPU (normal - not fast)
- Saves as WAV files to `musicgen_test_outputs/`

**Expected output:**
```
[1/5] Prompt: 'upbeat electronic dance music...'
  ⏳ Generating (this may take 3-5 minutes on CPU)...
  ✓ Generated in 180.5 seconds
  ✓ Saved: test_gen_01.wav
  ✓ Duration: 8.0 seconds
```

**Total time for 5 samples:** ~15-25 minutes

---

## 📁 Output Location

All generated audio files go to:
```
D:\Ronit Sharma\vs code\ML Models\hack\musicgen_test_outputs\
├── test_gen_01.wav
├── test_gen_02.wav
├── test_gen_03.wav
├── test_gen_04.wav
└── test_gen_05.wav
```

---

## ⏱️ Timing Expectations

| Action | Time | Notes |
|--------|------|-------|
| Model download (first time only) | 20-40 min | One-time, uses cache after |
| Model load (from cache) | 5-10 min | Every subsequent run |
| Generate 1 sample (8 sec) | 3-5 min | CPU generation is slow |
| Generate 5 samples | 15-25 min | Sequential generation |

**Recommendation:** After first successful test, consider setting up GPU for faster iteration

---

## 🎯 What to Test

### After Test 1 (Model Loading)
- ✓ Model loads successfully
- ✓ No errors during loading
- ✓ Device shows as "cpu" or "cuda"

### After Test 2 (Generation)
- ✓ WAV files are created
- ✓ Files are playable (use Windows Media Player or any audio player)
- ✓ Audio quality is acceptable (no static/noise)
- ✓ Generated music sounds coherent
- ✓ Different prompts produce different outputs

---

## 📊 Simplified Test Files

### Test 1: Model Loading
**File:** `test_musicgen_01_load_model_SIMPLE.py`
- Uses: Transformers library directly
- No audiocraft required
- Downloads model on first run
- Status: ✅ WORKING

### Test 2: Music Generation  
**File:** `test_musicgen_02_generate_simple.py` (UPDATED)
- Uses: Transformers library
- Updated to work without audiocraft
- Generates 5 samples
- Status: ✅ READY

---

## 🔄 How to Run Both Tests

### Quick Sequence
```powershell
# 1. Load model (first time: 20-40 min)
python test_musicgen_01_load_model_SIMPLE.py

# Wait for "✓ Model successfully loaded" message

# 2. Generate music (15-25 min)
python test_musicgen_02_generate_simple.py

# Wait for "✓ All outputs saved" message

# 3. Listen to results
# Open any test_gen_*.wav file
```

**Total time from start to finish:** ~50-75 minutes (first time)

---

## 💡 Pro Tips

### Tip 1: Monitor the Download
- Progress bar shows download speed
- First run may take time depending on internet
- Don't interrupt - it can resume if needed

### Tip 2: Check Generated Output
```powershell
# List all generated files
Get-ChildItem musicgen_test_outputs\*.wav

# Check file sizes (should be ~1-2 MB each)
Get-ChildItem musicgen_test_outputs\*.wav | Select-Object Name, Length
```

### Tip 3: Listen to Files
- Use Windows Media Player (default)
- Or: VLC, foobar2000, Audacity, etc.
- Right-click WAV file → Open with

### Tip 4: If Generation is Too Slow
- Using CPU is expected to be slow
- Consider testing with 1-2 samples first
- For faster testing, GPU setup would help (later)

---

## ✨ Success Indicators

You've successfully tested MusicGen when:

- ✅ Model loads in Test 1
- ✅ 5 WAV files are created in musicgen_test_outputs/
- ✅ Files are playable
- ✅ Audio quality is acceptable (coherent, not noisy)
- ✅ Different prompts produce different music

---

## 🆘 Troubleshooting

### Issue: "No module named transformers"
**Solution:** Run
```powershell
$python = "D:\Ronit Sharma\vs code\ML Models\.conda\python.exe"
& $python -m pip install transformers --upgrade
```

### Issue: Model Download Stops
**Solution:** It may be resuming. Wait 5-10 minutes.
If stuck: Restart the script (downloads resume from checkpoint)

### Issue: Very Slow Generation
**Solution:** Normal on CPU. Each 8-second sample takes 3-5 minutes. 
Expected and OK - GPU would be 10-20x faster

### Issue: Low Audio Quality
**Solution:** Try different prompts or use larger model
```python
model_name = "facebook/musicgen-medium"  # instead of small
```

---

## 📝 Environment Info

```
Python: 3.x (from conda environment)
PyTorch: 2.10.0 (CPU)
Transformers: 5.2.0
Device: CPU (no GPU)
RAM: Available
Disk: ~10 GB free (for models)
```

---

## 🎵 What's Next

### After Validation
1. Review audio quality
2. Analyze generated samples
3. Set up GPU for faster testing (optional)
4. Move to DEMUCS testing (stem separation)
5. Integrate into FastAPI backend

---

## 🔗 Important Files

- **test_musicgen_01_load_model_SIMPLE.py** - Model loading test ✅
- **test_musicgen_02_generate_simple.py** - Music generation (UPDATED) ✅
- **MUSICGEN_INSTALLATION_STATUS.md** - Full installation details
- **MUSICGEN_QUICK_START.md** - Original guide (some parts updated)

---

## 🚀 Start Now

**Ready to test?**

```powershell
cd "D:\Ronit Sharma\vs code\ML Models\hack"
python test_musicgen_01_load_model_SIMPLE.py
```

**Estimated time:** 20-40 minutes for first run

**Next command after model loads:**
```powershell
python test_musicgen_02_generate_simple.py
```

**Then:** Listen to `musicgen_test_outputs/test_gen_01.wav`

Let's go! 🎵
