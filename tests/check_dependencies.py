import sys

print("="*60)
print("DEPENDENCY CHECK")
print("="*60)

errors = []

# Check PyTorch
try:
    import torch
    print(f"[OK] PyTorch {torch.__version__}")
except Exception as e:
    print(f"[X] PyTorch: {e}")
    errors.append(f"PyTorch: {e}")

# Check Transformers
try:
    import transformers
    print(f"[OK] Transformers {transformers.__version__}")
except Exception as e:
    print(f"[X] Transformers: {e}")
    errors.append(f"Transformers: {e}")

# Check TorchAudio
try:
    import torchaudio
    print(f"[OK] TorchAudio")
except Exception as e:
    print(f"[X] TorchAudio: {e}")
    errors.append(f"TorchAudio: {e}")

# Check AudioCraft (Optional - we use transformers instead)
try:
    import audiocraft
    print(f"[OK] AudioCraft (optional)")
except Exception as e:
    print(f"[SKIP] AudioCraft: Not installed (optional - using transformers instead)")

# Check Librosa
try:
    import librosa
    print(f"[OK] Librosa")
except Exception as e:
    print(f"[X] Librosa: {e}")
    errors.append(f"Librosa: {e}")

# Check Core Libs
try:
    import numpy
    import scipy
    print(f"[OK] NumPy & SciPy")
except Exception as e:
    print(f"[X] NumPy/SciPy: {e}")
    errors.append(f"NumPy/SciPy: {e}")

print("\n" + "="*60)
if errors:
    print(f"[ERROR] ERRORS FOUND: {len(errors)}")
    for err in errors:
        print(f"  - {err}")
    sys.exit(1)
else:
    print("[OK] ALL DEPENDENCIES OK")
    sys.exit(0)
