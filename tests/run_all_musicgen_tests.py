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
