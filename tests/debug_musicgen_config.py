#!/usr/bin/env python3
"""
Debug MusicGen config loading issue
"""

from transformers import MusicgenForConditionalGeneration
import json

MODEL_NAME = "facebook/musicgen-small"

print("=" * 70)
print("DEBUGGING MusicGen Configuration Loading")
print("=" * 70)

# Try to load just the config
from transformers import AutoConfig

print(f"\n[INFO] Loading config from {MODEL_NAME}...")
try:
    config = AutoConfig.from_pretrained(MODEL_NAME)
    print(f"[OK] Config type: {type(config).__name__}")
    print(f"[OK] Config model_type: {config.model_type}")
    
    # Check attributes
    print(f"\n[CHECK] Config attributes:")
    print(f"  - has decoder: {hasattr(config, 'decoder')}")
    print(f"  - has text_encoder: {hasattr(config, 'text_encoder')}")
    print(f"  - has audio_encoder: {hasattr(config, 'audio_encoder')}")
    
    if hasattr(config, 'decoder'):
        print(f"  - decoder type: {type(config.decoder).__name__}")
    if hasattr(config, 'text_encoder'):
        print(f"  - text_encoder type: {type(config.text_encoder).__name__}")
    if hasattr(config, 'audio_encoder'):
        print(f"  - audio_encoder type: {type(config.audio_encoder).__name__}")
        
except Exception as e:
    print(f"[ERROR] {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
