#!/usr/bin/env python3
"""
MusicGen Test - DIRECT LOADING WORKAROUND
Bypass transformers v5.2.0 bugs by loading model components directly
"""

import sys
import torch
import os

print("=" * 70)
print("MusicGen Loading - Direct Model Component Approach")
print("=" * 70)

MODEL_NAME = "facebook/musicgen-small"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n[DEVICE] Using: {DEVICE}")
print(f"[PyTorch] {torch.__version__}")

try:
    print(f"\n[LOADING] Loading model components directly...")
    
    # Load each component separately to avoid the config class bug
    from transformers import AutoConfig, T5ForConditionalGeneration, EncodecModel, AutoProcessor
    
    print("[1/4] Loading configuration files...")
    # Load config files directly from HF Hub without using from_pretrained on the full model
    base_url = f"https://huggingface.co/{MODEL_NAME}/resolve/main"
    
    # Actually, let's just try importing and using the model despite the bugs
    # Apply a monkey patch first
    from transformers.models.musicgen.configuration_musicgen import MusicgenConfig
    
    original_from_pretrained = MusicgenConfig.from_pretrained
    
    def patched_from_pretrained(cls, *args, **kwargs):
        """Patched from_pretrained that handles config class mapping"""
        config = original_from_pretrained(*args, **kwargs)
        # Ensure components are properly initialized
        if not hasattr(config, 'text_encoder'):
            config.text_encoder = None
        if not hasattr(config, 'audio_encoder'):
            config.audio_encoder = None
        if not hasattr(config, 'decoder'):
            config.decoder = None
        return config
    
    # Apply patch
    MusicgenConfig.from_pretrained = classmethod(patched_from_pretrained)
    
    print("[2/4] Loading model...")
    from transformers import MusicgenForConditionalGeneration
    model = MusicgenForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    print("[3/4] Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    
    print("[4/4] Model ready!")
    
    print("\n" + "=" * 70)
    print("SUCCESS: MusicGen loaded successfully!")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {next(model.parameters()).device}")
    
except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
