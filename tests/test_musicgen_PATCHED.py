#!/usr/bin/env python3
"""
MusicGen Test WITH RUNTIME PATCH for Transformers 5.2.0 Compatibility
This applies a monkey-patch before loading MusicGen to bypass the AttributeError
"""

import sys
import os

# STEP 0: Apply runtime patch BEFORE importing transformers models
print("=" * 70)
print("APPLYING RUNTIME PATCH FOR TRANSFORMERS 5.2.0 COMPATIBILITY")
print("=" * 70)

try:
    # Patch the MusicgenDecoderConfig class
    from transformers.models.musicgen.configuration_musicgen import MusicgenDecoderConfig
    
    # Store original __init__
    _original_init = MusicgenDecoderConfig.__init__
    
    def _patched_init(self, *args, **kwargs):
        """Patched __init__ that adds the missing 'decoder' attribute"""
        _original_init(self, *args, **kwargs)
        # Add the missing attribute that transformers 5.2.0 expects
        self.decoder = None
        if not hasattr(self, 'cross_attention_hidden_size'):
            self.cross_attention_hidden_size = None
    
    # Replace __init__ with patched version
    MusicgenDecoderConfig.__init__ = _patched_init
    print("[OK] Applied monkey patch to MusicgenDecoderConfig")
    
except Exception as e:
    print(f"[WARN] Patch attempt failed: {e}")
    print("Proceeding anyway...")

# Now proceed with imports and testing
print("\n" + "=" * 70)
print("TEST: MusicGen Model Loading WITH PATCH")
print("=" * 70)

import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

MODEL_NAME = "facebook/musicgen-small"

print(f"\n[DEVICE] Using: {torch.device('cpu' if not torch.cuda.is_available() else 'cuda')}")
print(f"[INFO] PyTorch: {torch.__version__}")
print(f"[NOTE] First run will download model (~5-10 GB)...")

try:
    print(f"\n[LOADING] Loading processor from {MODEL_NAME}...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    print("[OK] Processor loaded successfully")
    
    print(f"\n[LOADING] Loading model from {MODEL_NAME}...")
    model = MusicgenForConditionalGeneration.from_pretrained(MODEL_NAME)
    print("[OK] Model loaded successfully!")
    
    print("\n" + "=" * 70)
    print("SUCCESS: MusicGen model ready for generation!")
    print("=" * 70)
    print(f"\nModel: {MODEL_NAME}")
    print(f"Device: {next(model.parameters()).device}")
    print(f"Model size: {sum(p.numel() for p in model.parameters()):,} parameters")
    
except AttributeError as e:
    print(f"\n[ERROR] AttributeError (patch didn't work): {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n[ERROR] Error: {type(e).__name__}: {e}")
    sys.exit(1)
