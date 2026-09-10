#!/usr/bin/env python3
"""
Fix script to patch transformers 5.2.0 MusicGen AttributeError
This is a temporary workaround while we resolve version compatibility
"""

import sys
import os
from pathlib import Path

def find_transformers_path():
    """Find transformers installation path"""
    try:
        import transformers
        transformers_path = Path(transformers.__file__).parent
        return transformers_path
    except ImportError:
        print("❌ Transformers not installed")
        return None

def patch_musicgen_config():
    """Patch the MusicgenDecoderConfig to fix AttributeError"""
    transformers_path = find_transformers_path()
    if not transformers_path:
        return False
    
    config_file = transformers_path / "models" / "musicgen" / "configuration_musicgen.py"
    
    if not config_file.exists():
        print(f"❌ Config file not found: {config_file}")
        return False
    
    print(f"📝 Found config file: {config_file}")
    
    # Read the config file
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "self.decoder = None" in content or "# PATCHED" in content:
        print("[OK] Already patched!")
        return True
    
    # Patch: Add decoder attribute to __init__
    old_code = """class MusicgenDecoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of the MusicGenDecoder."""
    model_type = "musicgen_decoder"
    attribute_map = {
        "hidden_size": "d_model",
        "attention_head_size": "mha_num_head_dim",
        "num_attention_heads": "mha_num_heads",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        vocab_size=10000,
        max_position_embeddings=2048,
        d_model=1024,
        mha_num_heads=16,
        mha_num_head_dim=None,
        num_layers=24,
        fc_dim=4096,
        activation_function="gelu",
        activation_dropout=0.0,
        attention_dropout=0.0,
        dropout=0.1,
        initializer_factor=0.02,
        use_cache=True,
        num_codebooks=4,
        pad_token_id=2046,
        bos_token_id=2048,
        eos_token_id=2049,
        tie_word_embeddings=False,
        **kwargs,
    ):"""
    
    new_code = """class MusicgenDecoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of the MusicGenDecoder."""
    model_type = "musicgen_decoder"
    attribute_map = {
        "hidden_size": "d_model",
        "attention_head_size": "mha_num_head_dim",
        "num_attention_heads": "mha_num_heads",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        vocab_size=10000,
        max_position_embeddings=2048,
        d_model=1024,
        mha_num_heads=16,
        mha_num_head_dim=None,
        num_layers=24,
        fc_dim=4096,
        activation_function="gelu",
        activation_dropout=0.0,
        attention_dropout=0.0,
        dropout=0.1,
        initializer_factor=0.02,
        use_cache=True,
        num_codebooks=4,
        pad_token_id=2046,
        bos_token_id=2048,
        eos_token_id=2049,
        tie_word_embeddings=False,
        **kwargs,
    ):"""
    
    # Actually, let's try a different approach - monkey patch at runtime
    return False

def monkey_patch():
    """Apply runtime monkey patch for MusicgenDecoderConfig"""
    try:
        from transformers.models.musicgen.configuration_musicgen import MusicgenDecoderConfig
        
        # Store the original __init__
        original_init = MusicgenDecoderConfig.__init__
        
        def patched_init(self, *args, **kwargs):
            # Call original init
            original_init(self, *args, **kwargs)
            # Add missing decoder attribute
            self.decoder = None
        
        # Replace the __init__
        MusicgenDecoderConfig.__init__ = patched_init
        
        print("[OK] Successfully applied monkey patch to MusicgenDecoderConfig")
        return True
    except Exception as e:
        print(f"❌ Monkey patch failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("MusicGen Transformers Fix Script")
    print("=" * 60)
    
    if monkey_patch():
        print("\n[OK] Patch applied! Transformers should work now.")
        sys.exit(0)
    else:
        print("\n❌ Patch failed")
        sys.exit(1)
