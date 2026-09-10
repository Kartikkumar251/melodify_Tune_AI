#!/usr/bin/env python3
"""
Check what's in the config.json on HuggingFace for musicgen-small
"""

import json
from pathlib import Path
from huggingface_hub import hf_hub_download

print("=" * 70)
print("Checking facebook/musicgen-small config.json from HuggingFace Hub")
print("=" * 70)

try:
    # Download the config.json
    config_path = hf_hub_download(
        repo_id="facebook/musicgen-small",
        filename="config.json"
    )
    
    print(f"\n[Downloaded] Config file location: {config_path}\n")
    
    # Read and print it
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    print("[Config Contents]")
    print(f"model_type:  {config_dict.get('model_type', 'NOT SET')}")
    print(f"Keys: {list(config_dict.keys())}")
    
    # Print full config
    print("\n[Full Config JSON]")
    print(json.dumps(config_dict, indent=2)[:1000])
    
except Exception as e:
    print(f"[ERROR] {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
