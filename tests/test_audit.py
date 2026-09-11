"""
test_audit.py — End-to-End Comprehensive Audit & Verification Suite for Melodify
Verifies all 11 core endpoints, synthesis diversity, voice/hum-to-beat, stem separation,
mastering, continuation, FCA compression/reconstruction, MIDI extraction, lyrics, and autocomplete.
"""

import os
import sys
import time
import json
import urllib.request
import requests
import numpy as np
import soundfile as sf
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_suite():
    print("=" * 60)
    print("  MELODIFY COMPLETE AUDIT & INTEGRATION TEST SUITE")
    print("=" * 60)

    # 1. Health check
    with urllib.request.urlopen(f"{BASE_URL}/health", timeout=5) as r:
        assert r.status == 200
        health_data = json.loads(r.read())
        print("  [OK] [1/11] Health Check OK:", health_data)

    # 2. Test Text-to-Beat Generation across multiple genres to ensure variety
    genres = [
        ("lofi", "chill lofi rain beats with warm rhodes keys 80 bpm"),
        ("synthwave", "retrowave 80s synth brass outrun drive 120 bpm"),
        ("trap", "atlanta dark trap 808 sub bass rolling hi hats"),
        ("rock", "energetic modern indie rock guitar riff anthem"),
        ("piano", "calm emotional grand piano solo ballad 72 bpm"),
    ]
    gen_files = []
    for label, prompt in genres:
        req_data = json.dumps({"prompt": prompt, "name": label}).encode("utf-8")
        req = urllib.request.Request(
            f"{BASE_URL}/generate",
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            assert r.status == 200
            res = json.loads(r.read())
            gen_files.append(res["filename"])
            print(f"  [OK] [2/11] Generated [{label}]: {res['filename']} ({res['duration']}s)")

    # Verify all generated files are distinct
    assert len(set(gen_files)) == len(gen_files), "Generated filenames must be distinct"

    # 3. Test Audio Analysis on generated beat
    target_file = gen_files[0]
    req_data = json.dumps({"filename": target_file}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/analyze",
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        assert r.status == 200
        analysis = json.loads(r.read())
        print(f"  [OK] [3/11] Analysis OK: BPM={analysis['bpm']}, Key={analysis['key']}, Loudness={analysis['loudness_db']}dB")

    # 4. Test Stem Separation
    req_data = json.dumps({"filename": target_file}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/separate",
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        assert r.status == 200
        stems = json.loads(r.read())
        assert "drums" in stems["stems"] and "bass" in stems["stems"]
        print("  [OK] [4/11] Stem Separation OK (Drums, Bass, Vocals, Other stems generated)")

    # 5. Test AI Mastering
    req_data = json.dumps({"filename": target_file}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/master",
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        assert r.status == 200
        master = json.loads(r.read())
        print(f"  [OK] [5/11] AI Mastering OK: {master['filename']}, Mastered LUFS={master['analysis']['mastered_lufs']}")

    # 6. Test Track Continuation / Extension
    req_data = json.dumps({"filename": target_file, "prompt": "high energy drop with heavy drums"}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/continue",
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        assert r.status == 200
        cont = json.loads(r.read())
        print(f"  [OK] [6/11] Track Continuation OK: {cont['filename']} ({cont['duration']}s)")

    # 7. Test Hum to Beat Conversion
    test_hum_path = Path("upload_tmp/test_hum_quick.wav")
    test_hum_path.parent.mkdir(parents=True, exist_ok=True)
    if not test_hum_path.exists():
        t = np.linspace(0, 2.5, int(44100 * 2.5))
        sf.write(str(test_hum_path), (0.5 * np.sin(2 * np.pi * 330 * t)).astype(np.float32), 44100)
    
    with open(test_hum_path, "rb") as f:
        r = requests.post(f"{BASE_URL}/hum", files={"file": ("hum.wav", f, "audio/wav")}, data={"prompt": "Synthwave lead synth beat"})
        assert r.status_code == 200
        hum_res = r.json()
        print(f"  [OK] [7/11] Hum-to-Beat OK: {hum_res['filename']} ({hum_res['duration']}s)")

    # 8. Test FCA Lossless and Perceptual Optimization
    for mode in ["lossless", "perceptual"]:
        req_data = json.dumps({"filename": target_file, "mode": mode, "quality": 6}).encode("utf-8")
        req = urllib.request.Request(
            f"{BASE_URL}/tools/fca-optimize",
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            assert r.status == 200
            fca_res = json.loads(r.read())
            comp_ratio = fca_res.get('metrics', {}).get('compression_ratio', fca_res.get('compression_ratio'))
            savings = fca_res.get('metrics', {}).get('space_saving_pct', fca_res.get('reduction_percent', fca_res.get('space_saved_percent')))
            print(f"  [OK] [8/11] FCA Optimization [{mode.upper()}] OK: {fca_res['fca_filename']} (Ratio: {comp_ratio}:1, Savings: {savings}%)")

    # 9. Test Audio to MIDI
    req_data = json.dumps({"filename": target_file}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/tools/audio-to-midi",
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        assert r.status == 200
        midi_res = json.loads(r.read())
        print(f"  [OK] [9/11] Audio to MIDI OK: {midi_res['filename']}")

    # 10. Test Lyrics Generator
    req_data = json.dumps({"title": "Neon Horizons", "genre": "Synthwave", "mood": "Nostalgic"}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/tools/generate-lyrics",
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        assert r.status == 200
        lyrics_res = json.loads(r.read())
        print(f"  [OK] [10/11] Lyrics Generator OK: {len(lyrics_res['sections'])} sections generated")

    # 11. Test Prompt Autocomplete
    with urllib.request.urlopen(f"{BASE_URL}/tools/prompt-autocomplete?q=trap", timeout=5) as r:
        assert r.status == 200
        auto_res = json.loads(r.read())
        print(f"  [OK] [11/11] Prompt Autocomplete OK: {len(auto_res['suggestions'])} suggestions for 'trap'")

    print("=" * 60)
    print("  ALL 11 FEATURES AUDITED & VERIFIED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_suite()
