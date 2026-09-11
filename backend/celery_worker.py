"""
celery_worker.py — Asynchronous Task Execution Queue for BeatFlow AI
Handles background heavy compute operations:
  - Generative AI beat synthesis
  - Demucs stem separation
  - Librosa acoustic feature extraction
Includes automated in-memory fakeredis fallback if external Redis broker is not running.
"""
from __future__ import annotations
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

_BACKEND_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _BACKEND_DIR.parent
for _p in [str(_BACKEND_DIR), str(_ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from celery import Celery


def _setup_broker() -> Tuple[str, str]:
    """
    Initializes Redis broker connection with automatic fallback to in-memory fake server.
    """
    real_broker = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    real_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

    try:
        import redis as _redis_lib
        r = _redis_lib.Redis.from_url(real_broker, socket_connect_timeout=1, socket_timeout=1)
        r.ping()
        return real_broker, real_backend
    except Exception:
        pass

    try:
        import fakeredis
        from fakeredis import FakeServer

        _fake_server = FakeServer()
        _fake_server.connected = True

        import redis

        def _fake_from_url(url: str, **kwargs: Any) -> fakeredis.FakeRedis:
            return fakeredis.FakeRedis(server=_fake_server, decode_responses=False)

        redis.Redis.from_url = staticmethod(_fake_from_url)
        redis.StrictRedis.from_url = staticmethod(_fake_from_url)

        fake_broker = "memory://localhost/"
        fake_backend = "cache+memory://"
        return fake_broker, fake_backend
    except ImportError:
        return real_broker, real_backend


BROKER, BACKEND = _setup_broker()

celery_app = Celery(
    "beatflow",
    broker=BROKER,
    backend=BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_time_limit=300,
    task_soft_time_limit=240,
    task_always_eager=False,
    broker_transport_options={"polling_interval": 0.1},
)


def _gpu_context():
    """Detect GPU availability and precision."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    return device, dtype


# ── Task 1: Generate Beat ─────────────────────────────────────────
@celery_app.task(bind=True, name="beatflow.generate_beat")
def generate_beat_task(self, prompt: str, label: str, commit_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Async MusicGen generation task.
    """
    self.update_state(state="PROGRESS", meta={"step": "loading model"})
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    import torch
    import soundfile as sf

    device, dtype = _gpu_context()
    t0 = time.time()
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained(
        "facebook/musicgen-small", torch_dtype=dtype
    ).to(device)
    model.eval()

    self.update_state(state="PROGRESS", meta={"step": "generating"})
    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)

    with torch.inference_mode():
        with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
            output = model.generate(**inputs, max_new_tokens=512)

    audio_np = output[0, 0].cpu().float().numpy()
    sample_rate = model.config.audio_encoder.sampling_rate
    duration = len(audio_np) / sample_rate

    ts = datetime.now().strftime("%H%M%S")
    filename = f"beat_{ts}.wav"
    out_path = Path("beat_outputs") / filename
    out_path.parent.mkdir(exist_ok=True)
    sf.write(str(out_path), audio_np, sample_rate)
    elapsed = round(time.time() - t0, 1)

    result = {
        "audio_url": f"/audio/{filename}",
        "duration": round(duration, 2),
        "elapsed": elapsed,
        "device": str(device),
    }

    if commit_id:
        try:
            from database import SessionLocal
            from models import Commit
            db = SessionLocal()
            commit = db.query(Commit).filter(Commit.id == commit_id).first()
            if commit:
                commit.audio_url = result["audio_url"]
                commit.duration = result["duration"]
                commit.elapsed_sec = elapsed
                db.commit()
            db.close()
        except Exception as e:
            pass

    return result


# ── Task 2: Stem Separation ───────────────────────────────────────
@celery_app.task(bind=True, name="beatflow.separate_stems")
def separate_stems_task(self, audio_path: str, commit_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Async Demucs stem separation task.
    """
    self.update_state(state="PROGRESS", meta={"step": "separating stems"})
    from audio_processing import separate_stems

    stems = separate_stems(audio_path)
    stems_dir = Path("stems_outputs")

    stem_urls: Dict[str, str] = {}
    for name, path in stems.items():
        rel = Path(path).relative_to(stems_dir)
        stem_urls[name] = f"/stems/{rel.as_posix()}"

    result = {"stems": stem_urls}

    if commit_id:
        try:
            from database import SessionLocal
            from models import Stem
            db = SessionLocal()
            for stem_type, url in stem_urls.items():
                s = Stem(commit_id=commit_id, type=stem_type, audio_url=url)
                db.add(s)
            db.commit()
            db.close()
        except Exception as e:
            pass

    return result


# ── Task 3: Analyze Audio ─────────────────────────────────────────
@celery_app.task(bind=True, name="beatflow.analyze_audio")
def analyze_audio_task(self, audio_path: str, commit_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Async Librosa audio analysis task.
    """
    self.update_state(state="PROGRESS", meta={"step": "analyzing"})
    from audio_processing import analyze_audio

    result = analyze_audio(audio_path)

    if commit_id:
        try:
            from database import SessionLocal
            from models import Commit
            db = SessionLocal()
            c = db.query(Commit).filter(Commit.id == commit_id).first()
            if c:
                c.bpm = result.get("bpm")
                c.key = result.get("key")
                c.energy = result.get("energy")
                db.commit()
            db.close()
        except Exception as e:
            pass

    return result
