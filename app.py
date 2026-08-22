"""
Reelwala - Video Processing Engine (app.py)
=============================================
Standalone FastAPI backend implementing both Reelwala generation pipelines
from the product prototype:

    Model 1 - URL to Short    : YouTube/Reel link -> translated, captioned 9:16 clip
    Model 2 - Script to Short : text prompt        -> AI-narrated, captioned 9:16 clip

Run locally
-----------
    pip install fastapi "uvicorn[standard]" python-multipart pydantic \
                yt-dlp openai-whisper torch deep-translator \
                edge-tts indic-transliteration requests

    # ffmpeg is a SYSTEM binary, not a pip package - install it separately:
    #   macOS:   brew install ffmpeg
    #   Ubuntu:  sudo apt install ffmpeg
    #   Windows: https://ffmpeg.org/download.html

    uvicorn app:app --reload --port 8000

Architecture notes
-------------------
This is a single-process prototype: job status and credit balances live
in memory (JobStore / CreditStore below), and the heavy pipeline work
runs on FastAPI/Starlette's background threadpool (BackgroundTasks
automatically off-loads plain `def` callables to a worker thread, so a
whisper transcription or an ffmpeg render never blocks the event loop).

For real production traffic you would swap:
  - in-memory JobStore/CreditStore  -> Postgres/Redis-backed tables
  - BackgroundTasks                 -> Celery/RQ workers (ideally GPU boxes
                                        for whisper + edge-tts)
  - direct credit top-up endpoint   -> a verified Razorpay webhook
  - local OUTPUT_DIR file serving   -> signed, expiring URLs on S3/GCS
The structure below is written so each of those is a drop-in swap - the
pipeline functions don't know or care how they were scheduled.

YouTube download notes
-----------------------
download_youtube_video() below tries multiple independent mechanisms in
order of reliability (see its docstring) instead of depending on any
single third-party server. At minimum, set YOUTUBE_COOKIES_FILE in
production - that's the one fix that resolves "Sign in to confirm you're
not a bot" for most cloud hosts. Full setup instructions are in that
function's docstring.
"""

import asyncio
import glob
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import edge_tts
import requests
import whisper
import yt_dlp
from deep_translator import GoogleTranslator
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, HttpUrl, field_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("reelwala")

# ============================================================================
# CONFIG
# ============================================================================

OUTPUT_DIR = os.path.join(os.getcwd(), "reelwala_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WHISPER_MODEL_SIZE = "small"   # tiny / base / small / medium / large-v3
STARTING_CREDITS = 9           # mirrors the frontend prototype's default balance

COST_URL_SHORT = 4
COST_SCRIPT_SHORT = 9

# Mirrors the ₹99 / ₹199 plans from the pricing sheet in the React prototype.
CREDIT_PLANS = {
    "basic": {"price_inr": 99, "credits": 15},
    "pro": {"price_inr": 199, "credits": 40},
}

# Per-language config: the machine-translation code, the edge-tts regional
# neural voice, and a font capable of rendering that script for burned-in
# subtitles. Hinglish reuses Hindi translation + a Hindi voice, then gets
# transliterated to Roman script at the caption layer (see to_hinglish()).
LANGUAGE_CONFIG = {
    "Hindi":    {"translate_code": "hi", "tts_voice": "hi-IN-SwaraNeural",    "font": "Noto Sans Devanagari"},
    "Tamil":    {"translate_code": "ta", "tts_voice": "ta-IN-PallaviNeural",  "font": "Noto Sans Tamil"},
    "Telugu":   {"translate_code": "te", "tts_voice": "te-IN-ShrutiNeural",   "font": "Noto Sans Telugu"},
    "Kannada":  {"translate_code": "kn", "tts_voice": "kn-IN-SapnaNeural",    "font": "Noto Sans Kannada"},
    "Bengali":  {"translate_code": "bn", "tts_voice": "bn-IN-TanishaaNeural", "font": "Noto Sans Bengali"},
    "Marathi":  {"translate_code": "mr", "tts_voice": "mr-IN-AarohiNeural",   "font": "Noto Sans Devanagari"},
    "Hinglish": {"translate_code": "hi", "tts_voice": "hi-IN-SwaraNeural",    "font": "Noto Sans"},
}

# Placeholder background palette per visual style, used until a real
# stock-footage / generative b-roll pipeline is wired in (see
# build_background_clip).
STYLE_PALETTES = {
    "Village/Traditional": ("8B4513", "D2A679"),
    "Storytelling": ("1B2A4A", "4A6FA5"),
    "High Energy": ("FF2E63", "FFB627"),
}

VIDEO_WIDTH, VIDEO_HEIGHT = 1080, 1920  # 9:16 output resolution

# --- YouTube download config (see download_youtube_video docstring) ---
YOUTUBE_COOKIES_FILE = os.environ.get("YOUTUBE_COOKIES_FILE")
YOUTUBE_PROXY = os.environ.get("YOUTUBE_PROXY")
COBALT_API_URL = os.environ.get("COBALT_API_URL")
COBALT_API_KEY = os.environ.get("COBALT_API_KEY")


# ============================================================================
# ENUMS & REQUEST/RESPONSE SCHEMAS
# ============================================================================

class Language(str, Enum):
    HINDI = "Hindi"
    TAMIL = "Tamil"
    TELUGU = "Telugu"
    KANNADA = "Kannada"
    BENGALI = "Bengali"
    MARATHI = "Marathi"
    HINGLISH = "Hinglish"


class VisualStyle(str, Enum):
    VILLAGE = "Village/Traditional"
    STORYTELLING = "Storytelling"
    HIGH_ENERGY = "High Energy"


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class UrlShortRequest(BaseModel):
    user_id: str
    youtube_url: HttpUrl
    language: Language
    duration: int = Field(..., description="Target duration in seconds (15, 30, or 60)")

    @field_validator("duration")
    @classmethod
    def validate_duration(cls, v: int) -> int:
        if v not in (15, 30, 60):
            raise ValueError("duration must be one of 15, 30, 60")
        return v


class ScriptShortRequest(BaseModel):
    user_id: str
    prompt: str = Field(..., min_length=5, max_length=2000)
    language: Language
    style: VisualStyle
    voiceover: bool = True


class TopupRequest(BaseModel):
    user_id: str
    plan: str  # "basic" | "pro"


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    credits_remaining: int


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    stage: str
    progress: int
    download_url: Optional[str] = None
    error: Optional[str] = None


class CreditsResponse(BaseModel):
    user_id: str
    credits: int


# ============================================================================
# IN-MEMORY STORES (credits + job tracking)
# Both are simple lock-guarded dicts. See the "Architecture notes" docstring
# above for how to swap these for real persistence.
# ============================================================================

@dataclass
class Job:
    id: str
    user_id: str
    kind: str  # "url" | "script"
    status: JobStatus = JobStatus.QUEUED
    stage: str = "Queued..."
    progress: int = 0
    output_path: Optional[str] = None
    error: Optional[str] = None
    credits_charged: int = 0
    created_at: float = field(default_factory=time.time)


class JobStore:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, user_id: str, kind: str, credits_charged: int) -> Job:
        job = Job(id=str(uuid.uuid4()), user_id=user_id, kind=kind, credits_charged=credits_charged)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in fields.items():
                setattr(job, key, value)

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)


class CreditStore:
    """Thread-safe credit ledger. Every operation takes a lock so concurrent
    requests from the same user (e.g. two tabs generating at once) can never
    both pass a balance check that only one of them should have passed."""

    def __init__(self, starting_balance: int):
        self._balances: Dict[str, int] = {}
        self._starting_balance = starting_balance
        self._lock = threading.Lock()

    def _ensure(self, user_id: str) -> None:
        if user_id not in self._balances:
            self._balances[user_id] = self._starting_balance

    def get_balance(self, user_id: str) -> int:
        with self._lock:
            self._ensure(user_id)
            return self._balances[user_id]

    def try_charge(self, user_id: str, amount: int) -> bool:
        """Atomically deduct credits. Returns False (no state change) if the
        user doesn't have enough - this is the check that should trigger the
        frontend's recharge sheet."""
        with self._lock:
            self._ensure(user_id)
            if self._balances[user_id] < amount:
                return False
            self._balances[user_id] -= amount
            return True

    def refund(self, user_id: str, amount: int) -> None:
        with self._lock:
            self._ensure(user_id)
            self._balances[user_id] += amount

    def topup(self, user_id: str, amount: int) -> None:
        with self._lock:
            self._ensure(user_id)
            self._balances[user_id] += amount


JOBS = JobStore()
CREDITS = CreditStore(starting_balance=STARTING_CREDITS)


# ============================================================================
# GENERIC HELPERS (shared by both pipelines)
# ============================================================================

def run_subprocess(cmd: List[str]) -> subprocess.CompletedProcess:
    """Runs an external command (ffmpeg, etc.), raising with the captured
    stderr on failure so job errors are actually diagnosable from the API."""
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        stderr_tail = result.stderr.decode(errors="ignore")[-2000:]
        raise RuntimeError(f"Command failed ({cmd[0]}): {stderr_tail}")
    return result


def escape_ffmpeg_path(path: str) -> str:
    """ffmpeg's filtergraph parser treats ':' and '\\' as special characters,
    which breaks Windows paths (and colon-containing tmp paths) passed to
    filters like `ass=...` unless escaped first."""
    return path.replace("\\", "/").replace(":", "\\:")


def format_ass_time(seconds: float) -> str:
    """Converts seconds -> ASS subtitle timestamp format H:MM:SS.CC"""
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


ASS_HEADER_TEMPLATE = """[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{font},76,&H00FFFFFF,&H00000000,&H90000000,-1,0,1,3,1,2,60,60,160,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def build_ass_subtitles(cues: List[dict], font_name: str, out_path: str) -> None:
    """Writes an .ass subtitle file with a soft fade-in/out on every cue -
    this is what gives the burned captions their "animated" feel when
    ffmpeg composites them with the `ass=` filter (requires libass)."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(ASS_HEADER_TEMPLATE.format(width=VIDEO_WIDTH, height=VIDEO_HEIGHT, font=font_name))
        for cue in cues:
            if cue["end"] <= cue["start"]:
                continue
            start = format_ass_time(cue["start"])
            end = format_ass_time(cue["end"])
            text = cue["text"].strip().replace("\n", "\\N")
            if not text:
                continue
            # \fad(in_ms, out_ms) - the animated fade the frontend preview mimics
            f.write(f"Dialogue: 0,{start},{end},Caption,,0,0,0,,{{\\fad(180,180)}}{text}\n")


def chunk_into_cues(segments: List[dict], max_chars: int = 42) -> List[dict]:
    """Groups word/segment-level {start, end, text} entries (from whisper or
    edge-tts word-boundary events) into short caption cues, matching the
    punchy 3-line captions shown in the frontend preview rather than dumping
    a whole sentence on screen at once."""
    cues: List[dict] = []
    current_text = ""
    current_start: Optional[float] = None
    current_end: Optional[float] = None

    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        if current_start is None:
            current_start = seg["start"]
        candidate = f"{current_text} {text}".strip()
        if len(candidate) > max_chars and current_text:
            cues.append({"start": current_start, "end": current_end, "text": current_text})
            current_text = text
            current_start = seg["start"]
        else:
            current_text = candidate
        current_end = seg["end"]

    if current_text:
        cues.append({"start": current_start, "end": current_end, "text": current_text})
    return cues


def to_hinglish(cues: List[dict]) -> List[dict]:
    """Transliterates Devanagari cue text to Roman script for the Hinglish
    option. Requires the optional `indic-transliteration` package; if it's
    not installed we fail soft and keep the Devanagari text rather than
    crashing the whole render."""
    try:
        from indic_transliteration import sanscript
