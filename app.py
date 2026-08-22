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
                edge-tts indic-transliteration

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
"""

import asyncio
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
        from indic_transliteration.sanscript import transliterate
    except ImportError:
        logger.warning("indic_transliteration not installed - skipping Hinglish transliteration")
        return cues

    out = []
    for cue in cues:
        roman = transliterate(cue["text"], sanscript.DEVANAGARI, sanscript.ITRANS)
        out.append({**cue, "text": roman})
    return out


# ============================================================================
# MODEL 1 PIPELINE - URL TO SHORT
# ============================================================================

def download_youtube_video(url: str, out_dir: str) -> str:
    """Downloads YouTube video with multi-instance fallback to avoid DNS and Bot errors."""
    import os
    import requests

    output_path = os.path.join(out_dir, "input_video.mp4")
    
    # List of active public Cobalt API instances to try sequentially
    instances = [
        "https://cobalt-api.kwiatekmom.tokyo/",
        "https://api.cobalt.red/",
        "https://cobalt.api.sc3.io/"
    ]

    payload = {
        "url": url,
        "videoQuality": "720",
        "downloadMode": "auto"
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    for instance in instances:
        try:
            res = requests.post(instance, json=payload, headers=headers, timeout=10)
            if res.status_code == 200:
                data = res.json()
                if data.get("status") in ["redirect", "tunnel", "picker"]:
                    video_url = data.get("url")
                    video_bytes = requests.get(video_url, stream=True, timeout=15)
                    with open(output_path, "wb") as f:
                        for chunk in video_bytes.iter_content(chunk_size=8192):
                            f.write(chunk)
                    return output_path
        except Exception:
            continue  # Try the next instance if one fails or times out

    raise Exception("All download instances failed. Please check video URL or try again.")
    """Lazily loads (and caches) the whisper model on first use - loading it
    at import time would slow down every reload/worker boot for no reason."""
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            logger.info("Loading whisper model '%s' (first request only)...", WHISPER_MODEL_SIZE)
            _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
        return _whisper_model


def transcribe_audio(audio_path: str) -> List[dict]:
    """Runs whisper with word-level timestamps enabled so downstream caption
    chunking can align tightly with speech rather than whole-sentence blocks."""
    model = get_whisper_model()
    result = model.transcribe(audio_path, word_timestamps=True, fp16=False)
    return result["segments"]  # each: {start, end, text, words: [...]}


def select_highlight_window(segments: List[dict], target_duration: int) -> (float, float):
    """Greedy heuristic: slide a `target_duration`-second window across the
    transcript and keep the one with the highest spoken-word density. This
    stands in for a real "best moment" / virality model - swap it for a
    learned scorer (audio energy, laughter detection, an LLM ranking
    transcript chunks, etc.) once you have data to train one on."""
    if not segments:
        return 0.0, float(target_duration)

    total_end = segments[-1]["end"]
    if total_end <= target_duration:
        return 0.0, total_end

    best_start, best_score = 0.0, -1
    t = 0.0
    step = 1.0
    while t + target_duration <= total_end:
        window_words = sum(
            len(s["text"].split()) for s in segments if s["start"] >= t and s["end"] <= t + target_duration
        )
        if window_words > best_score:
            best_score, best_start = window_words, t
        t += step

    return best_start, best_start + target_duration


def translate_cues(cues: List[dict], target_lang_code: str) -> List[dict]:
    """Translates each caption cue independently (rather than the whole
    transcript at once) so timestamps stay aligned with translated text.
    Note: for long videos, batch these calls or cache repeated phrases -
    one HTTP round-trip per cue is fine for short-form clips only."""
    translator = GoogleTranslator(source="auto", target=target_lang_code)
    translated = []
    for cue in cues:
        try:
            text = translator.translate(cue["text"])
        except Exception:
            logger.exception("Translation failed for cue, keeping original text")
            text = cue["text"]
        translated.append({**cue, "text": text or cue["text"]})
    return translated


def render_url_short(video_path: str, ass_path: str, start: float, end: float, out_path: str) -> None:
    """Crops the source to a centered 9:16 frame, trims to the selected
    highlight window, and burns the (already time-shifted) subtitle track -
    all in a single ffmpeg pass."""
    duration = end - start
    vf = f"crop=ih*9/16:ih,scale={VIDEO_WIDTH}:{VIDEO_HEIGHT},ass={escape_ffmpeg_path(ass_path)}"
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start), "-i", video_path, "-t", str(duration),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]
    run_subprocess(cmd)


# ============================================================================
# MODEL 2 PIPELINE - SCRIPT TO SHORT
# ============================================================================

def expand_prompt_to_script(prompt: str, style: str) -> str:
    """STUB: expands a short topic/prompt into a narrated script.
    In production, replace this with a call to an LLM (Claude/GPT) using a
    style-specific system prompt so tone actually matches
    Village/Storytelling/High-Energy. Kept local here so the pipeline has
    no external dependency beyond translation + TTS while that's wired up."""
    style_openers = {
        "Village/Traditional": "Ek chhote se gaon mein, ",
        "Storytelling": "It all began on a quiet morning, when ",
        "High Energy": "You will NOT believe what happened when ",
    }
    opener = style_openers.get(style, "")
    return f"{opener}{prompt.strip()}"


async def synthesize_speech(text: str, voice: str, out_audio_path: str) -> List[dict]:
    """Streams edge-tts audio to disk while collecting WordBoundary events,
    which give us free word-level timestamps in the *target* language -
    no separate forced-alignment step needed for the caption track.
    Offsets/durations arrive in 100-nanosecond units per the edge-tts API."""
    communicate = edge_tts.Communicate(text, voice)
    words: List[dict] = []
    with open(out_audio_path, "wb") as audio_file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                start = chunk["offset"] / 10_000_000
                dur = chunk["duration"] / 10_000_000
                words.append({"start": start, "end": start + dur, "text": chunk["text"]})
    return words


def estimate_word_timings(text: str, words_per_second: float = 2.5) -> List[dict]:
    """Fallback timing model used only when voiceover is switched off, so
    captions still have sane timing to render against a silent track."""
    cues, t = [], 0.0
    for word in text.split():
        dur = 1 / words_per_second
        cues.append({"start": t, "end": t + dur, "text": word})
        t += dur
    return cues


def build_silent_audio(duration: float, out_path: str) -> None:
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-t", str(duration), "-q:a", "9", "-acodec", "libmp3lame", out_path,
    ]
    run_subprocess(cmd)


def build_background_clip(style: str, duration: float, out_path: str) -> None:
    """Placeholder visual layer: a style-tinted solid frame with a slow Ken
    Burns zoom so it doesn't feel static. Swap this for real stock footage
    or a generative b-roll pipeline once you have licensed visual assets -
    everything downstream (captions, audio mux) is agnostic to where this
    file comes from."""
    color, _accent = STYLE_PALETTES.get(style, STYLE_PALETTES["Storytelling"])
    lavfi = (
        f"color=c=0x{color}:s=1350x2400:d={duration}:r=25,"
        f"zoompan=z='min(zoom+0.0006,1.15)':d=1:s={VIDEO_WIDTH}x{VIDEO_HEIGHT}:fps=25"
    )
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", lavfi, "-t", str(duration), out_path]
    run_subprocess(cmd)


def render_script_short(background_path: str, audio_path: str, ass_path: str, out_path: str, duration: float) -> None:
    """Muxes the generated voiceover onto the background clip and burns the
    caption track, trimmed to the audio's actual duration."""
    vf = f"ass={escape_ffmpeg_path(ass_path)}"
    cmd = [
        "ffmpeg", "-y",
        "-i", background_path,
        "-i", audio_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest", "-t", str(duration),
        out_path,
    ]
    run_subprocess(cmd)


# ============================================================================
# JOB RUNNERS
# Each wraps a full pipeline with progress updates (mirroring the stage
# copy shown in the frontend prototype), plus credit refund + cleanup on
# failure. These are plain `def`s so Starlette's BackgroundTasks runs them
# on its threadpool automatically instead of blocking the event loop.
# ============================================================================

def run_url_pipeline_job(job_id: str, user_id: str, youtube_url: str, language: str, duration: int, cost: int) -> None:
    work_dir = tempfile.mkdtemp(prefix=f"reelwala_{job_id}_")
    try:
        JOBS.update(job_id, status=JobStatus.PROCESSING, stage="Fetching source video...", progress=5)
        video_path = download_youtube_video(youtube_url, work_dir)

        JOBS.update(job_id, stage="Transcribing audio...", progress=25)
        audio_path = os.path.join(work_dir, "audio.wav")
        extract_audio(video_path, audio_path)
        segments = transcribe_audio(audio_path)

        JOBS.update(job_id, stage=f"Translating to {language}...", progress=50)
        lang_cfg = LANGUAGE_CONFIG[language]
        cues = chunk_into_cues(segments)
        translated_cues = translate_cues(cues, lang_cfg["translate_code"])
        if language == "Hinglish":
            translated_cues = to_hinglish(translated_cues)

        JOBS.update(job_id, stage="Generating captions...", progress=70)
        start, end = select_highlight_window(segments, duration)
        # Shift caption timestamps so they line up with the trimmed clip,
        # keeping only cues that actually fall inside the selected window.
        shifted_cues = [
            {**c, "start": max(0.0, c["start"] - start), "end": max(0.0, c["end"] - start)}
            for c in translated_cues
            if c["start"] < end and c["end"] > start
        ]
        ass_path = os.path.join(work_dir, "captions.ass")
        build_ass_subtitles(shifted_cues, lang_cfg["font"], ass_path)

        JOBS.update(job_id, stage="Rendering vertical short...", progress=85)
        output_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")
        render_url_short(video_path, ass_path, start, end, output_path)

        JOBS.update(job_id, status=JobStatus.DONE, stage="Done", progress=100, output_path=output_path)
    except Exception as exc:
        logger.exception("url-short job %s failed", job_id)
        CREDITS.refund(user_id, cost)
        JOBS.update(job_id, status=JobStatus.FAILED, error=str(exc), stage="Failed")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def run_script_pipeline_job(
    job_id: str, user_id: str, prompt: str, language: str, style: str, voiceover: bool, cost: int
) -> None:
    work_dir = tempfile.mkdtemp(prefix=f"reelwala_{job_id}_")
    try:
        JOBS.update(job_id, status=JobStatus.PROCESSING, stage="Understanding your story...", progress=5)
        raw_script = expand_prompt_to_script(prompt, style)

        JOBS.update(job_id, stage="Writing script beats...", progress=20)
        lang_cfg = LANGUAGE_CONFIG[language]
        translate_code = lang_cfg["translate_code"]
        translated_script = (
            GoogleTranslator(source="auto", target=translate_code).translate(raw_script)
            if translate_code != "en"
            else raw_script
        )
        translated_script = translated_script or raw_script

        audio_path = os.path.join(work_dir, "voice.mp3")
        if voiceover:
            JOBS.update(job_id, stage="Recording AI voiceover...", progress=40)
            # edge_tts is async; this thread has no running loop (it's a
            # BackgroundTasks worker thread), so asyncio.run() is safe here.
            words = asyncio.run(synthesize_speech(translated_script, lang_cfg["tts_voice"], audio_path))
        else:
            words = estimate_word_timings(translated_script)
            build_silent_audio(words[-1]["end"] if words else 6.0, audio_path)

        duration = max((words[-1]["end"] if words else 6.0), 6.0)

        JOBS.update(job_id, stage=f"Applying {style} style...", progress=60)
        background_path = os.path.join(work_dir, "background.mp4")
        build_background_clip(style, duration, background_path)

        JOBS.update(job_id, stage=f"Translating to {language}...", progress=75)
        caption_cues = chunk_into_cues(words)
        if language == "Hinglish":
            caption_cues = to_hinglish(caption_cues)
        ass_path = os.path.join(work_dir, "captions.ass")
        build_ass_subtitles(caption_cues, lang_cfg["font"], ass_path)

        JOBS.update(job_id, stage="Rendering vertical short...", progress=88)
        output_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")
        render_script_short(background_path, audio_path, ass_path, output_path, duration)

        JOBS.update(job_id, status=JobStatus.DONE, stage="Done", progress=100, output_path=output_path)
    except Exception as exc:
        logger.exception("script-short job %s failed", job_id)
        CREDITS.refund(user_id, cost)
        JOBS.update(job_id, status=JobStatus.FAILED, error=str(exc), stage="Failed")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="Reelwala Processing Engine", version="0.1.0")

# The React prototype runs on a different origin during development -
# lock allow_origins down to your real frontend domain(s) in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/generate/url-short", response_model=JobResponse)
def generate_url_short(req: UrlShortRequest, background_tasks: BackgroundTasks):
    """Model 1: URL to Short. Costs 4 credits, charged up front; refunded
    automatically if the pipeline throws."""
    if not CREDITS.try_charge(req.user_id, COST_URL_SHORT):
        raise HTTPException(status_code=402, detail=f"Insufficient credits. This reel costs {COST_URL_SHORT} credits.")

    job = JOBS.create(user_id=req.user_id, kind="url", credits_charged=COST_URL_SHORT)
    background_tasks.add_task(
        run_url_pipeline_job,
        job.id, req.user_id, str(req.youtube_url), req.language.value, req.duration, COST_URL_SHORT,
    )
    return JobResponse(job_id=job.id, status=job.status, credits_remaining=CREDITS.get_balance(req.user_id))


@app.post("/api/generate/script-short", response_model=JobResponse)
def generate_script_short(req: ScriptShortRequest, background_tasks: BackgroundTasks):
    """Model 2: Script to Short. Costs 9 credits, charged up front; refunded
    automatically if the pipeline throws."""
    if not CREDITS.try_charge(req.user_id, COST_SCRIPT_SHORT):
        raise HTTPException(status_code=402, detail=f"Insufficient credits. This reel costs {COST_SCRIPT_SHORT} credits.")

    job = JOBS.create(user_id=req.user_id, kind="script", credits_charged=COST_SCRIPT_SHORT)
    background_tasks.add_task(
        run_script_pipeline_job,
        job.id, req.user_id, req.prompt, req.language.value, req.style.value, req.voiceover, COST_SCRIPT_SHORT,
    )
    return JobResponse(job_id=job.id, status=job.status, credits_remaining=CREDITS.get_balance(req.user_id))


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    """Poll this from the frontend's progress bar / stage text."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    download_url = f"/api/download/{job_id}" if job.status == JobStatus.DONE else None
    return JobStatusResponse(
        job_id=job.id, status=job.status, stage=job.stage, progress=job.progress,
        download_url=download_url, error=job.error,
    )


@app.get("/api/download/{job_id}")
def download_job(job_id: str):
    """Serves the rendered MP4 once a job is done.
    NOTE: for production, replace this with a signed/expiring URL on
    object storage rather than serving straight off local disk."""
    job = JOBS.get(job_id)
    if job is None or job.status != JobStatus.DONE or not job.output_path:
        raise HTTPException(status_code=404, detail="Rendered file not ready")
    return FileResponse(job.output_path, media_type="video/mp4", filename=f"reelwala_{job_id}.mp4")


@app.get("/api/credits/{user_id}", response_model=CreditsResponse)
def get_credits(user_id: str):
    return CreditsResponse(user_id=user_id, credits=CREDITS.get_balance(user_id))


@app.post("/api/credits/topup", response_model=CreditsResponse)
def topup_credits(req: TopupRequest):
    """Mirrors the mock Razorpay flow in the frontend prototype for local
    testing.
    NOTE: in real production, never credit an account directly off a client
    call like this - verify the Razorpay payment signature server-side (or
    handle it entirely from Razorpay's webhook) before calling
    CREDITS.topup()."""
    plan = CREDIT_PLANS.get(req.plan)
    if plan is None:
        raise HTTPException(status_code=400, detail=f"Unknown plan '{req.plan}'. Valid plans: {list(CREDIT_PLANS)}")

    CREDITS.topup(req.user_id, plan["credits"])
    return CreditsResponse(user_id=req.user_id, credits=CREDITS.get_balance(req.user_id))


@app.get("/healthz")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
