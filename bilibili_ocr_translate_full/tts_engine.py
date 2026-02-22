"""
Text-to-speech engine for subtitle overlay.
Pluggable backends: Piper, XTTS v2, ElevenLabs, OpenAI. Fallback: macOS 'say'.

TTS can run in a separate process (use_subprocess=True) to avoid OCR/TTS CPU contention.
When OCR and TTS run in the same process, Vision OCR + Piper ONNX compete for CPU/GIL
and cause audio stutter. Subprocess isolation fixes this.
"""
import gc
import glob
import io
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import queue
import time

import numpy as np
import sounddevice as sd


def _cleanup_piper_temp():
    """Remove Piper temp files/dirs in /tmp to prevent clogging."""
    try:
        for path in glob.glob("/tmp/piper*"):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    os.remove(path)
                except OSError:
                    pass
    except Exception:
        pass


_PIPER_URL_FMT = "https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang_family}/{lang_code}/{voice_name}/{voice_quality}/{lang_code}-{voice_name}-{voice_quality}{ext}?download=true"
_PIPER_VOICE_RE = re.compile(r"^(?P<lang_family>[^-]+)_(?P<lang_region>[^-]+)-(?P<voice_name>[^-]+)-(?P<voice_quality>.+)$")


def _download_piper_voice_resilient(voice_id, download_dir, max_retries=3):
    """Download Piper voice with retries and proper User-Agent. Returns True on success."""
    m = _PIPER_VOICE_RE.match(voice_id.strip())
    if not m:
        return False
    lang_family = m.group("lang_family")
    lang_code = f"{lang_family}_{m.group('lang_region')}"
    voice_name = m.group("voice_name")
    voice_quality = m.group("voice_quality")
    fmt = {"lang_family": lang_family, "lang_code": lang_code, "voice_name": voice_name, "voice_quality": voice_quality}
    model_path = os.path.join(download_dir, f"{voice_id}.onnx")
    config_path = os.path.join(download_dir, f"{voice_id}.onnx.json")
    if os.path.isfile(model_path) and os.path.isfile(config_path) and os.path.getsize(model_path) > 0:
        return True
    try:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
    except ImportError:
        return False
    session = requests.Session()
    retries = Retry(total=max_retries, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    headers = {"User-Agent": "Mozilla/5.0 (compatible; PiperTTS/1.0)"}
    for ext, path in [(".onnx", model_path), (".onnx.json", config_path)]:
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            continue
        url = _PIPER_URL_FMT.format(ext=ext, **fmt)
        for attempt in range(max_retries):
            try:
                r = session.get(url, headers=headers, timeout=60)
                r.raise_for_status()
                with open(path, "wb") as f:
                    f.write(r.content)
                break
            except (requests.RequestException, OSError) as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
    return os.path.isfile(model_path) and os.path.getsize(model_path) > 0


# --- Persistent audio player: single stream, no churn, no concurrency ---
# One thread owns sounddevice. All TTS backends submit via this. Eliminates
# stream open/close per utterance and overlapping playback (machine gun stutter).

_TTS_SAMPLERATE = 22050
_TTS_CHANNELS = 1


class _PersistentAudioPlayer:
    """Single persistent OutputStream. Only place that touches sounddevice."""

    def __init__(self):
        self._q = queue.Queue()
        self._running = True
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def play(self, data, samplerate):
        """Enqueue float32 (n,1) audio. Resamples to _TTS_SAMPLERATE if needed."""
        if data is None or len(data) == 0:
            return
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.ndim == 2 and data.shape[1] > 1:
            data = data.mean(axis=1, keepdims=True)
        data = np.ascontiguousarray(data)
        sr = int(samplerate)
        if sr != _TTS_SAMPLERATE:
            try:
                from scipy.signal import resample
                n_new = int(data.shape[0] * _TTS_SAMPLERATE / sr)
                data = resample(data, n_new, axis=0).astype(np.float32)
            except Exception:
                pass
        self._q.put(data)

    def stop(self):
        """Clear pending and insert stop marker. Stream stays open."""
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        self._q.put(None)

    def _run(self):
        while self._running:
            try:
                with sd.OutputStream(
                    samplerate=_TTS_SAMPLERATE,
                    channels=_TTS_CHANNELS,
                    dtype="float32",
                ) as stream:
                    while self._running:
                        item = self._q.get()
                        if item is None:
                            continue
                        # Validate: no NaN/inf, clamp to [-1,1]
                        arr = np.asarray(item, dtype=np.float32)
                        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                            continue
                        arr = np.clip(arr, -1.0, 1.0)
                        arr = np.ascontiguousarray(arr)
                        try:
                            stream.write(arr)
                        except sd.PortAudioError as e:
                            print(f"[TTS] PortAudio error: {e}, restarting stream...", flush=True)
                            break
            except Exception as e:
                print(f"[TTS] Audio stream error: {e}", flush=True)
                time.sleep(0.3)


_audio_player = _PersistentAudioPlayer()
_audio_stopped = False


def _safe_play(data, samplerate):
    """Route all TTS playback through the persistent player."""
    if data is None or len(data) == 0 or _audio_stopped:
        return
    _audio_player.play(data, samplerate)


def _stop_audio():
    """Stop playback and clear queue. Use instead of sd.stop()."""
    global _audio_stopped
    _audio_stopped = True
    _audio_player.stop()
    _audio_stopped = False


def _play_audio_pcm(samples, sample_rate):
    """Play raw PCM int16 samples via sounddevice."""
    if samples is None or len(samples) == 0:
        return
    samples = np.asarray(samples)
    if samples.dtype == np.int16:
        samples = samples.astype(np.float32) / 32768.0
    _safe_play(samples, sample_rate)


def _play_audio_from_bytes(audio_bytes, format_hint="wav"):
    """Play audio from bytes (WAV or MP3) using pydub if available."""
    try:
        from pydub import AudioSegment
        if format_hint == "mp3":
            audio = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
        else:
            audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))
        # Get samples as numpy array
        samples = np.array(audio.get_array_of_samples())
        n_channels = audio.channels
        # Reshape for multi-channel: (n_samples, n_channels)
        if n_channels > 1:
            n_samples = len(samples) // n_channels
            samples = samples.reshape(n_samples, n_channels)
        else:
            samples = samples.reshape(-1, 1)  # mono: (n, 1)
        # Convert to float32 and normalize
        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0
        else:
            samples = samples.astype(np.float32)
        _safe_play(samples, audio.frame_rate)
    except ImportError:
        # Fallback: assume WAV, use wave
        import wave
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
            n_frames = wav.getnframes()
            n_channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            raw_data = wav.readframes(n_frames)
            data = np.frombuffer(raw_data, dtype=np.int16)
            # Reshape for multi-channel: (n_frames, n_channels)
            if n_channels > 1:
                data = data.reshape(n_frames, n_channels)
            else:
                data = data.reshape(-1, 1)  # mono: (n, 1)
            data = data.astype(np.float32) / 32768.0
            _safe_play(data, sample_rate)

# --- Backend: macOS say (fallback) ---

class SayBackend:
    """macOS 'say' command - offline, instant, no deps."""

    def __init__(self):
        self.current_process = None
        self.voices = {"zh": "Ting-Ting", "en": "Samantha", "ja": "Kyoko"}

    def speak(self, text, lang="en"):
        if not text or not str(text).strip():
            return
        voice = self.voices.get(lang, "Samantha")
        self.current_process = subprocess.Popen(
            ["say", "-v", voice, "-r", "180", str(text).strip()],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.current_process.wait()
        self.current_process = None

    def stop(self):
        if self.current_process:
            try:
                self.current_process.terminate()
                self.current_process.wait(timeout=1)
            except Exception:
                pass
            self.current_process = None


# --- Backend: Piper (local) ---

DEFAULT_PIPER_VOICE = "en_US-lessac-medium"
DEFAULT_PIPER_VOICE_DIR = os.path.expanduser("~/.local/share/piper/voices")

# Piper voices: (display_name, model_id)
PIPER_VOICES = [
    # English (US)
    ("Lessac (US, medium)", "en_US-lessac-medium"),
    ("Lessac (US, low)", "en_US-lessac-low"),
    ("Ryan (US, medium)", "en_US-ryan-medium"),
    ("Amy (US, medium)", "en_US-amy-medium"),
    ("Danny (US, low)", "en_US-danny-low"),
    # English (UK)
    ("Alan (UK, medium)", "en_GB-alan-medium"),
    ("Cori (UK, medium)", "en_GB-cori-medium"),
    ("Southern English Female (UK, low)", "en_GB-southern_english_female-low"),
    # German
    ("Eva (DE, x_low)", "de_DE-eva_k-x_low"),
    # French
    ("Gilles (FR, low)", "fr_FR-gilles-low"),
    # Spanish
    ("Sharvard (ES, medium)", "es_ES-sharvard-medium"),
    # Italian
    ("Riccardo (IT, x_low)", "it_IT-riccardo-x_low"),
]


# Voices that no longer exist on Hugging Face - map to valid alternative
PIPER_VOICE_FALLBACKS = {"en_US-danny-medium": "en_US-danny-low"}

class PiperBackend:
    """Piper TTS - local, ultra-lightweight. Auto-downloads voice on first use."""

    def __init__(self, voice_id=None, status_callback=None):
        raw = voice_id or DEFAULT_PIPER_VOICE
        self.voice_id = PIPER_VOICE_FALLBACKS.get(raw, raw)
        self.voice = None
        self.status_callback = status_callback
        self._load_model()

    def _find_model_path(self):
        """Find Piper model - check common locations. Only returns the requested voice_id, never a random fallback."""
        candidates = [
            os.path.join(DEFAULT_PIPER_VOICE_DIR, f"{self.voice_id}.onnx"),
            os.path.expanduser(f"~/.piper/voices/{self.voice_id}.onnx"),
            os.path.join(os.path.dirname(__file__), "piper_voices", f"{self.voice_id}.onnx"),
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p
        return None

    def _download_voice(self):
        """Download Piper voice - use resilient requests-based download, fallback to piper CLI."""
        os.makedirs(DEFAULT_PIPER_VOICE_DIR, exist_ok=True)
        print(f"[TTS] Downloading Piper voice ({self.voice_id})...")
        if self.status_callback:
            try:
                self.status_callback("Downloading Piper voice...")
            except Exception:
                pass
        try:
            if _download_piper_voice_resilient(self.voice_id, DEFAULT_PIPER_VOICE_DIR):
                print("[TTS] Piper voice downloaded.")
                return
        except Exception as e:
            print(f"[TTS] Resilient download failed: {e}, trying piper CLI...")
        try:
            subprocess.run(
                [sys.executable, "-m", "piper.download_voices", "--data-dir", DEFAULT_PIPER_VOICE_DIR, self.voice_id],
                check=True,
            )
            print("[TTS] Piper voice downloaded.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to download Piper voice: {e}")
        except FileNotFoundError:
            raise RuntimeError("piper-tts not installed. Run: pip install piper-tts")

    def _load_model(self):
        try:
            from piper import PiperVoice
            path = self._find_model_path()
            if not path:
                self._download_voice()
                path = self._find_model_path()
            if path:
                self.voice = PiperVoice.load(path)
            else:
                self.voice = None
        except ImportError:
            self.voice = None

    def speak(self, text, lang="en"):
        if not text or not str(text).strip():
            return
        if not self.voice:
            raise RuntimeError("Piper model not found. Run: pip install piper-tts")
        text = str(text).strip()

        # --- generate audio ---
        full_audio = b""
        sample_rate = 22050

        if hasattr(self.voice, "synthesize_stream_raw"):
            chunks = []
            for chunk in self.voice.synthesize_stream_raw(text):
                chunks.append(chunk)
            full_audio = b"".join(chunks)
            if hasattr(self.voice, "config"):
                sample_rate = getattr(self.voice.config, "sample_rate", 22050)
            print(f"[Piper] full_audio len: {len(full_audio)}, sample_rate: {sample_rate}", flush=True)
        else:
            audio_buffer = io.BytesIO()
            import wave
            with wave.open(audio_buffer, "wb") as wav_file:
                self.voice.synthesize_wav(text, wav_file)
            audio_buffer.seek(0)
            with wave.open(audio_buffer, "rb") as wav:
                sample_rate = wav.getframerate()
                full_audio = wav.readframes(wav.getnframes())
            print(f"[Piper] wav sample_rate: {sample_rate}, bytes len: {len(full_audio)}", flush=True)

        if not full_audio:
            print("[Piper] no audio produced", flush=True)
            return

        # --- convert/play via persistent player (no sd.play/sd.stop churn) ---
        data = np.frombuffer(full_audio, dtype=np.int16)
        data = data.astype(np.float32) / 32768.0
        data = data.reshape(-1, 1)
        data = np.ascontiguousarray(data)
        print(f"[Piper] numpy shape: {data.shape}, dtype: {data.dtype}", flush=True)
        _safe_play(data, sample_rate)

    def stop(self):
        _stop_audio()
        _cleanup_piper_temp()


def redownload_piper_voices(voice_ids=None):
    """Remove existing Piper voice models and re-download. voice_ids: list of model ids, or None for all PIPER_VOICES."""
    voice_ids = voice_ids or [vid for _, vid in PIPER_VOICES]
    dirs = [
        DEFAULT_PIPER_VOICE_DIR,
        os.path.expanduser("~/.piper/voices"),
    ]
    removed = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith(".onnx"):
                path = os.path.join(d, f)
                try:
                    os.remove(path)
                    removed.append(path)
                except OSError:
                    pass
    if removed:
        print(f"[TTS] Removed {len(removed)} Piper voice file(s): {removed}")
    for vid in voice_ids:
        print(f"[TTS] Downloading Piper voice: {vid}")
        try:
            os.makedirs(DEFAULT_PIPER_VOICE_DIR, exist_ok=True)
            subprocess.run(
                [sys.executable, "-m", "piper.download_voices", "--data-dir", DEFAULT_PIPER_VOICE_DIR, vid],
                check=True,
            )
            print(f"[TTS] Downloaded {vid}")
        except subprocess.CalledProcessError as e:
            print(f"[TTS] Failed to download {vid}: {e}")
        except FileNotFoundError:
            print("[TTS] piper-tts not installed. Run: pip install piper-tts")
            break


# --- Backend: XTTS v2 (local) ---

# XTTS uses language for voice; speaker cloning would need wav file
XTTS_VOICES = [
    ("Default (language-based)", "default"),
]


class XTTSBackend:
    """Coqui XTTS v2 - local, ~2GB model."""

    def __init__(self, voice_id=None):
        self.voice_id = voice_id or "default"
        self.tts = None
        self._load_model()

    def _load_model(self):
        try:
            from TTS.api import TTS
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        except Exception:
            self.tts = None

    def speak(self, text, lang="en"):
        if not text or not str(text).strip():
            return
        if not self.tts:
            raise RuntimeError("XTTS v2 failed to load. Install: pip install TTS")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            self.tts.tts_to_file(text=str(text).strip(), language=lang, file_path=path)
            from scipy.io import wavfile
            rate, data = wavfile.read(path)
            # Ensure data is in correct format for sounddevice
            # Convert to float32 and normalize if integer
            if np.issubdtype(data.dtype, np.integer):
                # Normalize int16/int32 to [-1, 1] float32
                max_val = np.iinfo(data.dtype).max
                data = data.astype(np.float32) / max_val
            else:
                data = data.astype(np.float32)
            # Ensure shape (n_samples, n_channels) for sounddevice
            if data.ndim == 1:
                data = data.reshape(-1, 1)  # mono
                n_channels = 1
            elif data.ndim == 2 and data.shape[1] <= 2:
                n_channels = data.shape[1]  # stereo/mono already correct
            else:
                data = data.flatten().reshape(-1, 1)
                n_channels = 1
            _safe_play(data, rate)
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass

    def stop(self):
        _stop_audio()


# --- Backend: ElevenLabs (API) ---

# ElevenLabs preset voices: (display_name, voice_id)
ELEVENLABS_VOICES = [
    ("Adam", "pNInz6obpgDQGcFmaJgB"),
    ("Rachel", "21m00Tcm4TlvDq8ikWAM"),
    ("Sam", "yoZ06aMxZJJ28mfd3POQ"),
    ("Bella", "EXAVITQu4vr4xnSDxMaL"),
    ("Antoni", "ErXwobaYiN019PkySvjV"),
    ("Josh", "TxGEqnHWrfWFTfGW9XjX"),
    ("Arnold", "VR6AewLTigWG4xSOukaG"),
    ("Emily", "LcfcDJNUP1GQjkzn1xUU"),
]


class ElevenLabsBackend:
    """ElevenLabs TTS - API, high quality. Supports turbo (English) and multilingual_v2 models."""

    def __init__(self, api_key=None, voice_id=None, model_id="eleven_flash_v2_5"):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        self.voice_id = voice_id or "pNInz6obpgDQGcFmaJgB"  # Adam
        self.model_id = model_id

    def speak(self, text, lang="en"):
        if not text or not str(text).strip():
            return
        if not self.api_key:
            raise RuntimeError("ELEVENLABS_API_KEY not set")
        import requests
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream"
        headers = {"xi-api-key": self.api_key, "Content-Type": "application/json"}
        payload = {
            "text": str(text).strip(),
            "model_id": self.model_id,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75, "style": 0.3, "use_speaker_boost": True},
        }
        response = requests.post(url, json=payload, headers=headers, stream=True)
        response.raise_for_status()
        audio_data = b"".join(response.iter_content(chunk_size=4096))
        try:
            _play_audio_from_bytes(audio_data, "mp3")
        except Exception as e:
            print(f"[TTS] ElevenLabs playback failed: {e}")
            # Fallback: try to play as raw data
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_mp3(io.BytesIO(audio_data))
                raw = audio.raw_data
                data = np.frombuffer(raw, dtype=np.int16)
                n_channels = audio.channels
                n_samples = len(data) // n_channels
                # Reshape for multi-channel: (n_samples, n_channels)
                if n_channels > 1:
                    data = data.reshape(n_samples, n_channels)
                else:
                    data = data.reshape(-1, 1)  # mono
                data = data.astype(np.float32) / 32768.0
                _safe_play(data, audio.frame_rate)
            except Exception as e2:
                print(f"[TTS] ElevenLabs fallback failed: {e2}")

    def stop(self):
        _stop_audio()


# --- Backend: OpenAI TTS (API) ---

# OpenAI voices (tts-1/tts-1-hd compatible)
OPENAI_VOICES = [
    ("Alloy", "alloy"),
    ("Echo", "echo"),
    ("Fable", "fable"),
    ("Onyx", "onyx"),
    ("Nova", "nova"),
    ("Shimmer", "shimmer"),
    ("Coral", "coral"),
    ("Sage", "sage"),
    ("Ash", "ash"),
    ("Ballad", "ballad"),
    ("Verse", "verse"),
    ("Cedar", "cedar"),
    ("Marin", "marin"),
]


class OpenAIBackend:
    """OpenAI TTS - API. Supports voice and speed."""

    def __init__(self, api_key=None, voice_id=None, speed=1.2):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.voice = voice_id or "alloy"
        self.speed = max(0.25, min(4.0, float(speed)))

    def speak(self, text, lang="en"):
        if not text or not str(text).strip():
            return
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        response = client.audio.speech.create(
            model="tts-1",
            voice=self.voice,
            input=str(text).strip(),
            speed=self.speed,
        )
        audio_data = response.content
        _play_audio_from_bytes(audio_data, "mp3")

    def stop(self):
        _stop_audio()


# --- Factory ---

def create_tts_backend(backend_id, voice_id=None, speed=1.2, status_callback=None, **kwargs):
    """Create TTS backend by id. Falls back to SayBackend on failure."""
    backend_id = (backend_id or "piper").lower()
    try:
        if backend_id == "piper":
            try:
                return PiperBackend(voice_id=voice_id, status_callback=status_callback)
            except Exception as e:
                msg = f"Piper TTS failed to initialize: {e}. Using macOS say."
                print(f"[TTS] {msg}")
                if status_callback:
                    try:
                        status_callback(msg, duration_sec=8, is_good_news=False)
                    except Exception:
                        status_callback(msg)
                return SayBackend()
        if backend_id == "xtts":
            try:
                return XTTSBackend(voice_id=voice_id)
            except Exception as e:
                msg = f"XTTS v2 failed to initialize: {e}. Using macOS say."
                print(f"[TTS] {msg}")
                if status_callback:
                    try:
                        status_callback(msg, duration_sec=8, is_good_news=False)
                    except Exception:
                        status_callback(msg)
                return SayBackend()
        if backend_id == "elevenlabs":
            key = kwargs.get("api_key") or os.environ.get("ELEVENLABS_API_KEY")
            if not key:
                msg = "ElevenLabs API key not set. Using macOS say. Add ELEVENLABS_API_KEY in API Keys."
                print(f"[TTS] {msg}")
                if status_callback:
                    try:
                        status_callback(msg, duration_sec=8, is_good_news=False)
                    except Exception:
                        status_callback(msg)
                return SayBackend()
            return ElevenLabsBackend(api_key=key, voice_id=voice_id, model_id="eleven_turbo_v2_5")
        if backend_id == "elevenlabs_multilingual_v2":
            key = kwargs.get("api_key") or os.environ.get("ELEVENLABS_API_KEY")
            if not key:
                msg = "ElevenLabs API key not set. Using macOS say. Add ELEVENLABS_API_KEY in API Keys."
                print(f"[TTS] {msg}")
                if status_callback:
                    try:
                        status_callback(msg, duration_sec=8, is_good_news=False)
                    except Exception:
                        status_callback(msg)
                return SayBackend()
            return ElevenLabsBackend(api_key=key, voice_id=voice_id, model_id="eleven_multilingual_v2")
        if backend_id == "openai":
            key = kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY")
            if not key:
                msg = "OpenAI API key not set. Using macOS say. Add OPENAI_API_KEY in API Keys."
                print(f"[TTS] {msg}")
                if status_callback:
                    try:
                        status_callback(msg, duration_sec=8, is_good_news=False)
                    except Exception:
                        status_callback(msg)
                return SayBackend()
            return OpenAIBackend(api_key=key, voice_id=voice_id, speed=speed)
    except Exception as e:
        msg = f"TTS backend '{backend_id}' failed: {e}. Using macOS say."
        print(f"[TTS] {msg}")
        if status_callback:
            try:
                status_callback(msg, duration_sec=8, is_good_news=False)
            except Exception:
                status_callback(msg)
    return SayBackend()


def create_tts_engine(backend_id="piper", voice_id=None, speed=1.2, status_callback=None, use_subprocess=None, **backend_kwargs):
    """Create TTSEngine. use_subprocess=True runs TTS in separate process (recommended for OCR mode)."""
    if use_subprocess is None:
        use_subprocess = os.environ.get("BILIOCR_TTS_IN_PROCESS", "").lower() not in ("1", "true", "yes")
    if use_subprocess:
        print("[TTS] Using subprocess mode (OCR/TTS isolated - prevents audio stutter)")
        return TTSEngineSubprocess(backend_id=backend_id, voice_id=voice_id, speed=speed, status_callback=status_callback, **backend_kwargs)
    return TTSEngine(backend_id=backend_id, voice_id=voice_id, speed=speed, status_callback=status_callback, **backend_kwargs)


def _tts_subprocess_worker(in_queue, backend_id, voice_id, speed):
    """Runs in subprocess. Own process = no GIL/CPU contention with OCR."""
    backend = None
    try:
        backend = create_tts_backend(backend_id, voice_id=voice_id, speed=speed, status_callback=None)
    except Exception as e:
        print(f"[TTS Subprocess] Failed to init backend: {e}", flush=True)
        return
    while True:
        try:
            item = in_queue.get(timeout=0.3)
        except Exception:
            continue
        if item is None:
            break
        if item == "stop":
            try:
                backend.stop()
            except Exception:
                pass
            try:
                while True:
                    in_queue.get_nowait()
            except Exception:
                pass
            continue
        if isinstance(item, tuple) and len(item) >= 2 and item[0] == "speak":
            text, lang = item[1], item[2] if len(item) > 2 else "en"
            if text and str(text).strip():
                try:
                    backend.speak(str(text).strip(), lang)
                except Exception as e:
                    print(f"[TTS Subprocess] Error: {e}", flush=True)


# --- Orchestrator (in-process) ---

class TTSEngine:
    """Queue-based TTS orchestrator. Delegates to pluggable backend. Runs in same process."""

    def __init__(self, backend_id="piper", voice_id=None, speed=1.2, status_callback=None, **backend_kwargs):
        self.tts_queue = queue.Queue()
        self.backend = create_tts_backend(backend_id, voice_id=voice_id, speed=speed, status_callback=status_callback, **backend_kwargs)
        self.is_speaking = False
        self._stop_requested = False

    def speak(self, text, lang="en"):
        """Add to queue, speak sequentially."""
        if not text or not str(text).strip():
            return
        self.tts_queue.put((str(text).strip(), lang))
        if not self.is_speaking:
            threading.Thread(target=self._process_queue, daemon=True).start()

    def _process_queue(self):
        while True:
            try:
                text, lang = self.tts_queue.get_nowait()
            except queue.Empty:
                break
            if self._stop_requested:
                break
            self.is_speaking = True
            try:
                print(f"[TTS] Speaking via {type(self.backend).__name__}: {text[:50]}...", flush=True)
                self.backend.speak(text, lang)
            except Exception as e:
                print(f"[TTS] Error: {e}")
            self.is_speaking = False
        self.is_speaking = False

    def stop(self):
        """Interrupt current speech and clear queue."""
        self._stop_requested = True
        try:
            self.backend.stop()
        except Exception:
            pass
        try:
            while True:
                self.tts_queue.get_nowait()
        except queue.Empty:
            pass
        self._stop_requested = False
        self.is_speaking = False


# --- Orchestrator (subprocess - isolates TTS from OCR) ---

class TTSEngineSubprocess:
    """TTS in separate process. No CPU/GIL contention with OCR. Independent queue."""

    def __init__(self, backend_id="piper", voice_id=None, speed=1.2, status_callback=None, **backend_kwargs):
        self._queue = multiprocessing.Queue(maxsize=100)
        self._process = multiprocessing.Process(
            target=_tts_subprocess_worker,
            args=(self._queue, backend_id, voice_id or "en_US-lessac-medium", speed or 1.2),
            daemon=True,
        )
        self._process.start()
        self.is_speaking = False  # Approximate; subprocess doesn't report back

    def speak(self, text, lang="en"):
        """Add to queue. Subprocess speaks independently of OCR."""
        if not text or not str(text).strip():
            return
        if not self._process.is_alive():
            return
        try:
            self._queue.put_nowait(("speak", str(text).strip(), lang))
        except Exception:
            try:
                self._queue.put(("speak", str(text).strip(), lang), timeout=1.0)
            except Exception:
                pass

    def stop(self):
        """Stop playback and clear queue in subprocess."""
        try:
            self._queue.put("stop", timeout=0.5)
        except Exception:
            pass

    def shutdown(self):
        """Terminate subprocess. Call on app exit."""
        try:
            self._queue.put(None, timeout=0.5)
        except Exception:
            pass
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
