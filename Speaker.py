"""
Speaker.py — Cartesia TTS + Recall.ai audio delivery

Supports two modes:
  - REST (_synthesise): Returns complete MP3. Used for fallback + pre-baked audio.
  - WebSocket streaming (_stream_tts): Yields PCM chunks. Used for Output Media.
"""

import os
import base64
import asyncio
import json
import httpx
import io
import hashlib
import platform
import re

if platform.system() == "Windows":
    os.environ.setdefault("FFMPEG_BINARY",  r"C:\Users\user\Downloads\ffmpeg-8.1-full_build\bin\ffmpeg.exe")
    os.environ.setdefault("FFPROBE_BINARY", r"C:\Users\user\Downloads\ffmpeg-8.1-full_build\bin\ffprobe.exe")

from pydub import AudioSegment

NOISE_FILE   = "freesound_community-office-ambience-24734 (1).mp3"
NOISE_SLICES = 20

# ── Number to words for TTS ──────────────────────────────────────────────────
_DIGIT_WORDS = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
}

def _prep_for_tts(text: str) -> str:
    """Convert numbers to spoken form for TTS clarity."""
    # SCRUM-15 → "SCRUM fifteen"
    def _ticket_repl(m):
        prefix = m.group(1)
        num = m.group(2)
        spoken = " ".join(_DIGIT_WORDS.get(d, d) for d in num)
        return f"{prefix} {spoken}"

    text = re.sub(r'\b([A-Z]+-?)(\d+)\b', _ticket_repl, text)

    # Standalone numbers: 123 → "one two three"
    def _num_repl(m):
        return " ".join(_DIGIT_WORDS.get(d, d) for d in m.group(0))

    text = re.sub(r'(?<![A-Za-z])\b(\d{2,})\b(?![A-Za-z])', _num_repl, text)
    return text


def _mix_noise(voice_bytes: bytes, noise_slices: list, text: str) -> tuple[bytes, int]:
    try:
        voice       = AudioSegment.from_file(io.BytesIO(voice_bytes)).fade_in(80)
        duration_ms = len(voice)
        hash_val    = int(hashlib.md5(text.encode()).hexdigest(), 16)
        slice_idx   = hash_val % len(noise_slices)
        noise_seg   = noise_slices[slice_idx]
        loops       = (duration_ms // len(noise_seg)) + 2
        noise       = (noise_seg * loops)[:duration_ms]
        noise       = noise + 3
        noise       = noise.low_pass_filter(4000)
        combined    = voice.overlay(noise, gain_during_overlay=-3)
        output      = io.BytesIO()
        combined.export(output, format="mp3", bitrate="64k")
        return output.getvalue(), duration_ms
    except Exception as e:
        print(f"[Speaker] Noise failed: {e}")
        return voice_bytes, get_duration_ms(voice_bytes)


def get_duration_ms(audio_bytes: bytes) -> int:
    try:
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        return len(seg)
    except Exception:
        return int((len(audio_bytes) * 8) / (48 * 1000) * 1000)


RECALL_REGION   = os.environ.get("RECALLAI_REGION", "ap-northeast-1")
RECALL_API_BASE = f"https://{RECALL_REGION}.recall.ai/api/v1"

CARTESIA_VOICE_ID = "79a125e8-cd45-4c13-8a67-188112f4dd22"
CARTESIA_MODEL    = "sonic-turbo"
CARTESIA_WS_URL   = "wss://api.cartesia.ai/tts/websocket"


class CartesiaSpeaker:
    def __init__(self, bot_id: str = None):
        import Speaker as _self_module
        print(f"[Speaker] Loaded from: {_self_module.__file__}")
        self.recall_key = os.environ["RECALLAI_API_KEY"]
        self.bot_id     = bot_id

        base_dir = os.path.dirname(os.path.abspath(__file__))
        noise_path = os.path.join(base_dir, NOISE_FILE)
        self._noise_slices = []
        try:
            full_noise = AudioSegment.from_file(noise_path)
            slice_len = len(full_noise) // NOISE_SLICES
            self._noise_slices = [full_noise[i * slice_len:(i + 1) * slice_len] for i in range(NOISE_SLICES)]
        except Exception as e:
            print(f"[Speaker] Noise load failed (not critical): {e}")

        self._base_noise = self._noise_slices if self._noise_slices else None
        limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)

        # Multi-key Cartesia setup
        self._cartesia_keys = []
        for key_name in ["CARTESIA_API_KEY", "CARTESIA_API_KEY_2", "CARTESIA_API_KEY_3", "CARTESIA_API_KEY_4", "CARTESIA_API_KEY_5"]:
            val = os.environ.get(key_name, "").strip()
            if val:
                self._cartesia_keys.append(val)

        if not self._cartesia_keys:
            raise ValueError("No CARTESIA_API_KEY found in environment")

        print(f"[Speaker] {len(self._cartesia_keys)} Cartesia key(s) loaded")
        self._key_index = 0

        self._cartesia_client = httpx.AsyncClient(timeout=30, limits=limits)
        self._recall_client   = httpx.AsyncClient(timeout=30, limits=limits)
        self._recall_headers  = {
            "Authorization": f"Token {self.recall_key}",
            "Content-Type":  "application/json",
            "accept":        "application/json",
        }

        # WebSocket connection (persistent, reused across TTS calls)
        self._cartesia_ws = None
        self._ws_lock = asyncio.Lock()
        self._context_counter = 0

    async def warmup(self):
        valid_keys = []
        for i, key in enumerate(self._cartesia_keys):
            try:
                headers = {"Authorization": f"Bearer {key}", "Cartesia-Version": "2025-04-16", "Content-Type": "application/json"}
                response = await self._cartesia_client.post(
                    "https://api.cartesia.ai/tts/bytes", headers=headers,
                    json={
                        "model_id": CARTESIA_MODEL, "transcript": "hi",
                        "voice": {"mode": "id", "id": CARTESIA_VOICE_ID},
                        "language": "en",
                        "output_format": {"container": "mp3", "sample_rate": 44100, "bit_rate": 192000},
                    },
                )
                if response.status_code in (200, 201):
                    valid_keys.append(key)
                    print(f"[Speaker] ✅ Cartesia key #{i+1} valid")
                else:
                    print(f"[Speaker] ❌ Cartesia key #{i+1} invalid ({response.status_code})")
            except Exception as e:
                print(f"[Speaker] ❌ Cartesia key #{i+1} failed: {e}")

        if valid_keys:
            self._cartesia_keys = valid_keys
            print(f"[Speaker] ✅ {len(valid_keys)} valid key(s), Cartesia warmed up")
        else:
            print(f"[Speaker] ⚠️  No valid Cartesia keys!")

    def _next_key(self) -> str:
        key = self._cartesia_keys[self._key_index % len(self._cartesia_keys)]
        self._key_index += 1
        return key

    def _next_cartesia_headers(self) -> dict:
        key = self._next_key()
        return {"Authorization": f"Bearer {key}", "Cartesia-Version": "2025-04-16", "Content-Type": "application/json"}

    # ══════════════════════════════════════════════════════════════════════════
    # REST TTS — returns complete MP3 (used for fallback + pre-baked audio)
    # ══════════════════════════════════════════════════════════════════════════

    async def _synthesise(self, text: str) -> bytes:
        text = _prep_for_tts(text)
        headers = self._next_cartesia_headers()
        key_num = (self._key_index - 1) % len(self._cartesia_keys) + 1
        print(f"[Speaker] TTS via Cartesia (key #{key_num})...")
        response = await self._cartesia_client.post(
            "https://api.cartesia.ai/tts/bytes", headers=headers,
            json={
                "model_id": CARTESIA_MODEL, "transcript": text,
                "voice": {"mode": "id", "id": CARTESIA_VOICE_ID},
                "language": "en",
                "output_format": {"container": "mp3", "sample_rate": 44100, "bit_rate": 192000},
            },
        )
        response.raise_for_status()
        return response.content

    # ══════════════════════════════════════════════════════════════════════════
    # WebSocket TTS — yields PCM chunks (used for Output Media streaming)
    # ══════════════════════════════════════════════════════════════════════════

    async def _ensure_ws_connected(self):
        """Connect to Cartesia WebSocket if not already connected."""
        if self._cartesia_ws is not None:
            try:
                # Check if still open
                await self._cartesia_ws.ping()
                return
            except Exception:
                self._cartesia_ws = None

        import websockets
        key = self._next_key()
        url = f"{CARTESIA_WS_URL}?api_key={key}&cartesia_version=2025-04-16"
        self._cartesia_ws = await websockets.connect(url, ping_interval=20, ping_timeout=10)
        print("[Speaker] ✅ Cartesia WebSocket connected")

    async def _stream_tts(self, text: str):
        """Stream TTS as PCM s16le chunks via Cartesia WebSocket.

        Yields: bytes — raw PCM s16le chunks at 24kHz mono.
        Each chunk is typically ~20ms of audio (960 bytes).
        """
        text = _prep_for_tts(text)
        self._context_counter += 1
        context_id = f"ctx-{self._context_counter}"

        async with self._ws_lock:
            try:
                await self._ensure_ws_connected()
            except Exception as e:
                print(f"[Speaker] ⚠️  Cartesia WS connect failed: {e}")
                raise

        # Send TTS request
        request = {
            "model_id": CARTESIA_MODEL,
            "transcript": text,
            "voice": {"mode": "id", "id": CARTESIA_VOICE_ID},
            "language": "en",
            "context_id": context_id,
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": 48000,
            },
            "continue": False,
        }

        try:
            await self._cartesia_ws.send(json.dumps(request))
            total_bytes = 0
            first_chunk = True

            while True:
                raw = await asyncio.wait_for(self._cartesia_ws.recv(), timeout=10)
                msg = json.loads(raw)

                if msg.get("context_id") != context_id:
                    continue

                if msg.get("type") == "chunk" and msg.get("data"):
                    pcm_bytes = base64.b64decode(msg["data"])
                    total_bytes += len(pcm_bytes)
                    if first_chunk:
                        print(f"[Speaker] 🔊 First PCM chunk ({len(pcm_bytes)} bytes)")
                        first_chunk = False
                    yield pcm_bytes

                if msg.get("done"):
                    break

            duration_ms = (total_bytes / 2 / 48000) * 1000
            print(f"[Speaker] ✅ Streamed {total_bytes} bytes ({duration_ms:.0f}ms audio)")

        except Exception as e:
            print(f"[Speaker] ⚠️  Stream TTS error: {e}")
            # Close broken connection so next call reconnects
            try:
                await self._cartesia_ws.close()
            except Exception:
                pass
            self._cartesia_ws = None
            raise

    # ══════════════════════════════════════════════════════════════════════════
    # Recall.ai audio injection (fallback mode)
    # ══════════════════════════════════════════════════════════════════════════

    async def _inject_into_meeting(self, b64_audio: str):
        if not self.bot_id:
            return
        if os.environ.get("DEBUG_SAVE_AUDIO", "").lower() in ("1", "true", "yes"):
            try:
                raw = base64.b64decode(b64_audio)
                self._debug_audio_counter = getattr(self, '_debug_audio_counter', 0) + 1
                fname = f"debug_inject_{self._debug_audio_counter:03d}.mp3"
                with open(fname, "wb") as f:
                    f.write(raw)
            except Exception:
                pass

        response = await self._recall_client.post(
            f"{RECALL_API_BASE}/bot/{self.bot_id}/output_audio/",
            headers=self._recall_headers,
            json={"kind": "mp3", "b64_data": b64_audio},
        )
        if response.status_code not in (200, 201):
            print(f"[Speaker] Inject error {response.status_code}: {response.text}")
        else:
            print("[Speaker] Audio injected")

    async def stop_audio(self):
        if not self.bot_id:
            return
        try:
            response = await self._recall_client.delete(
                f"{RECALL_API_BASE}/bot/{self.bot_id}/output_audio/",
                headers=self._recall_headers,
            )
            if response.status_code == 204:
                print("[Speaker] ⏹️  Audio stopped")
        except Exception as e:
            print(f"[Speaker] Stop audio error: {e}")

    async def close(self):
        if self._cartesia_ws:
            try:
                await self._cartesia_ws.close()
            except Exception:
                pass
        await asyncio.gather(
            self._cartesia_client.aclose(),
            self._recall_client.aclose(),
        )