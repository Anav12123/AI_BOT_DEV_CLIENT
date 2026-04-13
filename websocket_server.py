"""
websocket_server.py — Production voice pipeline with Output Media streaming

Audio delivery:
  Primary: Cartesia WebSocket → PCM chunks → audio page WebSocket → AudioWorklet
  Fallback: Cartesia REST → MP3 → output_audio API (if audio page not connected)
"""

import asyncio
import json
import time
import base64
import os
import re as _re
import random
from aiohttp import web
import aiohttp
from collections import deque

from Trigger import TriggerDetector
from Agent import PMAgent, FILLERS
from Speaker import CartesiaSpeaker, _mix_noise, get_duration_ms
from vad import RmsVAD
from JiraClient import JiraClient, JiraAuthError, JiraNotFoundError, JiraTransitionError, JiraPermissionError
from jira_prompts import AzureExtractor, JIRA_RESPONSE_PROMPT, JIRA_INTENT_PROMPT
from standup import StandupFlow
import session_store


def ts():
    return time.strftime("%H:%M:%S")

def elapsed(since: float) -> str:
    return f"{(time.time() - since)*1000:.0f}ms"

WORDS_PER_SECOND = 3.2
PCM_SAMPLE_RATE  = 48000  # Cartesia WebSocket output
PCM_BYTES_PER_SEC = PCM_SAMPLE_RATE * 2  # 16-bit mono

_ACK_PHRASES = frozenset({
    "sure", "ok", "okay", "yeah", "yes", "go ahead", "alright",
    "right", "hmm", "mhm", "cool", "got it", "fine", "yep", "yup",
    "carry on", "go on", "continue", "waiting", "i'm waiting",
    "i am waiting", "no problem", "take your time", "np",
    "hello", "hi", "hey", "huh", "what", "sorry",
})

_INTERRUPT_ACKS = [
    "Oh sorry, go ahead.",
    "My bad, what were you saying?",
    "Sure, I'm listening.",
    "Oh, go on.",
]

_TRANSCRIPTION_FIXES = [
    (_re.compile(r'\b(?:NF\s*Cloud|Enuf\s*Cloud|Enough\s*Cloud|Nav\s*Cloud|Anav\s*Cloud|Arnav\s*Cloud|Anab\s*Cloud|NFClouds?|EnoughClouds?|NavClouds?|AnavCloud)\b', _re.IGNORECASE), 'AnavClouds'),
    (_re.compile(r'\b(?:Sales\s*Force|Sells\s*Force|Cells\s*Force|SalesForce)\b', _re.IGNORECASE), 'Salesforce'),
]

def _fix_transcription(text):
    result = text
    for p, r in _TRANSCRIPTION_FIXES:
        result = p.sub(r, result)
    return result

def _is_ack(text):
    fragments = _re.split(r'[.!?,]+', text.strip().lower())
    return all(f.strip() in _ACK_PHRASES or f.strip() == "" for f in fragments) and text.strip() != ""


class BotSession:
    STRAGGLER_WAIT = 0.2   # Reduced from 0.4 for faster non-direct response
    STRAGGLER_DIRECT = 0.0  # No straggler for "Sam, ..." — direct address is complete
    WAIT_TIMEOUT   = 2.0

    def __init__(self, session_id, bot_id, server):
        self.session_id = session_id
        self.bot_id = bot_id
        self.server = server
        self.tag = f"[S:{session_id[:8]}]"

        self.username = ""
        self.meeting_url = ""
        self.mode = "client_call"
        self.started_at = time.time()

        self.agent = PMAgent()
        self.speaker = CartesiaSpeaker(bot_id=bot_id)
        self.trigger = TriggerDetector()
        self.vad = RmsVAD()

        self.jira = JiraClient()
        self.azure_extractor = AzureExtractor()

        self.speaking = False
        self.audio_playing = False
        self.convo_history = deque(maxlen=10)
        self.current_task = None
        self.current_text = ""
        self.current_speaker = ""
        self.interrupt_event = asyncio.Event()
        self.generation = 0

        self.buffer = []
        self.partial_text = ""
        self.partial_speaker = ""
        self.last_flushed_text = ""

        self.was_interrupted = False
        self.playing_ack = False
        self.eot_task = None
        self.searching = False

        self.audio_event_count = 0
        self.max_conf = 0.0
        self.debug_audio_file = None

        self._jira_context = ""

        # Output Media: audio page WebSocket connection
        self.audio_ws = None  # Set when audio page connects

        # Standup mode
        self.standup_flow = None
        self._standup_buffer = []
        self._standup_timer = None
        self._standup_finished = False  # Guard: prevent double finish
        self._auto_left = False         # Guard: prevent double leave

    @property
    def _streaming_mode(self) -> bool:
        """True if Output Media audio page is connected."""
        return self.audio_ws is not None and not self.audio_ws.closed

    async def setup(self):
        self.agent.start()
        await self.speaker.warmup()
        await self.vad.setup()
        if self.jira.enabled:
            await self.jira.test_connection()
            await self._sync_pending_tickets()
            await self._preload_jira_context()
        print(f"[{ts()}] {self.tag} ✅ Session ready (bot: {self.bot_id[:12]})")

    async def _sync_pending_tickets(self):
        pending = session_store.get_pending_tickets()
        if not pending:
            return
        print(f"[{ts()}] {self.tag} 🔄 Syncing {len(pending)} pending ticket(s)...")
        synced = 0
        for item in pending:
            try:
                await self.jira.create_ticket(
                    summary=item.get("summary", ""), issue_type=item.get("type", "Task"),
                    priority=item.get("priority", "Medium"), description=item.get("description", ""),
                    labels=item.get("labels", []),
                )
                synced += 1
            except Exception as e:
                print(f"[{ts()}] {self.tag} ⚠️  Pending sync failed: {e}")
                break
        if synced > 0:
            session_store.clear_pending_tickets()
            print(f"[{ts()}] {self.tag} ✅ Synced {synced} pending ticket(s)")

    async def _preload_jira_context(self):
        try:
            tickets = await self.jira.get_my_tickets(max_results=20)
            if tickets:
                lines = [f"  {t['key']}: {t['summary']} [{t['status']}] ({t['priority']}, {t['assignee']})" for t in tickets]
                self._jira_context = "JIRA TICKETS:\n" + "\n".join(lines)
                print(f"[{ts()}] {self.tag} 📥 Pre-loaded {len(tickets)} ticket(s)")
        except Exception as e:
            print(f"[{ts()}] {self.tag} ⚠️  Pre-load failed: {e}")

    async def cleanup(self):
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
        if self.eot_task and not self.eot_task.done():
            self.eot_task.cancel()

        # Save standup data if standup was in progress
        if self.standup_flow:
            self.standup_flow._cancel_silence_timer()
            if self._standup_timer and not self._standup_timer.done():
                self._standup_timer.cancel()
            if self.standup_flow.data.get("yesterday", {}).get("raw"):
                await self._finish_standup()

        try:
            await self.speaker.close()
        except Exception:
            pass

        if len(self.agent.rag._entries) > 3:
            await self._post_meeting_save(extract_jira=self.jira.enabled and self.azure_extractor.enabled)

        try:
            await self.jira.close()
        except Exception:
            pass
        try:
            await self.azure_extractor.close()
        except Exception:
            pass
        self.agent.reset()
        print(f"[{ts()}] {self.tag} 🧹 Session cleaned up")

    async def _post_meeting_save(self, extract_jira=True):
        print(f"[{ts()}] {self.tag} 📋 Meeting ended — processing session...")
        transcript_entries = [{"speaker": e.get("speaker", "?"), "text": e["text"], "time": e.get("time", 0)} for e in self.agent.rag._entries]
        transcript_text = "\n".join(e["text"] for e in self.agent.rag._entries if e.get("speaker", "").lower() != "sam")
        duration_min = int((time.time() - self.started_at) / 60)

        items = []
        if extract_jira:
            try:
                items = await self.azure_extractor.extract_action_items(transcript_text)
            except Exception as e:
                print(f"[{ts()}] {self.tag} ❌ Extraction failed: {e}")

        created_tickets = []
        if items and self.jira.enabled:
            user_cache = {}
            for item in items:
                try:
                    related = await self.jira.find_related_tickets(item.get("summary", ""))
                    if related:
                        rs = ", ".join(f"{r['key']} ({r['summary'][:40]})" for r in related[:3])
                        item["description"] = item.get("description", "") + f"\n\nRelated: {rs}"

                    assignee_id = None
                    an = item.get("assignee")
                    if an and an.lower() not in ("null", "none", "unassigned", ""):
                        assignee_id = user_cache.get(an) or await self.jira.search_user(an)
                        if an not in user_cache:
                            user_cache[an] = assignee_id

                    result = await self.jira.create_ticket(
                        summary=item["summary"], issue_type=item.get("type", "Task"),
                        priority=item.get("priority", "Medium"), description=item.get("description", ""),
                        labels=item.get("labels", ["client-feedback"]), assignee_id=assignee_id,
                    )
                    item["jira_key"] = result.get("key", "?")
                    created_tickets.append(item)
                    print(f"[{ts()}] {self.tag} 📤 Created {item['jira_key']}: {item['summary']}")
                except JiraAuthError:
                    session_store.save_pending_ticket(item)
                    break
                except Exception as e:
                    print(f"[{ts()}] {self.tag} ⚠️  Failed — saving locally: {e}")
                    session_store.save_pending_ticket(item)

        session_data = {
            "session_id": self.session_id, "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "user": self.username, "mode": self.mode,
            "project": self.jira.project if self.jira.enabled else "",
            "meeting_url": self.meeting_url, "duration_minutes": duration_min,
            "transcript": transcript_entries,
            "summary": "; ".join(e["text"][:80] for e in transcript_entries if not e["text"].startswith("Sam:"))[:200],
            "feedback_count": len(items), "tickets_created": len(created_tickets),
            "action_items": [{"type": i.get("type"), "summary": i.get("summary"), "priority": i.get("priority"), "jira_key": i.get("jira_key", ""), "assignee": i.get("assignee", "")} for i in items],
        }
        try:
            session_store.save_session(session_data)
        except Exception as e:
            print(f"[{ts()}] {self.tag} ⚠️  Save session failed: {e}")

    # ── Event dispatch ────────────────────────────────────────────────────────

    async def handle_event(self, raw):
        t = time.time()
        try:
            payload = json.loads(raw)
        except Exception:
            return
        event = payload.get("event", "")

        if event == "transcript.data":
            inner = payload.get("data", {}).get("data", {})
            words = inner.get("words", [])
            text = " ".join(w.get("text", "") for w in words).strip()
            speaker = inner.get("participant", {}).get("name", "Unknown")
            if not text or speaker.lower() == "sam":
                return
            text = _fix_transcription(text)
            text = _re.sub(r'\s+', ' ', text).strip().lstrip("-–— ").strip()
            if not text:
                return

            self.agent.log_exchange(speaker, text)
            print(f"\n[{ts()}] {self.tag} [{speaker}] {text}")

            # ── Standup mode: buffer transcript with state tag, wait 800ms ──
            if self.standup_flow and not self.standup_flow.is_done:
                self._standup_buffer.append((text, self.standup_flow.state))
                # Only start/restart timer if not currently processing
                if not self.standup_flow._processing:
                    if self._standup_timer and not self._standup_timer.done():
                        self._standup_timer.cancel()
                    self._standup_timer = asyncio.create_task(self._flush_standup_buffer(speaker))
                return

            if self.was_interrupted:
                self.was_interrupted = False
                self.buffer.clear()
                self.partial_text = ""
                await self._play_interrupt_ack()
                return

            self.partial_text = ""
            self.partial_speaker = ""

            if self.last_flushed_text:
                flushed_w = set(self.last_flushed_text.lower().split())
                incoming_w = set(text.lower().split())
                sim = len(flushed_w & incoming_w) / max(len(flushed_w), len(incoming_w), 1)
                if sim >= 0.7 or text.lower().strip() in self.last_flushed_text.lower():
                    self.last_flushed_text = ""
                    return
            self.last_flushed_text = ""

            if self.speaking and not self.playing_ack:
                if self.current_speaker == speaker and _is_ack(text.lstrip("-–— ")):
                    return
                if self.audio_playing:
                    await self._stop_all_audio()
                    self.interrupt_event.set()
                    if self.current_task and not self.current_task.done():
                        self.current_task.cancel()
                    self.speaking = False
                    self.audio_playing = False
                    self.buffer.clear()
                    self.vad.end_turn()
                    await self._play_interrupt_ack()
                    return
                elif self.current_speaker != speaker or len(text.split()) > 8:
                    if self.current_task and not self.current_task.done():
                        self.current_task.cancel()
                    self.interrupt_event.set()
                    await asyncio.sleep(0.05)
                    self.speaking = False
                    self.buffer.append((speaker, text, t))
                    self._schedule_eot_check(speaker)
                    return

            self.buffer.append((speaker, text, t))
            self._schedule_eot_check(speaker)

        elif event == "transcript.partial_data":
            inner = payload.get("data", {}).get("data", {})
            text = " ".join(w.get("text", "") for w in inner.get("words", [])).strip()
            speaker = inner.get("participant", {}).get("name", "Unknown")
            if text and speaker.lower() != "sam":
                self.partial_text = _fix_transcription(text)
                self.partial_speaker = speaker
                if self.eot_task and not self.eot_task.done():
                    self.eot_task.cancel()

        elif event == "participant_events.speech_off":
            pass
        elif event == "participant_events.speech_on":
            speaker = payload.get("data", {}).get("data", {}).get("participant", {}).get("name", "Unknown")
            # Skip interrupt during standup — user speech gets buffered instead
            if self.standup_flow and not self.standup_flow.is_done:
                return
            if self.speaking and self.audio_playing and self.current_speaker != speaker:
                await self._stop_all_audio()
                self.interrupt_event.set()
                if self.current_task and not self.current_task.done():
                    self.current_task.cancel()
                self.speaking = False
                self.audio_playing = False
                self.was_interrupted = True

        elif event == "audio_mixed_raw.data":
            if not self.vad.ready or self.audio_playing:
                return
            audio_b64 = payload.get("data", {}).get("data", {}).get("buffer", "")
            if audio_b64:
                try:
                    pcm = base64.b64decode(audio_b64)
                    for rms in self.vad.process_chunk(pcm):
                        self.vad.update_state(rms)
                    self.audio_event_count += 1
                    if self.audio_event_count == 1:
                        print(f"[{ts()}] {self.tag} 🔊 First audio received ({len(pcm)} bytes)")
                except Exception:
                    pass

        elif event == "participant_events.join":
            name = payload.get("data", {}).get("data", {}).get("participant", {}).get("name", "Unknown")
            if name and name.lower() != "sam":
                print(f"[{ts()}] {self.tag} 👋 {name} joined")
                if self.mode == "standup":
                    asyncio.create_task(self._start_standup(name))
                else:
                    asyncio.create_task(self._greet(name, t))

        elif event == "participant_events.leave":
            name = payload.get("data", {}).get("data", {}).get("participant", {}).get("name", "Unknown")
            if name and name.lower() != "sam":
                print(f"[{ts()}] {self.tag} 👋 {name} left")

    # ── EOT ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_direct_address(text: str) -> bool:
        """Check if text starts with 'Sam' — direct address, skip EOT + straggler."""
        t = text.strip().lower()
        return t.startswith("sam,") or t.startswith("sam ") or t == "sam" or \
               t.startswith("hey sam") or t.startswith("hi sam") or t.startswith("hello sam")

    def _schedule_eot_check(self, speaker):
        if self.eot_task and not self.eot_task.done():
            self.eot_task.cancel()
        self.eot_task = asyncio.create_task(self._run_eot_check(speaker))

    async def _run_eot_check(self, speaker):
        try:
            result = self._get_buffer_text()
            if not result or self.speaking:
                return
            spk, full_text, t0 = result
            context = "\n".join(self.convo_history)

            # ── Fast path: direct address skips EOT classifier + straggler ──
            if self._is_direct_address(full_text):
                print(f"[{ts()}] {self.tag} ⚡ Direct address — skipping EOT + straggler")
                # No straggler wait at all
            else:
                # Normal path: run EOT classifier
                decision = await self.agent.check_end_of_turn(full_text, context)
                if decision == "RESPOND":
                    await asyncio.sleep(self.STRAGGLER_WAIT)  # 200ms
                else:
                    await asyncio.sleep(self.WAIT_TIMEOUT)

            if self.speaking or not self.buffer:
                return
            result = self._get_buffer_text()
            if not result:
                return
            spk, full_text, t0 = result
            self.buffer.clear()
            self.partial_text = ""
            self.vad.end_turn()
            self.last_flushed_text = full_text
            self._start_process(full_text, spk, t0)
        except asyncio.CancelledError:
            return

    def _get_buffer_text(self):
        if not self.buffer and not self.partial_text:
            return None
        if self.buffer:
            speaker = self.buffer[-1][0]
            t0 = self.buffer[0][2]
            full_text = " ".join(txt for _, txt, _ in self.buffer)
            if self.partial_text and self.partial_text not in full_text:
                full_text += " " + self.partial_text
        else:
            speaker = self.partial_speaker or "Unknown"
            t0 = time.time()
            full_text = self.partial_text
        return speaker, full_text, t0

    def _start_process(self, text, speaker, t0):
        self.generation += 1
        self.current_text = text
        self.current_speaker = speaker
        self.interrupt_event.clear()
        self.convo_history.append(f"{speaker}: {text}")
        self.current_task = asyncio.create_task(self._process(text, speaker, t0, self.generation))

    # ── Audio helpers ─────────────────────────────────────────────────────────

    async def _stop_all_audio(self):
        """Stop audio in both streaming and fallback mode."""
        if self._streaming_mode:
            try:
                await self.audio_ws.send_str(json.dumps({"type": "stop"}))
            except Exception:
                pass
        try:
            await self.speaker.stop_audio()
        except Exception:
            pass

    async def _stream_and_relay(self, text: str, my_gen: int) -> float:
        """Stream TTS via Cartesia WebSocket → relay PCM to audio page.
        Returns duration in seconds. Used in streaming mode."""
        total_bytes = 0
        t0 = time.time()
        try:
            async for pcm_chunk in self.speaker._stream_tts(text):
                if self.interrupt_event.is_set() or my_gen != self.generation:
                    return 0
                await self.audio_ws.send_bytes(pcm_chunk)
                total_bytes += len(pcm_chunk)
            # Send flush to let AudioWorklet know this utterance is done
            await self.audio_ws.send_str(json.dumps({"type": "flush"}))
        except Exception as e:
            print(f"[{ts()}] {self.tag} ⚠️  Stream relay error: {e}")
            return 0

        duration = total_bytes / PCM_BYTES_PER_SEC
        print(f"[{ts()}] {self.tag} ⏱ Streamed {total_bytes} bytes ({duration:.1f}s) in {elapsed(t0)}")
        return duration

    async def _wait_for_playback(self, duration_sec: float, my_gen: int) -> bool:
        """Wait for audio playback to finish, interruptible."""
        if duration_sec <= 0:
            return True
        self.audio_playing = True
        try:
            # Add 200ms for the AudioWorklet buffer delay
            await asyncio.wait_for(self.interrupt_event.wait(), timeout=duration_sec + 0.2)
            self.audio_playing = False
            return False  # interrupted
        except asyncio.TimeoutError:
            self.audio_playing = False
            return True  # completed

    async def _speak_streaming(self, text: str, my_gen: int) -> bool:
        """Stream TTS + wait for playback. Returns True if completed, False if interrupted."""
        duration = await self._stream_and_relay(text, my_gen)
        if duration <= 0:
            return False
        return await self._wait_for_playback(duration, my_gen)

    async def _speak_fallback(self, text: str, label: str, my_gen: int) -> bool:
        """Fallback: REST TTS + MP3 inject. Returns True if completed."""
        try:
            async with self.server._tts_semaphore:
                audio = await self.speaker._synthesise(text)
        except Exception as e:
            print(f"[{ts()}] {self.tag} ⚠️  Fallback TTS error: {e}")
            return True

        if self.interrupt_event.is_set() or my_gen != self.generation:
            return False

        try:
            await self.speaker.stop_audio()
        except Exception:
            pass
        b64 = base64.b64encode(audio).decode("utf-8")
        await self.speaker._inject_into_meeting(b64)
        self.audio_playing = True

        play_dur = max(500, get_duration_ms(audio))
        try:
            await asyncio.wait_for(self.interrupt_event.wait(), timeout=play_dur / 1000)
            self.audio_playing = False
            return False
        except asyncio.TimeoutError:
            self.audio_playing = False
            return True

    async def _speak(self, text: str, label: str, my_gen: int) -> bool:
        """Speak text using best available method. Returns True if completed."""
        if self._streaming_mode:
            return await self._speak_streaming(text, my_gen)
        else:
            return await self._speak_fallback(text, label, my_gen)

    async def _greet(self, name, t0):
        await asyncio.sleep(1.0)
        if self.speaking:
            return
        greeting = f"Hey {name}, welcome to the call!"
        self._log_sam(greeting)
        self.speaking = True
        try:
            await self._speak(greeting, "greeting", self.generation)
        except Exception as e:
            print(f"[{ts()}] {self.tag} ⚠️  Greet error: {e}")
        finally:
            self.speaking = False
            self.audio_playing = False

    async def _start_standup(self, developer_name: str):
        """Initialize and start the standup flow for a developer."""
        await asyncio.sleep(1.0)
        if self.speaking:
            return
        print(f"[{ts()}] {self.tag} 📋 Starting standup for {developer_name}")

        # Create standup flow with speaker function
        async def speak_fn(text, label, gen):
            self._log_sam(text)
            return await self._speak(text, label, gen)

        self.standup_flow = StandupFlow(
            developer_name=developer_name,
            agent=self.agent,
            speaker_fn=speak_fn,
            jira_client=self.jira if self.jira.enabled else None,
            jira_context=self._jira_context,
            azure_extractor=self.azure_extractor if self.azure_extractor.enabled else None,
        )

        self.generation += 1
        self.speaking = True
        try:
            await self.standup_flow.start(self.generation)
        except Exception as e:
            print(f"[{ts()}] {self.tag} ⚠️  Standup start error: {e}")
        finally:
            self.speaking = False
            self.audio_playing = False

    async def _finish_standup(self):
        """Save basic standup + leave immediately + background extract/Jira."""
        if self._standup_finished:
            return
        if not self.standup_flow:
            return
        self._standup_finished = True

        # 1. Save basic standup data (raw answers, before extraction)
        result = self.standup_flow.get_result()
        result["session_id"] = self.session_id
        result["user"] = self.username
        result["mode"] = "standup"
        session_store.save_standup(result)
        print(f"[{ts()}] {self.tag} ✅ Standup saved (basic)")

        # 2. Leave meeting immediately (2 second pause for last audio)
        if self.standup_flow.is_done:
            asyncio.create_task(self._auto_leave_after_standup())

        # 3. Background: Azure extraction + Jira (fire and forget)
        if self.standup_flow.data.get("completed"):
            asyncio.create_task(self._background_standup_work())

    async def _auto_leave_after_standup(self):
        """Leave meeting 2 seconds after standup completes."""
        if self._auto_left:
            return
        self._auto_left = True
        try:
            await asyncio.sleep(2.0)
            print(f"[{ts()}] {self.tag} 🚪 Auto-leaving after standup")
            if self.bot_id:
                import httpx
                RECALL_REGION = os.environ.get("RECALLAI_REGION", "ap-northeast-1")
                RECALL_API_BASE = f"https://{RECALL_REGION}.recall.ai/api/v1"
                headers = {
                    "Authorization": f"Token {os.environ['RECALLAI_API_KEY']}",
                    "Content-Type": "application/json",
                }
                async with httpx.AsyncClient(timeout=10) as client:
                    await client.post(f"{RECALL_API_BASE}/bot/{self.bot_id}/leave_call/", headers=headers)
                print(f"[{ts()}] {self.tag} ✅ Bot left meeting")
        except Exception as e:
            print(f"[{ts()}] {self.tag} ⚠️  Auto-leave failed: {e}")

    async def _background_standup_work(self):
        """Background: Azure extraction + Jira comments + transitions.
        Runs AFTER bot leaves. User is already gone."""
        try:
            # Wait a moment for bot to leave cleanly
            await asyncio.sleep(3.0)

            print(f"[{ts()}] {self.tag} 🔧 Background: processing standup data...")
            await self.standup_flow.background_finalize()

            # Re-save with enriched data (summaries, jira_ids, status_updates)
            result = self.standup_flow.get_result()
            result["session_id"] = self.session_id
            result["user"] = self.username
            result["mode"] = "standup"
            session_store.save_standup(result)
            print(f"[{ts()}] {self.tag} ✅ Standup re-saved (enriched)")

        except Exception as e:
            print(f"[{ts()}] {self.tag} ⚠️  Background standup work failed: {e}")

    async def _flush_standup_buffer(self, speaker: str):
        """Wait 800ms for more transcript chunks, then send combined text to standup flow."""
        try:
            await asyncio.sleep(0.8)  # Straggler wait — let Deepgram finish the sentence
            if not self._standup_buffer:
                return

            # Wait for Sam to finish speaking before processing (max 30 seconds)
            for _ in range(300):
                if not self.speaking:
                    break
                await asyncio.sleep(0.1)

            if not self._standup_buffer:
                return

            # State-tagged filtering: only process text from current state
            current_state = self.standup_flow.state
            relevant = [t for t, s in self._standup_buffer if s == current_state]
            stale = [(t, s) for t, s in self._standup_buffer if s != current_state]
            self._standup_buffer.clear()

            if stale:
                stale_states = set(s.name for _, s in stale)
                stale_texts = [t for t, _ in stale]
                print(f"[{ts()}] {self.tag} 🗑️ Discarded {len(stale)} stale input(s) from {stale_states}: {' | '.join(t[:30] for t in stale_texts)}")

            if not relevant:
                return

            full_text = " ".join(relevant)
            print(f"[{ts()}] {self.tag} 📋 Standup input: {full_text[:80]}")

            # Clear any stale interrupt from speech_on events
            self.interrupt_event.clear()

            self.generation += 1
            self.speaking = True
            try:
                still_active = await self.standup_flow.handle(full_text, speaker, self.generation)
                if not still_active:
                    await self._finish_standup()
            except Exception as e:
                print(f"[{ts()}] {self.tag} ⚠️  Standup error: {e}")
            finally:
                self.speaking = False

            # After processing, check if more text arrived during processing
            if self._standup_buffer and self.standup_flow and not self.standup_flow.is_done:
                self._standup_timer = asyncio.create_task(self._flush_standup_buffer(speaker))

        except asyncio.CancelledError:
            pass

    def _log_sam(self, text):
        self.convo_history.append(f"Sam: {text}")
        self.agent.log_exchange("Sam", text)
        print(f"[{ts()}] {self.tag} 🗣️ Sam: {text[:100]}")

    async def _play_interrupt_ack(self):
        if not self.server._interrupt_ack_audio:
            return
        self.interrupt_event.clear()
        self.generation += 1
        self.speaking = True
        self.playing_ack = True
        try:
            text, audio = random.choice(self.server._interrupt_ack_audio)
            # For acks, always use fallback (pre-baked MP3, instant)
            await asyncio.sleep(0.5)  # Wait for Recall to fully stop previous audio
            b64 = base64.b64encode(audio).decode("utf-8")
            await self.speaker._inject_into_meeting(b64)
            self.audio_playing = True
            play_dur = get_duration_ms(audio)
            try:
                await asyncio.wait_for(self.interrupt_event.wait(), timeout=play_dur / 1000)
            except asyncio.TimeoutError:
                pass
        except Exception as e:
            print(f"[{ts()}] {self.tag} ⚠️  Ack error: {e}")
        finally:
            self.speaking = False
            self.audio_playing = False
            self.playing_ack = False

    # ── Background search ─────────────────────────────────────────────────────

    async def _search_and_speak(self, text, context, my_gen):
        summary = await self.agent.search_and_summarize(text, context)
        sentences = self.agent._split_sentences(summary)
        for sent in sentences:
            if self.interrupt_event.is_set() or my_gen != self.generation:
                return ""
            await self._speak(sent, "search", my_gen)
        return " ".join(sentences)

    # ── Jira handler ──────────────────────────────────────────────────────────

    async def _handle_jira_read(self, text, context, my_gen):
        try:
            context_block = ""
            if context:
                lines = [l for l in context.strip().split('\n') if l.strip()][-3:]
                if lines:
                    context_block = "Recent conversation:\n" + "\n".join(lines) + "\n\n"

            t0 = time.time()
            response = await self.agent.client.chat.completions.create(
                model=self.agent.model,
                messages=[{"role": "system", "content": JIRA_INTENT_PROMPT.format(project_key=self.jira.project, text=text, context_block=context_block)}, {"role": "user", "content": text}],
                temperature=0.0, max_tokens=30,
            )
            intent = response.choices[0].message.content.strip()
            print(f"[{ts()}] {self.tag} 🎫 Intent: \"{intent}\" ({(time.time()-t0)*1000:.0f}ms)")

            if intent == "MY_TICKETS":
                tickets = await self.jira.get_my_tickets()
                return {"tickets": tickets, "count": len(tickets)}
            elif intent == "SPRINT_STATUS":
                return await self.jira.get_sprint_status()
            elif intent.startswith("TICKET:"):
                ids = _re.findall(r'[A-Z]+-\d+', intent.split(":", 1)[1])
                if not ids:
                    return {"tickets": await self.jira.get_my_tickets(), "count": 0}
                if len(ids) == 1:
                    return await self.jira.get_ticket(ids[0])
                results = []
                for tid in ids[:5]:
                    try:
                        results.append(await self.jira.get_ticket(tid))
                    except Exception:
                        pass
                return {"tickets": results, "count": len(results)}
            elif intent.startswith("TRANSITION:"):
                parts = intent.split(":")
                if len(parts) >= 3:
                    tid, status = parts[1].strip(), parts[2].strip()
                    if not _re.match(r'^[A-Z]+-\d+$', tid):
                        return {"error": f"Invalid ID: {tid}"}
                    try:
                        r = await self.jira.transition_ticket(tid, status)
                        if r.get("action") == "already_done":
                            return {"action": "already_done", "ticket": tid, "message": f"{tid} already at '{r['already_at']}'."}
                        return {"action": "transition", "ticket": tid, "new_status": r["new_status"]}
                    except JiraTransitionError as e:
                        return {"action": "transition_error", "ticket": tid, "error": str(e)}
            elif intent.startswith("SEARCH:"):
                q = intent.split(":", 1)[1].strip()
                tickets = await self.jira.search_text(q, max_results=5)
                return {"tickets": tickets, "count": len(tickets)} if tickets else {"error": f"No tickets for '{q}'."}
            else:
                return {"tickets": await self.jira.get_my_tickets(), "count": 0}
        except Exception as e:
            print(f"[{ts()}] {self.tag} ⚠️  Jira error: {e}")
            return {"error": str(e)}

    # ── Main pipeline ─────────────────────────────────────────────────────────

    async def _process(self, text, speaker, t0, generation=0):
        if self.speaking:
            return
        self.speaking = True
        self.interrupt_event.clear()
        my_gen = generation
        is_direct = self._is_direct_address(text)

        try:
            context = "\n".join(self.convo_history)
            t1 = time.time()
            mode = "streaming" if self._streaming_mode else "fallback"
            print(f"[{ts()}] {self.tag} 🔊 Mode: {mode}")

            # ── Trigger: skip for direct address ──────────────────────
            if is_direct:
                print(f"  ⚡ Direct address — trigger skipped")
                should = True
            else:
                trigger_task = asyncio.create_task(
                    self.trigger.should_respond(text, speaker, context,
                                                [e["text"] for e in self.agent.rag._entries[-20:]]))
                should = await trigger_task

            if not should:
                return

            # ── Speculative execution: start router + PM LLM in parallel ──
            router_task = asyncio.create_task(self.agent._route(text, context))

            # Speculatively start PM LLM (80% of queries go to PM)
            speculative_queue = asyncio.Queue()
            speculative_llm = asyncio.create_task(
                self.agent.stream_sentences_to_queue(text, context, speculative_queue))

            route = await router_task
            print(f"[{ts()}] {self.tag} Route: [{route}] ({elapsed(t1)})")

            # ── [FT] — cancel speculative LLM ──
            if route == "FT":
                speculative_llm.cancel()
                try:
                    await speculative_llm
                except (asyncio.CancelledError, Exception):
                    pass

                self.searching = True
                filler = random.choice(FILLERS)
                await self._speak(filler, "filler", my_gen)

                if not self.interrupt_event.is_set():
                    full_text = await self._search_and_speak(text, context, my_gen)
                    if full_text:
                        self._log_sam(full_text)
                        self.trigger.mark_responded()

            # ── [JIRA] — cancel speculative LLM ──
            elif route == "JIRA":
                speculative_llm.cancel()
                try:
                    await speculative_llm
                except (asyncio.CancelledError, Exception):
                    pass

                if not self.jira.enabled:
                    await self._speak("Sorry, Jira isn't configured.", "jira-disabled", my_gen)
                    return

                await self._speak("Let me check Jira real quick.", "jira-filler", my_gen)
                if self.interrupt_event.is_set():
                    return

                jira_result = await self._handle_jira_read(text, context, my_gen)
                if self.interrupt_event.is_set():
                    return

                if jira_result:
                    jira_str = json.dumps(jira_result, indent=2, default=str)[:1500]
                    system = JIRA_RESPONSE_PROMPT.format(jira_data=jira_str)
                    try:
                        answer = await self.agent._llm_call(text, system, max_tokens=120)
                    except Exception:
                        answer = "Got data but couldn't format it."
                else:
                    answer = "Couldn't find that in Jira."

                await self._speak(answer, "jira-response", my_gen)
                self._log_sam(answer)
                self.trigger.mark_responded()

            # ── [PM] — use speculative LLM (already running!) ──
            else:
                # LLM has been running since before router finished
                # Sentences may already be in the queue
                all_sentences = []

                while True:
                    if self.interrupt_event.is_set() or my_gen != self.generation:
                        speculative_llm.cancel()
                        return
                    try:
                        item = await asyncio.wait_for(speculative_queue.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        break
                    if item is None:
                        break
                    if item == "__FLUSH__":
                        continue
                    all_sentences.append(item)

                    # Speak each sentence as it arrives (streaming mode)
                    if self._streaming_mode:
                        ok = await self._speak(item, f"sentence-{len(all_sentences)}", my_gen)
                        if not ok:
                            self._log_sam(" ".join(all_sentences) + " [interrupted]")
                            self.trigger.mark_responded()
                            return

                # In fallback mode, all sentences collected — speak as single combined
                if not self._streaming_mode and all_sentences:
                    if len(all_sentences) == 1:
                        await self._speak(all_sentences[0], "single", my_gen)
                    else:
                        from pydub import AudioSegment as _AS
                        parts = []
                        for s in all_sentences:
                            try:
                                async with self.server._tts_semaphore:
                                    parts.append(await self.speaker._synthesise(s))
                            except Exception:
                                pass
                        if parts:
                            combined = parts[0] if len(parts) == 1 else self._combine_audio(parts)
                            b64 = base64.b64encode(combined).decode("utf-8")
                            try:
                                await self.speaker.stop_audio()
                            except Exception:
                                pass
                            await self.speaker._inject_into_meeting(b64)
                            self.audio_playing = True
                            dur = get_duration_ms(combined)
                            try:
                                await asyncio.wait_for(self.interrupt_event.wait(), timeout=dur / 1000)
                            except asyncio.TimeoutError:
                                pass
                            self.audio_playing = False

                if all_sentences:
                    print(f"[{ts()}] {self.tag} 📊 TOTAL: {elapsed(t0)}")
                    self._log_sam(" ".join(all_sentences))
                    self.trigger.mark_responded()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            import traceback
            print(f"[{ts()}] {self.tag} ❌ Error: {e}")
            traceback.print_exc()
        finally:
            self.audio_playing = False
            self.speaking = False
            self.searching = False

    def _combine_audio(self, audio_list):
        from pydub import AudioSegment
        import io
        combined = AudioSegment.empty()
        for ab in audio_list:
            combined += AudioSegment.from_file(io.BytesIO(ab), format="mp3")
        output = io.BytesIO()
        combined.export(output, format="mp3", bitrate="192k")
        return output.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# WebSocketServer
# ══════════════════════════════════════════════════════════════════════════════

class WebSocketServer:
    def __init__(self, port=8000):
        self.port = port
        self.sessions = {}
        self._tts_semaphore = asyncio.Semaphore(4)
        self._interrupt_ack_audio = []
        self.debug_save_audio = os.environ.get("DEBUG_SAVE_AUDIO", "").lower() in ("1", "true", "yes")
        self.app = web.Application()
        self.app.router.add_get("/ws/{session_id}", self.handle_websocket)
        self.app.router.add_get("/audio/{session_id}", self.handle_audio_ws)
        self.app.router.add_get("/health", self.handle_health)

    async def handle_health(self, request):
        return web.json_response({"status": "ok", "sessions": len(self.sessions)})

    async def handle_websocket(self, request):
        """Recall.ai transcript/events WebSocket."""
        session_id = request.match_info.get("session_id", "")
        session = self.sessions.get(session_id)
        if not session:
            return web.Response(status=404)
        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)
        print(f"[{ts()}] {session.tag} ✅ Recall.ai WebSocket connected")
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        await session.handle_event(msg.data)
                    except Exception as e:
                        print(f"[{ts()}] {session.tag} ⚠️  Event error: {e}")
                elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING):
                    break
        except Exception as e:
            print(f"[{ts()}] {session.tag} WS error: {e}")
        finally:
            print(f"[{ts()}] {session.tag} WebSocket disconnected")
            await self.remove_session(session_id)
        return ws

    async def handle_audio_ws(self, request):
        """Output Media audio page WebSocket — receives PCM chunks to play."""
        session_id = request.match_info.get("session_id", "")
        session = self.sessions.get(session_id)
        if not session:
            return web.Response(status=404, text="Session not found")
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        session.audio_ws = ws
        print(f"[{ts()}] {session.tag} 🔊 Audio page connected (streaming mode ON)")
        try:
            async for msg in ws:
                pass  # Audio page doesn't send us data
        except Exception:
            pass
        finally:
            session.audio_ws = None
            print(f"[{ts()}] {session.tag} 🔇 Audio page disconnected (fallback mode)")
        return ws

    def create_session(self, session_id, bot_id):
        session = BotSession(session_id, bot_id, self)
        self.sessions[session_id] = session
        print(f"[{ts()}] 📦 Session created: {session_id[:12]}")
        return session

    async def remove_session(self, session_id):
        session = self.sessions.pop(session_id, None)
        if session:
            await session.cleanup()
            print(f"[{ts()}] 🗑️  Session removed: {session_id[:12]}")

    async def start(self):
        print(f"[{ts()}] Pre-baking interrupt ack audio...")
        temp = CartesiaSpeaker(bot_id=None)
        await temp.warmup()
        for phrase in _INTERRUPT_ACKS:
            try:
                async with self._tts_semaphore:
                    audio = await temp._synthesise(phrase)
                self._interrupt_ack_audio.append((phrase, audio))
            except Exception as e:
                print(f"[{ts()}] ⚠️  Pre-bake failed: {e}")
        await temp.close()
        print(f"[{ts()}] ✅ {len(self._interrupt_ack_audio)} acks pre-baked")

        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.port)
        await site.start()
        print(f"[{ts()}] WebSocket ready on ws://0.0.0.0:{self.port}/ws/{{session_id}}")
        print(f"[{ts()}] Audio relay on ws://0.0.0.0:{self.port}/audio/{{session_id}}")
        print(f"[{ts()}] Health: http://localhost:{self.port}/health\n")