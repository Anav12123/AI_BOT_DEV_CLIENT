# """
# Agent.py — Groq Llama 3.3 70B + In-Memory RAG

# Memory architecture:
#   1. RAG store: Every exchange is embedded (Azure OpenAI text-embedding-3-small)
#      and stored in-memory. On query, cosine similarity finds relevant past exchanges.
#      Catches semantic matches like "money" → "budget" that keyword search misses.
#   2. Meeting log: Full text of every exchange (fallback + debugging)
#   3. Recent history: Last 10 LLM turns for conversation flow

# Embedding happens async in background — never blocks the response pipeline.
# If embeddings fail, falls back to keyword search automatically.
# """

# import os
# import asyncio
# import re
# import time
# import numpy as np
# from openai import AsyncOpenAI
# from typing import List, Optional


# # ── Debug logger — writes all prompt inputs to file ──────────────────────────
# DEBUG_PROMPTS_FILE = "debug_prompts.txt"

# def _debug_log(label: str, **kwargs):
#     """Append a debug entry to debug_prompts.txt with timestamp and all variables."""
#     try:
#         ts = time.strftime("%H:%M:%S")
#         with open(DEBUG_PROMPTS_FILE, "a", encoding="utf-8") as f:
#             f.write(f"\n{'='*80}\n")
#             f.write(f"[{ts}] {label}\n")
#             f.write(f"{'='*80}\n")
#             for key, val in kwargs.items():
#                 val_str = str(val) if val else "(EMPTY)"
#                 f.write(f"  {key}:\n    {val_str}\n")
#             f.write(f"\n")
#     except Exception:
#         pass  # never crash on debug logging


# # ══════════════════════════════════════════════════════════════════════════════
# # IN-MEMORY RAG STORE
# # ══════════════════════════════════════════════════════════════════════════════

# class MeetingRAG:
#     """In-memory vector store for meeting transcripts.
#     Uses fastembed (free, local, ~200MB). No API key needed.
#     Model: BAAI/bge-small-en-v1.5 — 130MB, 384-dim, fast on CPU.
#     """

#     def __init__(self):
#         self._entries: list[dict] = []
#         self._embed_queue: asyncio.Queue = asyncio.Queue()
#         self._embed_task: Optional[asyncio.Task] = None
#         self._model = None
#         self._ready = False

#         try:
#             from fastembed import TextEmbedding
#             self._model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
#             self._ready = True
#             print("[RAG] Local embeddings ready (BAAI/bge-small-en-v1.5, fastembed)")
#         except ImportError:
#             print("[RAG] ⚠️  fastembed not installed — keyword fallback only")
#         except Exception as e:
#             print(f"[RAG] ⚠️  Model load failed: {e} — keyword fallback only")

#     def start_background_embedder(self):
#         if self._ready and not self._embed_task:
#             self._embed_task = asyncio.create_task(self._embedding_worker())

#     async def _embedding_worker(self):
#         loop = asyncio.get_event_loop()
#         while True:
#             try:
#                 entry = await self._embed_queue.get()
#                 vector = await loop.run_in_executor(
#                     None, self._embed_sync, entry["text"]
#                 )
#                 if vector is not None:
#                     entry["vector"] = vector
#                     self._entries.append(entry)
#             except asyncio.CancelledError:
#                 break
#             except Exception as e:
#                 print(f"[RAG] Embed worker error: {e}")

#     def _embed_sync(self, text: str) -> Optional[np.ndarray]:
#         if not self._model:
#             return None
#         try:
#             # fastembed returns a generator — get first result
#             vectors = list(self._model.embed([text]))
#             return np.array(vectors[0], dtype=np.float32)
#         except Exception as e:
#             print(f"[RAG] Embedding failed: {e}")
#             return None

#     def add(self, speaker: str, text: str):
#         """Queue an exchange for embedding (non-blocking)."""
#         entry = {
#             "text": f"{speaker}: {text}",
#             "speaker": speaker,
#             "time": time.time(),
#             "vector": None,
#         }
#         if self._ready:
#             try:
#                 self._embed_queue.put_nowait(entry)
#             except Exception:
#                 pass
#         else:
#             self._entries.append(entry)

#     async def search(self, query: str, top_k: int = 5) -> List[str]:
#         """Find relevant past exchanges by cosine similarity.
#         Falls back to keyword matching if embeddings unavailable.
#         """
#         if not self._entries:
#             return []

#         # Vector search
#         if self._ready and self._model:
#             loop = asyncio.get_event_loop()
#             query_vector = await loop.run_in_executor(
#                 None, self._embed_sync, query
#             )

#             if query_vector is not None:
#                 scored = []
#                 for entry in self._entries:
#                     if entry["vector"] is not None:
#                         sim = self._cosine_sim(query_vector, entry["vector"])
#                         scored.append((sim, entry["text"]))

#                 if scored:
#                     scored.sort(key=lambda x: x[0], reverse=True)
#                     results = [text for sim, text in scored[:top_k] if sim > 0.3]
#                     if results:
#                         print(f"[RAG] Vector search: {len(results)} hits for \"{query[:50]}\"")
#                         return results

#         # Fallback: keyword search
#         return self._keyword_search(query, top_k)

#     def _keyword_search(self, query: str, top_k: int = 5, exclude_text: str = "") -> List[str]:
#         stop = {"the", "a", "an", "is", "are", "was", "were", "what", "who",
#                 "how", "when", "where", "why", "did", "do", "does", "can",
#                 "could", "would", "should", "we", "i", "you", "they", "it",
#                 "about", "tell", "me", "something", "discuss", "talked", "sam"}
#         query_words = {w for w in query.lower().split() if w not in stop and len(w) > 2}
#         if not query_words:
#             return []
#         # Normalize exclude text for comparison
#         exclude_lower = exclude_text.lower().strip() if exclude_text else ""
#         scored = []
#         for entry in self._entries:
#             entry_lower = entry["text"].lower().strip()
#             # Skip if this entry IS the current utterance (prevent echo)
#             if exclude_lower and (exclude_lower in entry_lower or entry_lower.endswith(exclude_lower)):
#                 continue
#             hits = sum(1 for w in query_words if w in entry_lower)
#             if hits > 0:
#                 scored.append((hits, entry["text"]))
#         scored.sort(key=lambda x: x[0], reverse=True)
#         results = [text for _, text in scored[:top_k]]
#         if results:
#             print(f"[RAG] Keyword fallback: {len(results)} hits for \"{query[:50]}\"")
#         return results

#     @staticmethod
#     def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
#         dot = np.dot(a, b)
#         norm = np.linalg.norm(a) * np.linalg.norm(b)
#         return float(dot / norm) if norm > 0 else 0.0

#     @property
#     def size(self) -> int:
#         return len(self._entries)

#     def clear(self):
#         self._entries.clear()


# # ══════════════════════════════════════════════════════════════════════════════
# # PROMPTS
# # ══════════════════════════════════════════════════════════════════════════════

# # ── FAST ROUTER — classifies before main LLM ────────────────────────────────
# ROUTER_PROMPT = """Classify this message. Reply with ONLY one tag:
# [PM] — greetings, small talk, personal questions, jokes, opinions, PM work topics (agenda, blockers, sprint status, timeline, team updates), conversation about ongoing work, AND any question about what was said earlier in THIS conversation (memory, recall, "what did I ask", "repeat that", "come again", "previous question").
# [FT] — ANY question needing verifiable facts from the OUTSIDE WORLD: who is someone (CEO, founder, president), company info (rates, pricing, services), real-world knowledge (people, places, events, news, weather, science, history, dates, numbers, statistics).

# IMPORTANT:
# - If the user asks WHO someone is (CEO, founder, manager), or asks about PRICES/RATES/COSTS — always [FT], even if they say "our company".
# - If the user asks about what THEY said earlier, what was discussed, or asks you to repeat — always [PM], this is conversation memory not web search.

# Examples:
# "Hi Sam, how are you?" → [PM]
# "What's on the agenda?" → [PM]
# "Who is the CEO of our company?" → [FT]
# "What are the hourly rates?" → [FT]
# "Who is the prime minister of India?" → [FT]
# "Tell me about yourself" → [PM]
# "What's the founder's name?" → [FT]
# "How's the sprint going?" → [PM]
# "What services does our company offer?" → [FT]
# "What was my previous question?" → [PM]
# "Can you repeat that?" → [PM]
# "What did I just ask you?" → [PM]
# "Come again?" → [PM]
# "Describe yourself in three words" → [PM]

# ONLY reply with [PM] or [FT]. Nothing else."""

# # ── PM PROMPT — only handles personality + PM answers ────────────────────────
# PM_PROMPT = """You are Sam, a senior PM at AnavClouds Software Solutions. You're on a live voice call in a meeting.

# HOW YOU TALK:
# - Like a real person in a meeting, not a chatbot. Use "yeah", "honestly", "look", "so basically".
# - React to what the person ACTUALLY said before answering. Show you're listening.
# - Use their name naturally (not every time — that's weird).
# - Contractions always. Never say "I am" when you can say "I'm".
# - Throw in light humor when it fits. You're the fun PM, not the boring one.
# - If they ask for a joke, deliver the FULL joke with setup AND punchline in one response.

# YOUR BACKGROUND:
# - Senior PM at AnavClouds (Salesforce + AI company). You handle sprints, budgets, timelines, CRM rollouts.
# - Confident, warm, slightly sarcastic. You deflect weird questions with humor.
# - You remember what was said in the meeting (see MEETING MEMORY below).

# RULES:
# - 1-2 sentences. Max 20 words per sentence.
# - If user asks to REPEAT something: rephrase your last answer differently.
# - Say something NEW each time — don't repeat yourself.
# - No markdown, no lists, no asterisks, no bullet points, no parenthetical actions like (laughs) or (sighs) — this is voice output, not a script.

# MEETING MEMORY: Use this to answer about past discussions if provided."""

# SEARCH_SUMMARY_PROMPT = """You are Sam, a PM on a live voice call. You just searched the web for the user.

# Search results:
# {search_results}

# Rules:
# - Skip any filler — the user already heard one.
# - Jump straight to the answer.
# - 2-3 SHORT sentences max. Keep each under 18 words.
# - Sound like you're telling a coworker, not reading a report.
# - If the results don't have the answer, just say so naturally — don't apologize."""

# INTERRUPT_PROMPT = """You are Sam, a witty senior PM. You were interrupted.
# Reply in ONE sentence — 15 words max. Be quick, natural.
# Start with: "Oh," / "Right," / "Sure," / "Got it," — then pivot to their question."""

# SEARCH_QUERY_PROMPT = """Convert the user's message into a DESCRIPTIVE Google search query (8-15 words).
# Make the query specific and detailed so Google returns the best results.
# Replace 'our company'/'my company'/'our'/'we' with 'AnavClouds Software Solutions'.
# If the user does NOT mention the company, do NOT add AnavClouds.
# Remove 'Sam' and filler words.
# If the message is about conversation memory (what did I ask, repeat, previous question) — output: SKIP
# For multi-part questions, include all parts in a single descriptive query.

# Examples:
# "What are the hourly rates?" → "What is the hourly rate and pricing of AnavClouds Software Solutions"
# "Who is the CEO?" → "Who is the CEO and founder of AnavClouds Software Solutions company"
# "Who is the prime minister of India?" → "Who is the current prime minister of India 2025"
# "What's the weather in Delhi?" → "What is the weather forecast in Delhi India today"
# "Tell me about the company" → "AnavClouds Software Solutions company overview services and products"
# "What is the density of India and who founded your company?" → "Population density of India and founder of AnavClouds Software Solutions"
# "How many employees?" → "How many employees work at AnavClouds Software Solutions team size"
# "What was my previous question?" → SKIP

# Output ONLY the search query. No quotes, no explanation."""

# FILLERS = [
#     "Hmm, let me look that up real quick.",
#     "Right, give me one sec to check on that.",
#     "Uh, good question — let me pull that up.",
#     "Yeah, hold on, let me find that for you.",
#     "Well, let me check on that real quick.",
# ]


# # ── End-of-Turn Classifier (RESPOND vs WAIT) ────────────────────────────────
# EOT_PROMPT = """You are Sam, an AI participant in a live meeting. Someone is speaking to you. Based on the conversation, decide:

# Should you RESPOND now, or WAIT for them to continue?

# {context_block}Current utterance: "{text}"

# RESPOND if ANY of these are true:
# - The utterance contains a question (even rhetorical like "Right?" or "You know?")
# - The utterance is a request or command ("Tell me...", "Can you...")
# - The utterance is a short reaction or statement directed at you
# - The speaker seems to be done talking and waiting for your input
# - You are unsure — RESPOND is always safer than making someone wait

# WAIT only if ALL of these are true:
# - The utterance is clearly mid-sentence (grammatically incomplete)
# - OR the speaker is obviously setting up a longer explanation and hasn't asked anything yet
# - AND there is no question, request, or floor-handoff signal

# Reply with one word: RESPOND or WAIT"""


# # ══════════════════════════════════════════════════════════════════════════════
# # PM AGENT
# # ══════════════════════════════════════════════════════════════════════════════

# class PMAgent:
#     def __init__(self):
#         self.client = AsyncOpenAI(
#             api_key=os.environ["GROQ_API_KEY"],
#             base_url="https://api.groq.com/openai/v1",
#         )
#         self.model = "llama-3.1-8b-instant"

#         # Recent LLM history — last 10 turns
#         self.history: list[dict] = []

#         # RAG store — embeds + retrieves meeting exchanges
#         self.rag = MeetingRAG()

#     def start(self):
#         """Call once after event loop is running to start background embedder + warmup."""
#         self.rag.start_background_embedder()
#         asyncio.create_task(self._warmup())

#     async def _warmup(self):
#         """Pre-establish TCP connection to Groq — saves ~300ms on first real call."""
#         try:
#             await self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[{"role": "user", "content": "hi"}],
#                 max_tokens=1,
#             )
#             print("[Agent] ✅ Groq connection warmed up")
#         except Exception:
#             pass

#     def _get_web_search(self):
#         if not hasattr(self, '_web_search') or self._web_search is None:
#             from WebSearch import WebSearch
#             self._web_search = WebSearch()
#         return self._web_search

#     # ── Memory ────────────────────────────────────────────────────────────────

#     def log_exchange(self, speaker: str, text: str):
#         """Store an exchange in RAG. Called by websocket_server for every transcript."""
#         self.rag.add(speaker, text)

#     async def _build_context(self, user_text: str, context: str) -> str:
#         """Build meeting context for the system prompt (NOT the user message).
#         Returns a string to append to the system prompt with relevant memory + recent convo.
#         Filters current utterance from RAG to prevent echo."""
#         parts = []

#         # Fast keyword search — exclude current utterance to prevent echo
#         rag_results = self.rag._keyword_search(user_text, top_k=2, exclude_text=user_text)
#         if rag_results:
#             parts.append("MEETING MEMORY (relevant past exchanges):\n" + "\n".join(rag_results))

#         # Recent conversation — last 4 lines for flow
#         if context:
#             recent = "\n".join(context.split("\n")[-4:])
#             parts.append(f"RECENT CONVERSATION:\n{recent}")

#         if not parts:
#             return ""

#         full_context = "\n\n".join(parts)

#         # _debug_log("BUILD CONTEXT (system prompt appendix)",
#         #            user_text=user_text,
#         #            convo_history_raw=context or "(EMPTY)",
#         #            rag_results=rag_results or "(NONE)",
#         #            built_context=full_context)

#         return full_context

#     # ── Search signal ─────────────────────────────────────────────────────────

#     def _is_search_signal(self, text: str) -> bool:
#         upper = text.strip().upper()
#         return upper.strip("[]").strip() == "SEARCH" or "[SEARCH]" in upper

#     # ── Fast Router — [PM] or [FT] classification ──────────────────────────

#     async def _route(self, user_text: str) -> str:
#         """Ultra-fast classification: [PM] or [FT]. ~100-150ms on 8b."""
#         import time as _t
#         t0 = _t.time()
#         # _debug_log("ROUTER", user_text=user_text)
#         try:
#             response = await self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[
#                     {"role": "system", "content": ROUTER_PROMPT},
#                     {"role": "user", "content": user_text},
#                 ],
#                 temperature=0.0,
#                 max_tokens=5,
#             )
#             tag = response.choices[0].message.content.strip().upper()
#             ms = (_t.time() - t0) * 1000
#             route = "FT" if "[FT]" in tag or "FT" in tag else "PM"
#             print(f"[Agent] ⏱ Router: [{route}] ({ms:.0f}ms)")
#             return route
#         except Exception as e:
#             print(f"[Agent] Router failed: {e} — defaulting PM")
#             return "PM"

#     # ── End-of-Turn Classifier ────────────────────────────────────────────────

#     async def check_end_of_turn(self, text: str, context: str = "") -> str:
#         """Decide if the speaker expects a response now or is still talking.
#         Returns 'RESPOND' or 'WAIT'.
#         Uses conversation context for better decisions.
#         Defaults to RESPOND on timeout/error (don't leave user hanging)."""
#         import time as _t
#         import re as _re
#         t0 = _t.time()
#         # Clean transcript artifacts
#         clean_text = _re.sub(r'\s+', ' ', text).strip()

#         # Build context block (last 3 exchanges)
#         context_block = ""
#         if context:
#             lines = [l for l in context.strip().split('\n') if l.strip()][-3:]
#             if lines:
#                 context_block = "Recent conversation:\n" + "\n".join(lines) + "\n\n"

#         # _debug_log("EOT CHECK",
#         #            utterance=clean_text,
#         #            context_raw=context or "(EMPTY)",
#         #            context_block=context_block or "(EMPTY)",
#         #            full_prompt=EOT_PROMPT.format(text=clean_text, context_block=context_block))

#         try:
#             response = await asyncio.wait_for(
#                 self.client.chat.completions.create(
#                     model=self.model,
#                     messages=[
#                         {"role": "system", "content": EOT_PROMPT.format(
#                             text=clean_text,
#                             context_block=context_block
#                         )},
#                         {"role": "user", "content": "RESPOND or WAIT?"},
#                     ],
#                     temperature=0.0,
#                     max_tokens=3,
#                 ),
#                 timeout=0.5,
#             )
#             result = response.choices[0].message.content.strip().upper()
#             # Parse: check for WAIT first (since RESPOND doesn't contain WAIT)
#             if "WAIT" in result:
#                 decision = "WAIT"
#             else:
#                 decision = "RESPOND"
#             ms = (_t.time() - t0) * 1000
#             emoji = "🟢" if decision == "RESPOND" else "🟡"
#             print(f"[EOT] {emoji} {decision} ({ms:.0f}ms): \"{clean_text[:60]}\"")
#             return decision
#         except asyncio.TimeoutError:
#             print(f"[EOT] ⏱ Timeout — defaulting RESPOND")
#             return "RESPOND"
#         except Exception as e:
#             print(f"[EOT] Error: {e} — defaulting RESPOND")
#             return "RESPOND"

#     # ── LLM search query conversion ──────────────────────────────────────────

#     async def _to_english_search_query(self, user_text: str, context: str) -> str:
#         clean = re.sub(r'\[LANG:\w+\]\s*', '', user_text).strip()
#         context_hint = ""
#         if context:
#             recent = context.split("\n")[-3:]
#             context_hint = "\nRecent conversation:\n" + "\n".join(recent)

#         # _debug_log("SEARCH QUERY CONVERSION",
#         #            user_text=clean,
#         #            context_hint=context_hint or "(EMPTY)",
#         #            full_system_prompt=SEARCH_QUERY_PROMPT + context_hint)
#         try:
#             response = await self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[
#                     {"role": "system", "content": SEARCH_QUERY_PROMPT + context_hint},
#                     {"role": "user", "content": clean},
#                 ],
#                 temperature=0.0,
#                 max_tokens=30,
#             )
#             query = response.choices[0].message.content.strip().strip('"\'')
#             # Safety: only use first line (LLM sometimes outputs multiple)
#             query = query.split('\n')[0].strip()
#             print(f"[Agent] LLM search query: \"{clean}\" → \"{query}\"")
#             return query
#         except Exception as e:
#             print(f"[Agent] Query conversion failed: {e}")
#             return clean

#     # ── Background search (runs independently, survives interrupts) ──────────

#     async def search_and_summarize(self, user_text: str, context: str) -> str:
#         """Full search pipeline. Returns summary text. Safe to run as background task."""
#         search_query = await self._to_english_search_query(user_text, context)
#         try:
#             results = await self._get_web_search().search(search_query)
#             if not results:
#                 # _debug_log("SEARCH SUMMARY", search_query=search_query, results="(NONE)")
#                 return "Hmm, couldn't find that online right now."
#             system = SEARCH_SUMMARY_PROMPT.format(search_results=results[:800])
#             # _debug_log("SEARCH SUMMARY",
#             #            search_query=search_query,
#             #            results_preview=results[:200],
#             #            user_text=user_text)
#             response = await self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[
#                     {"role": "system", "content": system},
#                     {"role": "user", "content": user_text},
#                 ],
#                 temperature=0.5,
#                 max_tokens=120,
#             )
#             answer = response.choices[0].message.content.strip()
#             self.history.append({"role": "user", "content": user_text})
#             self.history.append({"role": "assistant", "content": answer})
#             return answer
#         except Exception as e:
#             print(f"[Agent] search_and_summarize failed: {e}")
#             return "Hmm, I couldn't look that up right now."

#     # ── Core: respond (non-streaming) ────────────────────────────────────────

#     async def respond(self, user_text: str) -> str:
#         return await self.respond_with_context(user_text, "")

#     async def respond_with_context(self, user_text: str, context: str, interrupted: bool = False) -> str:
#         meeting_context = await self._build_context(user_text, context)

#         if interrupted:
#             system = INTERRUPT_PROMPT
#             if meeting_context:
#                 system = INTERRUPT_PROMPT + "\n\n" + meeting_context
#             return await self._llm_call(user_text, system, max_tokens=25)

#         # Fast route: [PM] or [FT]?
#         route = await self._route(user_text)

#         if route == "FT":
#             # Skip LLM entirely — go straight to search
#             print(f"[Agent] Router → [FT] — searching: {user_text}")
#             search_query = await self._to_english_search_query(user_text, context)
#             try:
#                 results = await self._get_web_search().search(search_query)
#                 if not results:
#                     return "Hmm, couldn't find that online right now."
#                 system = SEARCH_SUMMARY_PROMPT.format(search_results=results[:800])
#                 return await self._llm_call(user_text, system, max_tokens=120)
#             except Exception as e:
#                 print(f"[Agent] Web search failed: {e}")
#                 return "Hmm, I couldn't look that up right now."

#         # PM path — answer directly, context in system prompt
#         system = PM_PROMPT
#         if meeting_context:
#             system = PM_PROMPT + "\n\n" + meeting_context
#         return await self._llm_call(user_text, system, max_tokens=120)

#     # ── Core: streaming (used by websocket_server) ───────────────────────────

#     async def stream_sentences_to_queue(self, user_text: str, context: str, queue: asyncio.Queue):
#         """Stream PM response sentences to queue. PM-only — [FT] handled by websocket_server."""
#         import time as _t

#         t0 = _t.time()
#         meeting_context = await self._build_context(user_text, context)
#         rag_ms = (_t.time() - t0) * 1000
#         print(f"[Agent] ⏱ RAG context: {rag_ms:.0f}ms")

#         # Store CLEAN user text in history (not context blocks!)
#         self.history.append({"role": "user", "content": user_text})
#         if len(self.history) > 6:
#             self.history = self.history[-6:]

#         # Build system prompt: PM personality + meeting context
#         system_prompt = PM_PROMPT
#         if meeting_context:
#             system_prompt = PM_PROMPT + "\n\n" + meeting_context

#         total_chars = len(system_prompt) + sum(len(m["content"]) for m in self.history)
#         print(f"[Agent] ⏱ Context size: {total_chars} chars (~{total_chars//4} tokens)")

#         # _debug_log("PM STREAM",
#         #            user_text=user_text,
#         #            convo_history_raw=context or "(EMPTY)",
#         #            meeting_context=meeting_context or "(NONE)",
#         #            llm_history=[f"{m['role']}: {m['content'][:80]}" for m in self.history],
#         #            system_prompt_preview=system_prompt[-300:])

#         try:
#             t1 = _t.time()
#             stream = await self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[{"role": "system", "content": system_prompt}] + self.history,
#                 temperature=0.7,
#                 max_tokens=120,
#                 stream=True,
#             )
#             stream_open_ms = (_t.time() - t1) * 1000
#             print(f"[Agent] ⏱ Stream opened: {stream_open_ms:.0f}ms")

#             buffer = ""
#             full_response = ""
#             first_token_time = None
#             sentence_count = 0
#             async for chunk in stream:
#                 token = chunk.choices[0].delta.content if chunk.choices else None
#                 if not token:
#                     continue

#                 if first_token_time is None:
#                     first_token_time = _t.time()
#                     ttft_ms = (first_token_time - t1) * 1000
#                     print(f"[Agent] ⏱ First token: {ttft_ms:.0f}ms")

#                 buffer += token
#                 full_response += token

#                 while True:
#                     indices = [buffer.find(c) for c in ".!?" if buffer.find(c) != -1]
#                     if not indices:
#                         break
#                     idx = min(indices)
#                     sentence = buffer[:idx+1].strip()
#                     buffer = buffer[idx+1:].lstrip()
#                     if sentence and len(sentence) > 2:  # skip punctuation-only fragments like "."
#                         # Strip emote/action markers — TTS would say "(laughs)" literally
#                         sentence = re.sub(r'\([^)]*\)', '', sentence).strip()
#                         if not sentence or len(sentence) <= 2:
#                             continue
#                         sentence_count += 1
#                         sent_ms = (_t.time() - t1) * 1000
#                         print(f"[Agent] ⏱ Sentence {sentence_count} ready: {sent_ms:.0f}ms")
#                         await queue.put(sentence)

#             llm_total_ms = (_t.time() - t1) * 1000
#             print(f"[Agent] ⏱ LLM total: {llm_total_ms:.0f}ms ({len(full_response.split())} words)")

#             if buffer.strip() and len(buffer.strip()) > 2:
#                 clean_buf = re.sub(r'\([^)]*\)', '', buffer).strip()
#                 if clean_buf and len(clean_buf) > 2:
#                     await queue.put(clean_buf)
#             self.history.append({"role": "assistant", "content": full_response.strip()})

#         except Exception as e:
#             print(f"[Agent] LLM error: {e}")
#             await queue.put("Hmm, something went wrong on my end.")
#         finally:
#             await queue.put(None)

#     # ── Helpers ───────────────────────────────────────────────────────────────

#     async def _llm_call(self, user_msg: str, system: str, max_tokens: int = 60) -> str:
#         """LLM call with clean history. user_msg goes as clean text, not context dump."""
#         self.history.append({"role": "user", "content": user_msg})
#         if len(self.history) > 6:
#             self.history = self.history[-6:]

#         stream = await self.client.chat.completions.create(
#             model=self.model,
#             messages=[{"role": "system", "content": system}] + self.history,
#             temperature=0.7,
#             max_tokens=max_tokens,
#             stream=True,
#         )

#         tokens = []
#         async for chunk in stream:
#             t = chunk.choices[0].delta.content if chunk.choices else None
#             if t:
#                 tokens.append(t)

#         result = "".join(tokens).strip()
#         self.history.append({"role": "assistant", "content": result})
#         return result

#     def _split_sentences(self, text: str) -> list[str]:
#         parts = re.split(r'(?<=[.!?])\s+', text.strip())
#         return [p.strip() for p in parts if p.strip()]

#     def reset(self):
#         self.history.clear()
#         self.rag.clear()

"""
Agent.py — Groq Llama 3.3 70B + In-Memory RAG

Memory architecture:
  1. RAG store: Every exchange is embedded (Azure OpenAI text-embedding-3-small)
     and stored in-memory. On query, cosine similarity finds relevant past exchanges.
     Catches semantic matches like "money" → "budget" that keyword search misses.
  2. Meeting log: Full text of every exchange (fallback + debugging)
  3. Recent history: Last 10 LLM turns for conversation flow

Embedding happens async in background — never blocks the response pipeline.
If embeddings fail, falls back to keyword search automatically.
"""

import os
import asyncio
import re
import time
import numpy as np
from openai import AsyncOpenAI
from typing import List, Optional


# ── Debug logger — writes all prompt inputs to file ──────────────────────────
DEBUG_PROMPTS_FILE = "debug_prompts.txt"

def _debug_log(label: str, **kwargs):
    """Append a debug entry to debug_prompts.txt with timestamp and all variables."""
    if os.environ.get("DEBUG_SAVE_AUDIO", "").lower() not in ("1", "true", "yes"):
        return
    try:
        ts = time.strftime("%H:%M:%S")
        with open(DEBUG_PROMPTS_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"[{ts}] {label}\n")
            f.write(f"{'='*80}\n")
            for key, val in kwargs.items():
                val_str = str(val) if val else "(EMPTY)"
                f.write(f"  {key}:\n    {val_str}\n")
            f.write(f"\n")
    except Exception:
        pass  # never crash on debug logging


# ══════════════════════════════════════════════════════════════════════════════
# IN-MEMORY RAG STORE
# ══════════════════════════════════════════════════════════════════════════════

class MeetingRAG:
    """In-memory vector store for meeting transcripts.
    Uses fastembed (free, local, ~200MB). No API key needed.
    Model: BAAI/bge-small-en-v1.5 — 130MB, 384-dim, fast on CPU.
    """

    def __init__(self):
        self._entries: list[dict] = []
        self._embed_queue: asyncio.Queue = asyncio.Queue()
        self._embed_task: Optional[asyncio.Task] = None
        self._model = None
        self._ready = False

        try:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            self._ready = True
            print("[RAG] Local embeddings ready (BAAI/bge-small-en-v1.5, fastembed)")
        except ImportError:
            print("[RAG] ⚠️  fastembed not installed — keyword fallback only")
        except Exception as e:
            print(f"[RAG] ⚠️  Model load failed: {e} — keyword fallback only")

    def start_background_embedder(self):
        if self._ready and not self._embed_task:
            self._embed_task = asyncio.create_task(self._embedding_worker())

    async def _embedding_worker(self):
        loop = asyncio.get_event_loop()
        while True:
            try:
                entry = await self._embed_queue.get()
                vector = await loop.run_in_executor(
                    None, self._embed_sync, entry["text"]
                )
                if vector is not None:
                    entry["vector"] = vector
                    self._entries.append(entry)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[RAG] Embed worker error: {e}")

    def _embed_sync(self, text: str) -> Optional[np.ndarray]:
        if not self._model:
            return None
        try:
            # fastembed returns a generator — get first result
            vectors = list(self._model.embed([text]))
            return np.array(vectors[0], dtype=np.float32)
        except Exception as e:
            print(f"[RAG] Embedding failed: {e}")
            return None

    def add(self, speaker: str, text: str):
        """Queue an exchange for embedding (non-blocking)."""
        entry = {
            "text": f"{speaker}: {text}",
            "speaker": speaker,
            "time": time.time(),
            "vector": None,
        }
        if self._ready:
            try:
                self._embed_queue.put_nowait(entry)
            except Exception:
                pass
        else:
            self._entries.append(entry)

    async def search(self, query: str, top_k: int = 5) -> List[str]:
        """Find relevant past exchanges by cosine similarity.
        Falls back to keyword matching if embeddings unavailable.
        """
        if not self._entries:
            return []

        # Vector search
        if self._ready and self._model:
            loop = asyncio.get_event_loop()
            query_vector = await loop.run_in_executor(
                None, self._embed_sync, query
            )

            if query_vector is not None:
                scored = []
                for entry in self._entries:
                    if entry["vector"] is not None:
                        sim = self._cosine_sim(query_vector, entry["vector"])
                        scored.append((sim, entry["text"]))

                if scored:
                    scored.sort(key=lambda x: x[0], reverse=True)
                    results = [text for sim, text in scored[:top_k] if sim > 0.3]
                    if results:
                        print(f"[RAG] Vector search: {len(results)} hits for \"{query[:50]}\"")
                        return results

        # Fallback: keyword search
        return self._keyword_search(query, top_k)

    def _keyword_search(self, query: str, top_k: int = 5, exclude_text: str = "") -> List[str]:
        stop = {"the", "a", "an", "is", "are", "was", "were", "what", "who",
                "how", "when", "where", "why", "did", "do", "does", "can",
                "could", "would", "should", "we", "i", "you", "they", "it",
                "about", "tell", "me", "something", "discuss", "talked", "sam"}
        query_words = {w for w in query.lower().split() if w not in stop and len(w) > 2}
        if not query_words:
            return []
        # Normalize exclude text for comparison
        exclude_lower = exclude_text.lower().strip() if exclude_text else ""
        scored = []
        for entry in self._entries:
            entry_lower = entry["text"].lower().strip()
            # Skip if this entry IS the current utterance (prevent echo)
            if exclude_lower and (exclude_lower in entry_lower or entry_lower.endswith(exclude_lower)):
                continue
            hits = sum(1 for w in query_words if w in entry_lower)
            if hits > 0:
                scored.append((hits, entry["text"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [text for _, text in scored[:top_k]]
        if results:
            print(f"[RAG] Keyword fallback: {len(results)} hits for \"{query[:50]}\"")
        return results

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(dot / norm) if norm > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._entries)

    def clear(self):
        self._entries.clear()


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

# ── FAST ROUTER — classifies before main LLM ────────────────────────────────
ROUTER_PROMPT = """Classify this message. Reply with ONLY one tag:
[PM] — greetings, small talk, personal questions, jokes, opinions, PM work topics (agenda, blockers, sprint status, timeline, team updates), conversation about ongoing work, AND any question about what was said earlier in THIS conversation (memory, recall, "what did I ask", "repeat that", "come again", "previous question").
[FT] — ANY question needing verifiable facts from the OUTSIDE WORLD: who is someone (CEO, founder, president), company info (rates, pricing, services), real-world knowledge (people, places, events, news, weather, science, history, dates, numbers, statistics).
[JIRA] — ONLY when the user explicitly asks to CHECK, QUERY, or MANAGE Jira: checking ticket status by ID, listing open tickets, sprint status, moving/transitioning tickets. NOT for reporting bugs, requesting features, or discussing issues — those are [PM] (they get saved and uploaded to Jira after the meeting ends automatically).

IMPORTANT:
- If the user asks WHO someone is (CEO, founder, manager), or asks about PRICES/RATES/COSTS — always [FT], even if they say "our company".
- If the user asks about what THEY said earlier, what was discussed, or asks you to repeat — always [PM], this is conversation memory not web search.
- If Sam ALREADY answered a similar question in the recent conversation, route to [PM] — no need to search again.
- If the question is about Sam personally (who are you, what do you do, who do you work for) — always [PM], Sam knows his own identity.
- [JIRA] is ONLY for explicit Jira operations: "check my tickets", "sprint status", "status of ticket 123", "move ticket to done". If user is REPORTING a problem ("login is crashing") or REQUESTING a feature ("we need CSV export"), that is [PM] — NOT [JIRA].

Examples:
"Hi Sam, how are you?" → [PM]
"What's on the agenda?" → [PM]
"Who is the CEO of our company?" → [FT]
"What are the hourly rates?" → [FT]
"Who is the prime minister of India?" → [FT]
"Tell me about yourself" → [PM]
"What's the founder's name?" → [FT]
"How's the sprint going?" → [JIRA]
"What services does our company offer?" → [FT]
"What was my previous question?" → [PM]
"Can you repeat that?" → [PM]
"What did I just ask you?" → [PM]
"Come again?" → [PM]
"Describe yourself in three words" → [PM]
"Whom do you work for?" → [PM]
"Where do you work?" → [PM]
"What do you do?" → [PM]
"What's your role?" → [PM]
"What's the status of project 123?" → [JIRA]
"What are my open tickets?" → [JIRA]
"Move project 456 to done" → [JIRA]
"What's the sprint status?" → [JIRA]
"How many tickets are in progress?" → [JIRA]
"The login page is crashing" → [PM]
"We need to add CSV export" → [PM]
"There's a bug with the payment page" → [PM]
"Can we add dark mode?" → [PM]
"The API is slow" → [PM]
"Create a ticket for the login bug" → [JIRA]

ONLY reply with [PM], [FT], or [JIRA]. Nothing else."""

# ── PM PROMPT — only handles personality + PM answers ────────────────────────
PM_PROMPT = """You are Sam, a senior PM at AnavClouds Software Solutions. You're on a live voice call in a meeting.

HOW YOU TALK:
- Like a real person in a meeting, not a chatbot. Use "yeah", "honestly", "look", "so basically".
- React to what the person ACTUALLY said before answering. Show you're listening.
- Use their name naturally (not every time — that's weird).
- Contractions always. Never say "I am" when you can say "I'm".
- Throw in light humor when it fits. You're the fun PM, not the boring one.
- If they ask for a joke, deliver the FULL joke with setup AND punchline in one response.

YOUR BACKGROUND:
- Senior PM at AnavClouds (Salesforce + AI company). You handle sprints, budgets, timelines, CRM rollouts.
- Confident, warm, slightly sarcastic. You deflect weird questions with humor.
- You remember what was said in the meeting (see MEETING MEMORY below).

RULES:
- 1-2 sentences. Max 20 words per sentence.
- If user asks to REPEAT something: rephrase your last answer differently.
- Say something NEW each time — don't repeat yourself.
- No markdown, no lists, no asterisks, no bullet points, no parenthetical actions like (laughs) or (sighs) — this is voice output, not a script.

MEETING MEMORY: Use this to answer about past discussions if provided."""

SEARCH_SUMMARY_PROMPT = """You are Sam, a PM on a live voice call. You just searched the web for the user.

Search results:
{search_results}

Rules:
- Skip any filler — the user already heard one.
- Jump straight to the answer.
- 2-3 SHORT sentences max. Keep each under 18 words.
- Sound like you're telling a coworker, not reading a report.
- If the results don't have the answer, just say so naturally — don't apologize."""

INTERRUPT_PROMPT = """You are Sam, a witty senior PM. You were interrupted.
Reply in ONE sentence — 15 words max. Be quick, natural.
Start with: "Oh," / "Right," / "Sure," / "Got it," — then pivot to their question."""

SEARCH_QUERY_PROMPT = """Convert the user's message into a DESCRIPTIVE Google search query (8-15 words).
Make the query specific and detailed so Google returns the best results.
Replace 'our company'/'my company'/'our'/'we' with 'AnavClouds Software Solutions'.
If the user does NOT mention the company, do NOT add AnavClouds.
Remove 'Sam' and filler words.
If the message is about conversation memory (what did I ask, repeat, previous question) — output: SKIP
For multi-part questions, include all parts in a single descriptive query.

Examples:
"What are the hourly rates?" → "What is the hourly rate and pricing of AnavClouds Software Solutions"
"Who is the CEO?" → "Who is the CEO and founder of AnavClouds Software Solutions company"
"Who is the prime minister of India?" → "Who is the current prime minister of India 2025"
"What's the weather in Delhi?" → "What is the weather forecast in Delhi India today"
"Tell me about the company" → "AnavClouds Software Solutions company overview services and products"
"What is the density of India and who founded your company?" → "Population density of India and founder of AnavClouds Software Solutions"
"How many employees?" → "How many employees work at AnavClouds Software Solutions team size"
"What was my previous question?" → SKIP

Output ONLY the search query. No quotes, no explanation."""

FILLERS = [
    "Hmm, let me look that up real quick.",
    "Right, give me one sec to check on that.",
    "Uh, good question — let me pull that up.",
    "Yeah, hold on, let me find that for you.",
    "Well, let me check on that real quick.",
]


# ── End-of-Turn Classifier (RESPOND vs WAIT) ────────────────────────────────
EOT_PROMPT = """You are Sam, an AI participant in a live meeting. Someone is speaking to you. Based on the conversation, decide:

Should you RESPOND now, or WAIT for them to continue?

{context_block}Current utterance: "{text}"

RESPOND if ANY of these are true:
- The utterance contains a question (even rhetorical like "Right?" or "You know?")
- The utterance is a request or command ("Tell me...", "Can you...")
- The utterance is a short reaction or statement directed at you
- The speaker seems to be done talking and waiting for your input
- You are unsure — RESPOND is always safer than making someone wait

WAIT only if ALL of these are true:
- The utterance is clearly mid-sentence (grammatically incomplete)
- OR the speaker is obviously setting up a longer explanation and hasn't asked anything yet
- AND there is no question, request, or floor-handoff signal

Reply with one word: RESPOND or WAIT"""


# ══════════════════════════════════════════════════════════════════════════════
# PM AGENT
# ══════════════════════════════════════════════════════════════════════════════

class PMAgent:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = "llama-3.1-8b-instant"

        # Recent LLM history — last 10 turns
        self.history: list[dict] = []

        # RAG store — embeds + retrieves meeting exchanges
        self.rag = MeetingRAG()

    def start(self):
        """Call once after event loop is running to start background embedder + warmup."""
        self.rag.start_background_embedder()
        asyncio.create_task(self._warmup())

    async def _warmup(self):
        """Pre-establish TCP connection to Groq — saves ~300ms on first real call."""
        try:
            await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            print("[Agent] ✅ Groq connection warmed up")
        except Exception:
            pass

    def _get_web_search(self):
        if not hasattr(self, '_web_search') or self._web_search is None:
            from WebSearch import WebSearch
            self._web_search = WebSearch()
        return self._web_search

    # ── Memory ────────────────────────────────────────────────────────────────

    def log_exchange(self, speaker: str, text: str):
        """Store an exchange in RAG. Called by websocket_server for every transcript."""
        self.rag.add(speaker, text)

    async def _build_context(self, user_text: str, context: str) -> str:
        """Build meeting context for the system prompt (NOT the user message).
        Returns a string to append to the system prompt with relevant memory + recent convo.
        Filters current utterance from RAG to prevent echo."""
        parts = []

        # Fast keyword search — exclude current utterance to prevent echo
        rag_results = self.rag._keyword_search(user_text, top_k=2, exclude_text=user_text)
        if rag_results:
            parts.append("MEETING MEMORY (relevant past exchanges):\n" + "\n".join(rag_results))

        # Recent conversation — last 4 lines for flow
        if context:
            recent = "\n".join(context.split("\n")[-4:])
            parts.append(f"RECENT CONVERSATION:\n{recent}")

        if not parts:
            return ""

        full_context = "\n\n".join(parts)

        _debug_log("BUILD CONTEXT (system prompt appendix)",
                   user_text=user_text,
                   convo_history_raw=context or "(EMPTY)",
                   rag_results=rag_results or "(NONE)",
                   built_context=full_context)

        return full_context

    # ── Search signal ─────────────────────────────────────────────────────────

    def _is_search_signal(self, text: str) -> bool:
        upper = text.strip().upper()
        return upper.strip("[]").strip() == "SEARCH" or "[SEARCH]" in upper

    # ── Fast Router — [PM] or [FT] classification ──────────────────────────

    async def _route(self, user_text: str, context: str = "") -> str:
        """Ultra-fast classification: [PM] or [FT]. ~100-150ms on 8b."""
        import time as _t
        t0 = _t.time()
        _debug_log("ROUTER", user_text=user_text)

        # Add recent conversation so router can see what's been discussed
        ctx_hint = ""
        if context:
            lines = [l for l in context.strip().split('\n') if l.strip()][-3:]
            if lines:
                ctx_hint = "\n\nRecent conversation:\n" + "\n".join(lines) + "\n\nNow classify the LATEST message only:"

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ROUTER_PROMPT + ctx_hint},
                    {"role": "user", "content": user_text},
                ],
                temperature=0.0,
                max_tokens=5,
            )
            tag = response.choices[0].message.content.strip().upper()
            ms = (_t.time() - t0) * 1000
            if "[JIRA]" in tag or "JIRA" in tag:
                route = "JIRA"
            elif "[FT]" in tag or "FT" in tag:
                route = "FT"
            else:
                route = "PM"
            print(f"[Agent] ⏱ Router: [{route}] ({ms:.0f}ms)")
            return route
        except Exception as e:
            print(f"[Agent] Router failed: {e} — defaulting PM")
            return "PM"

    # ── End-of-Turn Classifier ────────────────────────────────────────────────

    async def check_end_of_turn(self, text: str, context: str = "") -> str:
        """Decide if the speaker expects a response now or is still talking.
        Returns 'RESPOND' or 'WAIT'.
        Uses conversation context for better decisions.
        Defaults to RESPOND on timeout/error (don't leave user hanging)."""
        import time as _t
        import re as _re
        t0 = _t.time()
        # Clean transcript artifacts
        clean_text = _re.sub(r'\s+', ' ', text).strip()

        # Build context block (last 3 exchanges)
        context_block = ""
        if context:
            lines = [l for l in context.strip().split('\n') if l.strip()][-3:]
            if lines:
                context_block = "Recent conversation:\n" + "\n".join(lines) + "\n\n"

        _debug_log("EOT CHECK",
                   utterance=clean_text,
                   context_raw=context or "(EMPTY)",
                   context_block=context_block or "(EMPTY)",
                   full_prompt=EOT_PROMPT.format(text=clean_text, context_block=context_block))

        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": EOT_PROMPT.format(
                            text=clean_text,
                            context_block=context_block
                        )},
                        {"role": "user", "content": "RESPOND or WAIT?"},
                    ],
                    temperature=0.0,
                    max_tokens=3,
                ),
                timeout=0.5,
            )
            result = response.choices[0].message.content.strip().upper()
            # Parse: check for WAIT first (since RESPOND doesn't contain WAIT)
            if "WAIT" in result:
                decision = "WAIT"
            else:
                decision = "RESPOND"
            ms = (_t.time() - t0) * 1000
            emoji = "🟢" if decision == "RESPOND" else "🟡"
            print(f"[EOT] {emoji} {decision} ({ms:.0f}ms): \"{clean_text[:60]}\"")
            return decision
        except asyncio.TimeoutError:
            print(f"[EOT] ⏱ Timeout — defaulting RESPOND")
            return "RESPOND"
        except Exception as e:
            print(f"[EOT] Error: {e} — defaulting RESPOND")
            return "RESPOND"

    # ── LLM search query conversion ──────────────────────────────────────────

    async def _to_english_search_query(self, user_text: str, context: str) -> str:
        clean = re.sub(r'\[LANG:\w+\]\s*', '', user_text).strip()
        context_hint = ""
        if context:
            recent = context.split("\n")[-3:]
            context_hint = "\nRecent conversation:\n" + "\n".join(recent)

        _debug_log("SEARCH QUERY CONVERSION",
                   user_text=clean,
                   context_hint=context_hint or "(EMPTY)",
                   full_system_prompt=SEARCH_QUERY_PROMPT + context_hint)
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SEARCH_QUERY_PROMPT + context_hint},
                    {"role": "user", "content": clean},
                ],
                temperature=0.0,
                max_tokens=30,
            )
            query = response.choices[0].message.content.strip().strip('"\'')
            # Safety: only use first line (LLM sometimes outputs multiple)
            query = query.split('\n')[0].strip()
            print(f"[Agent] LLM search query: \"{clean}\" → \"{query}\"")
            return query
        except Exception as e:
            print(f"[Agent] Query conversion failed: {e}")
            return clean

    # ── Background search (runs independently, survives interrupts) ──────────

    async def search_and_summarize(self, user_text: str, context: str) -> str:
        """Full search pipeline. Returns summary text. Safe to run as background task."""
        search_query = await self._to_english_search_query(user_text, context)
        try:
            results = await self._get_web_search().search(search_query)
            if not results:
                _debug_log("SEARCH SUMMARY", search_query=search_query, results="(NONE)")
                return "Hmm, couldn't find that online right now."
            system = SEARCH_SUMMARY_PROMPT.format(search_results=results[:800])
            _debug_log("SEARCH SUMMARY",
                       search_query=search_query,
                       results_preview=results[:200],
                       user_text=user_text)
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_text},
                ],
                temperature=0.5,
                max_tokens=120,
            )
            answer = response.choices[0].message.content.strip()
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": answer})
            return answer
        except Exception as e:
            print(f"[Agent] search_and_summarize failed: {e}")
            return "Hmm, I couldn't look that up right now."

    # ── Core: respond (non-streaming) ────────────────────────────────────────

    async def respond(self, user_text: str) -> str:
        return await self.respond_with_context(user_text, "")

    async def respond_with_context(self, user_text: str, context: str, interrupted: bool = False) -> str:
        meeting_context = await self._build_context(user_text, context)

        if interrupted:
            system = INTERRUPT_PROMPT
            if meeting_context:
                system = INTERRUPT_PROMPT + "\n\n" + meeting_context
            return await self._llm_call(user_text, system, max_tokens=25)

        # Fast route: [PM] or [FT]?
        route = await self._route(user_text, context)

        if route == "FT":
            # Skip LLM entirely — go straight to search
            print(f"[Agent] Router → [FT] — searching: {user_text}")
            search_query = await self._to_english_search_query(user_text, context)
            try:
                results = await self._get_web_search().search(search_query)
                if not results:
                    return "Hmm, couldn't find that online right now."
                system = SEARCH_SUMMARY_PROMPT.format(search_results=results[:800])
                return await self._llm_call(user_text, system, max_tokens=120)
            except Exception as e:
                print(f"[Agent] Web search failed: {e}")
                return "Hmm, I couldn't look that up right now."

        # PM path — answer directly, context in system prompt
        system = PM_PROMPT
        if meeting_context:
            system = PM_PROMPT + "\n\n" + meeting_context
        return await self._llm_call(user_text, system, max_tokens=120)

    # ── Core: streaming (used by websocket_server) ───────────────────────────

    async def stream_sentences_to_queue(self, user_text: str, context: str, queue: asyncio.Queue):
        """Stream PM response sentences to queue. PM-only — [FT] handled by websocket_server."""
        import time as _t

        t0 = _t.time()
        meeting_context = await self._build_context(user_text, context)
        rag_ms = (_t.time() - t0) * 1000
        print(f"[Agent] ⏱ RAG context: {rag_ms:.0f}ms")

        # Store CLEAN user text in history (not context blocks!)
        self.history.append({"role": "user", "content": user_text})
        if len(self.history) > 6:
            self.history = self.history[-6:]

        # Build system prompt: PM personality + meeting context
        system_prompt = PM_PROMPT
        if meeting_context:
            system_prompt = PM_PROMPT + "\n\n" + meeting_context

        total_chars = len(system_prompt) + sum(len(m["content"]) for m in self.history)
        print(f"[Agent] ⏱ Context size: {total_chars} chars (~{total_chars//4} tokens)")

        _debug_log("PM STREAM",
                   user_text=user_text,
                   convo_history_raw=context or "(EMPTY)",
                   meeting_context=meeting_context or "(NONE)",
                   llm_history=[f"{m['role']}: {m['content'][:80]}" for m in self.history],
                   system_prompt_preview=system_prompt[-300:])

        try:
            t1 = _t.time()
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt}] + self.history,
                temperature=0.7,
                max_tokens=120,
                stream=True,
            )
            stream_open_ms = (_t.time() - t1) * 1000
            print(f"[Agent] ⏱ Stream opened: {stream_open_ms:.0f}ms")

            buffer = ""
            full_response = ""
            first_token_time = None
            sentence_count = 0
            async for chunk in stream:
                token = chunk.choices[0].delta.content if chunk.choices else None
                if not token:
                    continue

                if first_token_time is None:
                    first_token_time = _t.time()
                    ttft_ms = (first_token_time - t1) * 1000
                    print(f"[Agent] ⏱ First token: {ttft_ms:.0f}ms")

                buffer += token
                full_response += token

                while True:
                    indices = [buffer.find(c) for c in ".!?" if buffer.find(c) != -1]
                    if not indices:
                        break
                    idx = min(indices)
                    sentence = buffer[:idx+1].strip()
                    buffer = buffer[idx+1:].lstrip()
                    if sentence and len(sentence) > 2:  # skip punctuation-only fragments like "."
                        # Strip emote/action markers — TTS would say "(laughs)" literally
                        sentence = re.sub(r'\([^)]*\)', '', sentence).strip()
                        if not sentence or len(sentence) <= 2:
                            continue
                        sentence_count += 1
                        sent_ms = (_t.time() - t1) * 1000
                        print(f"[Agent] ⏱ Sentence {sentence_count} ready: {sent_ms:.0f}ms")
                        await queue.put(sentence)

            llm_total_ms = (_t.time() - t1) * 1000
            print(f"[Agent] ⏱ LLM total: {llm_total_ms:.0f}ms ({len(full_response.split())} words)")

            if buffer.strip() and len(buffer.strip()) > 2:
                clean_buf = re.sub(r'\([^)]*\)', '', buffer).strip()
                if clean_buf and len(clean_buf) > 2:
                    await queue.put(clean_buf)
            self.history.append({"role": "assistant", "content": full_response.strip()})

        except Exception as e:
            print(f"[Agent] LLM error: {e}")
            await queue.put("Hmm, something went wrong on my end.")
        finally:
            await queue.put(None)

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _llm_call(self, user_msg: str, system: str, max_tokens: int = 60) -> str:
        """LLM call with clean history. user_msg goes as clean text, not context dump."""
        self.history.append({"role": "user", "content": user_msg})
        if len(self.history) > 6:
            self.history = self.history[-6:]

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}] + self.history,
            temperature=0.7,
            max_tokens=max_tokens,
            stream=True,
        )

        tokens = []
        async for chunk in stream:
            t = chunk.choices[0].delta.content if chunk.choices else None
            if t:
                tokens.append(t)

        result = "".join(tokens).strip()
        self.history.append({"role": "assistant", "content": result})
        return result

    def _split_sentences(self, text: str) -> list[str]:
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p.strip() for p in parts if p.strip()]

    def reset(self):
        self.history.clear()
        self.rag.clear()