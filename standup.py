"""
standup.py — Developer Standup State Machine (Production)

Architecture:
  CONVERSATION (Groq — fast, user-facing):
    Q&A:        classify + ack (parallel, ~200ms)
    Summary:    from raw answers (no extraction, fast)
    Corrections: update raw → re-summarize (fast)
    Confirm → bot leaves in 2 seconds

  BACKGROUND (Azure — reliable, after bot leaves):
    Extract structured data from confirmed raw answers
    Comment on Jira tickets
    Transition ticket statuses
    Save enriched standup data

User never waits for Azure. Entire conversation is Groq-speed.
"""

import asyncio
import time
import re
import json
import os
from enum import Enum, auto


class StandupState(Enum):
    GREETING      = auto()
    ASK_YESTERDAY = auto()
    ASK_TODAY     = auto()
    ASK_BLOCKERS  = auto()
    SUMMARY       = auto()
    CONFIRM       = auto()
    DONE          = auto()


_JIRA_ID_PATTERN = re.compile(r'\b([A-Z][A-Z0-9]+-\d+)\b')


# ══════════════════════════════════════════════════════════════════════════════
# GROQ PROMPTS (fast, user-facing — no JSON extraction)
# ══════════════════════════════════════════════════════════════════════════════

META_COMMAND_PROMPT = """You are Sam, an AI PM running a standup.

Current standup state: asking about {current_question}

CONVERSATION SO FAR:
{context}

Developer said: "{text}"

Is the developer trying to CONTROL THE STANDUP PROCESS, or are they ANSWERING THE QUESTION?

- REDO — the developer clearly and explicitly wants to restart the entire standup from scratch
- STOP — the developer clearly and explicitly wants to cancel/end the standup session entirely
- NONE — the developer is answering the question, thinking aloud, starting a sentence, giving a filler, or doing anything other than controlling the standup process

RULES:
- Default to NONE. The vast majority of inputs are answers, not commands.
- Incomplete utterances, partial sentences, and single words are NONE.
- Only return REDO or STOP when intent is unmistakably clear. If ANY ambiguity, return NONE.

Return ONLY: REDO, STOP, or NONE"""

CLASSIFY_PROMPT = """You are Sam, an AI PM running a standup. Classify the developer's response.

CONVERSATION SO FAR:
{context}

The question was about: {topic}
Developer said: "{answer}"

Classify as ONE of:
- ANSWER — a real, substantive answer (even if brief or messy speech-to-text)
- FILLER — a filler word, acknowledgment, or incomplete utterance that does NOT answer the question
- COPIES_PREVIOUS — developer said they'll do the same as yesterday / continuing previous work
- EMPTY — developer indicated nothing/none/no blockers (only valid for blockers question)

Default to ANSWER if the developer said anything meaningful about work, tasks, or tickets.

Return ONLY: ANSWER, FILLER, COPIES_PREVIOUS, or EMPTY"""

ACK_PROMPT = """You are Sam, an AI PM running a quick standup.
The developer just answered a question. Give a BRIEF acknowledgment (1 short sentence max).
Be casual, warm, specific — reference what they said. Do NOT ask the next question.

CONVERSATION SO FAR:
{context}

Developer just said: "{answer}"
Question was about: {topic}

Your brief acknowledgment (1 sentence only):"""

SUMMARY_PROMPT = """You are Sam, an AI PM wrapping up a standup. Summarize what the developer said.

Developer: {developer}

What they said for each question:
  Yesterday: "{yesterday}"
  Today: "{today}"
  Blockers: "{blockers}"

STRICT RULES:
- Exactly 3 short sentences: one for yesterday, one for today, one for blockers
- Paraphrase their words clearly — fix any speech-to-text errors
- Do NOT add commentary, analysis, suggestions, or extra detail
- End with: "Does this sound right, or do you want to change anything?"
- Keep it under 40 words before the confirmation question

Your 3-sentence summary then ask for confirmation:"""

CONFIRM_PROMPT = """You are Sam, an AI PM. The developer responded to the standup summary confirmation.
You just asked: "Does this sound right, or do you want to change anything?"

CONVERSATION SO FAR:
{context}

Current standup:
  Yesterday: "{yesterday}"
  Today: "{today}"
  Blockers: "{blockers}"

Developer just said: "{response}"

RULES:
- "Yes", "looks good", "correct", "sounds good", "yep", "right" = CONFIRMED
- "No", "nope", "not correct" WITHOUT specifying what to change = UNCLEAR
- Mentions changing/fixing yesterday's work = CORRECTION_YESTERDAY
- Mentions changing/fixing today's plan = CORRECTION_TODAY
- Mentions changing/fixing blockers = CORRECTION_BLOCKERS
- Wants to restart entirely = REDO
- Anything ambiguous = UNCLEAR

Return ONLY: CONFIRMED, CORRECTION_YESTERDAY, CORRECTION_TODAY, CORRECTION_BLOCKERS, REDO, UNCLEAR"""

# ══════════════════════════════════════════════════════════════════════════════
# AZURE PROMPT (background extraction after bot leaves)
# ══════════════════════════════════════════════════════════════════════════════

FULL_EXTRACT_PROMPT = """You are an AI PM assistant. Extract structured standup data from a developer's confirmed answers.

Developer: {developer}
Jira project key: {project_key}
Date: {date}

AVAILABLE JIRA TICKETS:
{available_tickets}

CONFIRMED STANDUP ANSWERS:
  Yesterday: "{yesterday}"
  Today: "{today}"
  Blockers: "{blockers}"

Extract structured data from ALL THREE answers:
1. Create a clean summary for each
2. Match task descriptions to available Jira tickets above
3. Detect status: "completed/finished/done/resolved" = done, "will work on/starting/continuing" = in_progress, "blocked by/stuck" = blocked

RULES:
- Only include Jira IDs that match the developer's described work to available tickets
- If "ticket fourteen" → {project_key}-14
- Do NOT invent ticket IDs not in the available list
- Include status_updates ONLY for clear status changes

Return ONLY valid JSON:
{{
  "yesterday": {{
    "summary": "clean summary",
    "tasks": ["task 1"],
    "jira_ids": ["{project_key}-14"],
    "status_updates": [{{"ticket": "{project_key}-14", "action": "done"}}]
  }},
  "today": {{
    "summary": "clean summary",
    "tasks": ["task 1"],
    "jira_ids": ["{project_key}-23"],
    "status_updates": [{{"ticket": "{project_key}-23", "action": "in_progress"}}]
  }},
  "blockers": {{
    "summary": "No blockers",
    "items": [],
    "jira_ids": []
  }}
}}"""


class StandupFlow:
    """Production standup: Groq-speed conversation, Azure background extraction."""

    def __init__(self, developer_name: str, agent, speaker_fn,
                 jira_client=None, jira_context: str = "", azure_extractor=None):
        self.developer = developer_name
        self.agent = agent
        self.speak = speaker_fn
        self.jira = jira_client
        self.azure = azure_extractor
        self._jira_context = jira_context or "(no tickets loaded)"

        self._project_key = "SCRUM"
        if self.jira and hasattr(self.jira, 'project') and self.jira.project:
            self._project_key = self.jira.project

        self.state = StandupState.GREETING
        self.data = {
            "developer": developer_name,
            "date": time.strftime("%Y-%m-%d"),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "yesterday": {"summary": "", "tasks": [], "jira_ids": [], "raw": ""},
            "today":     {"summary": "", "tasks": [], "jira_ids": [], "raw": ""},
            "blockers":  {"summary": "", "items": [], "jira_ids": [], "raw": ""},
            "completed": False,
        }

        self._silence_task = None
        self._generation = 0
        self._all_jira_ids = set()
        self._all_status_updates = []
        self._processing = False
        self._history = []

    @property
    def is_done(self) -> bool:
        return self.state == StandupState.DONE

    def _add_history(self, speaker: str, text: str):
        self._history.append(f"{speaker}: {text}")
        if len(self._history) > 20:
            self._history = self._history[-20:]

    def _get_context(self) -> str:
        return "\n".join(self._history) if self._history else "(standup just started)"

    # ── Groq (fast, user-facing) ──────────────────────────────────────────────

    async def _groq(self, system: str, user_msg: str, max_tokens: int = 100) -> str:
        response = await self.agent.client.chat.completions.create(
            model=self.agent.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
            temperature=0.3, max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    # ── Azure (reliable, background only) ─────────────────────────────────────

    async def _azure(self, system: str, user_msg: str, max_tokens: int = 500) -> str:
        if not self.azure or not self.azure.enabled:
            return await self._groq(system, user_msg, max_tokens)

        import httpx
        url = f"{self.azure.endpoint}/openai/deployments/{self.azure.deployment}/chat/completions?api-version={self.azure.api_version}"
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.post(url,
                        headers={"api-key": self.azure.api_key, "Content-Type": "application/json"},
                        json={"messages": [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
                              "temperature": 0.2, "max_tokens": max_tokens})
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                if attempt < 2:
                    print(f"[Standup] ⚠️  Azure attempt {attempt+1}/3: {e}")
                    await asyncio.sleep(1)
                else:
                    print(f"[Standup] ❌ Azure failed, falling back to Groq")
                    return await self._groq(system, user_msg, max_tokens)

    def _current_question_label(self) -> str:
        return {
            StandupState.ASK_YESTERDAY: "yesterday's work",
            StandupState.ASK_TODAY: "today's plan",
            StandupState.ASK_BLOCKERS: "blockers",
            StandupState.CONFIRM: "confirming the summary",
        }.get(self.state, "standup")

    # ══════════════════════════════════════════════════════════════════════════
    # USER-FACING FLOW (all Groq, fast)
    # ══════════════════════════════════════════════════════════════════════════

    async def start(self, gen: int):
        self._generation = gen
        greeting = f"Hey {self.developer}, let's do your standup real quick. What did you work on yesterday?"
        self._add_history("Sam", greeting)
        await self.speak(greeting, "standup-greeting", gen)
        self.state = StandupState.ASK_YESTERDAY
        self._start_silence_timer()

    async def handle(self, text: str, speaker: str, gen: int) -> bool:
        self._generation = gen
        self._cancel_silence_timer()
        if speaker.lower() == "sam":
            return not self.is_done
        self._add_history(speaker, text)
        if self._processing:
            return not self.is_done
        self._processing = True
        try:
            if self.state in (StandupState.ASK_YESTERDAY, StandupState.ASK_TODAY, StandupState.ASK_BLOCKERS):
                await self._handle_question(text, gen)
            elif self.state == StandupState.CONFIRM:
                await self._handle_confirmation(text, gen)
        finally:
            self._processing = False
        return not self.is_done

    # ── Q&A (classify + ack, all parallel, no extraction) ─────────────────────

    async def _handle_question(self, text: str, gen: int):
        topic = self._current_question_label()
        field = {StandupState.ASK_YESTERDAY: "yesterday",
                 StandupState.ASK_TODAY: "today",
                 StandupState.ASK_BLOCKERS: "blockers"}[self.state]

        # All 3 in parallel
        meta_task = asyncio.create_task(self._get_meta(text))
        classify_task = asyncio.create_task(self._classify(text, topic))
        ack_task = asyncio.create_task(self._get_ack(text, topic))

        meta = await meta_task

        if meta == "REDO":
            classify_task.cancel(); ack_task.cancel()
            r = "No problem, let's start over. What did you work on yesterday?"
            self._add_history("Sam", r)
            await self.speak(r, "standup-redo", gen)
            self._reset_data()
            self.state = StandupState.ASK_YESTERDAY
            self._start_silence_timer()
            return

        if meta == "STOP":
            classify_task.cancel(); ack_task.cancel()
            r = "Okay, standup cancelled. Let me know if you want to do it later."
            self._add_history("Sam", r)
            await self.speak(r, "standup-stop", gen)
            self.state = StandupState.DONE
            return

        classification = await classify_task

        if classification == "FILLER":
            ack_task.cancel()
            reprompts = {
                "yesterday": "Sorry, I didn't catch that. What tasks or tickets did you work on yesterday?",
                "today": "Sorry, could you repeat that? What are you planning to work on today?",
                "blockers": "Sorry, I didn't get that. Are there any blockers, or are you all clear?",
            }
            r = reprompts[field]
            self._add_history("Sam", r)
            await self.speak(r, f"standup-clarify-{field}", gen)
            self._start_silence_timer()
            return

        # Store raw answer
        if classification == "COPIES_PREVIOUS" and field == "today" and self.data["yesterday"]["raw"]:
            ack_task.cancel()
            self.data["today"]["raw"] = f"Same as yesterday: {self.data['yesterday']['raw']}"
            response = "Got it, continuing from yesterday."
        elif classification == "EMPTY" and field == "blockers":
            ack_task.cancel()
            self.data["blockers"]["raw"] = "No blockers"
            response = "All clear, no blockers."
        else:
            self.data[field]["raw"] = text
            response = await ack_task

        # Advance state
        if self.state == StandupState.ASK_YESTERDAY:
            response += " What's on your plate for today?"
            self._add_history("Sam", response)
            await self.speak(response, "standup-ack-yesterday", gen)
            self.state = StandupState.ASK_TODAY
        elif self.state == StandupState.ASK_TODAY:
            response += " Any blockers?"
            self._add_history("Sam", response)
            await self.speak(response, "standup-ack-today", gen)
            self.state = StandupState.ASK_BLOCKERS
        elif self.state == StandupState.ASK_BLOCKERS:
            self._add_history("Sam", response)
            await self.speak(response, "standup-ack-blockers", gen)
            # Go straight to summary (from raw answers, no extraction)
            self.state = StandupState.SUMMARY
            await self._speak_summary(gen)
            self.state = StandupState.CONFIRM

        self._start_silence_timer()

    # ── Summary (Groq, from raw answers — fast) ──────────────────────────────

    async def _speak_summary(self, gen: int):
        yesterday = self.data["yesterday"]["raw"] or "(no answer)"
        today = self.data["today"]["raw"] or "(no answer)"
        blockers = self.data["blockers"]["raw"] or "No blockers"

        try:
            summary = await self._groq(
                SUMMARY_PROMPT.format(developer=self.developer,
                                       yesterday=yesterday, today=today, blockers=blockers),
                "Summarize standup", max_tokens=80)
        except Exception:
            summary = (f"Yesterday: {yesterday}. Today: {today}. {blockers}. "
                       f"Does this sound right, or do you want to change anything?")
        self._add_history("Sam", summary)
        await self.speak(summary, "standup-summary", gen)

    # ── Confirmation ──────────────────────────────────────────────────────────

    async def _handle_confirmation(self, text: str, gen: int):
        yesterday = self.data["yesterday"]["raw"]
        today = self.data["today"]["raw"]
        blockers = self.data["blockers"]["raw"] or "No blockers"

        try:
            intent = await self._groq(
                CONFIRM_PROMPT.format(context=self._get_context(),
                                       yesterday=yesterday, today=today, blockers=blockers, response=text),
                text, max_tokens=20)
            intent = intent.strip().upper().replace(" ", "_")
            print(f"[Standup] 🔍 Confirm intent: {intent}")
        except Exception:
            intent = "UNCLEAR"

        if intent == "CONFIRMED":
            r = "Great, standup saved. Have a productive day!"
            self._add_history("Sam", r)
            await self.speak(r, "standup-confirmed", gen)
            self.data["completed"] = True
            self.data["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self.state = StandupState.DONE
            print(f"[Standup] ✅ {self.developer}'s standup confirmed")

        elif intent == "REDO":
            r = "No problem, let's start over. What did you work on yesterday?"
            self._add_history("Sam", r)
            await self.speak(r, "standup-redo", gen)
            self._reset_data()
            self.state = StandupState.ASK_YESTERDAY
            self._start_silence_timer()

        elif intent.startswith("CORRECTION_YESTERDAY"):
            await self._apply_correction("yesterday", text, gen)
        elif intent.startswith("CORRECTION_TODAY"):
            await self._apply_correction("today", text, gen)
        elif intent.startswith("CORRECTION_BLOCKER"):
            await self._apply_correction("blockers", text, gen)
        else:
            r = "Got it, what would you like to change? You can update yesterday, today, or blockers."
            self._add_history("Sam", r)
            await self.speak(r, "standup-unclear", gen)
            self._start_silence_timer()

    # ── Correction (update raw answer, re-summarize — Groq, fast) ─────────────

    async def _apply_correction(self, field: str, correction_text: str, gen: int):
        field_label = {"yesterday": "yesterday's work", "today": "today's plan", "blockers": "blockers"}[field]
        print(f"[Standup] ✏️  Correcting {field}: {correction_text[:60]}")

        # Update raw answer with correction
        current_raw = self.data[field]["raw"]
        self.data[field]["raw"] = f"{current_raw}. Correction: {correction_text}"

        r = f"Got it, I've updated {field_label}."
        self._add_history("Sam", r)
        await self.speak(r, "standup-correction-ack", gen)
        await self._speak_summary(gen)
        self.state = StandupState.CONFIRM
        self._start_silence_timer()

    # ══════════════════════════════════════════════════════════════════════════
    # BACKGROUND PROCESSING (Azure, after bot leaves)
    # ══════════════════════════════════════════════════════════════════════════

    async def background_finalize(self):
        """Called AFTER bot leaves. Extracts data, comments, transitions.
        Creates fresh JiraClient since session's client gets closed during cleanup."""

        print(f"[Standup] 🔧 Background: extracting structured data...")
        t0 = time.time()

        try:
            raw = await self._azure(
                FULL_EXTRACT_PROMPT.format(
                    developer=self.developer,
                    project_key=self._project_key,
                    date=self.data["date"],
                    available_tickets=self._jira_context,
                    yesterday=self.data["yesterday"]["raw"],
                    today=self.data["today"]["raw"],
                    blockers=self.data["blockers"]["raw"] or "No blockers",
                ),
                f"Extract standup for {self.developer}",
                max_tokens=500,
            )
            print(f"[Standup] ⏱ Azure extraction: {(time.time()-t0)*1000:.0f}ms")

            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r'^```(?:json)?\s*', '', raw)
                raw = re.sub(r'\s*```$', '', raw)
            extracted = json.loads(raw)

            for field in ("yesterday", "today", "blockers"):
                section = extracted.get(field, {})
                self.data[field]["summary"] = section.get("summary", self.data[field]["raw"])
                self.data[field]["tasks"] = section.get("tasks", [])
                ids = self._filter_jira_ids(section.get("jira_ids", []))
                self.data[field]["jira_ids"] = ids
                self._all_jira_ids.update(ids)
                if field == "blockers":
                    self.data[field]["items"] = section.get("items", section.get("tasks", []))

                for su in section.get("status_updates", []):
                    ticket, action = su.get("ticket", ""), su.get("action", "")
                    if ticket and action:
                        filtered = self._filter_jira_ids([ticket])
                        if filtered:
                            self._all_status_updates.append({"ticket": filtered[0], "action": action})
                            print(f"[Standup] 📌 Status: {filtered[0]} → {action}")

        except Exception as e:
            print(f"[Standup] ⚠️  Azure extraction failed: {e}")
            for field in ("yesterday", "today", "blockers"):
                if not self.data[field]["summary"]:
                    self.data[field]["summary"] = self.data[field]["raw"]

        # Create fresh JiraClient for background work (session's client is closed)
        bg_jira = None
        if self.jira and self.jira.enabled:
            try:
                from JiraClient import JiraClient
                bg_jira = JiraClient()
                if not bg_jira.enabled:
                    bg_jira = None
            except Exception as e:
                print(f"[Standup] ⚠️  Background JiraClient failed: {e}")

        if bg_jira:
            await self._auto_comment_jira(bg_jira)
            await self._auto_transition_jira(bg_jira)
            await self._auto_assign_sprint(bg_jira)
            await bg_jira.close()

        print(f"[Standup] ✅ Background processing complete ({(time.time()-t0)*1000:.0f}ms total)")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _reset_data(self):
        self.data["yesterday"] = {"summary": "", "tasks": [], "jira_ids": [], "raw": ""}
        self.data["today"]     = {"summary": "", "tasks": [], "jira_ids": [], "raw": ""}
        self.data["blockers"]  = {"summary": "", "items": [], "jira_ids": [], "raw": ""}
        self._all_jira_ids.clear()
        self._all_status_updates.clear()

    def _filter_jira_ids(self, ids: list) -> list:
        conversation_text = " ".join(self._history).upper()
        valid = []
        for tid in ids:
            if tid.upper().startswith(self._project_key.upper() + "-"):
                valid.append(tid)
            elif tid.upper() in conversation_text:
                valid.append(tid)
            else:
                print(f"[Standup] 🚫 Filtered hallucinated ID: {tid}")
        return valid

    async def _get_meta(self, text: str) -> str:
        try:
            r = await self._groq(
                META_COMMAND_PROMPT.format(current_question=self._current_question_label(),
                                           context=self._get_context(), text=text),
                text, max_tokens=10)
            result = r.strip().upper()
            print(f"[Standup] 🔍 Meta: {result}")
            return result
        except Exception:
            return "NONE"

    async def _classify(self, text: str, topic: str) -> str:
        try:
            r = await self._groq(
                CLASSIFY_PROMPT.format(context=self._get_context(), topic=topic, answer=text),
                text, max_tokens=10)
            c = r.strip().upper()
            print(f"[Standup] 📋 Classify: {c}")
            return c
        except Exception:
            return "ANSWER"

    async def _get_ack(self, text: str, topic: str) -> str:
        try:
            return (await self._groq(
                ACK_PROMPT.format(context=self._get_context(), answer=text, topic=topic),
                text, max_tokens=30)).strip()
        except Exception:
            return "Got it."

    # ── Jira (background only) ────────────────────────────────────────────────

    async def _auto_comment_jira(self, jira_client=None):
        jira = jira_client or self.jira
        if not jira or not jira.enabled or not self._all_jira_ids:
            return
        date_str = time.strftime("%B %d, %Y")
        for tid in self._all_jira_ids:
            contexts = []
            if tid in self.data["yesterday"]["jira_ids"]:
                contexts.append(f"Yesterday: {self.data['yesterday']['summary']}")
            if tid in self.data["today"]["jira_ids"]:
                contexts.append(f"Today: {self.data['today']['summary']}")
            if tid in self.data["blockers"]["jira_ids"]:
                contexts.append(f"Blocker: {self.data['blockers']['summary']}")
            if not contexts:
                contexts.append("Mentioned in standup")
            comment = f"📋 *Standup Update — {date_str}*\nDeveloper: {self.developer}\n" + "\n".join(contexts)
            try:
                await jira.add_comment(tid, comment)
                print(f"[Standup] 💬 Commented on {tid}")
            except Exception as e:
                print(f"[Standup] ⚠️  Comment on {tid} failed: {e}")

    async def _auto_transition_jira(self, jira_client=None):
        jira = jira_client or self.jira
        if not jira or not jira.enabled or not self._all_status_updates:
            return
        final = {}
        for su in self._all_status_updates:
            final[su["ticket"]] = su["action"]
        ACTION_MAP = {"done": "Done", "in_progress": "In Progress", "blocked": None}
        for tid, action in final.items():
            target = ACTION_MAP.get(action)
            if not target:
                if action == "blocked":
                    print(f"[Standup] ⚠️  {tid} blocked — noted in comment")
                continue
            try:
                result = await jira.transition_ticket(tid, target)
                if result.get("action") == "already_done":
                    print(f"[Standup] ℹ️  {tid} already at '{result['already_at']}'")
                else:
                    print(f"[Standup] 🔄 {tid}: → {result.get('new_status', target)}")
            except Exception as e:
                print(f"[Standup] ⚠️  Transition {tid} → {target} failed: {e}")

    async def _auto_assign_sprint(self, jira_client=None):
        """Assign all mentioned tickets to active sprint."""
        jira = jira_client or self.jira
        if not jira or not jira.enabled or not self._all_jira_ids:
            return
        ticket_ids = list(self._all_jira_ids)
        try:
            success = await jira.move_to_sprint(ticket_ids)
            if not success:
                print(f"[Standup] ⚠️  No active sprint — tickets remain in backlog")
        except Exception as e:
            print(f"[Standup] ⚠️  Sprint assignment failed: {e}")

    # ── Silence timer ─────────────────────────────────────────────────────────

    def _start_silence_timer(self):
        self._cancel_silence_timer()
        self._silence_task = asyncio.create_task(self._silence_reprompt())

    def _cancel_silence_timer(self):
        if self._silence_task and not self._silence_task.done():
            self._silence_task.cancel()

    async def _silence_reprompt(self):
        try:
            await asyncio.sleep(10.0)
            prompts = {
                StandupState.ASK_YESTERDAY: "Still there? What did you work on yesterday?",
                StandupState.ASK_TODAY: "What's your plan for today?",
                StandupState.ASK_BLOCKERS: "Any blockers holding you up?",
                StandupState.CONFIRM: "Is the summary correct, or do you want to change something?",
            }
            prompt = prompts.get(self.state)
            if prompt:
                print(f"[Standup] ⏰ Re-prompting ({self.state.name})")
                self._add_history("Sam", prompt)
                await self.speak(prompt, "standup-reprompt", self._generation)
                self._start_silence_timer()
        except asyncio.CancelledError:
            pass

    def get_result(self) -> dict:
        return {
            "developer": self.data["developer"],
            "date": self.data["date"],
            "started_at": self.data["started_at"],
            "completed_at": self.data.get("completed_at", ""),
            "completed": self.data["completed"],
            "yesterday": {
                "summary": self.data["yesterday"].get("summary") or self.data["yesterday"]["raw"],
                "tasks": self.data["yesterday"]["tasks"],
                "jira_ids": self.data["yesterday"]["jira_ids"],
            },
            "today": {
                "summary": self.data["today"].get("summary") or self.data["today"]["raw"],
                "tasks": self.data["today"]["tasks"],
                "jira_ids": self.data["today"]["jira_ids"],
            },
            "blockers": {
                "summary": self.data["blockers"].get("summary") or self.data["blockers"]["raw"],
                "items": self.data["blockers"]["items"],
                "jira_ids": self.data["blockers"]["jira_ids"],
            },
            "all_jira_ids": list(self._all_jira_ids),
            "status_updates": self._all_status_updates,
        }