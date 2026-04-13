"""
jira_prompts.py — Jira integration prompts + Azure GPT-4o mini extraction

During meeting: JIRA_INTENT_PROMPT (Groq) classifies what Jira op to run
                JIRA_RESPONSE_PROMPT (Groq) formats Jira data for voice
Post-meeting:   EXTRACTION_PROMPT (Azure GPT-4o mini) extracts action items
"""

import os
import json
import httpx


# ── Prompt for Jira read responses (Groq, during meeting) ────────────────────

JIRA_RESPONSE_PROMPT = """You are Sam, a PM on a live voice call. You just looked up Jira ticket info.

Jira data:
{jira_data}

Rules:
- Speak the info naturally, like telling a coworker.
- If there are MORE than 5 tickets, read the top 5 by priority, then say "Want me to go through the rest?"
- If 5 or fewer, read all of them.
- For each ticket mention: ticket ID, summary, status, and assignee if assigned.
- Don't read raw JSON — summarize it conversationally.
- If a transition was successful, confirm it naturally.
- If a transition failed, explain what moves are available.
- If the ticket is already at the requested status, say so naturally.
- No markdown, no lists — this is voice output.
- Keep it under 30 seconds of speaking time."""


# ── Jira intent classifier (Groq, during meeting) ────────────────────────────

JIRA_INTENT_PROMPT = """You are a Jira intent classifier. Given a user message from a meeting, determine what Jira operation they want.

The Jira project key is: {project_key}

{context_block}User message: "{text}"

Classify into EXACTLY one of these formats:

MY_TICKETS — user wants to see their open/assigned tickets, any tickets in the project, available tickets
SPRINT_STATUS — user wants sprint overview, progress, how many done/in-progress
TICKET:ID — user wants details about a specific ticket (e.g. TICKET:SCRUM-123). For MULTIPLE tickets use comma: TICKET:SCRUM-15,SCRUM-12
TRANSITION:ID:STATUS — user wants to move a ticket to a new status (e.g. TRANSITION:SCRUM-123:Done). Status must be one of: Done, In Progress, To Do
SEARCH:QUERY — user wants to find tickets by topic/description (e.g. SEARCH:login page bug)

Rules:
- For ticket IDs: convert spoken numbers to actual IDs. "project one two three" = {project_key}-123, "scrum six" = {project_key}-6, "ticket 456" = {project_key}-456
- If user says "define those tickets", "tell me more about those", "what are they about" — look at the RECENT CONVERSATION for ticket IDs that were just mentioned and return TICKET:ID,ID
- If user says "check one more time" or similar without a real ticket reference, use MY_TICKETS
- If user asks about "open tickets", "any tickets", "available tickets" — use MY_TICKETS
- If unsure, default to MY_TICKETS

Reply with ONLY the classification. Nothing else."""


# ── Prompt for post-meeting extraction (Azure GPT-4o mini) ───────────────────

EXTRACTION_PROMPT = """You are an AI assistant that extracts action items from meeting transcripts.

Analyze the following meeting transcript and extract all actionable items including:
- Bugs reported (anything broken, crashing, slow, not working)
- Feature requests (new functionality requested)
- Tasks assigned (someone asked to do something)
- Decisions made that need tracking
- Blockers mentioned

For each item, output a JSON object with these fields:
- "type": one of "Bug", "Story", "Task"
- "summary": short title (max 80 chars)
- "description": detailed description with context from the meeting
- "priority": "Highest", "High", "Medium", or "Low"
- "labels": list of relevant labels (see rules below)
- "assignee": person's name if someone was explicitly assigned this task, otherwise null

Rules:
- Only extract ACTIONABLE items, not general discussion
- ONLY extract items from what HUMAN participants said — IGNORE everything Sam said (lines starting with "Sam:")
- IGNORE any bot errors, connection failures, DNS issues, or technical problems with Sam himself — these are NOT action items
- Include meeting context in descriptions (who said what, why it matters)
- If someone was assigned a task, mention them in the description
- If no action items found, return empty list []
- Be specific in summaries — "Login page crash on Chrome Android" not "Bug fix needed"
- Labels must include "client-feedback" and "session-{meeting_date}" for all items
- For bugs also add "bug" label
- For feature requests add "feature-request" label
- For blockers add "blocker" label
- For urgent/critical items add "critical" label

Respond with ONLY a JSON array. No markdown, no explanation, no backticks.

Meeting date: {meeting_date}

Meeting transcript:
{transcript}

Extract action items as JSON array:"""


# ── Azure GPT-4o mini client ─────────────────────────────────────────────────

class AzureExtractor:
    """Post-meeting action item extractor using Azure OpenAI GPT-4o mini."""

    def __init__(self):
        self.endpoint   = os.environ.get("AZURE_ENDPOINT", "").strip().rstrip("/")
        self.api_key    = os.environ.get("AZURE_API_KEY", "").strip()
        self.deployment = os.environ.get("AZURE_DEPLOYMENT", "gpt-4o-mini").strip()
        self.api_version = os.environ.get("AZURE_API_VERSION", "2024-02-15-preview").strip()

        if not self.endpoint or not self.api_key:
            print("[Azure] ⚠️  Missing AZURE_ENDPOINT or AZURE_API_KEY — post-meeting extraction disabled")
            self.enabled = False
            return

        self.enabled = True
        self._client = httpx.AsyncClient(timeout=60)
        print(f"[Azure] ✅ Configured: {self.endpoint} (deployment: {self.deployment})")

    async def extract_action_items(self, transcript: str, date_str: str = "") -> list[dict]:
        if not self.enabled:
            print("[Azure] ⚠️  Extraction disabled — no Azure credentials")
            return []

        if not transcript or len(transcript.strip()) < 50:
            print("[Azure] ⚠️  Transcript too short — skipping extraction")
            return []

        import time
        if not date_str:
            date_str = time.strftime("%Y-%m-%d")

        prompt = EXTRACTION_PROMPT.format(
            transcript=transcript[:12000],
            meeting_date=date_str,
        )

        url = f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"

        try:
            print(f"[Azure] 🤖 Extracting action items ({len(transcript)} chars transcript)...")
            t0 = time.time()

            response = None
            for attempt in range(3):
                try:
                    response = await self._client.post(
                        url,
                        headers={"api-key": self.api_key, "Content-Type": "application/json"},
                        json={
                            "messages": [
                                {"role": "system", "content": "You extract action items from meeting transcripts. Respond with JSON only."},
                                {"role": "user", "content": prompt},
                            ],
                            "temperature": 0.2,
                            "max_tokens": 2000,
                        },
                    )
                    break
                except Exception as net_err:
                    if "getaddrinfo" in str(net_err) or "ConnectError" in str(net_err):
                        if attempt < 2:
                            import asyncio
                            wait = 2.0 * (attempt + 1)
                            print(f"[Azure] ⚠️  DNS error (attempt {attempt+1}/3), retrying in {wait}s...")
                            await asyncio.sleep(wait)
                            continue
                    raise

            if response is None:
                print("[Azure] ❌ All retry attempts failed")
                return []

            if response.status_code != 200:
                print(f"[Azure] ❌ API error {response.status_code}: {response.text[:300]}")
                return []

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                print(f"[Azure] ❌ No choices in response: {str(data)[:300]}")
                return []
            content = choices[0].get("message", {}).get("content", "")
            if not content:
                print(f"[Azure] ❌ Empty content in response")
                return []

            ms = (time.time() - t0) * 1000
            print(f"[Azure] ⏱ Extraction done: {ms:.0f}ms")
            print(f"[Azure] 📝 Raw response: {content[:200]}")

            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1]
                content = content.rsplit("```", 1)[0]
            content = content.strip()

            items = json.loads(content)
            if not isinstance(items, list):
                print(f"[Azure] ⚠️  Expected list, got {type(items)}")
                return []

            valid_items = []
            for item in items:
                if isinstance(item, dict) and "summary" in item and "type" in item:
                    item.setdefault("priority", "Medium")
                    item.setdefault("description", item["summary"])
                    item.setdefault("labels", [])
                    if "client-feedback" not in item["labels"]:
                        item["labels"].append("client-feedback")
                    session_label = f"session-{date_str}"
                    if session_label not in item["labels"]:
                        item["labels"].append(session_label)
                    valid_items.append(item)

            print(f"[Azure] ✅ Extracted {len(valid_items)} action item(s)")
            for i, item in enumerate(valid_items):
                print(f"[Azure]   {i+1}. [{item['type']}] {item['summary']} ({item['priority']}) labels={item['labels']}")

            return valid_items

        except json.JSONDecodeError as e:
            print(f"[Azure] ❌ JSON parse failed: {e}")
            print(f"[Azure]   Raw content: {content[:200]}")
            return []
        except Exception as e:
            print(f"[Azure] ❌ Extraction failed: {e}")
            return []

    async def close(self):
        if self.enabled:
            await self._client.aclose()