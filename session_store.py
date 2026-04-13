"""
session_store.py — JSON file-based persistence

Stores:
  - sessions.json: meeting history (transcript, action items, Jira keys)
  - settings.json: Jira/Azure config (editable from UI)
  - pending_tickets.json: failed Jira tickets for retry
"""

import json
import os
import time
import threading

SESSIONS_FILE = os.environ.get("SESSIONS_FILE", "sessions.json")
SETTINGS_FILE = os.environ.get("SETTINGS_FILE", "settings.json")
PENDING_FILE  = os.environ.get("PENDING_FILE", "pending_tickets.json")
_lock = threading.Lock()


# ── Generic JSON helpers ──────────────────────────────────────────────────────

def _load_json(path: str, default=None):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"[Store] ⚠️  Load {path} failed: {e}")
    return default if default is not None else {}

def _save_json(path: str, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    except Exception as e:
        print(f"[Store] ⚠️  Save {path} failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# SESSION HISTORY
# ══════════════════════════════════════════════════════════════════════════════

def save_session(session_data: dict):
    with _lock:
        sessions = _load_json(SESSIONS_FILE, [])
        sessions.insert(0, session_data)
        if len(sessions) > 200:
            sessions = sessions[:200]
        _save_json(SESSIONS_FILE, sessions)
    sid = session_data.get("session_id", "?")[:12]
    print(f"[SessionStore] 💾 Saved session {sid} ({len(session_data.get('action_items', []))} items)")

def get_sessions(limit: int = 50, user: str = None) -> list[dict]:
    sessions = _load_json(SESSIONS_FILE, [])
    if user:
        sessions = [s for s in sessions if s.get("user") == user]
    summaries = []
    for s in sessions[:limit]:
        summaries.append({
            "session_id": s.get("session_id", ""),
            "date": s.get("date", ""),
            "user": s.get("user", ""),
            "mode": s.get("mode", "client_call"),
            "project": s.get("project", ""),
            "meeting_url": s.get("meeting_url", ""),
            "duration_minutes": s.get("duration_minutes", 0),
            "summary": s.get("summary", ""),
            "feedback_count": s.get("feedback_count", 0),
            "tickets_created": s.get("tickets_created", 0),
            "action_items": s.get("action_items", []),
        })
    return summaries

def get_session_detail(session_id: str) -> dict | None:
    sessions = _load_json(SESSIONS_FILE, [])
    for s in sessions:
        if s.get("session_id") == session_id:
            return s
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS (Jira/Azure config, editable from UI)
# ══════════════════════════════════════════════════════════════════════════════

def load_settings() -> dict:
    """Load saved settings. Falls back to env vars if no settings file."""
    saved = _load_json(SETTINGS_FILE, {})
    return {
        "jira_url":     saved.get("jira_url",     os.environ.get("JIRA_BASE_URL", "")),
        "jira_email":   saved.get("jira_email",   os.environ.get("JIRA_EMAIL", "")),
        "jira_token":   saved.get("jira_token",   os.environ.get("JIRA_API_TOKEN", "")),
        "jira_project": saved.get("jira_project", os.environ.get("JIRA_DEFAULT_PROJECT", "")),
        "jira_sprint":  saved.get("jira_sprint",  ""),
        "azure_endpoint":   saved.get("azure_endpoint",   os.environ.get("AZURE_ENDPOINT", "")),
        "azure_key":        saved.get("azure_key",        os.environ.get("AZURE_API_KEY", "")),
        "azure_deployment": saved.get("azure_deployment", os.environ.get("AZURE_DEPLOYMENT", "gpt-4o-mini")),
    }

def save_settings(settings: dict):
    """Save settings to file AND update env vars for current process."""
    with _lock:
        _save_json(SETTINGS_FILE, settings)
    # Also update env vars so JiraClient picks them up on next init
    env_map = {
        "jira_url":     "JIRA_BASE_URL",
        "jira_email":   "JIRA_EMAIL",
        "jira_token":   "JIRA_API_TOKEN",
        "jira_project": "JIRA_DEFAULT_PROJECT",
        "azure_endpoint":   "AZURE_ENDPOINT",
        "azure_key":        "AZURE_API_KEY",
        "azure_deployment": "AZURE_DEPLOYMENT",
    }
    for key, env_name in env_map.items():
        val = settings.get(key, "")
        if val:
            os.environ[env_name] = val
    print(f"[Settings] 💾 Saved + env updated (project: {settings.get('jira_project', '?')})")


# ══════════════════════════════════════════════════════════════════════════════
# PENDING TICKETS (failed Jira creates — retry later)
# ══════════════════════════════════════════════════════════════════════════════

def save_pending_ticket(ticket_data: dict):
    """Save a failed ticket for later retry."""
    with _lock:
        pending = _load_json(PENDING_FILE, [])
        ticket_data["failed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        pending.append(ticket_data)
        _save_json(PENDING_FILE, pending)
    print(f"[Pending] 💾 Saved pending ticket: {ticket_data.get('summary', '?')[:50]}")

def get_pending_tickets() -> list[dict]:
    return _load_json(PENDING_FILE, [])

def clear_pending_tickets():
    with _lock:
        _save_json(PENDING_FILE, [])
    print("[Pending] 🗑️  Cleared all pending tickets")

def remove_pending_ticket(index: int):
    """Remove a single pending ticket by index after successful sync."""
    with _lock:
        pending = _load_json(PENDING_FILE, [])
        if 0 <= index < len(pending):
            pending.pop(index)
            _save_json(PENDING_FILE, pending)


# ══════════════════════════════════════════════════════════════════════════════
# STANDUPS (structured developer standup data)
# ══════════════════════════════════════════════════════════════════════════════

STANDUPS_FILE = os.environ.get("STANDUPS_FILE", "standups.json")


def save_standup(standup_data: dict):
    """Save a completed standup."""
    with _lock:
        standups = _load_json(STANDUPS_FILE, [])
        standups.insert(0, standup_data)
        if len(standups) > 500:
            standups = standups[:500]
        _save_json(STANDUPS_FILE, standups)
    dev = standup_data.get("developer", "?")
    date = standup_data.get("date", "?")
    print(f"[Standup] 💾 Saved standup for {dev} on {date}")


def get_team_standups(date: str = None, user: str = None) -> list[dict]:
    """Get all standups for a given date (defaults to today).
    Returns list of standup summaries for the PM dashboard."""
    if not date:
        date = time.strftime("%Y-%m-%d")
    standups = _load_json(STANDUPS_FILE, [])
    results = []
    for s in standups:
        if s.get("date") != date:
            continue
        results.append({
            "developer": s.get("developer", "?"),
            "date": s.get("date", ""),
            "completed": s.get("completed", False),
            "started_at": s.get("started_at", ""),
            "completed_at": s.get("completed_at", ""),
            "yesterday": s.get("yesterday", {}).get("summary", ""),
            "today": s.get("today", {}).get("summary", ""),
            "blockers": s.get("blockers", {}).get("summary", "No blockers"),
            "blocker_count": len(s.get("blockers", {}).get("items", [])),
            "jira_ids": s.get("all_jira_ids", []),
        })
    return results


def get_standup_detail(developer: str, date: str = None) -> dict | None:
    """Get full standup detail for a specific developer on a date."""
    if not date:
        date = time.strftime("%Y-%m-%d")
    standups = _load_json(STANDUPS_FILE, [])
    for s in standups:
        if s.get("developer") == developer and s.get("date") == date:
            return s
    return None


def get_previous_standup(developer: str) -> dict | None:
    """Get the most recent standup for a developer (for 'same as yesterday')."""
    standups = _load_json(STANDUPS_FILE, [])
    for s in standups:
        if s.get("developer") == developer and s.get("completed"):
            return s
    return None