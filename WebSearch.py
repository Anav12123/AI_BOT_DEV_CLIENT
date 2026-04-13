"""
WebSearch.py — SerpAPI Google Search
Query conversion handled by Agent.py (LLM-based).
Rotates between 2 API keys to avoid rate limits.
"""

import os
import re
import httpx
from typing import Optional


class WebSearch:
    def __init__(self):
        self._keys = []
        k1 = os.environ.get("SERPAPI_KEY_1", "").strip().strip('"\'')
        k2 = os.environ.get("SERPAPI_KEY_2", "").strip().strip('"\'')
        if k1: self._keys.append(k1)
        if k2: self._keys.append(k2)
        if not self._keys:
            print("[WebSearch] ⚠️  No SERPAPI keys — web search disabled")
        else:
            print(f"[WebSearch] {len(self._keys)} key(s) loaded (key1: {k1[:8]}...)")
        self._key_index = 0
        self._client = httpx.AsyncClient(timeout=20.0)

    def _next_key(self) -> str:
        key = self._keys[self._key_index % len(self._keys)]
        self._key_index += 1
        return key

    def _trim_query(self, query: str, max_words: int = 20) -> str:
        """Clean up query — strip tags, prefixes, cap length."""
        clean = re.sub(r'\[LANG:\w+\]\s*', '', query).strip()
        for prefix in ["sam,", "sam ", "hey sam,", "hey sam ",
                        "can you tell me", "could you tell me",
                        "please tell me", "do you know",
                        "i want to know", "tell me"]:
            if clean.lower().startswith(prefix):
                clean = clean[len(prefix):].strip().lstrip(",. ")
        words = clean.split()
        return " ".join(words[:max_words])

    async def search(self, query: str) -> Optional[str]:
        if not self._keys:
            return None

        trimmed = self._trim_query(query)
        api_key = self._next_key()
        print(f"[WebSearch] SerpAPI query: \"{trimmed}\" (key #{self._key_index})")

        try:
            resp = await self._client.get(
                "https://serpapi.com/search.json",
                params={"engine": "google", "q": trimmed, "api_key": api_key, "num": 3},
            )
            if resp.status_code != 200:
                print(f"[WebSearch] HTTP {resp.status_code}: {resp.text[:200]}")
                return None
            data = resp.json()

            # 1. Answer box
            ab = data.get("answer_box", {})
            answer = ab.get("answer", "") or ab.get("snippet", "")
            if answer:
                print(f"[WebSearch] Answer box ({len(answer)} chars)")
                return answer[:800]

            # 2. Knowledge graph
            kg = data.get("knowledge_graph", {})
            if kg.get("description"):
                title = kg.get("title", "")
                result = f"{title}: {kg['description']}" if title else kg["description"]
                print(f"[WebSearch] Knowledge graph ({len(result)} chars)")
                return result[:800]

            # 3. AI overview
            ai = data.get("ai_overview", {})
            if ai:
                parts = [b.get("snippet", "") for b in ai.get("text_blocks", []) if b.get("snippet")]
                if parts:
                    combined = " ".join(parts)[:800]
                    print(f"[WebSearch] AI overview ({len(combined)} chars)")
                    return combined

            # 4. Organic results
            organic = data.get("organic_results", [])
            parts = [r.get("snippet", "") for r in organic[:3] if r.get("snippet")]
            if parts:
                combined = " ".join(parts)[:800]
                print(f"[WebSearch] Organic results ({len(combined)} chars)")
                return combined

            return None

        except httpx.TimeoutException:
            print(f"[WebSearch] TIMEOUT: {trimmed}")
            return None
        except Exception as e:
            print(f"[WebSearch] Error: {type(e).__name__}: {e}")
            return None

    async def close(self):
        await self._client.aclose()
