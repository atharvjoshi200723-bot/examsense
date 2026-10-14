"""
Generate study flashcards from past exam paper content via RAG + Gemini.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

from google import genai

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import GEMINI_API_KEY, GEMINI_MODEL  # noqa: E402
from processors.rag_embedder import search  # noqa: E402

RAG_QUERY = "definitions concepts theory important terms"
SEARCH_TOP_K = 15
TOP_CONTEXT_CHUNKS = 15

_gemini_client: genai.Client | None = None
if GEMINI_API_KEY:
    try:
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Error initializing Gemini client in flashcard_gen: {e}")


def _build_context(chunks: list[dict[str, Any]]) -> str:
    sections: list[str] = []
    for index, hit in enumerate(chunks[:TOP_CONTEXT_CHUNKS], start=1):
        meta = hit.get("metadata") or {}
        header = f"[Excerpt {index} | {meta.get('filename', 'unknown')}]"
        sections.append(f"{header}\n{hit.get('text', '')}")
    return "\n\n---\n\n".join(sections)


def _parse_flashcards_json(raw_text: str) -> list[dict[str, Any]]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")
    flashcards = parsed.get("flashcards", [])
    if not isinstance(flashcards, list):
        raise ValueError("'flashcards' must be a list.")
    return flashcards


def _build_prompt(college: str, course: str, count: int, context: str) -> str:
    return f"""Create {count} flashcards from these {college} {course} exam papers.
Focus on most repeated concepts.

Past paper excerpts:
{context}

Return ONLY this JSON, nothing else:
{{
  "flashcards": [
    {{
      "id": 1,
      "front": "question or term",
      "back": "answer or definition",
      "topic": "topic name",
      "difficulty": "easy"
    }}
  ]
}}"""


def generate_flashcards(
    college: str, course: str, semester: str, count: int = 20,
) -> list[dict[str, Any]]:
    if not _gemini_client:
        print("generate_flashcards: GEMINI_API_KEY is not set.")
        return []
    try:
        try:
            hits = search(RAG_QUERY, college, course, semester, top_k=SEARCH_TOP_K)
        except Exception as exc:
            print(f"generate_flashcards: RAG search failed: {exc}")
            return []
        if not hits:
            print("generate_flashcards: no past paper content found.")
            return []
        chunk_list = hits.get("hits", []) if isinstance(hits, dict) else hits
        if not chunk_list:
            print("generate_flashcards: no chunks found in search result.")
            return []
        context = _build_context(chunk_list)
        prompt = _build_prompt(college, course, count, context)
        response = _gemini_client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt,
        )
        raw_text = (response.text or "").strip()
        if not raw_text:
            print("generate_flashcards: Gemini returned empty text.")
            return []
        try:
            return _parse_flashcards_json(raw_text)
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"generate_flashcards: JSON parse failed: {exc}")
            return []
    except Exception as exc:
        print(f"generate_flashcards failed: {exc}")
        return []
