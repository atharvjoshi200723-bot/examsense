"""
Answer student questions using RAG retrieval over past exam papers + Gemini.
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

SEARCH_TOP_K = 8

# ── Gemini client (new google-genai library) ──────────────────────────────────
_gemini_client: genai.Client | None = None
if GEMINI_API_KEY:
    try:
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Error initializing Gemini client in ask_engine: {e}")


def _error_response(message: str) -> dict[str, Any]:
    """Return a helpful JSON-shaped response when something goes wrong."""
    return {
        "answer": message,
        "related_topics": [],
        "past_frequency": 0,
        "exam_tips": [
            "Try rephrasing your question with a specific topic or unit name.",
            "Make sure past papers for your college, course, and semester are indexed.",
        ],
        "confidence": "low",
        "error": message,
    }


def _build_context(chunks: list[dict[str, Any]]) -> str:
    """Format the top RAG hits as context for Gemini."""
    sections: list[str] = []
    for index, hit in enumerate(chunks[:SEARCH_TOP_K], start=1):
        meta = hit.get("metadata") or {}
        header = (
            f"[Paper {index} | {meta.get('filename', 'unknown')} | "
            f"year {meta.get('year', '?')} | similarity {hit.get('similarity', 0):.2f}]"
        )
        sections.append(f"{header}\n{hit.get('text', '')}")
    return "\n\n---\n\n".join(sections)


def _build_prompt(college: str, course: str, user_question: str, context: str) -> str:
    """Prompt Gemini to answer using only retrieved past paper excerpts."""
    return f"""You are a helpful exam tutor for {college} {course} students.

A student asked: {user_question}

Based ONLY on these past exam papers, answer clearly and helpfully.

Past paper excerpts:
{context}

Return ONLY this JSON, nothing else:
{{
  "answer": "detailed answer here",
  "related_topics": ["topic1", "topic2"],
  "past_frequency": 0,
  "exam_tips": [
    "tip for writing good exam answer",
    "another tip"
  ],
  "confidence": "high"
}}

Set confidence to one of: high, medium, low.
Set past_frequency to how many times this topic appears to show up across the excerpts."""


def _parse_answer_json(raw_text: str) -> dict[str, Any]:
    """Parse Gemini JSON, stripping markdown code fences if present."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")

    parsed.setdefault("answer", "")
    parsed.setdefault("related_topics", [])
    parsed.setdefault("past_frequency", 0)
    parsed.setdefault("exam_tips", [])
    parsed.setdefault("confidence", "medium")
    return parsed


def ask_question(
    college: str,
    course: str,
    semester: str,
    user_question: str,
) -> dict[str, Any]:
    """
    Answer a student's custom question using past paper context from ChromaDB.
    """
    question = (user_question or "").strip()
    if not question:
        return _error_response("Please enter a question.")

    if not _gemini_client:
        return _error_response(
            "The tutor is unavailable right now because GEMINI_API_KEY is not configured."
        )

    try:
        try:
            hits = search(question, college, course, semester, top_k=SEARCH_TOP_K)
        except Exception as exc:
            return _error_response(f"Could not search past papers: {exc}")

        if not hits:
            return _error_response(
                f"No past papers found for {college} / {course} / {semester}. "
                "Upload papers or run ingestion first."
            )

        chunk_list = hits.get("hits", []) if isinstance(hits, dict) else hits
        if not chunk_list:
            return _error_response(
                f"No past papers found for {college} / {course} / {semester}."
            )

        context = _build_context(chunk_list)

        prompt = _build_prompt(college, course, question, context)
        response = _gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        raw_text = (response.text or "").strip()

        if not raw_text:
            return _error_response(
                "I couldn't generate an answer. Please try again in a moment."
            )

        try:
            return _parse_answer_json(raw_text)
        except (json.JSONDecodeError, ValueError) as exc:
            return _error_response(
                f"I found relevant papers but could not format the answer ({exc}). "
                "Please try rephrasing your question."
            )

    except Exception as exc:
        return _error_response(f"Something went wrong while answering your question: {exc}")
