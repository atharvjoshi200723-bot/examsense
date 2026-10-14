"""
Generate a full formatted model exam paper from hot topics using Gemini.
"""

import sys
from pathlib import Path

from google import genai

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import GEMINI_API_KEY, GEMINI_MODEL  # noqa: E402

# ── Gemini client (new google-genai library) ──────────────────────────────────
_gemini_client: genai.Client | None = None
if GEMINI_API_KEY:
    try:
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Error initializing Gemini client in model_paper: {e}")


def _format_hot_topics(hot_topics: list[str] | str) -> str:
    """Normalize hot_topics into a comma-separated string for the prompt."""
    if isinstance(hot_topics, str):
        return hot_topics.strip()
    return ", ".join(str(topic).strip() for topic in hot_topics if str(topic).strip())


def _build_prompt(college: str, course: str, semester: str, hot_topics: str) -> str:
    """Build the Gemini prompt for a university-style model paper."""
    return f"""Create a complete model exam paper for {college} {course} Semester {semester}.

Format EXACTLY like a real Indian university exam paper:

[{college.upper()}]
End Semester Examination
Course: {course} | Semester: {semester}
Time: 3 Hours | Max Marks: 70

SECTION A (10 x 2 = 20 Marks)
Answer all questions
(10 short questions, 2 marks each)

SECTION B (5 x 6 = 30 Marks)
Answer any 5 questions
(7 medium questions, 6 marks each)

SECTION C (2 x 10 = 20 Marks)
Answer any 2 questions
(3 long questions, 10 marks each)

Base all questions on these important topics: {hot_topics}

Return as plain formatted text only."""


def generate_model_paper(
    college: str,
    course: str,
    semester: str,
    hot_topics: list[str] | str,
) -> str | None:
    """
    Generate a complete formatted model exam paper.

    Returns the paper as plain text, or None if generation fails.
    """
    if not _gemini_client:
        print("generate_model_paper: GEMINI_API_KEY is not set.")
        return None

    topics_text = _format_hot_topics(hot_topics)
    if not topics_text:
        print("generate_model_paper: hot_topics is empty.")
        return None

    try:
        prompt = _build_prompt(college, course, semester, topics_text)
        response = _gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        paper = (response.text or "").strip()

        if not paper:
            print("generate_model_paper: Gemini returned empty text.")
            return None

        return paper

    except Exception as exc:
        print(f"generate_model_paper failed: {exc}")
        return None
