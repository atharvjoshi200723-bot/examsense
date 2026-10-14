"""
Deterministic IDs for individual ExamSense question occurrences.

The same question from the same paper will always receive
the same question_id, so re-running ingestion does not create
random IDs.
"""

import hashlib
import re


def _normalize(value) -> str:
    """
    Normalize text before hashing so small formatting differences
    do not unnecessarily change the ID.
    """
    if value is None:
        return ""

    value = str(value).strip().lower()

    # Collapse whitespace
    value = re.sub(r"\s+", " ", value)

    return value


def generate_question_id(question) -> str:
    """
    Generate a deterministic ID for one question occurrence.

    The ID is based on:
        - source PDF
        - question number
        - sub-question
        - question text

    This identifies an individual occurrence of a question
    inside a specific historical paper.
    """

    source_path = _normalize(question.source_path)
    question_number = _normalize(question.question_number)
    sub_question = _normalize(question.sub_question)
    question_text = _normalize(question.question_text)

    identity = "|".join([
        source_path,
        question_number,
        sub_question,
        question_text,
    ])

    digest = hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()

    return f"q_{digest[:24]}"