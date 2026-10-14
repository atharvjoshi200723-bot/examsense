
from pydantic import BaseModel, Field


class QuestionOccurrence(BaseModel):
    """
    One question/sub-question extracted from one historical exam paper.
    """

    question_number: int
    sub_question: str | None = None

    question_text: str

    marks: float | None = None

    page_start: int | None = None
    page_end: int | None = None

    # Filled later by the ingestion pipeline
    college: str | None = None
    degree: str | None = None
    branch: str | None = None
    semester: str | None = None
    subject: str | None = None
    course_code: str | None = None

    exam_type: str | None = None
    attempt: str | None = None
    exam_year: int | None = None
    exam_month: str | None = None

    source_path: str | None = None

    # Stable identifier generated later
    question_id: str | None = None
