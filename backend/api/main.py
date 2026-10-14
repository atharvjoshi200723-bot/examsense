"""
ExamSense FastAPI server — connects processors to the frontend.
"""

import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import COLLEGE_NAMES, PAPERS_DIR  # noqa: E402
from processors import (  # noqa: E402
    ask_engine,
    flashcard_gen,
    model_paper,
    pdf_reader,
    predictor,
    quiz_gen,
    rag_embedder,
)
from scrapers import upload_handler  # noqa: E402

PAPERS_ROOT = (BACKEND_DIR / PAPERS_DIR.lstrip("./")).resolve()

# ── FastAPI app + CORS ────────────────────────────────────────────────────────
app = FastAPI(title="ExamSense API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8000"],
    allow_origin_regex=r"https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    print("ExamSense API running!")


# ── Defaults for college dropdowns when filesystem archives are empty ─────────
DEFAULT_COURSES = {
    "GraphicEra": ["BTech", "BCA", "BBA", "MBA", "MCA", "BSc", "BPharma"],
    "VIT": ["BTech", "MTech", "BCA", "MCA", "BSc"],
    "SRM": ["BTech", "MTech", "BCA", "BBA"],
    "BITS": ["BTech", "MTech", "MSc", "BPharm"],
    "UPES": ["BTech", "BCA", "BBA", "LLB", "MBA"],
    "Manipal": ["BTech", "MTech", "BCA", "BBA"]
}
DEFAULT_YEARS = ["Year 1", "Year 2", "Year 3", "Year 4"]
DEFAULT_SEMESTERS = ["Sem 1", "Sem 2", "Sem 3", "Sem 4", "Sem 5", "Sem 6", "Sem 7", "Sem 8"]


# ── Request models ────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    college: str = Field(..., min_length=1)
    course: str = Field(..., min_length=1)
    year: str = ""
    semester: str = ""


class ModelPaperRequest(BaseModel):
    college: str = Field(..., min_length=1)
    course: str = Field(..., min_length=1)
    year: str = ""
    semester: str = ""


class FlashcardsRequest(BaseModel):
    college: str = Field(..., min_length=1)
    course: str = Field(..., min_length=1)
    semester: str = ""


class QuizRequest(BaseModel):
    college: str = Field(..., min_length=1)
    course: str = Field(..., min_length=1)
    semester: str = ""


class AskRequest(BaseModel):
    college: str = Field(..., min_length=1)
    course: str = Field(..., min_length=1)
    semester: str = ""
    question: str = Field(..., min_length=1)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _list_subfolders(path: Path) -> list[str]:
    """Return sorted subdirectory names under a papers path."""
    if not path.exists() or not path.is_dir():
        return []
    return sorted(entry.name for entry in path.iterdir() if entry.is_dir())


def _safe_path_part(value: str, label: str) -> str:
    """Reject path traversal in folder scan parameters."""
    if not value or value.strip() in (".", ".."):
        raise HTTPException(status_code=400, detail=f"Invalid {label}.")
    if "/" in value or "\\" in value:
        raise HTTPException(status_code=400, detail=f"Invalid {label}.")
    return value.strip()


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict[str, Any]:
    """Health check with college list and indexed paper count."""
    try:
        stats = rag_embedder.get_stats()
        total_papers = stats.get("total_documents", 0)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read index stats: {exc}",
        ) from exc

    return {
        "status": "ok",
        "colleges": COLLEGE_NAMES,
        "total_papers": total_papers,
    }


@app.get("/colleges")
def get_colleges() -> dict[str, list[str]]:
    """Return supported colleges for the frontend dropdown."""
    return {"colleges": COLLEGE_NAMES}


@app.get("/courses")
def get_courses(college: str) -> dict[str, list[str]]:
    """List course folders or fall back to ChromaDB metadata / defaults."""
    if not college or not college.strip():
        raise HTTPException(status_code=400, detail="college parameter is required.")
    college = _safe_path_part(college, "college")
    courses = _list_subfolders(PAPERS_ROOT / college)
    if not courses:
        # Check ChromaDB metadatas
        try:
            metas = rag_embedder.collection.get(where={"college": college}, include=["metadatas"])["metadatas"]
            discovered = set()
            for m in metas:
                c = m.get("course")
                if c and c.lower() not in ("all papers", "general", "unknown"):
                    discovered.add(c.title() if c.islower() else c)
            if discovered:
                courses = sorted(list(discovered))
        except Exception:
            pass

    if not courses:
        courses = DEFAULT_COURSES.get(college, DEFAULT_COURSES["GraphicEra"])

    return {"courses": courses}


@app.get("/years")
def get_years(college: str, course: str) -> dict[str, list[str]]:
    """List year folders or fall back to defaults."""
    if not college or not course:
        raise HTTPException(status_code=400, detail="college and course parameters are required.")
    college = _safe_path_part(college, "college")
    course = _safe_path_part(course, "course")
    years = _list_subfolders(PAPERS_ROOT / college / course)
    if not years:
        years = DEFAULT_YEARS
    return {"years": years}


@app.get("/semesters")
def get_semesters(college: str, course: str, year: str = "") -> dict[str, list[str]]:
    """List semester folders or fall back to defaults."""
    if not college or not course:
        raise HTTPException(status_code=400, detail="college and course parameters are required.")
    college = _safe_path_part(college, "college")
    course = _safe_path_part(course, "course")
    semesters = []
    if year:
        clean_year = _safe_path_part(year, "year")
        semesters = _list_subfolders(PAPERS_ROOT / college / course / clean_year)
    if not semesters:
        semesters = DEFAULT_SEMESTERS
    return {"semesters": semesters}


@app.post("/predict")
def predict(body: PredictRequest) -> dict[str, Any]:
    """Predict likely exam questions from indexed past papers."""
    if not body.college.strip() or not body.course.strip():
        raise HTTPException(status_code=400, detail="college and course are required.")
    try:
        return predictor.predict_questions(
            body.college.strip(),
            body.course.strip(),
            body.year.strip(),
            body.semester.strip(),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc


@app.post("/model-paper")
def generate_model_paper_endpoint(body: ModelPaperRequest) -> dict[str, Any]:
    """Generate a formatted model exam paper based on predicted hot topics."""
    try:
        prediction = predictor.predict_questions(
            body.college,
            body.course,
            body.year,
            body.semester,
        )
        hot_topics = prediction.get("hot_topics") or []
        if not hot_topics:
            raise HTTPException(
                status_code=404,
                detail="No hot topics found. Run predictions after indexing past papers.",
            )

        paper = model_paper.generate_model_paper(
            body.college,
            body.course,
            body.semester,
            hot_topics,
        )
        if paper is None:
            raise HTTPException(
                status_code=500,
                detail="Model paper generation failed.",
            )

        return {"paper": paper}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model paper generation failed: {exc}",
        ) from exc


@app.post("/flashcards")
def flashcards(body: FlashcardsRequest) -> dict[str, list[dict[str, Any]]]:
    """Generate flashcards from past paper content."""
    try:
        cards = flashcard_gen.generate_flashcards(
            body.college,
            body.course,
            body.semester,
        )
        return {"flashcards": cards}
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Flashcard generation failed: {exc}",
        ) from exc


@app.post("/quiz")
def quiz(body: QuizRequest) -> dict[str, list[dict[str, Any]]]:
    """Generate an MCQ quiz from past paper content."""
    try:
        questions = quiz_gen.generate_quiz(
            body.college,
            body.course,
            body.semester,
        )
        return {"questions": questions}
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Quiz generation failed: {exc}",
        ) from exc


@app.post("/ask")
def ask(body: AskRequest) -> dict[str, Any]:
    """Answer a student question using past paper context."""
    try:
        return ask_engine.ask_question(
            body.college,
            body.course,
            body.semester,
            body.question,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Ask engine failed: {exc}",
        ) from exc


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    college: str = Form(...),
    course: str = Form(...),
    year: str = Form(...),
    semester: str = Form(...),
) -> dict[str, Any]:
    """
    Upload a student PDF, extract text, embed into ChromaDB, then delete temp file.
    """
    filepath: str | None = None

    try:
        upload_result = upload_handler.handle_upload(
            file,
            college,
            course,
            year,
            semester,
        )
        if not upload_result.get("success"):
            raise HTTPException(
                status_code=400,
                detail=upload_result.get("error", "Upload validation failed."),
            )

        filepath = upload_result["filepath"]
        metadata = upload_result["metadata"]

        extraction = pdf_reader.extract_text(filepath)
        if not extraction.get("success"):
            raise HTTPException(
                status_code=422,
                detail=extraction.get("error", "Could not extract text from PDF."),
            )

        chunks_stored = rag_embedder.embed_and_store(extraction["text"], metadata)

        return {
            "success": True,
            "message": "Paper uploaded and processed!",
            "chunks_stored": chunks_stored,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Upload processing failed: {exc}",
        ) from exc
    finally:
        if filepath:
            try:
                Path(filepath).unlink(missing_ok=True)
            except OSError:
                pass
