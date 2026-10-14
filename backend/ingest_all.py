"""
ExamSense Ingestion Pipeline - Phase 2

Pipeline:
    GitHub PDF
        ↓
    Metadata extraction
        ↓
    Native text / EasyOCR fallback
        ↓
    Question extraction
        ↓
    QuestionOccurrence records
        ↓
    Stable question IDs
        ↓
    One vector per question/sub-question
        ↓
    ChromaDB

Important:
- No 400-word chunking for GEHU questions.
- Each question/sub-question becomes one embedding.
- Old legacy embed_and_store() is not used for GEHU.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------

from scrapers.github_scraper import GitHubScraper, COLLEGE_REPOS
from processors.pdf_reader import extract_text
from processors.rag_embedder import embed_questions, get_stats
from question_extractor import extract_questions
from question_ids import generate_question_id


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def is_text_useful(text: str) -> bool:
    """
    Check whether extracted text contains enough useful content.
    """
    if not text:
        return False

    cleaned = "".join(
        c for c in text
        if c.isprintable() and not c.isspace()
    )

    return len(cleaned) > 100


def attach_metadata(question, metadata):
    """
    Copy PDF-level metadata onto a QuestionOccurrence record.
    """

    question.college = metadata.get("college")
    question.degree = metadata.get("degree")
    question.branch = metadata.get("branch")
    question.semester = metadata.get("semester")
    question.subject = metadata.get("subject")
    question.course_code = metadata.get("course_code")
    question.exam_type = metadata.get("exam_type")
    question.attempt = metadata.get("attempt")
    question.exam_year = metadata.get("exam_year")
    question.exam_month = metadata.get("exam_month")
    question.source_path = metadata.get("repo_path") or metadata.get("source_path")

    # Generate deterministic occurrence ID
    question.question_id = generate_question_id(question)

    return question


# ---------------------------------------------------------
# Process one PDF
# ---------------------------------------------------------

def process_and_delete(filepath, metadata):
    """
    Process a single downloaded PDF.

    PDF
      → text/OCR
      → individual questions
      → metadata
      → stable IDs
      → ChromaDB vectors
    """

    filename = metadata.get("filename", filepath)

    try:
        print("\n" + "-" * 60)
        print(f"📄 Processing: {filename}")

        # -------------------------------------------------
        # 1. Extract text
        # -------------------------------------------------

        print("    → Extracting text...")

        result = extract_text(filepath)

        if not result.get("success"):
            print(
                f"❌ {filename} - "
                f"Text extraction failed: {result.get('error', 'unknown error')}"
            )
            return

        text = result.get("text", "")

        if not is_text_useful(text):
            print(f"❌ {filename} - No usable text extracted")
            return

        print(
            f"    → Extraction method: "
            f"{result.get('method', 'unknown')}"
        )

        # -------------------------------------------------
        # 2. Extract individual questions
        # -------------------------------------------------

        print("    → Extracting individual questions...")

        pages = result.get("pages")

        if not pages:
            print(
                f"❌ {filename} - "
                "No page-aware extraction available"
            )
            return

        questions = extract_questions(pages)

        if not questions:
            print(
                f"❌ {filename} - "
                "No questions could be extracted"
            )
            return

        print(
            f"    → Extracted {len(questions)} "
            f"question records"
        )

        # -------------------------------------------------
        # 3. Attach metadata + generate IDs
        # -------------------------------------------------

        valid_questions = []

        for question in questions:

            question = attach_metadata(
                question,
                metadata
            )

            if not question.question_text.strip():
                continue

            valid_questions.append(question)

        if not valid_questions:
            print(
                f"❌ {filename} - "
                "No valid question records"
            )
            return

        # -------------------------------------------------
        # 4. Store one vector per question
        # -------------------------------------------------

        print(
            f"    → Embedding {len(valid_questions)} "
            "individual questions..."
        )

        stored = embed_questions(valid_questions)

        print(
            f"✅ {filename} - "
            f"{stored} question vectors stored"
        )

        # -------------------------------------------------
        # 5. Show a small preview
        # -------------------------------------------------

        for question in valid_questions[:3]:

            label = f"Q{question.question_number}"

            if question.sub_question:
                label += f"({question.sub_question})"

            print(
                f"       {label}: "
                f"{question.question_text[:100]}"
            )

        if len(valid_questions) > 3:
            print(
                f"       ... "
                f"{len(valid_questions) - 3} more"
            )

    except Exception as e:

        print(
            f"❌ {filename} - "
            f"Error: {type(e).__name__}: {e}"
        )

    finally:

        # -------------------------------------------------
        # Delete temporary PDF
        # -------------------------------------------------

        try:
            if os.path.exists(filepath):
                os.remove(filepath)

        except Exception as cleanup_error:
            print(
                f"⚠️ Could not delete temporary file: "
                f"{cleanup_error}"
            )


# ---------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------

def run_ingestion():

    print("=" * 60)
    print("🚀 ExamSense Phase 2 Ingestion Started")
    print("=" * 60)

    # -----------------------------------------------------
    # Initial stats
    # -----------------------------------------------------

    stats_before = get_stats()

    print("\n[ChromaDB Initial Stats]")
    print(
        f"Total vectors: "
        f"{stats_before.get('total_chunks', 0)}"
    )

    print(
        f"Total documents: "
        f"{stats_before.get('total_documents', 0)}"
    )

    # -----------------------------------------------------
    # GEHU
    # -----------------------------------------------------

    print("\n📚 Scraping GitHub (GraphicEra / GEHU)...")

    try:

        ge_repo = COLLEGE_REPOS.get(
            "GraphicEra",
            "gehuhaldwani/pyqs"
        )

        scraper = GitHubScraper(
            "GraphicEra",
            ge_repo
        )

        scraper.run(process_and_delete)

    except Exception as e:

        print(
            f"❌ Scraping error: "
            f"{type(e).__name__}: {e}"
        )

    # -----------------------------------------------------
    # Final stats
    # -----------------------------------------------------

    stats_after = get_stats()

    print("\n" + "=" * 60)
    print("✅ Ingestion Complete!")
    print("=" * 60)

    print(
        f"Total vectors: "
        f"{stats_after.get('total_chunks', 0)}"
    )

    print(
        f"Total documents: "
        f"{stats_after.get('total_documents', 0)}"
    )

    print("=" * 60)


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    run_ingestion()