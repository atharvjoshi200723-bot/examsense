"""
Handle student PDF uploads for colleges without public paper repositories (e.g. BITS).
"""

import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

TEMP_UPLOAD_DIR = Path("/tmp/examsense/uploads")

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "application/acrobat",
    "applications/vnd.pdf",
    "application/vnd.adobe.pdf",
    "text/pdf",
    "text/x-pdf",
}


def _safe_part(value: str) -> str:
    """Sanitize a string for use in filenames."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", str(value).strip())
    return cleaned or "unknown"


def _failure(error: str) -> dict[str, Any]:
    """Standard error response."""
    return {"success": False, "error": error}


def _is_pdf_content(content: bytes) -> bool:
    """Check PDF magic bytes (%PDF) at the start of the file."""
    return content[:4] == b"%PDF"


def _read_upload(file: Any) -> tuple[bytes, str, str | None]:
    """
    Read bytes from an upload object, local path, or raw bytes.

    Supports FastAPI UploadFile, pathlib paths, and file-like objects.
    """
    if isinstance(file, (str, Path)):
        path = Path(file)
        return path.read_bytes(), path.name, "application/pdf"

    if isinstance(file, bytes):
        return file, "upload.pdf", "application/pdf"

    filename = getattr(file, "filename", None) or getattr(file, "name", None) or "upload.pdf"
    content_type = getattr(file, "content_type", None)

    # FastAPI UploadFile wraps an underlying synchronous file in .file
    spooled = getattr(file, "file", None)
    if spooled is not None and hasattr(spooled, "read") and callable(spooled.read):
        content = spooled.read()
        if hasattr(spooled, "seek") and callable(spooled.seek):
            try:
                spooled.seek(0)
            except Exception:
                pass
        return content, filename, content_type

    if hasattr(file, "read") and callable(file.read):
        content = file.read()
        if isinstance(content, str):
            content = content.encode("utf-8")
        if hasattr(file, "seek") and callable(file.seek):
            try:
                file.seek(0)
            except Exception:
                pass
        return content, filename, content_type

    raise TypeError("Unsupported file type. Pass a path, bytes, or file-like upload.")


def handle_upload(
    file: Any,
    college: str,
    course: str,
    year: str,
    semester: str,
) -> dict[str, Any]:
    """
    Validate and save an uploaded exam PDF to a temporary folder.

    Args:
        file: Upload object (e.g. FastAPI UploadFile), path, or bytes
        college: College name (e.g. "BITS")
        course: Course code or name
        year: Exam year
        semester: Semester identifier (e.g. "sem4")

    Returns:
        Success dict with filepath + metadata, or error dict on failure.
    """
    try:
        content, original_name, content_type = _read_upload(file)
    except Exception as exc:
        return _failure(f"Could not read uploaded file: {exc}")

    if not content:
        return _failure("Uploaded file is empty.")

    # Step 1: Validate PDF extension
    if not original_name.lower().endswith(".pdf"):
        return _failure("Only PDF files are allowed. Please upload a .pdf file.")

    # Step 2: Validate content type when the client provides one
    if content_type and content_type.lower() not in ALLOWED_CONTENT_TYPES:
        return _failure(f"Invalid content type: {content_type}. Expected application/pdf.")

    # Step 3: Validate actual PDF bytes (not just extension)
    if not _is_pdf_content(content):
        return _failure("File does not appear to be a valid PDF.")

    # Step 4: Build a safe, unique filename
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_filename = (
        f"{_safe_part(college)}_{_safe_part(course)}_{_safe_part(year)}_"
        f"{_safe_part(semester)}_{timestamp}.pdf"
    )

    # Step 5: Save temporarily under /tmp/examsense/uploads/
    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = TEMP_UPLOAD_DIR / safe_filename

    try:
        dest_path.write_bytes(content)
    except OSError as exc:
        return _failure(f"Could not save file: {exc}")

    metadata = {
        "college": college,
        "course": course,
        "year": year,
        "semester": semester,
        "filename": safe_filename,
    }

    return {
        "success": True,
        "filepath": str(dest_path),
        "metadata": metadata,
    }
