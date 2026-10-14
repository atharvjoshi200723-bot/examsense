"""PDF text extraction for ExamSense.

Policy:
- Prefer selectable/native PDF text whenever it is readable.
- Do not reject native text because question labels are missing or nonconsecutive.
- OCR only pages with weak/empty native text, so mixed PDFs are supported.
- Preserve page boundaries for downstream question extraction.
"""

import sys
from pathlib import Path
from typing import Any

try:
    import pymupdf as fitz
except ImportError:  # Compatibility with older installations
    import fitz  # type: ignore

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

_ocr_reader = None


def get_ocr_reader():
    """Load EasyOCR lazily and reuse its reader."""
    global _ocr_reader
    if _ocr_reader is None:
        print("Loading EasyOCR model (first time only)...")
        import easyocr
        _ocr_reader = easyocr.Reader(["en"], gpu=False)
        print("EasyOCR loaded!")
    return _ocr_reader


def is_text_usable(text: str, min_chars: int = 100, min_alpha: int = 50) -> bool:
    """Check textual readability only; never assess question numbering here."""
    if not text:
        return False
    compact = " ".join(text.split())
    if len(compact) < min_chars:
        return False
    alphabetic = sum(ch.isalpha() for ch in compact)
    non_space = sum(not ch.isspace() for ch in compact)
    if alphabetic < min_alpha or non_space == 0:
        return False
    return alphabetic / non_space >= 0.20


def _page_native_usable(text: str) -> bool:
    """A lighter page-level check used to choose pages requiring OCR."""
    if not text:
        return False
    compact = " ".join(text.split())
    alpha = sum(ch.isalpha() for ch in compact)
    non_space = sum(not ch.isspace() for ch in compact)
    return len(compact) >= 80 and alpha >= 35 and non_space > 0 and alpha / non_space >= 0.20


def has_question_structure(text: str) -> bool:
    """Legacy diagnostic only. Not used to decide whether native text is usable."""
    if not text:
        return False
    import re
    patterns = (
        r"(?im)^\s*(?:question|ques|que|q)\s*\.?\s*\d{1,2}\b",
        r"(?m)^\s*\d{1,2}\s*[.)]\s+\S",
        r"(?m)^\s*\(\s*[a-hA-H]\s*\)",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def is_native_exam_text_usable(text: str) -> bool:
    """Backward-compatible wrapper: readability only, not marker continuity."""
    return is_text_usable(text)


def extract_native_text(doc) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for page_index, page in enumerate(doc):
        try:
            page_text = page.get_text("text", sort=True).strip()
        except Exception as exc:
            print(f"    → Native extraction failed on page {page_index + 1}: {exc}")
            page_text = ""
        pages.append({"page_number": page_index + 1, "text": page_text})
    return pages


def combine_pages(pages: list[dict[str, Any]]) -> str:
    parts = []
    for page in pages:
        page_text = (page.get("text", "") or "").strip()
        if not page_text:
            continue
        page_number = page.get("page_number", len(parts) + 1)
        parts.append(f"[PAGE {page_number}]\n{page_text}")
    return "\n\n".join(parts).strip()


def extract_native(filepath: str | Path) -> dict[str, Any]:
    """Extract native text and report readability; do not inspect question order."""
    with fitz.open(filepath) as doc:
        pages = extract_native_text(doc)
        page_count = len(doc)
    combined = combine_pages(pages)
    readable = is_text_usable(combined)
    return {
        "pages": pages,
        "text": combined,
        "page_count": page_count,
        "usable": readable,
        "readable": readable,
        "structurally_usable": has_question_structure(combined),
    }


def group_ocr_results_into_lines(results) -> list[str]:
    """Sort EasyOCR boxes top-to-bottom and left-to-right into text lines."""
    detections = []
    for result in results:
        if len(result) < 2:
            continue
        bbox, detected = result[0], str(result[1]).strip()
        if not detected or bbox is None:
            continue
        try:
            xs = [float(point[0]) for point in bbox]
            ys = [float(point[1]) for point in bbox]
            left, top, bottom = min(xs), min(ys), max(ys)
            height = max(1.0, bottom - top)
            detections.append({
                "text": detected,
                "x": left,
                "top": top,
                "bottom": bottom,
                "center_y": (top + bottom) / 2.0,
                "height": height,
            })
        except (TypeError, ValueError, IndexError, KeyError):
            continue

    detections.sort(key=lambda item: (item["center_y"], item["x"]))
    groups: list[dict[str, Any]] = []
    for item in detections:
        best_group = None
        best_distance = float("inf")
        for group in groups:
            center = sum(group["centers"]) / len(group["centers"])
            heights = sorted(group["heights"])
            median_height = heights[len(heights) // 2]
            tolerance = max(8.0, 0.55 * max(item["height"], median_height))
            distance = abs(item["center_y"] - center)
            if distance <= tolerance and distance < best_distance:
                best_group, best_distance = group, distance
        if best_group is None:
            best_group = {"items": [], "centers": [], "heights": []}
            groups.append(best_group)
        best_group["items"].append(item)
        best_group["centers"].append(item["center_y"])
        best_group["heights"].append(item["height"])

    groups.sort(key=lambda group: min(item["top"] for item in group["items"]))
    lines = []
    for group in groups:
        group["items"].sort(key=lambda item: item["x"])
        line = " ".join(item["text"] for item in group["items"]).strip()
        if line:
            lines.append(line)
    return lines


def _ocr_pages(filepath: str | Path, pages_to_ocr: set[int], existing_pages: list[dict[str, Any]] | None = None):
    """OCR selected zero-based page indexes while retaining good native pages."""
    page_count = 0
    pages = existing_pages[:] if existing_pages is not None else []
    reader = get_ocr_reader()
    with fitz.open(filepath) as doc:
        page_count = len(doc)
        if not pages:
            pages = [{"page_number": i + 1, "text": ""} for i in range(page_count)]
        for page_index in sorted(pages_to_ocr):
            if page_index < 0 or page_index >= page_count:
                continue
            try:
                page = doc[page_index]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                results = reader.readtext(pix.tobytes("png"))
                page_lines = group_ocr_results_into_lines(results)
                page_text = "\n".join(page_lines).strip()
                print(
                    f"    → OCR page {page_index + 1}: {len(page_lines)} lines, "
                    f"{len(page_text)} characters"
                )
                pages[page_index] = {"page_number": page_index + 1, "text": page_text}
            except Exception as exc:
                print(f"    → OCR failed on page {page_index + 1}: {exc}")
                pages[page_index] = {"page_number": page_index + 1, "text": ""}
    combined = combine_pages(pages)
    success = bool(combined.strip())
    return {
        "text": combined,
        "pages": pages,
        "method": "hybrid_ocr" if existing_pages is not None else "easyocr",
        "page_count": page_count,
        "filepath": str(filepath),
        "success": success,
        "error": None if success else "No text extracted by native extraction or OCR",
    }


def extract_with_easyocr(filepath: str | Path) -> dict[str, Any]:
    """Backward-compatible helper that OCRs every page."""
    try:
        with fitz.open(filepath) as doc:
            page_count = len(doc)
        return _ocr_pages(filepath, set(range(page_count)))
    except Exception as exc:
        return {
            "text": "", "pages": [], "method": "easyocr_failed",
            "page_count": 0, "filepath": str(filepath), "success": False,
            "error": str(exc),
        }


def extract_text(filepath: str | Path) -> dict[str, Any]:
    """Use native text by default; OCR only pages with inadequate native text."""
    filepath = str(filepath)
    try:
        native = extract_native(filepath)
        pages = native["pages"]
        if not pages:
            return extract_with_easyocr(filepath)

        weak_indexes = {
            index for index, page in enumerate(pages)
            if not _page_native_usable(page.get("text", ""))
        }

        if not weak_indexes:
            print("    → Native extraction accepted; all pages have readable text.")
            return {
                "text": native["text"], "pages": pages, "method": "direct",
                "page_count": native["page_count"], "filepath": filepath,
                "success": True, "error": None,
            }

        if len(weak_indexes) == len(pages):
            print("    → Native text is empty/weak; using EasyOCR for all pages...")
            return _ocr_pages(filepath, weak_indexes)

        print(
            f"    → Hybrid extraction: keeping native text on {len(pages) - len(weak_indexes)} page(s); "
            f"using EasyOCR on {len(weak_indexes)} weak page(s)."
        )
        result = _ocr_pages(filepath, weak_indexes, existing_pages=pages)
        result["method"] = "hybrid"
        return result

    except Exception as exc:
        return {
            "text": "", "pages": [], "method": "failed", "page_count": 0,
            "filepath": filepath, "success": False, "error": str(exc),
        }


if __name__ == "__main__":
    print("ExamSense PDF reader loaded.")
    print("Policy: accept readable native text; OCR only weak pages; preserve page boundaries.")
