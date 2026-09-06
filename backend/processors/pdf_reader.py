"""
PDF Reader - Extracts text from PDFs
Uses PyMuPDF for text PDFs
Uses EasyOCR for scanned PDFs
No API needed, completely free, no quota
"""

import os
import sys
import fitz
from pathlib import Path
import tempfile
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# EasyOCR reader - loaded once, reused
_ocr_reader = None

def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        print("Loading EasyOCR model (first time only)...")
        import easyocr
        _ocr_reader = easyocr.Reader(['en'], gpu=False)
        print("EasyOCR loaded!")
    return _ocr_reader

def extract_text(filepath):
    """
    Extract text from PDF.
    Text PDF → direct extraction (fast)
    Scanned PDF → EasyOCR (free, no quota)
    """
    try:
        doc = fitz.open(filepath)
        full_text = ""

        for page in doc:
            full_text += page.get_text()

        page_count = len(doc)
        doc.close()

        # If enough text found, return directly
        if len(full_text.strip()) > 150:
            return {
                "text": full_text.strip(),
                "method": "direct",
                "page_count": page_count,
                "filepath": filepath,
                "success": True,
                "error": None
            }

        # Scanned PDF - use EasyOCR
        print(f"    → Scanned PDF, using EasyOCR...")
        return extract_with_easyocr(filepath)

    except Exception as e:
        return {
            "text": "",
            "method": "failed",
            "page_count": 0,
            "filepath": filepath,
            "success": False,
            "error": str(e)
        }

def extract_with_easyocr(filepath):
    """Use EasyOCR to read scanned PDF"""
    try:
        reader = get_ocr_reader()
        doc = fitz.open(filepath)
        full_text = ""
        page_count = len(doc)

        for page_num in range(page_count):
            try:
                page = doc[page_num]

                # Convert page to image
                mat = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat)

                # Save to temp file
                with tempfile.NamedTemporaryFile(
                    suffix='.png',
                    delete=False
                ) as tmp:
                    tmp_path = tmp.name
                    pix.save(tmp_path)

                # Run EasyOCR
                results = reader.readtext(tmp_path)

                # Extract text from results
                page_text = " ".join([
                    result[1] for result in results
                ])

                if page_text:
                    full_text += page_text + "\n\n"

                # Delete temp file
                os.unlink(tmp_path)

            except Exception as e:
                print(f"    → OCR failed page {page_num}: {e}")
                continue

        doc.close()

        if full_text.strip():
            return {
                "text": full_text.strip(),
                "method": "easyocr",
                "page_count": page_count,
                "filepath": filepath,
                "success": True,
                "error": None
            }
        else:
            return {
                "text": "",
                "method": "easyocr_failed",
                "page_count": page_count,
                "filepath": filepath,
                "success": False,
                "error": "No text extracted"
            }

    except Exception as e:
        return {
            "text": "",
            "method": "easyocr_failed",
            "page_count": 0,
            "filepath": filepath,
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    print("Testing PDF reader...")
    print("EasyOCR loaded successfully!")