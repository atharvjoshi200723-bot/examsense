"""
ExamSense Ingestion Pipeline - Fixed Version
- Silences PyTorch MPS warnings
- Two-layer OCR forcing system
- 384-dimension local embeddings
- Emergency fallback chunking
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from scrapers import github_scraper
from processors.pdf_reader import extract_text
from processors.rag_embedder import embed_and_store, get_stats


def is_text_useful(text):
    """
    Check if extracted text is actually useful.
    Filters out empty strings and hidden control characters.
    """
    if not text:
        return False
    cleaned = ''.join(
        c for c in text
        if c.isprintable() and not c.isspace()
    )
    return len(cleaned) > 100


def process_and_delete(filepath, metadata):
    """
    Main callback for each PDF.
    Two-layer system:
    Layer 1 → Try direct extraction
    Layer 2 → Force EasyOCR if text too short
    """
    try:
        filename = metadata.get('filename', filepath)

        # LAYER 1: Direct text extraction (extract_text already falls back to OCR if scanned)
        result = extract_text(filepath)
        text = result.get("text", "")

        # Fallback to OCR only if direct text was too short and OCR wasn't already attempted
        if not is_text_useful(text) and result.get("method") != "easyocr":
            print(f"    → Weak text detected, attempting EasyOCR...")
            from processors.pdf_reader import extract_with_easyocr
            result = extract_with_easyocr(filepath)
            text = result.get("text", "")

        # Store in ChromaDB
        if is_text_useful(text):
            chunks_stored = embed_and_store(text, metadata)
            print(f"✅ {filename} - {chunks_stored} chunks stored")
        else:
            print(f"❌ {filename} - No usable text extracted")

    except Exception as e:
        print(f"❌ {metadata.get('filename', 'unknown')} - Error: {e}")

    finally:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass


def run_ingestion():
    print("=" * 50)
    print("🚀 ExamSense Ingestion Started")
    print("=" * 50)

    stats_before = get_stats()
    print(f"\n[ChromaDB Initial Stats]")
    print(f"Total chunks: {stats_before['total_chunks']}")
    print(f"Total papers: {stats_before['total_documents']}")

    print("\nScraping GitHub (GraphicEra)...")
    try:
        from scrapers.github_scraper import GitHubScraper, COLLEGE_REPOS
        ge_repo = COLLEGE_REPOS.get("GraphicEra", "gehuhaldwani/pyqs")
        scraper = GitHubScraper("GraphicEra", ge_repo)
        scraper.run(process_and_delete)
    except Exception as e:
        print(f"Scraping error: {e}")

    stats_after = get_stats()
    print("\n" + "=" * 50)
    print("✅ Ingestion Complete!")
    print(f"Total chunks stored: {stats_after['total_chunks']}")
    print("=" * 50)


if __name__ == "__main__":
    run_ingestion()