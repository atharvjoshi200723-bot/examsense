"""
RAG Embedder - Local embeddings, zero API quota
Uses sentence-transformers (runs on laptop, free forever)
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import CHROMA_DIR
import chromadb
from sentence_transformers import SentenceTransformer

# Load model once, reuse forever
print("Loading embedding model...")
_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded!")

# ChromaDB setup
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(
    name="examsense",
    metadata={"hnsw:space": "cosine"}
)

def chunk_text(text, chunk_size=400, overlap=60):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
        if start >= len(words):
            break
    return chunks

def get_embedding(text):
    """Get embedding using local model - no API needed"""
    try:
        embedding = _model.encode(text)
        return embedding.tolist()
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def embed_and_store(text, metadata):
    """Embed text and store in ChromaDB"""
    if not text or len(text.strip()) < 50:
        return 0

    chunks = chunk_text(text)

    # Emergency fallback if chunker returns 0
    if len(chunks) == 0 and len(text.strip()) > 0:
        print(f"    → Emergency fallback chunk created")
        chunks = [text.strip()[:1000]]

    stored = 0

    for i, chunk in enumerate(chunks):
        try:
            chunk_id = (
                f"{metadata.get('college','X')}_"
                f"{metadata.get('course','X')}_"
                f"{metadata.get('year','X')}_"
                f"{metadata.get('semester','X')}_"
                f"{metadata.get('filename','X')}_"
                f"chunk{i}"
            )

            existing = collection.get(ids=[chunk_id])
            if existing and existing['ids']:
                continue

            embedding = get_embedding(chunk)
            if embedding is None:
                continue

            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    "college": metadata.get("college", ""),
                    "course": metadata.get("course", ""),
                    "year": str(metadata.get("year", "")),
                    "semester": metadata.get("semester", ""),
                    "filename": metadata.get("filename", ""),
                    "chunk_index": i
                }]
            )
            stored += 1

        except Exception as e:
            print(f"Error storing chunk {i}: {e}")
            continue

    return stored


def search(query, college, course=None,
           semester=None, top_k=10):
    """
    Search ChromaDB for similar content.
    Only filters by college because VIT papers
    are stored as course='all papers'
    """
    try:
        query_vector = get_embedding(query)
        if query_vector is None:
            return None

        # Only filter by college
        # Handles VIT structure where course = "all papers"
        where_filter = {
            "college": {"$eq": college}
        }

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_filter
        )

        return results

    except Exception as e:
        print(f"Search error: {e}")
        return None


def get_topic_frequency(topic, college, course=None):
    """Find how many years a topic appeared"""
    try:
        results = search(
            topic, college,
            top_k=50
        )
        if not results:
            return {
                "topic": topic,
                "count": 0,
                "years_found": [],
                "total_years": 0
            }

        years = set()
        for meta in results.get("metadatas", [[]])[0]:
            year = meta.get("year", "")
            if year and year != "unknown":
                years.add(year)

        return {
            "topic": topic,
            "years_found": list(years),
            "count": len(years),
            "total_years": len(years)
        }

    except Exception as e:
        print(f"Frequency error: {e}")
        return {
            "topic": topic,
            "count": 0,
            "years_found": [],
            "total_years": 0
        }


def get_stats():
    """Get ChromaDB statistics"""
    try:
        count = collection.count()
        return {
            "total_chunks": count,
            "total_documents": count,
            "by_college": {}
        }
    except Exception as e:
        print(f"Stats error: {e}")
        return {
            "total_chunks": 0,
            "total_documents": 0,
            "by_college": {}
        }