"""
ExamSense RAG Embedder - Phase 2

Local embeddings using sentence-transformers.
No external API required.

Two storage paths are supported:

1. Legacy:
   embed_and_store()
   - Keeps old chunk-based ingestion working for VIT/SRM.

2. Phase 2:
   embed_questions()
   - Stores ONE vector per extracted question/sub-question.
   - Used by the new GEHU ingestion pipeline.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import CHROMA_DIR

import chromadb
from sentence_transformers import SentenceTransformer


# =========================================================
# Embedding Model
# =========================================================

print("Loading embedding model...")

_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded!")


# =========================================================
# ChromaDB
# =========================================================

chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

collection = chroma_client.get_or_create_collection(
    name="examsense",
    metadata={
        "hnsw:space": "cosine"
    }
)


# =========================================================
# Embeddings
# =========================================================

def get_embedding(text):
    """
    Generate a local embedding.

    all-MiniLM-L6-v2 produces 384-dimensional vectors.
    """

    if not text or not text.strip():
        return None

    try:
        embedding = _model.encode(
            text,
            convert_to_numpy=True
        )

        return embedding.tolist()

    except Exception as e:
        print(f"Embedding error: {e}")
        return None


# =========================================================
# Legacy Chunking
# =========================================================

def chunk_text(text, chunk_size=400, overlap=60):
    """
    Legacy chunking function.

    Kept for backward compatibility with older
    VIT/SRM ingestion.
    
    GEHU Phase 2 ingestion does NOT use this.
    """

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


# =========================================================
# Legacy Embedding Storage
# =========================================================

def embed_and_store(text, metadata):
    """
    Legacy PDF/chunk storage.

    This is intentionally preserved so existing
    VIT/SRM functionality does not break.

    New GEHU ingestion should use embed_questions().
    """

    if not text or len(text.strip()) < 50:
        return 0

    chunks = chunk_text(text)

    if len(chunks) == 0 and text.strip():

        print(
            "    → Emergency fallback chunk created"
        )

        chunks = [
            text.strip()[:1000]
        ]

    stored = 0

    for i, chunk in enumerate(chunks):

        try:

            chunk_id = (
                f"{metadata.get('college', 'X')}_"
                f"{metadata.get('course', 'X')}_"
                f"{metadata.get('year', 'X')}_"
                f"{metadata.get('semester', 'X')}_"
                f"{metadata.get('filename', 'X')}_"
                f"chunk{i}"
            )

            existing = collection.get(
                ids=[chunk_id]
            )

            if existing and existing.get("ids"):
                continue

            embedding = get_embedding(chunk)

            if embedding is None:
                continue

            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[
                    {
                        "college": metadata.get(
                            "college", ""
                        ),
                        "course": metadata.get(
                            "course", ""
                        ),
                        "year": str(
                            metadata.get("year", "")
                        ),
                        "semester": metadata.get(
                            "semester", ""
                        ),
                        "filename": metadata.get(
                            "filename", ""
                        ),
                        "chunk_index": i
                    }
                ]
            )

            stored += 1

        except Exception as e:

            print(
                f"Error storing chunk {i}: {e}"
            )

    return stored


# =========================================================
# Phase 2 - Question Embedding
# =========================================================

def _question_metadata(question):
    """
    Convert a QuestionOccurrence object into
    Chroma-compatible metadata.

    Chroma metadata values must be primitive types.
    """

    return {
        "college": question.college or "",
        "degree": question.degree or "",
        "branch": question.branch or "",
        "semester": question.semester or "",
        "subject": question.subject or "",
        "course_code": question.course_code or "",
        "exam_type": question.exam_type or "",
        "attempt": question.attempt or "",
        "exam_year": (
            int(question.exam_year)
            if question.exam_year is not None
            else -1
        ),
        "exam_month": question.exam_month or "",
        "question_number": int(
            question.question_number
        ),
        "sub_question": (
            question.sub_question
            if question.sub_question
            else ""
        ),
        "page_start": (
            int(question.page_start)
            if question.page_start is not None
            else -1
        ),
        "page_end": (
            int(question.page_end)
            if question.page_end is not None
            else -1
        ),
        "source_path": question.source_path or "",
        "question_id": question.question_id or ""
    }


def embed_questions(questions):
    """
    Phase 2 question-level vector storage.

    Each QuestionOccurrence becomes exactly ONE vector.

    Example:

        Q1(a) → vector
        Q1(b) → vector
        Q1(c) → vector
        Q2(a) → vector

    Returns:
        Number of newly stored questions.
    """

    if not questions:
        return 0

    stored = 0

    for question in questions:

        try:

            # -------------------------------------------------
            # Validate
            # -------------------------------------------------

            question_text = (
                question.question_text or ""
            ).strip()

            if len(question_text) < 10:
                print(
                    "    ⚠️ Skipping very short question"
                )
                continue

            if not question.question_id:
                print(
                    "    ⚠️ Skipping question without "
                    "question_id"
                )
                continue

            # -------------------------------------------------
            # Check duplicate
            # -------------------------------------------------

            existing = collection.get(
                ids=[question.question_id]
            )

            if existing and existing.get("ids"):

                print(
                    f"    ↪ Already exists: "
                    f"{question.question_id}"
                )

                continue

            # -------------------------------------------------
            # Generate embedding
            # -------------------------------------------------

            embedding = get_embedding(
                question_text
            )

            if embedding is None:
                print(
                    f"    ❌ Embedding failed for "
                    f"{question.question_id}"
                )
                continue

            # -------------------------------------------------
            # Metadata
            # -------------------------------------------------

            metadata = _question_metadata(
                question
            )

            # -------------------------------------------------
            # Store in Chroma
            # -------------------------------------------------

            collection.add(
                ids=[
                    question.question_id
                ],
                embeddings=[
                    embedding
                ],
                documents=[
                    question_text
                ],
                metadatas=[
                    metadata
                ]
            )

            stored += 1

        except Exception as e:

            print(
                f"    ❌ Error storing "
                f"{question.question_id}: {e}"
            )

    return stored


# =========================================================
# Search
# =========================================================

def search(
    query,
    college,
    course=None,
    semester=None,
    top_k=10
):
    """
    Semantic search with metadata filtering.
    """

    try:

        query_vector = get_embedding(query)

        if query_vector is None:
            return None

        # -------------------------------------------------
        # Build metadata filters
        # -------------------------------------------------

        filters = [
            {
                "college": {
                    "$eq": college
                }
            }
        ]

        if (
            course
            and str(course).strip()
            and str(course).strip().lower()
            not in ("all papers", "general", "all")
        ):

            filters.append(
                {
                    "course": {
                        "$eq": str(course).strip()
                    }
                }
            )

        if semester and str(semester).strip():

            sem_val = str(
                semester
            ).strip()

            sem_clean = (
                sem_val
                .lower()
                .replace(" ", "")
            )

            filters.append(
                {
                    "semester": {
                        "$in": [
                            sem_val,
                            sem_clean
                        ]
                    }
                }
            )

        where_filter = (
            filters[0]
            if len(filters) == 1
            else {"$and": filters}
        )

        # -------------------------------------------------
        # Query
        # -------------------------------------------------

        results = None

        try:

            results = collection.query(
                query_embeddings=[
                    query_vector
                ],
                n_results=top_k,
                where=where_filter
            )

        except Exception as filter_error:

            print(
                "Filtered query failed, "
                f"falling back to college filter: "
                f"{filter_error}"
            )

        # -------------------------------------------------
        # Fallback
        # -------------------------------------------------

        if (
            not results
            or not results.get("documents")
            or not results["documents"][0]
        ):

            results = collection.query(
                query_embeddings=[
                    query_vector
                ],
                n_results=top_k,
                where={
                    "college": {
                        "$eq": college
                    }
                }
            )

        # -------------------------------------------------
        # Normalize results
        # -------------------------------------------------

        docs = (
            results.get("documents") or [[]]
        )[0]

        metas = (
            results.get("metadatas") or [[]]
        )[0]

        ids = (
            results.get("ids") or [[]]
        )[0]

        dists = (
            results.get("distances") or [[]]
        )[0]

        hits = []

        for i, doc in enumerate(docs):

            meta = (
                metas[i]
                if i < len(metas)
                else {}
            )

            result_id = (
                ids[i]
                if i < len(ids)
                else None
            )

            dist = (
                dists[i]
                if i < len(dists)
                else 0.0
            )

            similarity = max(
                0.0,
                1.0 - float(dist)
            )

            hits.append(
                {
                    "id": result_id,
                    "text": doc,
                    "metadata": meta,
                    "similarity": similarity
                }
            )

        results["hits"] = hits

        return results

    except Exception as e:

        print(
            f"Search error: {e}"
        )

        return None


# =========================================================
# Topic Frequency
# =========================================================

def get_topic_frequency(
    topic,
    college,
    course=None
):
    """
    Legacy frequency helper.

    This will eventually be replaced by the
    deterministic historical-statistics system
    in Phase 4.
    """

    try:

        results = search(
            topic,
            college,
            course=course,
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

        for meta in results.get(
            "metadatas",
            [[]]
        )[0]:

            year = meta.get(
                "exam_year",
                meta.get("year", "")
            )

            if (
                year
                and year != "unknown"
                and year != -1
            ):

                years.add(
                    str(year)
                )

        return {
            "topic": topic,
            "years_found": sorted(years),
            "count": len(years),
            "total_years": len(years)
        }

    except Exception as e:

        print(
            f"Frequency error: {e}"
        )

        return {
            "topic": topic,
            "count": 0,
            "years_found": [],
            "total_years": 0
        }


# =========================================================
# Statistics
# =========================================================

def get_stats():
    """
    Get ChromaDB statistics.
    """

    try:

        count = collection.count()

        return {
            "total_chunks": count,
            "total_documents": count,
            "by_college": {}
        }

    except Exception as e:

        print(
            f"Stats error: {e}"
        )

        return {
            "total_chunks": 0,
            "total_documents": 0,
            "by_college": {}
        }