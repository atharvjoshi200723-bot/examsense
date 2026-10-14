"""
Predictor - Predicts exam questions using RAG
"""

import sys
import os
import json
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import re
from config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_MODELS
from processors.rag_embedder import search
from google import genai

_gemini_client = None

def _get_client():
    global _gemini_client
    if _gemini_client is None and GEMINI_API_KEY:
        try:
            _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        except Exception as e:
            print(f"Error initializing Gemini client: {e}")
            _gemini_client = None
    return _gemini_client

def predict_questions(college, course, year, semester):
    """Predict exam questions from past papers"""
    try:
        # Search with multiple queries, including course context
        queries = [
            f"{course} important questions exam topics marks",
            f"{course} unit wise questions syllabus",
            f"{course} most asked questions previous year"
        ]

        all_chunks = []
        unique_papers = set()
        for query in queries:
            results = search(
                query,
                college=college,
                course=course,
                semester=semester,
                top_k=8
            )
            if results and results.get("documents"):
                docs = results["documents"]
                metas = results.get("metadatas", [[]])
                if docs and len(docs) > 0:
                    for idx, doc in enumerate(docs[0]):
                        all_chunks.append(doc)
                        if metas and len(metas[0]) > idx:
                            fn = metas[0][idx].get("filename")
                            if fn:
                                unique_papers.add(fn)

        # Remove duplicates
        seen = set()
        unique_chunks = []
        for chunk in all_chunks:
            if chunk not in seen:
                seen.add(chunk)
                unique_chunks.append(chunk)

        if not unique_chunks:
            return {
                "predicted_questions": [],
                "hot_topics": [],
                "safe_to_skip": [],
                "study_tip": f"No past papers found for {college} ({course} {semester}).",
                "total_papers_analyzed": 0
            }

        client = _get_client()
        if not client:
            return {
                "predicted_questions": [],
                "hot_topics": [],
                "safe_to_skip": [],
                "study_tip": "Gemini API key is not configured or failed to initialize.",
                "total_papers_analyzed": len(unique_papers) or len(unique_chunks)
            }

        # Take top 15 chunks as context
        context = "\n\n---\n\n".join(unique_chunks[:15])

        prompt = f"""You are an expert exam analyst for {college} university India.

Analyze these past {course} exam papers carefully.

Return ONLY a valid JSON object with exactly this structure, no other text:
{{
    "predicted_questions": [
        {{
            "question": "full question text here",
            "topic": "topic name",
            "years_appeared": [2021, 2022, 2023],
            "frequency": 3,
            "total_years": 5,
            "probability_percent": 85,
            "priority": "MUST_PREPARE",
            "marks": 10,
            "tip": "one line exam writing tip"
        }}
    ],
    "hot_topics": ["topic1", "topic2", "topic3"],
    "safe_to_skip": ["topic4", "topic5"],
    "study_tip": "one overall study advice",
    "total_papers_analyzed": {max(len(unique_papers), len(unique_chunks[:15]))}
}}

Priority must be one of: MUST_PREPARE, IMPORTANT, OPTIONAL
Give up to 15 predicted questions.
Sort by probability_percent descending.
Base everything on these past papers:

{context}

Return ONLY the JSON. No explanation. No markdown."""

        import time
        response = None
        for model_name in GEMINI_MODELS:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                if response and response.text:
                    break
            except Exception as model_err:
                print(f"Model {model_name} error: {model_err}")
                continue

        if not response or not response.text:
            raise RuntimeError("Gemini service is currently unavailable. Please try again in a few moments.")

        text = (response.text or "").strip()
        # Extract json safely using regex
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            text = json_match.group(0)

        result = json.loads(text)
        result["total_papers_analyzed"] = max(len(unique_papers), 1)
        return result

    except json.JSONDecodeError as jde:
        print(f"JSON decode error in predict_questions: {jde}")
        return {
            "predicted_questions": [],
            "hot_topics": [],
            "safe_to_skip": [],
            "study_tip": "Error parsing predictions from AI. Please try again.",
            "total_papers_analyzed": 0
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        return {
            "predicted_questions": [],
            "hot_topics": [],
            "safe_to_skip": [],
            "study_tip": f"Prediction error: {str(e)}",
            "total_papers_analyzed": 0
        }