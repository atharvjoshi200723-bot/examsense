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

from config import GEMINI_API_KEY
from processors.rag_embedder import search
from google import genai

client = genai.Client(api_key=GEMINI_API_KEY)

def predict_questions(college, course, year, semester):
    """Predict exam questions from past papers"""
    try:
        # Search with multiple queries
        queries = [
            "important questions exam topics marks",
            "unit wise questions syllabus",
            "most asked questions previous year"
        ]

        all_chunks = []
        for query in queries:
            results = search(
                query,
                college=college,
                top_k=8
            )
            if results and results.get("documents"):
                docs = results["documents"]
                if docs and len(docs) > 0:
                    all_chunks.extend(docs[0])

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
                "study_tip": "No papers found for this college yet.",
                "total_papers_analyzed": 0
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
    "total_papers_analyzed": 15
}}

Priority must be one of: MUST_PREPARE, IMPORTANT, OPTIONAL
Give exactly 15 predicted questions.
Sort by probability_percent descending.
Base everything on these past papers:

{context}

Return ONLY the JSON. No explanation. No markdown."""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        text = response.text.strip()
        text = text.replace("```json", "").replace("```", "").strip()

        result = json.loads(text)
        return result

    except json.JSONDecodeError:
        return {
            "predicted_questions": [],
            "hot_topics": [],
            "safe_to_skip": [],
            "study_tip": "Error parsing predictions. Try again.",
            "total_papers_analyzed": 0
        }
    except Exception as e:
        return {
            "predicted_questions": [],
            "hot_topics": [],
            "safe_to_skip": [],
            "study_tip": f"Error: {str(e)}",
            "total_papers_analyzed": 0
        }