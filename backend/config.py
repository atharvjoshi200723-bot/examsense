import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_MODELS = [GEMINI_MODEL, "gemini-3.5-flash", "gemini-flash-latest"]

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Paths
PAPERS_DIR = str(BASE_DIR / "data" / "papers")
CHROMA_DIR = str(BASE_DIR / "data" / "chromadb")

# All college data sources
COLLEGE_SOURCES = {
    "VIT": {
        "type": "github",
        "github_repo": "puneet-chandna/VIT-PYQPs-Paaji"
    },
    "SRM": {
        "type": "github",
        "github_repo": "srmist-2022-26/Study-Materials-2022"
    },
    "GraphicEra": {
        "type": "github",
        "github_repo": "gehuhaldwani/pyqs"
    },
    "BITS": {
        "type": "github",
        "github_repo": "thenicekat/BITSHYD-Past-Papers"
    },
    "UPES": {
        "type": "website",
        "base_url": "https://library.ddn.upes.ac.in/questionbank/"
    },
    "Manipal": {
        "type": "website",
        "base_url": "https://libportal.manipal.edu/mit/Question%20Paper.aspx"
    }
}

# List of college names for frontend dropdown
COLLEGE_NAMES = list(COLLEGE_SOURCES.keys())
