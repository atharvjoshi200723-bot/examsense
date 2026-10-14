"""
GitHub scraper for VIT, SRM, and GraphicEra question-paper PDFs.

Each PDF is downloaded to /tmp, passed to a processing callback, then deleted.
"""

import hashlib
import re
import sys
from pathlib import Path
from typing import Callable

import requests

# Allow `from config import ...` when run as: python scrapers/github_scraper.py
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import GITHUB_TOKEN  # noqa: E402

# ── College → GitHub repo mapping (exact repos for this scraper) ──────────────
COLLEGE_REPOS = {
    "VIT": "puneet-chandna/VIT-PYQPs-Paaji",
    "SRM": "srmist-2022-26/Study-Materials-2022",
    "GraphicEra": "gehuhaldwani/pyqs",
}

GITHUB_API = "https://api.github.com"
RAW_BASE = "https://raw.githubusercontent.com"
TEMP_ROOT = Path("/tmp/examsense")

# Regex helpers to pull year / semester tokens from folder names
YEAR_PATTERN = re.compile(
    r"year[\s_-]?(\d+)|y(?:ear)?[\s_-]?(\d+)|(\d+)(?:st|nd|rd|th)[\s_-]?year",
    re.IGNORECASE,
)
SEM_PATTERN = re.compile(
    r"sem(?:ester)?[\s_-]?(\d+)|s(?:em)?[\s_-]?(\d+)",
    re.IGNORECASE,
)
SKIP_DIR_NAMES = frozenset(
    {
        "pyq",
        "pyqs",
        "papers",
        "question",
        "questions",
        "bank",
        "qb",
        "previous",
        "past",
        "docs",
        "documents",
        "files",
        "assets",
        "data",
        "download",
        "downloads",
    }
)


def process_and_delete(filepath: str | Path, metadata: dict) -> None:
    """
    Process one downloaded PDF, then rely on the caller to delete the temp file.

    Replace or wrap this function when wiring R2 upload / indexing / embeddings.
    """
    # Placeholder — downstream pipeline will index the paper.
    _ = filepath, metadata


def _safe_segment(name: str) -> str:
    """Make a path segment safe for local filesystem use."""
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", name.strip())
    return cleaned or "unknown"


def parse_path_metadata(college: str, repo_path: str) -> dict:
    """
    Build metadata from the PDF's path inside the GitHub repo.

    Example: BTech/Year2/Sem4/paper.pdf → course, year2, sem4
    """
    parts = Path(repo_path).parts[:-1]
    filename = Path(repo_path).name
    course = "general"
    year = "unknown"
    semester = "unknown"
    subject = "general"

    # Check 4-digit calendar year in filename or path
    year_file_match = re.search(r"\b(20[0-2]\d)\b", filename)
    if year_file_match:
        year = year_file_match.group(1)

    for part in parts:
        year_match = YEAR_PATTERN.search(part)
        if year_match and year == "unknown":
            n = next(g for g in year_match.groups() if g is not None)
            year = n if len(n) == 4 else f"year{n}"
            continue

        sem_match = SEM_PATTERN.search(part)
        if sem_match:
            n = next(g for g in sem_match.groups() if g is not None)
            semester = f"sem{n}"
            continue

        normalized = part.lower().replace("-", "").replace("_", "")
        if normalized in SKIP_DIR_NAMES or part.startswith(".") or normalized in ("end", "mid", "old", "seta", "setb", "setc"):
            continue

        if course == "general":
            course = _safe_segment(part)
        elif subject == "general":
            subject = _safe_segment(part)

    # Normalize course capitalization for consistency with frontend
    c_lower = course.lower().replace(".", "").replace(" ", "")
    if c_lower == "btech":
        course = "BTech"
    elif c_lower == "bca":
        course = "BCA"
    elif c_lower == "bba":
        course = "BBA"
    elif c_lower == "mba":
        course = "MBA"
    elif c_lower == "mca":
        course = "MCA"
    elif "pharma" in c_lower:
        course = "BPharma"

    return {
        "college": college,
        "course": _safe_segment(course),
        "subject": _safe_segment(subject),
        "year": _safe_segment(year),
        "semester": _safe_segment(semester),
        "filename": _safe_segment(filename),
        "repo_path": repo_path,
    }


class GitHubScraper:
    """Lists and downloads PDFs from a single GitHub repository."""

    def __init__(self, college_name: str, repo_slug: str, token: str | None = None):
        self.college_name = college_name
        self.owner, self.repo = repo_slug.split("/", 1)
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/vnd.github.v3+json"
        if token:
            # Authenticated requests get a much higher GitHub API rate limit.
            self.session.headers["Authorization"] = f"Bearer {token}"

    def _get(self, url: str, timeout: int = 60) -> requests.Response:
        response = self.session.get(url, timeout=timeout)
        response.raise_for_status()
        return response

    def get_default_branch(self) -> str:
        """Step 1a: Read the repo's default branch (usually main or master)."""
        url = f"{GITHUB_API}/repos/{self.owner}/{self.repo}"
        return self._get(url, timeout=30).json()["default_branch"]

    def list_pdf_files(self) -> list[tuple[str, str]]:
        """
        Step 1b–1c: Use the Git Trees API with recursive=1 to list every file,
        then keep only paths ending in .pdf.
        """
        branch = self.get_default_branch()

        ref_url = f"{GITHUB_API}/repos/{self.owner}/{self.repo}/git/ref/heads/{branch}"
        commit_sha = self._get(ref_url, timeout=30).json()["object"]["sha"]

        commit_url = f"{GITHUB_API}/repos/{self.owner}/{self.repo}/git/commits/{commit_sha}"
        tree_sha = self._get(commit_url, timeout=30).json()["tree"]["sha"]

        tree_url = (
            f"{GITHUB_API}/repos/{self.owner}/{self.repo}/git/trees/{tree_sha}?recursive=1"
        )
        tree_data = self._get(tree_url, timeout=120).json()

        if tree_data.get("truncated"):
            print(
                f"{self.college_name}: Warning — repo tree was truncated; "
                "some PDFs may be missing."
            )

        pdfs: list[tuple[str, str]] = []
        for item in tree_data.get("tree", []):
            if item.get("type") != "blob":
                continue
            path = item.get("path", "")
            if not path.lower().endswith(".pdf"):
                continue
            download_url = f"{RAW_BASE}/{self.owner}/{self.repo}/{branch}/{path}"
            pdfs.append((path, download_url))

        return pdfs

    def _download_to_temp(self, download_url: str, repo_path: str, temp_dir: Path) -> Path:
        """Step 3a: Stream the PDF bytes into /tmp/examsense/{college}/."""
        # Hash the full repo path so two files with the same name don't collide.
        unique = hashlib.md5(repo_path.encode(), usedforsecurity=False).hexdigest()[:12]
        filename = _safe_segment(Path(repo_path).name)
        dest = temp_dir / f"{unique}_{filename}"

        response = self.session.get(download_url, stream=True, timeout=120)
        response.raise_for_status()

        with open(dest, "wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)

        return dest

    def run(
        self,
        process_fn: Callable[[str | Path, dict], None] = process_and_delete,
    ) -> int:
        """
        List all PDFs, download each to /tmp, call process_fn, then delete the temp file.
        Returns the number of PDFs successfully processed.
        """
        # ── Step 1: Discover every PDF in the repository ────────────────────
        try:
            pdf_files = self.list_pdf_files()
        except requests.RequestException as exc:
            print(f"{self.college_name}: Failed to list repo — {exc}")
            return 0

        total = len(pdf_files)
        print(f"{self.college_name}: Found {total} PDFs")

        if total == 0:
            print(f"{self.college_name}: Done! Processed 0 papers")
            return 0

        temp_dir = TEMP_ROOT / _safe_segment(self.college_name)
        temp_dir.mkdir(parents=True, exist_ok=True)
        processed = 0

        # ── Step 2–4: Download → process → delete, one file at a time ───────
        for index, (repo_path, download_url) in enumerate(pdf_files, start=1):
            filename = Path(repo_path).name
            metadata = parse_path_metadata(self.college_name, repo_path)
            temp_path: Path | None = None

            print(
                f"{self.college_name}: Processing {index}/{total} - {filename}",
                flush=True,
            )

            try:
                # Download to a temporary path under /tmp
                temp_path = self._download_to_temp(download_url, repo_path, temp_dir)

                # Hand off to the pipeline (upload, index, etc.)
                process_fn(temp_path, metadata)
                processed += 1

            except requests.RequestException as exc:
                print(f"  Skipped (download failed): {filename} — {exc}")
            except Exception as exc:
                print(f"  Skipped (processing failed): {filename} — {exc}")
            finally:
                # Always remove the local copy — we only needed it temporarily
                if temp_path is not None and temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError as exc:
                        print(f"  Warning: could not delete temp file — {exc}")

        print(f"{self.college_name}: Done! Processed {processed} papers")
        return processed


def run_all(
    process_fn: Callable[[str | Path, dict], None] = process_and_delete,
) -> int:
    """
    Run the GitHub scraper for VIT, SRM, and GraphicEra, one college at a time.
    """
    token = GITHUB_TOKEN
    if token:
        print("Using GITHUB_TOKEN from config (higher API rate limits).")
    else:
        print("No GITHUB_TOKEN in config — unauthenticated API (lower rate limits).")

    total = 0
    for college_name, repo_slug in COLLEGE_REPOS.items():
        scraper = GitHubScraper(college_name, repo_slug, token=token)
        total += scraper.run(process_fn=process_fn)

    print(f"\nAll colleges finished. {total} paper(s) processed in total.")
    return total


if __name__ == "__main__":
    run_all()
