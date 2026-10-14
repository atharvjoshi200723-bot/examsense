
"""
GitHub scraper for VIT, SRM, and GraphicEra question-paper PDFs.

Downloads PDFs one at a time to /tmp, extracts structured metadata,
passes them to the downstream processing pipeline, then deletes them.
"""

import hashlib
import re
import sys
from pathlib import Path
from typing import Callable

import requests


# ---------------------------------------------------------------------------
# Backend import setup
# ---------------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import GITHUB_TOKEN  # noqa: E402


# ---------------------------------------------------------------------------
# College -> GitHub repository mapping
# ---------------------------------------------------------------------------

COLLEGE_REPOS = {
    "VIT": "puneet-chandna/VIT-PYQPs-Paaji",
    "SRM": "srmist-2022-26/Study-Materials-2022",
    "GraphicEra": "gehuhaldwani/pyqs",
}


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GITHUB_API = "https://api.github.com"
RAW_BASE = "https://raw.githubusercontent.com"

TEMP_ROOT = Path("/tmp/examsense")


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

YEAR_PATTERN = re.compile(
    r"\byear[\s_-]?(\d+)\b"
    r"|\by[\s_-]?(\d+)\b"
    r"|\b(\d+)(?:st|nd|rd|th)[\s_-]?year\b",
    re.IGNORECASE,
)

SEM_PATTERN = re.compile(
    r"\bsem(?:ester)?[\s_-]?(\d+)\b"
    r"|\bs(?:em)?[\s_-]?(\d+)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Directory names that are not useful metadata
# ---------------------------------------------------------------------------

SKIP_DIR_NAMES = frozenset(
    {
        "pyq",
        "pyqs",
        "papers",
        "paper",
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


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _safe_segment(name: str) -> str:
    """Make a path segment safe for local filesystem use."""
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", name.strip())
    return cleaned or "unknown"


def _clean_name(value: str) -> str:
    """Clean whitespace while preserving useful characters."""
    value = value.strip()
    value = re.sub(r"\s+", " ", value)
    return value


def _slug_name(value: str) -> str:
    """Create a normalized lowercase representation."""
    value = _clean_name(value)
    value = value.replace("_", " ")
    value = value.replace("-", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip().lower()


# ---------------------------------------------------------------------------
# Metadata normalization
# ---------------------------------------------------------------------------

def _normalize_degree(value: str) -> str:
    """Normalize degree/program names."""

    cleaned = value.strip().lower()
    compact = re.sub(r"[\s._-]+", "", cleaned)

    mapping = {
        "btech": "BTech",
        "be": "BE",
        "bca": "BCA",
        "bba": "BBA",
        "mba": "MBA",
        "mca": "MCA",
        "mtech": "MTech",
        "bpharma": "BPharma",
        "mpharma": "MPharma",
    }

    if cleaned in mapping:
        return mapping[cleaned]

    if compact in mapping:
        return mapping[compact]

    return value.strip()


def _normalize_semester(value: str) -> str:
    """Convert semester representations to canonical form."""

    match = SEM_PATTERN.search(value)

    if not match:
        return "unknown"

    number = next(
        group for group in match.groups()
        if group is not None
    )

    return f"Sem {int(number)}"


# ---------------------------------------------------------------------------
# Year extraction
# ---------------------------------------------------------------------------

def _extract_year_of_study(parts: list[str]) -> str:
    """
    Extract academic year of study.

    Examples:
        Year 1 -> Year 1
        year2 -> Year 2
        3rd year -> Year 3

    IMPORTANT:
    2025 is an exam/calendar year, NOT a year of study.
    """

    for part in parts:
        match = YEAR_PATTERN.search(part)

        if not match:
            continue

        number = next(
            group for group in match.groups()
            if group is not None
        )

        if len(number) == 4:
            continue

        return f"Year {int(number)}"

    return "unknown"


def _extract_exam_year(
    filename: str,
    repo_path: str,
) -> str:
    """
    Extract calendar/exam year.

    Prefer filename.

    Example:
        tcs503_endsem_2025_jul_3.pdf
        -> 2025
    """

    filename_matches = re.findall(
        r"(?<!\d)(20\d{2})(?!\d)",
        filename,
    )

    if filename_matches:
        return filename_matches[0]

    path_matches = re.findall(
        r"(?<!\d)(20\d{2})(?!\d)",
        repo_path,
    )

    if path_matches:
        return path_matches[0]

    return "unknown"


# ---------------------------------------------------------------------------
# Filename metadata
# ---------------------------------------------------------------------------

def _extract_course_code(filename: str) -> str:
    """
    Extract course code.

    Examples:
        tcs503_endsem_2025_jul_3.pdf -> TCS503
        cse201_mid_2024.pdf          -> CSE201
    """

    stem = Path(filename).stem

    matches = re.findall(
        r"(?<![A-Za-z0-9])"
        r"([A-Za-z]{2,8}\d{2,4})"
        r"(?![A-Za-z0-9])",
        stem,
        flags=re.IGNORECASE,
    )

    if not matches:
        return "unknown"

    return matches[0].upper()


def _extract_exam_month(filename: str) -> str:
    """Extract exam month from filename."""

    months = {
        "jan": "jan",
        "january": "jan",
        "feb": "feb",
        "february": "feb",
        "mar": "mar",
        "march": "mar",
        "apr": "apr",
        "april": "apr",
        "may": "may",
        "jun": "jun",
        "june": "jun",
        "jul": "jul",
        "july": "jul",
        "aug": "aug",
        "august": "aug",
        "sep": "sep",
        "sept": "sep",
        "september": "sep",
        "oct": "oct",
        "october": "oct",
        "nov": "nov",
        "november": "nov",
        "dec": "dec",
        "december": "dec",
    }

    tokens = re.split(
        r"[_\-\s.]+",
        Path(filename).stem.lower(),
    )

    for token in tokens:
        if token in months:
            return months[token]

    return "unknown"


def _extract_exam_type(repo_path: str) -> str:
    """Detect exam type from path and filename."""

    text = repo_path.lower()

    if re.search(
        r"\b(end[_\-\s]?sem|endsem|end)\b",
        text,
    ):
        return "endsem"

    if re.search(
        r"\b(mid[_\-\s]?sem|midsem|mid)\b",
        text,
    ):
        return "midsem"

    if re.search(r"\bsessional\b", text):
        return "sessional"

    if re.search(r"\bquiz\b", text):
        return "quiz"

    if re.search(r"\binternal\b", text):
        return "internal"

    return "unknown"

def _extract_attempt(repo_path: str) -> str:
    """Detect examination attempt from the repository path."""

    text = repo_path.lower()

    # Check special attempts FIRST.
    # Underscores are common separators in filenames, so \b is not
    # sufficient here because "_" is treated as a word character.
    if re.search(
        r"(^|[_\-\s])(back|backlog|reappear|reappearence|reappearance)(?=[_\-\s.]|$)",
        text,
    ):
        return "back"

    if re.search(
        r"(^|[_\-\s])(supp|supplementary|supply)(?=[_\-\s.]|$)",
        text,
    ):
        return "supplementary"

    if re.search(
        r"(^|[_\-\s])regular(?=[_\-\s.]|$)",
        text,
    ):
        return "regular"

    # No special marker means regular.
    return "regular"



# ---------------------------------------------------------------------------
# MAIN METADATA PARSER
# ---------------------------------------------------------------------------

def parse_path_metadata(
    college: str,
    repo_path: str,
) -> dict:
    """
    Parse structured metadata from a question-paper path.

    Example:

        btech/CSE/sem 5/dbms/end/
        tcs503_endsem_2025_jul_3.pdf

    Produces:

        degree       = BTech
        branch       = CSE
        semester     = Sem 5
        subject      = dbms
        course_code  = TCS503
        exam_type    = endsem
        attempt      = regular
        exam_year    = 2025
        exam_month   = jul

    The parser is structural and is NOT hardcoded to CSE.
    """

    path = Path(repo_path)
    filename = path.name

    raw_parts = list(path.parts[:-1])

    parts = [
        _clean_name(part)
        for part in raw_parts
        if part.strip()
        and not part.startswith(".")
    ]

    metadata = {
        "college": college,
        "university": college,

        "degree": "unknown",
        "branch": "unknown",

        "semester": "unknown",

        "academic_year": "unknown",
        "year_of_study": "unknown",

        "subject": "unknown",
        "course_code": "unknown",

        "exam_type": "unknown",
        "attempt": "regular",

        "exam_year": "unknown",
        "exam_month": "unknown",

        "filename": _safe_segment(filename),
        "repo_path": repo_path,
    }

    # -----------------------------------------------------------------------
    # 1. DEGREE
    # -----------------------------------------------------------------------

    degree_index = None

    degree_values = {
        "btech",
        "b.tech",
        "be",
        "b.e",
        "bca",
        "bba",
        "mba",
        "mca",
        "mtech",
        "m.tech",
        "bpharma",
        "b.pharma",
        "mpharma",
        "m.pharma",
    }

    for index, part in enumerate(parts):
        normalized = _slug_name(part)

        if normalized in degree_values:
            metadata["degree"] = _normalize_degree(part)
            degree_index = index
            break

    # -----------------------------------------------------------------------
    # 2. SEMESTER
    # -----------------------------------------------------------------------

    for part in parts:
        match = SEM_PATTERN.search(part)

        if match:
            metadata["semester"] = _normalize_semester(part)
            break

    # -----------------------------------------------------------------------
    # 3. YEAR OF STUDY
    # -----------------------------------------------------------------------

    year_of_study = _extract_year_of_study(parts)

    metadata["year_of_study"] = year_of_study
    metadata["academic_year"] = year_of_study

    # -----------------------------------------------------------------------
    # 4. BRANCH
    # -----------------------------------------------------------------------

    if degree_index is not None:
        after_degree = parts[degree_index + 1:]

        if after_degree:
            candidate = after_degree[0]
            normalized = _slug_name(candidate)

            is_year = YEAR_PATTERN.search(candidate) is not None
            is_semester = SEM_PATTERN.search(candidate) is not None

            is_generic = normalized in {
                "common",
                "general",
                "all",
            }

            if (
                not is_year
                and not is_semester
                and normalized not in SKIP_DIR_NAMES
                and not is_generic
            ):
                metadata["branch"] = _clean_name(candidate)

    # -----------------------------------------------------------------------
    # 5. SUBJECT
    # -----------------------------------------------------------------------

    exam_index = None

    exam_directory_names = {
        "end",
        "endsem",
        "end sem",
        "mid",
        "midsem",
        "mid sem",
        "sessional",
        "quiz",
        "internal",
    }

    for index, part in enumerate(parts):
        normalized = _slug_name(part)

        if normalized in exam_directory_names:
            exam_index = index
            break

    if exam_index is not None and exam_index > 0:
        candidate = parts[exam_index - 1]
        normalized = _slug_name(candidate)

        if (
            normalized not in SKIP_DIR_NAMES
            and not YEAR_PATTERN.search(candidate)
            and not SEM_PATTERN.search(candidate)
        ):
            metadata["subject"] = _clean_name(candidate)

    # -----------------------------------------------------------------------
    # 6. FILENAME INFORMATION
    # -----------------------------------------------------------------------

    metadata["course_code"] = _extract_course_code(filename)

    metadata["exam_year"] = _extract_exam_year(
        filename,
        repo_path,
    )

    metadata["exam_month"] = _extract_exam_month(filename)

    metadata["exam_type"] = _extract_exam_type(repo_path)

    metadata["attempt"] = _extract_attempt(repo_path)

    # -----------------------------------------------------------------------
    # 7. BACKWARDS COMPATIBILITY
    # -----------------------------------------------------------------------

    metadata["course"] = metadata["degree"]
    metadata["year"] = metadata["exam_year"]

    if metadata["semester"] != "unknown":
        match = re.search(
            r"(\d+)",
            metadata["semester"],
        )

        if match:
            metadata["semester_legacy"] = (
                f"sem{match.group(1)}"
            )
        else:
            metadata["semester_legacy"] = "unknown"
    else:
        metadata["semester_legacy"] = "unknown"

    # -----------------------------------------------------------------------
    # 8. Final cleanup
    # -----------------------------------------------------------------------

    fields_to_clean = [
        "degree",
        "branch",
        "semester",
        "academic_year",
        "year_of_study",
        "subject",
        "course_code",
        "exam_type",
        "attempt",
        "exam_year",
        "exam_month",
        "course",
        "year",
        "semester_legacy",
    ]

    for key in fields_to_clean:
        if metadata[key] != "unknown":
            metadata[key] = _clean_name(
                str(metadata[key])
            )

    return metadata


# ---------------------------------------------------------------------------
# Existing processing callback
# ---------------------------------------------------------------------------

def process_and_delete(
    filepath: str | Path,
    metadata: dict,
) -> None:
    """
    Process one downloaded PDF.

    The downstream ingestion pipeline will replace/wrap this.
    """
    _ = filepath, metadata


# ---------------------------------------------------------------------------
# GitHub scraper
# ---------------------------------------------------------------------------

class GitHubScraper:
    """Lists and downloads PDFs from a single GitHub repository."""

    def __init__(
        self,
        college_name: str,
        repo_slug: str,
        token: str | None = None,
    ):
        self.college_name = college_name

        self.owner, self.repo = repo_slug.split(
            "/",
            1,
        )

        self.session = requests.Session()

        self.session.headers["Accept"] = (
            "application/vnd.github.v3+json"
        )

        if token:
            self.session.headers["Authorization"] = (
                f"Bearer {token}"
            )

    def _get(
        self,
        url: str,
        timeout: int = 60,
    ) -> requests.Response:

        response = self.session.get(
            url,
            timeout=timeout,
        )

        response.raise_for_status()

        return response

    def get_default_branch(self) -> str:
        """Get repository default branch."""

        url = (
            f"{GITHUB_API}/repos/"
            f"{self.owner}/{self.repo}"
        )

        return self._get(
            url,
            timeout=30,
        ).json()["default_branch"]

    def list_pdf_files(
        self,
    ) -> list[tuple[str, str]]:
        """List all PDF files in the repository."""

        branch = self.get_default_branch()

        ref_url = (
            f"{GITHUB_API}/repos/"
            f"{self.owner}/{self.repo}"
            f"/git/ref/heads/{branch}"
        )

        commit_sha = self._get(
            ref_url,
            timeout=30,
        ).json()["object"]["sha"]

        commit_url = (
            f"{GITHUB_API}/repos/"
            f"{self.owner}/{self.repo}"
            f"/git/commits/{commit_sha}"
        )

        tree_sha = self._get(
            commit_url,
            timeout=30,
        ).json()["tree"]["sha"]

        tree_url = (
            f"{GITHUB_API}/repos/"
            f"{self.owner}/{self.repo}"
            f"/git/trees/{tree_sha}"
            f"?recursive=1"
        )

        tree_data = self._get(
            tree_url,
            timeout=120,
        ).json()

        if tree_data.get("truncated"):
            print(
                f"{self.college_name}: Warning — "
                "repo tree was truncated; some PDFs "
                "may be missing."
            )

        pdfs = []

        for item in tree_data.get("tree", []):
            if item.get("type") != "blob":
                continue

            path = item.get("path", "")

            if not path.lower().endswith(".pdf"):
                continue

            download_url = (
                f"{RAW_BASE}/"
                f"{self.owner}/"
                f"{self.repo}/"
                f"{branch}/"
                f"{path}"
            )

            pdfs.append(
                (
                    path,
                    download_url,
                )
            )

        return pdfs

    def _download_to_temp(
        self,
        download_url: str,
        repo_path: str,
        temp_dir: Path,
    ) -> Path:
        """Download PDF into temporary directory."""

        unique = hashlib.md5(
            repo_path.encode(),
            usedforsecurity=False,
        ).hexdigest()[:12]

        filename = _safe_segment(
            Path(repo_path).name
        )

        dest = temp_dir / (
            f"{unique}_{filename}"
        )

        response = self.session.get(
            download_url,
            stream=True,
            timeout=120,
        )

        response.raise_for_status()

        with open(dest, "wb") as handle:
            for chunk in response.iter_content(
                chunk_size=8192
            ):
                if chunk:
                    handle.write(chunk)

        return dest

    def run(
        self,
        process_fn: Callable[
            [str | Path, dict],
            None,
        ] = process_and_delete,
    ) -> int:
        """
        List PDFs, download one at a time,
        process, then delete.
        """

        try:
            pdf_files = self.list_pdf_files()

        except requests.RequestException as exc:
            print(
                f"{self.college_name}: "
                f"Failed to list repo — {exc}"
            )
            return 0

        total = len(pdf_files)

        print(
            f"{self.college_name}: "
            f"Found {total} PDFs"
        )

        if total == 0:
            print(
                f"{self.college_name}: "
                "Done! Processed 0 papers"
            )
            return 0

        temp_dir = (
            TEMP_ROOT
            / _safe_segment(
                self.college_name
            )
        )

        temp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        processed = 0

        for index, (
            repo_path,
            download_url,
        ) in enumerate(
            pdf_files,
            start=1,
        ):
            filename = Path(repo_path).name

            metadata = parse_path_metadata(
                self.college_name,
                repo_path,
            )

            temp_path: Path | None = None

            print(
                f"{self.college_name}: "
                f"Processing {index}/{total} - "
                f"{filename}",
                flush=True,
            )

            try:
                temp_path = self._download_to_temp(
                    download_url,
                    repo_path,
                    temp_dir,
                )

                process_fn(
                    temp_path,
                    metadata,
                )

                processed += 1

            except requests.RequestException as exc:
                print(
                    f"  Skipped "
                    f"(download failed): "
                    f"{filename} — {exc}"
                )

            except Exception as exc:
                print(
                    f"  Skipped "
                    f"(processing failed): "
                    f"{filename} — {exc}"
                )

            finally:
                if (
                    temp_path is not None
                    and temp_path.exists()
                ):
                    try:
                        temp_path.unlink()
                    except OSError as exc:
                        print(
                            "  Warning: could not "
                            f"delete temp file — {exc}"
                        )

        print(
            f"{self.college_name}: "
            f"Done! Processed {processed} papers"
        )

        return processed


# ---------------------------------------------------------------------------
# Run all colleges
# ---------------------------------------------------------------------------

def run_all(
    process_fn: Callable[
        [str | Path, dict],
        None,
    ] = process_and_delete,
) -> int:
    """Run scraper for all configured colleges."""

    token = GITHUB_TOKEN

    if token:
        print(
            "Using GITHUB_TOKEN from config "
            "(higher API rate limits)."
        )
    else:
        print(
            "No GITHUB_TOKEN in config — "
            "unauthenticated API "
            "(lower rate limits)."
        )

    total = 0

    for college_name, repo_slug in COLLEGE_REPOS.items():
        scraper = GitHubScraper(
            college_name,
            repo_slug,
            token=token,
        )

        total += scraper.run(
            process_fn=process_fn
        )

    print(
        f"\nAll colleges finished. "
        f"{total} paper(s) processed in total."
    )

    return total


# ---------------------------------------------------------------------------
# TEST / MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Metadata parser tests only.
    # These DO NOT download the repositories.

    test_paths = [
        (
            "btech/CSE/sem 5/dbms/end/"
            "tcs503_endsem_2025_jul_3.pdf"
        ),
        (
            "btech/IT/sem 3/data_structures/end/"
            "xyz301_endsem_2024_dec.pdf"
        ),
        (
            "btech/Year 1/common/sem 1/mathematics/end/"
            "mth101_endsem_2025_jan.pdf"
        ),
        (
            "btech/CSE/sem 5/dbms/end/"
            "tcs503_endsem_back_2025_jul.pdf"
        ),
        (
            "btech/CSE/sem 4/operating_system/mid/"
            "cse401_midsem_2024_apr.pdf"
        ),
    ]

    for test_path in test_paths:
        print("\n" + "=" * 80)
        print(test_path)
        print("=" * 80)

        result = parse_path_metadata(
            "GraphicEra",
            test_path,
        )

        for key, value in result.items():
            print(
                f"{key:20} : {value}"
            )