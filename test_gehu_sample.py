"""Download and validate three representative GEHU PYQs without writing to Chroma."""
from pathlib import Path
from urllib.parse import quote
import sys
import requests

BACKEND = Path(__file__).resolve().parent / "backend"
if BACKEND.exists():
    sys.path.insert(0, str(BACKEND))
else:
    # Also works when copied into the repository root or run with PYTHONPATH=backend.
    sys.path.insert(0, str(Path.cwd() / "backend"))

try:
    from processors.pdf_reader import extract_text
    from question_extractor import extract_questions
except Exception as exc:
    print("IMPORT FAILED:", repr(exc))
    print("Run this from the ExamSense repository root with PYTHONPATH=backend.")
    raise SystemExit(1)

REPO = "gehuhaldwani/pyqs"
BRANCH = "main"
OUT = Path("/tmp/examsense_gehu_sample_test")
OUT.mkdir(parents=True, exist_ok=True)


def category(path: str) -> str | None:
    parts = [p.lower() for p in path.split("/")]
    all_parts = parts + [Path(path).stem.lower()]
    if any("sessional" in p for p in all_parts):
        return "sessional"
    if any(p in {"mid", "midsem", "mid-sem", "mid-semester"} or "midsem" in p for p in all_parts):
        return "mid-semester"
    if any(p == "end" or p in {"endsem", "end-sem", "end-semester"} or "endsem" in p for p in all_parts):
        return "end-semester"
    return None


tree_url = f"https://api.github.com/repos/{REPO}/git/trees/{BRANCH}?recursive=1"
print("Finding a small sample in the public GEHU PYQ repository...")
response = requests.get(tree_url, timeout=45)
response.raise_for_status()
tree_data = response.json()
if tree_data.get("truncated"):
    print("WARNING: GitHub returned a truncated repository listing; sample may be incomplete.")

pdf_paths = [
    item["path"] for item in tree_data.get("tree", [])
    if item.get("type") == "blob" and item.get("path", "").lower().endswith(".pdf")
]
selected = []
for wanted in ("end-semester", "mid-semester", "sessional"):
    match = next((p for p in pdf_paths if category(p) == wanted and p not in selected), None)
    if match:
        selected.append(match)

if len(selected) < 3:
    for path in pdf_paths:
        if path not in selected:
            selected.append(path)
        if len(selected) == 3:
            break

if not selected:
    print("FAILED: No PDFs were found in the repository tree.")
    raise SystemExit(1)

print(f"Selected {len(selected)} PDF(s) for a read-only extraction test. No Chroma writes will occur.")
failures = 0

for index, repo_path in enumerate(selected, start=1):
    filename = f"{index}_{Path(repo_path).name}"
    destination = OUT / filename
    download_url = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{quote(repo_path, safe='/')}"
    print("\n" + "=" * 78)
    print(f"SAMPLE {index}: {repo_path}")
    try:
        if not destination.exists() or destination.stat().st_size == 0:
            r = requests.get(download_url, timeout=60)
            r.raise_for_status()
            destination.write_bytes(r.content)
        result = extract_text(str(destination))
        questions = extract_questions(result["pages"])
        labels = []
        contaminated = []
        short = []
        for q in questions:
            label = f"Q{q.question_number}"
            if q.sub_question:
                label += f"({q.sub_question})"
            labels.append(label)
            text = (q.question_text or "").strip()
            lower = text.lower()
            if any(flag in lower for flag in (
                "all the questions are compulsory",
                "maximum marks",
                "answer any two sub questions",
                "each sub-question carries",
            )):
                contaminated.append(label)
            if len(text) < 15:
                short.append(label)
        print(f"Method: {result.get('method')} | Pages: {result.get('page_count')} | Records: {len(questions)}")
        print("Labels:", ", ".join(labels[:30]) if labels else "(none)")
        for q in questions[:4]:
            label = f"Q{q.question_number}" + (f"({q.sub_question})" if q.sub_question else "")
            print(f"  {label}: {(q.question_text or '').strip()[:140]}")
        warnings = []
        if len(questions) < 3:
            warnings.append("fewer than 3 records")
        if contaminated:
            warnings.append("paper instructions leaked into: " + ", ".join(contaminated))
        if short:
            warnings.append("very short records: " + ", ".join(short))

        # OCR can return legible text in the wrong column/order. A basic
        # record-count check alone incorrectly calls such a paper a pass.
        main_numbers_in_order = []
        for q in questions:
            if not main_numbers_in_order or main_numbers_in_order[-1] != q.question_number:
                main_numbers_in_order.append(q.question_number)
        regressions = [
            (left, right) for left, right in zip(main_numbers_in_order, main_numbers_in_order[1:])
            if right < left
        ]
        if regressions:
            warnings.append(
                "main-question numbering goes backwards "
                + ", ".join(f"Q{left}->Q{right}" for left, right in regressions[:4])
                + " (possible OCR reading-order/label issue)"
            )

        unique_numbers = set(main_numbers_in_order)
        if unique_numbers:
            missing_numbers = [n for n in range(1, max(unique_numbers) + 1) if n not in unique_numbers]
            if len(missing_numbers) >= 2:
                preview = ", ".join(f"Q{n}" for n in missing_numbers[:8])
                warnings.append(
                    "question-number gaps detected: " + preview
                    + (" ..." if len(missing_numbers) > 8 else "")
                )

        parent_numbers = {q.question_number for q in questions if not q.sub_question}
        child_numbers = {q.question_number for q in questions if q.sub_question}
        overlapping = sorted(parent_numbers & child_numbers)
        if overlapping:
            warnings.append(
                "both parent and subpart records exist for "
                + ", ".join(f"Q{n}" for n in overlapping[:6])
                + " (verify these were not misread labels)"
            )
        if warnings:
            failures += 1
            print("REVIEW NEEDED:", "; ".join(warnings))
        else:
            print("BASIC QUALITY CHECK: PASS (manual content review still needed)")
    except Exception as exc:
        failures += 1
        print("SAMPLE FAILED:", repr(exc))

print("\n" + "=" * 78)
print(f"SAMPLES FLAGGED FOR REVIEW/ERROR: {failures} of {len(selected)}")
print("This test does not modify ChromaDB and does not start full ingestion.")
print("Inspect each sample's labels and question text before approving full ingestion.")
