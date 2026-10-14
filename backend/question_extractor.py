"""Question extraction for ExamSense.

Supports two common paper layouts:
1. Q-style papers: Q1 -> (a), (b), ...
2. Numbered MCQ/objective papers: 1. question -> A./B./C./D. options

The parser deliberately does not invent missing question numbers. It preserves
options in the question text instead of turning them into independent questions.
"""

import re
from typing import Any

from question_models import QuestionOccurrence

# Main question markers such as Q1, Q.1, Question 1, Que 1.
Q_MAIN_RE = re.compile(
    r"^\s*(?:question|ques|que|q)\s*\.?\s*(\d{1,2})(?!\d)"
    r"(?=\s|[.):\-–—]|$)\s*[.):\-–—]?\s*(.*)$",
    re.IGNORECASE,
)

# OCR sometimes reads Q1 as QI, Ql, or Q|. Keep this correction narrow.
OCR_Q1_RE = re.compile(
    r"^\s*q\s*[iIl|](?=\s|[.):\-–—]|$)\s*[.):\-–—]?\s*(.*)$",
    re.IGNORECASE,
)

# Inline subpart labels: Q1(a) text, Q1 (b): text.
INLINE_Q_SUB_RE = re.compile(
    r"^\s*(?:question|ques|que|q)\s*\.?\s*(\d{1,2})\s*"
    r"\(\s*([a-hA-H])\s*\)\s*[:.)\-–—]?\s*(.*)$",
    re.IGNORECASE,
)

# Numbered independent questions: 1. text / 2) text. Decimal numbers like 5.5
# do not match because punctuation must be followed by whitespace or EOL.
NUMBERED_MAIN_RE = re.compile(
    r"^\s*(\d{1,2})\s*[.)](?=\s|$)\s*(.*)$"
)

# Lowercase subparts are common in descriptive papers.
SUBPART_RE = re.compile(
    r"^\s*(?:\(\s*(?P<paren>[a-h])\s*\)|"
    r"(?P<close>[a-h])\s*[)]|(?P<dot>[a-h])\s*[.,](?=\s|$))"
    r"\s*[:.)\-–—]?\s*(?P<rest>.*)$"
)
DOT_SUBPART_RE = re.compile(
    r"^\s*(?P<letter>[a-d])\s*[.,]\s*(?P<rest>.+?)\s*$"
)
MARKS_ONLY_RE = re.compile(
    r"^\s*(?:\(?\s*)?\d+(?:\.\d+)?\s*"
    r"(?:marks?|m)\s*[).\]]?\s*$",
    re.IGNORECASE,
)

# MCQ options. These are retained in the question body in numbered/MCQ mode.
OPTION_RE = re.compile(
    r"^\s*(?:\(\s*(?P<paren>[A-Da-d])\s*\)|"
    r"(?P<dot>[A-D])\s*[.)]|(?P<close>[A-D])\s*[)])"
    r"\s*[:.)\-–—]?\s*(?P<rest>.*)$"
)

COURSE_OUTCOME_RE = re.compile(r"^\s*C\s*O?\s*0?\d+\s*$", re.IGNORECASE)
PAGE_RE = re.compile(
    r"^\s*Page\s+[IVXLCDM0-9]+\s+of\s+[IVXLCDM0-9]+\s*$",
    re.IGNORECASE,
)
SECTION_MARKS_RE = re.compile(
    r"^\s*\(?\s*\d+\s*[x×*]\s*\d+\s*=\s*\d+(?:\.\d+)?\s*"
    r"(?:marks?)?\s*\)?\s*$",
    re.IGNORECASE,
)
LEADING_MARKS_RE = re.compile(
    r"^[\(\[]\s*(?:\d+\s*[x×*]\s*\d+\s*=\s*)?"
    r"\d+(?:\.\d+)?\s*(?:marks?|m)?\s*[\)\]]\s*",
    re.IGNORECASE,
)
OR_RE = re.compile(r"^\s*OR\s*$", re.IGNORECASE)
SECTION_RE = re.compile(
    r"^\s*(?:section\s*[-:]?\s*[A-Za-z0-9]{1,2}|part\s*[-:]?\s*[A-Za-z0-9]{1,2})"
    r"\s*(?:[)\]:(\-–—].*)?$",
    re.IGNORECASE,
)

HEADER_PATTERNS = [
    re.compile(r"^\s*Graphic\s*$", re.IGNORECASE),
    re.compile(r"^\s*Era\s*$", re.IGNORECASE),
    re.compile(r"^\s*Graphic\s*Era\b.*$", re.IGNORECASE),
    re.compile(r"^\s*Established\b.*$", re.IGNORECASE),
    re.compile(r"^\s*End Term\b.*$", re.IGNORECASE),
    re.compile(r"^\s*GEHU/.*$", re.IGNORECASE),
    re.compile(r"^\s*Roll\s*no\b.*$", re.IGNORECASE),
    re.compile(r"^\s*Time\s*:.*$", re.IGNORECASE),
    re.compile(r"^\s*Maximum Marks\b.*$", re.IGNORECASE),
]

MCQ_HEADER_RE = re.compile(
    r"\b(?:multiple\s*[- ]?choice|\bmcq\b|objective\s+questions?)\b",
    re.IGNORECASE,
)
INSTRUCTION_RE = re.compile(
    r"^\s*(?:answer\b|attempt\b|solve\b|choose\b|note\b|"
    r"all\s+questions\b|any\s+(?:one|two|three|four|five|\d+)\b|"
    r"each\b|assume\b|short\s+questions?\s*:?|"
    r"long\s+questions?\s*:?|multiple\s*[- ]?choice\s+questions?\s*:?|"
    r"objective\s+questions?\s*:?)",
    re.IGNORECASE,
)


def clean_line(line: str) -> str:
    """Normalize whitespace and remove clearly column-aligned CO labels.

    PDF text extraction sometimes appends a right-column CO1/COI label to the
    end of a question line. Only strip it when separated by a visible wide gap,
    avoiding accidental removal of ordinary in-sentence terms such as CO2.
    """
    line = line.replace("\r", " ")
    line = re.sub(
        r"\s{2,}(?:C\s*O\s*[1-5]|C0[1-5]|CO[Il|])\s*$",
        "",
        line,
        flags=re.IGNORECASE,
    )
    line = re.sub(r"[ \t]+", " ", line)
    return line.strip()


def is_noise_line(line: str) -> bool:
    if not line:
        return True
    if COURSE_OUTCOME_RE.match(line) or PAGE_RE.match(line):
        return True
    if SECTION_MARKS_RE.match(line) or OR_RE.match(line) or SECTION_RE.match(line):
        return True
    return any(pattern.match(line) for pattern in HEADER_PATTERNS)


def parse_main_question(line: str, allow_numbered: bool = True) -> tuple[int, str] | None:
    """Return (number, remainder) for a main marker, or None."""
    match = Q_MAIN_RE.match(line)
    if match:
        number = int(match.group(1))
        return (number, match.group(2).strip()) if number > 0 else None

    match = OCR_Q1_RE.match(line)
    if match:
        return 1, match.group(1).strip()

    if allow_numbered:
        match = NUMBERED_MAIN_RE.match(line)
        if match:
            number = int(match.group(1))
            return (number, match.group(2).strip()) if number > 0 else None
    return None


def parse_subquestion(line: str) -> tuple[str | None, str]:
    match = SUBPART_RE.match(line)
    if not match:
        return None, ""
    letter = match.group("paren") or match.group("close") or match.group("dot")
    return letter, (match.group("rest") or "").strip()


def _is_short_option_sequence(lines: list[tuple[int, str]], start: int) -> bool:
    """Conservatively detect a/b/c/d dot-prefixed answer choices.

    Short items like ``a. PLP`` / ``b. NAD+`` are much more likely MCQ options
    than descriptive subquestions. Require at least three distinct option
    letters in increasing order and compact text; this avoids classifying full
    prose questions as choices.
    """
    found: list[tuple[str, str]] = []
    expected_order = {"a": 0, "b": 1, "c": 2, "d": 3}
    last_rank = -1
    question_like_start = re.compile(
        r"^(?:what|why|how|when|where|who|which|whose|whom|"
        r"define|explain|describe|discuss|list|state|derive|calculate|"
        r"compute|determine|write|mention|illustrate|give|identify|"
        r"compare|differentiate|enumerate|name|show|find|prove|"
        r"elaborate|briefly|is|are|do|does|can|could|should|would|will)\b",
        re.IGNORECASE,
    )
    for _, line in lines[start:min(len(lines), start + 8)]:
        if line == GROUP_BOUNDARY:
            break
        match = DOT_SUBPART_RE.match(line)
        if not match:
            # An option sequence should be adjacent except for a wrapped line.
            if found and len(line) < 55 and not re.match(r"^[A-D][.)]\s", line):
                continue
            break
        letter = match.group("letter")
        rest = match.group("rest").strip()
        if (
            expected_order[letter] <= last_rank
            or len(rest) > 100
            or question_like_start.match(rest)
            or rest.endswith("?")
        ):
            break
        found.append((letter, rest))
        last_rank = expected_order[letter]
        if letter == "d":
            break
    return len(found) >= 3 and found[0][0] == "a" and found[-1][0] in {"c", "d"}


def _looks_like_mcq_stem(lines: list[tuple[int, str]]) -> bool:
    """Detect an interrogative stem that is likely followed by answer choices."""
    stem = " ".join(value for _, value in lines).strip()
    if not stem:
        return False
    return bool(
        re.match(
            r"^(?:what|which|who|where|when|how|select|choose|identify|"
            r"name the|pick the)\b",
            stem,
            re.IGNORECASE,
        )
        or stem.rstrip().endswith("?")
        or re.search(r"\bwhich\s+(?:enzyme|coenzyme|of the following|one)\b", stem, re.IGNORECASE)
    )


def _looks_like_compact_option(text: str) -> bool:
    """Return true for a compact choice, not a full descriptive subquestion."""
    value = text.strip()
    if not value or len(value) > 110 or value.endswith("?"):
        return False
    return not bool(re.match(
        r"^(?:what|why|how|when|where|who|which|whose|whom|"
        r"define|explain|describe|discuss|list|state|derive|calculate|"
        r"compute|determine|write|mention|illustrate|give|identify|"
        r"compare|differentiate|enumerate|name|show|find|prove|"
        r"elaborate|briefly|is|are|do|does|can|could|should|would|will)\b",
        value,
        re.IGNORECASE,
    ))


def _is_non_question_stub(text: str) -> bool:
    value = text.strip()
    return not value or bool(MARKS_ONLY_RE.match(value)) or bool(SECTION_MARKS_RE.match(value))


def extract_marks(text: str) -> tuple[str, float | None]:
    patterns = [
        r"\[\s*(\d+(?:\.\d+)?)\s*(?:marks?|M)?\s*\]\s*$",
        r"\(\s*(\d+(?:\.\d+)?)\s*marks?\s*\)\s*$",
        r"\b(\d+(?:\.\d+)?)\s*marks?\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            cleaned = text[:match.start()].rstrip(" \t-–—:")
            try:
                marks = float(match.group(1))
            except ValueError:
                marks = None
            return cleaned, marks
    return text, None


def is_instruction_text(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 10:
        return True
    return bool(INSTRUCTION_RE.match(stripped))


def strip_leading_marks(text: str) -> str:
    """Remove a leading marks allocation such as '(2 x 10 = 20 marks)'."""
    return LEADING_MARKS_RE.sub("", text or "", count=1).strip()


def _build_question(
    question_number: int,
    sub_question: str | None,
    lines: list[tuple[int, str]],
) -> QuestionOccurrence | None:
    text = "\n".join(value for _, value in lines).strip()
    if _is_non_question_stub(text):
        return None
    text, marks = extract_marks(text)
    if _is_non_question_stub(text):
        return None
    pages = [page for page, _ in lines]
    return QuestionOccurrence(
        question_number=question_number,
        sub_question=sub_question,
        question_text=text.strip(),
        marks=marks,
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
    )


class _Collector:
    def __init__(self) -> None:
        self.results: list[QuestionOccurrence] = []
        self.number: int | None = None
        self.subpart: str | None = None
        self.lines: list[tuple[int, str]] = []
        self.current_mcq = False

    def flush(self, drop_instruction_stem: bool = False) -> None:
        lines, self.lines = self.lines, []
        if self.number is None or not lines:
            return
        question = _build_question(self.number, self.subpart, lines)
        if question is None:
            return
        if drop_instruction_stem and self.subpart is None and is_instruction_text(question.question_text):
            return
        self.results.append(question)

    def start_main(self, number: int, remainder: str, page: int) -> None:
        self.flush()
        self.number = number
        self.subpart = None
        self.current_mcq = bool(MCQ_HEADER_RE.search(remainder))
        # The remainder can itself start with a subpart, e.g. "Q2. (a) ...".
        # Route it through the same parser as a separate line rather than
        # accidentally storing "(a)" as literal question text.
        if remainder:
            self.add(page, remainder)

    def start_subpart(self, letter: str, remainder: str, page: int) -> None:
        # In a known MCQ block, a/b/c/d are answer options, not new questions.
        if self.current_mcq:
            label = f"({letter})"
            self.lines.append((page, f"{label} {remainder}".strip()))
            return
        self.flush(drop_instruction_stem=True)
        self.subpart = letter
        if remainder:
            self.lines.append((page, remainder))

    def add(self, page: int, line: str) -> None:
        if self.number is None:
            return
        if MCQ_HEADER_RE.search(line):
            self.current_mcq = True
        option_match = OPTION_RE.match(line)
        if option_match and self.current_mcq:
            label = option_match.group("paren") or option_match.group("dot") or option_match.group("close")
            rest = (option_match.group("rest") or "").strip()
            self.lines.append((page, f"{label.upper()}. {rest}".strip()))
            return
        letter, remainder = parse_subquestion(line)
        if letter is not None:
            self.start_subpart(letter, remainder, page)
            return
        self.lines.append((page, line))


GROUP_BOUNDARY = "__QUESTION_GROUP_BOUNDARY__"


def _line_items(pages: list[dict[str, Any]]) -> list[tuple[int, str]]:
    """Flatten page-aware text while preserving marks lines as group boundaries.

    Some GEHU PDFs lose the Q2/Q3 glyphs in native extraction but retain the
    repeated marks-allocation line. Keeping that as a structural token lets
    the parser recover a missing main label without guessing from text alone.
    """
    output: list[tuple[int, str]] = []
    for page in pages:
        text = page.get("text", "") or ""
        page_number = int(page.get("page_number", 1))
        for raw_line in text.splitlines():
            line = clean_line(raw_line)
            if not line:
                continue
            if SECTION_MARKS_RE.match(line):
                output.append((page_number, GROUP_BOUNDARY))
                continue
            if not is_noise_line(line):
                output.append((page_number, line))
    return output


def _choose_mode(lines: list[tuple[int, str]]) -> str:
    """Choose a dominant numbering style rather than letting one OCR typo decide."""
    q_count = 0
    numbered_count = 0
    mcq_header = any(MCQ_HEADER_RE.search(line) for _, line in lines)
    for _, line in lines:
        if Q_MAIN_RE.match(line) or OCR_Q1_RE.match(line) or INLINE_Q_SUB_RE.match(line):
            q_count += 1
        if NUMBERED_MAIN_RE.match(line):
            numbered_count += 1

    # Mixed papers can contain a numbered MCQ section and Q-style descriptive
    # sections. Parse those regions independently instead of choosing one
    # format for the entire paper.
    if mcq_header and numbered_count >= 1 and q_count >= 1:
        return "mixed"
    # A section explicitly identified as MCQ/objective commonly has Q1 as a
    # section header and 1., 2., ... as the actual individual questions.
    if mcq_header and numbered_count >= 1:
        return "numbered"
    if q_count >= 2:
        return "qstyle"
    if numbered_count >= 2 and numbered_count > q_count:
        return "numbered"
    if q_count:
        return "qstyle"
    if numbered_count:
        return "numbered"
    return "none"


def _extract_numbered(lines: list[tuple[int, str]]) -> list[QuestionOccurrence]:
    collector = _Collector()
    # Numbered questions are independent records. Options such as A./B./C./D.
    # stay embedded in the current question; they are never new records.
    for line_index, (page, line) in enumerate(lines):
        if line == GROUP_BOUNDARY:
            collector.flush()
            continue
        match = NUMBERED_MAIN_RE.match(line)
        if match:
            number = int(match.group(1))
            remainder = strip_leading_marks(match.group(2) or "")
            collector.flush()
            collector.number = number
            collector.subpart = None
            collector.current_mcq = True
            if remainder:
                collector.lines.append((page, remainder))
            continue

        if collector.number is None:
            continue

        # Preserve options including both A. and (a) OCR variants.
        option = OPTION_RE.match(line)
        if option:
            label = option.group("paren") or option.group("dot") or option.group("close")
            rest = (option.group("rest") or "").strip()
            collector.lines.append((page, f"{label.upper()}. {rest}".strip()))
            continue

        letter, remainder = parse_subquestion(line)
        if letter is not None:
            collector.lines.append((page, f"({letter}) {remainder}".strip()))
            continue

        # Ignore obvious section instructions if no actual numbered question
        # text has started yet, but keep all continuations once a question exists.
        collector.lines.append((page, line))

    collector.flush()
    return collector.results


def _extract_qstyle(lines: list[tuple[int, str]]) -> list[QuestionOccurrence]:
    """Extract Q-style papers, including papers whose Q-label glyphs are lost.

    On the affected GEHU PDF, printed Q2 and Q3 are missing from native text,
    but the marks-allocation separators and (a)/(b)/(c) pattern remain. A
    separator followed by (a) establishes the next sequential main question.
    If (a) is missing but an unlabeled stem is followed by (b), that stem is
    stored as subpart (a).
    """
    results: list[QuestionOccurrence] = []
    number: int | None = None
    subpart: str | None = None
    current_lines: list[tuple[int, str]] = []
    seen_subparts: set[str] = set()
    boundary_pending = False

    def flush() -> None:
        nonlocal current_lines
        if number is not None and current_lines:
            question = _build_question(number, subpart, current_lines)
            # Section headings such as "Short questions; attempt any four" or
            # "Long Questions: attempt any one" are context, not question records.
            if question is not None and not (
                subpart is None and is_instruction_text(question.question_text)
            ):
                results.append(question)
        current_lines = []

    def set_main(new_number: int) -> None:
        nonlocal number, subpart, seen_subparts, boundary_pending
        flush()
        number = new_number
        subpart = None
        seen_subparts = set()
        boundary_pending = False

    def start_subpart(letter: str, remainder: str, page: int) -> None:
        nonlocal number, subpart, seen_subparts, boundary_pending, current_lines
        letter = letter.lower()
        if number is None:
            return

        # A separator immediately before (a) signals a new question group.
        if letter == "a" and boundary_pending:
            flush()
            number += 1
            subpart = None
            seen_subparts = set()
            boundary_pending = False
        # Fallback if labels are lost but the previous a/b/c triplet survived.
        elif letter == "a" and subpart == "c" and {"a", "b", "c"}.issubset(seen_subparts):
            flush()
            number += 1
            subpart = None
            seen_subparts = set()

        # This GEHU paper's first (a) marker is missing. When (b) arrives with
        # an unlabelled stem buffered, save that stem as (a) before starting b.
        if letter == "b" and subpart is None and current_lines:
            stem_text = " ".join(value for _, value in current_lines).strip()
            if stem_text and not is_instruction_text(stem_text):
                question = _build_question(number, "a", current_lines)
                if question is not None:
                    results.append(question)
            current_lines = []
            seen_subparts.add("a")
        else:
            # Discard a standalone instruction/header stem before explicit (a).
            flush()

        subpart = letter
        seen_subparts.add(letter)
        boundary_pending = False
        if remainder:
            current_lines.append((page, remainder))

    for line_index, (page, line) in enumerate(lines):
        if line == GROUP_BOUNDARY:
            if number is not None and current_lines:
                flush()
            if number is not None and subpart is not None:
                boundary_pending = True
            continue

        inline = INLINE_Q_SUB_RE.match(line)
        if inline:
            explicit_number = int(inline.group(1))
            letter = inline.group(2).lower()
            remainder = strip_leading_marks((inline.group(3) or "").strip())
            if number != explicit_number:
                set_main(explicit_number)
            start_subpart(letter, remainder, page)
            continue

        main = parse_main_question(line, allow_numbered=False)
        if main is not None:
            explicit_number, remainder = main
            set_main(explicit_number)
            remainder = strip_leading_marks(remainder)
            if remainder:
                inline_part = SUBPART_RE.match(remainder)
                if inline_part:
                    start_subpart(
                        inline_part.group("paren") or inline_part.group("close") or inline_part.group("dot"),
                        (inline_part.group("rest") or "").strip(),
                        page,
                    )
                else:
                    current_lines.append((page, remainder))
            continue

        letter, remainder = parse_subquestion(line)
        if letter is not None:
            # Lowercase-dot labels are ambiguous: descriptive subparts often
            # begin with a command ("a. Define..."), while OCR'd options are
            # compact ("b. NAD+"). Keep compact choices inside interrogative
            # stems so they are not indexed as separate questions.
            has_already_seen_option = any(
                re.match(r"^[A-D]\.\s", value.strip()) for _, value in current_lines
            )
            if (
                letter.lower() in {"a", "b", "c", "d"}
                and current_lines
                and (_looks_like_mcq_stem(current_lines) or has_already_seen_option)
                and _looks_like_compact_option(remainder)
            ):
                current_lines.append((page, f"{letter.upper()}. {remainder}".strip()))
                continue
            if (
                letter.lower() in {"a", "b", "c", "d"}
                and current_lines
                and _is_short_option_sequence(lines, line_index)
            ):
                # Only the current option is appended here; the same test runs
                # on the next option line, avoiding accidental loss of wrapped text.
                current_lines.append((page, f"{letter.upper()}. {remainder}".strip()))
                continue
            start_subpart(letter, remainder, page)
            continue

        if number is not None:
            current_lines.append((page, line))

    flush()
    return results


def _extract_mixed(lines: list[tuple[int, str]]) -> list[QuestionOccurrence]:
    """Parse numbered items inside MCQ regions and Q-style items elsewhere."""
    q_collector = _Collector()
    numbered_collector = _Collector()
    in_mcq_section = False

    for page, line in lines:
        if line == GROUP_BOUNDARY:
            q_collector.flush()
            numbered_collector.flush()
            continue
        inline = INLINE_Q_SUB_RE.match(line)
        main = parse_main_question(line, allow_numbered=False)

        if main is not None:
            number, remainder = main
            remainder = strip_leading_marks(remainder)
            if MCQ_HEADER_RE.search(remainder):
                q_collector.flush(drop_instruction_stem=True)
                numbered_collector.flush()
                in_mcq_section = True
                continue
            if in_mcq_section:
                numbered_collector.flush()
                in_mcq_section = False
            q_collector.start_main(number, remainder, page)
            continue

        if inline is not None and not in_mcq_section:
            number = int(inline.group(1))
            letter = inline.group(2).lower()
            remainder = (inline.group(3) or "").strip()
            if q_collector.number != number:
                q_collector.flush()
                q_collector.number = number
                q_collector.subpart = None
                q_collector.current_mcq = False
            q_collector.start_subpart(letter, remainder, page)
            continue

        if MCQ_HEADER_RE.search(line):
            q_collector.flush(drop_instruction_stem=True)
            numbered_collector.flush()
            in_mcq_section = True
            continue

        if in_mcq_section:
            numeric = NUMBERED_MAIN_RE.match(line)
            if numeric:
                numbered_collector.flush()
                numbered_collector.number = int(numeric.group(1))
                numbered_collector.subpart = None
                numbered_collector.current_mcq = True
                remainder = strip_leading_marks(numeric.group(2) or "")
                if remainder:
                    numbered_collector.lines.append((page, remainder))
                continue

            if numbered_collector.number is None:
                # Ignore section instructions and unnumbered text before the
                # first reliable MCQ stem rather than making a fake question.
                continue

            option = OPTION_RE.match(line)
            if option:
                label = option.group("paren") or option.group("dot") or option.group("close")
                remainder = (option.group("rest") or "").strip()
                numbered_collector.lines.append((page, f"{label.upper()}. {remainder}".strip()))
            else:
                letter, remainder = parse_subquestion(line)
                if letter is not None:
                    numbered_collector.lines.append((page, f"({letter}) {remainder}".strip()))
                else:
                    numbered_collector.lines.append((page, line))
            continue

        q_collector.add(page, line)

    q_collector.flush()
    numbered_collector.flush()
    combined = q_collector.results + numbered_collector.results

    # Preserve the document's reading order even when an MCQ section and a
    # descriptive section both restart numbering at Q1/Q2 on the same page.
    def source_position(question: QuestionOccurrence) -> tuple[int, int]:
        first_line = (question.question_text.splitlines() or [""])[0].strip()
        page = question.page_start or 1
        for index, (line_page, source_line) in enumerate(lines):
            if line_page != page or len(first_line) < 6:
                continue
            if first_line in source_line or source_line.endswith(first_line):
                return page, index
        return page, len(lines)

    return sorted(combined, key=source_position)



def extract_questions(pages: list[dict[str, Any]]) -> list[QuestionOccurrence]:
    """Extract question-level records from page-aware PDF text.

    The function selects a dominant layout from evidence in the document.
    It does not require consecutive question numbers, and it does not invent
    missing questions when OCR loses numbering.
    """
    lines = _line_items(pages)
    mode = _choose_mode(lines)

    # If a paper declares an MCQ/objective section but no numbered individual
    # stems survive extraction, do not store the entire section as one fake
    # question. This is a deliberate quality guard: OCR may have lost the
    # boundaries, and inventing them would pollute historical frequency data.
    has_mcq_header = any(MCQ_HEADER_RE.search(line) for _, line in lines)
    numbered_markers = sum(1 for _, line in lines if NUMBERED_MAIN_RE.match(line))
    q_main_markers = sum(
        1 for _, line in lines if Q_MAIN_RE.match(line) or OCR_Q1_RE.match(line)
    )
    if has_mcq_header and numbered_markers == 0 and q_main_markers <= 1:
        print("    → MCQ/objective section found but individual numbered stems were not detected; skipping rather than storing a merged question.")
        return []

    if mode == "mixed":
        return _extract_mixed(lines)
    if mode == "numbered":
        return _extract_numbered(lines)
    if mode == "qstyle":
        return _extract_qstyle(lines)
    return []


def _print_questions(title: str, pages: list[dict[str, Any]]) -> None:
    print("#" * 60)
    print(title)
    print("#" * 60)
    questions = extract_questions(pages)
    print(f"Total questions found: {len(questions)}")
    for question in questions:
        label = f"Q{question.question_number}"
        if question.sub_question:
            label += f"({question.sub_question})"
        print("=" * 60)
        print(label, f"[pages {question.page_start}-{question.page_end}]", f"[marks={question.marks}]")
        print(question.question_text)


if __name__ == "__main__":
    _print_questions(
        "Q-style descriptive paper",
        [{"page_number": 1, "text": """Graphic Era Hill University
Q1
(a) Explain Bernoulli's theorem.
(b) Define viscosity.
Q2. Explain laminar flow.
Q3
(a) Derive the equation.
"""}],
    )
    _print_questions(
        "Numbered MCQ paper",
        [{"page_number": 1, "text": """Multiple Choice Questions: Attempt all questions
1. The study of microscopic structure is called:
A. Anatomy
B. Physiology
C. Histology
D. Pathology
2) The midsagittal plane divides the body into:
(a) Unequal right and left portions
(b) Equal anterior and posterior portions
"""}],
    )
