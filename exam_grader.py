import onnxruntime as ort
ort.set_default_logger_severity(3)  # show errors only, hide warnings

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from exam_reader import (
    pdf_to_images,
    process_exam_sheet,
    prepare_blank_reference,
    load_glyph_baseline,
)
from exam_solution_processor import (
    extract_solution_data,
    load_solution_keys,
    process_solution_pdf,
)
from canvas_gradebook import (
    load_roster_from_csv,
    write_grades_csv,
)
from exam_combine_reports import create_combined_pdf

DEBUG = False

# Most exams are 55 questions (50 + 5 extra credit); pass --questions
# for others (e.g. 110 or 120 for a final).
DEFAULT_QUESTION_COUNT = 55

# Standard file names, looked for in the folder you run the grader
# from (your current directory), not the folder this script lives in.
STUDENT_PDF_NAME = "StudentAnswers"
SOLUTION_PDF_NAME = "SolutionKeys"
DEFAULT_SOLUTIONS_DIRECTORY = Path("ExamSolutionJSONs")

# A glyph baseline kept next to this script is used automatically
# when neither --glyph-baseline nor --blank-sheet is given.
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_GLYPH_BASELINE = SCRIPT_DIRECTORY / "exam_glyph_baseline.json"

# Points taken off for each problem with a sheet: an instructor-entered
# student ID, an instructor-entered version, or a reason the sheet was
# set aside in BadSheets. One deduction per field, however many digits
# of the ID were missing.
DEDUCTION_PER_ISSUE = 2.5

# Sheets set aside for a later --rerun-bad, one 2-page PDF per sheet,
# plus a record of why and what was already entered.
BAD_SHEETS_DIRECTORY = Path("BadSheets")
BAD_SHEETS_RECORD = "bad_sheets.json"

# Record of every graded sheet (so a rerun can spot a duplicate ID
# without regrading everything). Lives in the reports folder.
GRADED_RECORD = "_graded_sheets.json"


def max_score_for(question_count: int) -> int:
    """
    Highest score allowed, with 5 extra-credit questions per 55:
    55 questions -> 50, 110 -> 100, 120 -> 110 (still 10 extra).
    """

    return question_count - 5 * (question_count // 55)


def final_score(
    correct_count: int,
    max_score: float,
    deductions: float,
) -> float:
    """
    Deduct first, then cap at the max (extra credit can't exceed it);
    never below 0. 55 correct with a 2.5 deduction -> 52.5 -> capped 50.
    """

    return max(0.0, min(correct_count - deductions, max_score))


def format_points(value: float) -> str:
    """47.5 -> '47.5', 50.0 -> '50'."""

    return f"{value:g}"


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def clean_value(value: Any) -> str:
    """
    Convert a reader result into a plain string.

    The reader may return either:

        "answer_01_A"

    or:

        {"selected": "answer_01_A"}
    """

    if value is None:
        return ""

    if isinstance(value, dict):
        value = value.get("selected", value)

    if isinstance(value, list):
        return ", ".join(
            clean_value(item)
            for item in value
            if clean_value(item)
        )

    return str(value).strip()


def strip_group_prefix(group_name: str, value: Any) -> str:
    """
    Convert values such as:

        exam_F          -> F
        answer_071_A    -> A
        version_C       -> C
    """

    value = clean_value(value)

    if not value:
        return ""

    prefix = f"{group_name}_"

    if value.startswith(prefix):
        return value[len(prefix):].strip()

    return value


def make_output_name(
    input_path: str | Path,
    suffix: str,
    extension: str,
    levels: int = 3,
) -> Path:
    """
    Build an output path next to the input file, named after the last
    `levels` directories that contain it, e.g. an input in

        /workspaces/ExamGrader/ASTR C1001/F26 18937/ME2/

    gives ASTR_C1001_F26_18937_ME2_<suffix><extension> in that same
    ME2 folder. Spaces become underscores, and characters that aren't
    safe in file names are dropped. If the input sits fewer than
    `levels` directories deep, whatever directories exist are used.
    """

    folder = Path(input_path).resolve().parent
    # Skip the root ("/" or "C:\\"); keep only real folder names.
    names = list(folder.parts[1:] if folder.anchor else folder.parts)

    cleaned = []

    for name in names[-levels:]:
        name = re.sub(r"\s+", "_", name.strip())
        name = re.sub(r"[^A-Za-z0-9_.-]", "", name)

        if name:
            cleaned.append(name)

    return folder / ("_".join(cleaned + [suffix]) + extension)


def find_standard_input(base_name: str) -> Path | None:
    """
    Find a standard input PDF in the current directory, accepting
    either the plain name or the folder-prefixed name:

        StudentAnswers.pdf
        ASTR_C1001_F26_18937_ME2_StudentAnswers.pdf

    A plain-named file is renamed to the prefixed name (same scheme as
    the Grades/StudentReports outputs), so every file in the folder
    carries the course/section/exam in its name. If BOTH names exist,
    this stops rather than guess which one is the right scan.

    Returns the path to use, or None if neither file exists.
    """

    plain = Path(f"{base_name}.pdf")
    prefixed = make_output_name(plain, base_name, ".pdf")

    if plain.exists() and prefixed.exists():
        raise FileExistsError(
            f"Both {plain} and {prefixed.name} are in {Path.cwd()}. "
            "Remove or rename the one you don't want graded, then "
            "run again."
        )

    if plain.exists():
        plain.rename(prefixed)
        print(f"Renamed {plain} -> {prefixed.name}")
        return prefixed

    if prefixed.exists():
        return prefixed

    return None


# Files that are never the roster even if they sit in the folder.
NON_ROSTER_SUFFIXES = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff", ".json",
    ".py", ".zip", ".docx", ".xlsx", ".tex",
}


def looks_like_text(path: Path) -> bool:
    """True if the start of the file has no NUL bytes (i.e. not binary)."""

    try:
        with path.open("rb") as file:
            return b"\0" not in file.read(4096)
    except OSError:
        return False


def find_default_roster() -> Path:
    """
    Find the roster one directory above the current one, e.g. for

        .../ASTR C1001/F26 18937/ME2   (current directory)

    look in ".../ASTR C1001/F26 18937". The roster is the text file
    there (plain text with or without an extension, or a Canvas CSV).
    If there are several text files, the one whose name shares a
    number with that folder's name ("18937" in "F26 18937") wins.
    """

    folder = Path.cwd().parent

    candidates = [
        path
        for path in sorted(folder.iterdir())
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() not in NON_ROSTER_SUFFIXES
        and path.stat().st_size > 0
        and looks_like_text(path)
    ]

    if len(candidates) == 1:
        return candidates[0]

    folder_numbers = set(re.findall(r"\d+", folder.name))

    matching = [
        path
        for path in candidates
        if folder_numbers & set(re.findall(r"\d+", path.name))
    ]

    if len(matching) == 1:
        return matching[0]

    if not candidates:
        raise FileNotFoundError(
            f"No roster file found in {folder}. Put the roster there "
            "or pass --roster."
        )

    raise FileNotFoundError(
        f"Couldn't tell which file in {folder} is the roster: "
        + ", ".join(path.name for path in (matching or candidates))
        + ". Pass --roster to choose."
    )


def normalize_student_id(value: Any) -> str:
    """Keep digits only and remove leading zeroes."""

    digits = re.sub(r"\D", "", clean_value(value))

    if not digits:
        return ""

    return digits.lstrip("0") or "0"


# ---------------------------------------------------------------------
# Roster handling
# ---------------------------------------------------------------------

def load_student_roster(
    roster_path: str | Path,
) -> dict[str, str]:
    """
    Load a roster containing lines such as:

        2061993    Afurkeeva, Anna
        2236979    Bitcon-Tupa, Gavin P

    The function accepts tabs or spaces between the ID and name.
    """

    roster_path = Path(roster_path)

    text = None

    for encoding in (
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin-1",
    ):
        try:
            text = roster_path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        raise RuntimeError(
            f"Could not decode roster file: {roster_path}"
        )

    roster: dict[str, str] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if line.casefold().startswith("wait list"):
            break

        # Seven-digit ID followed by whitespace and the student's name.
        match = re.match(
            r"^(\d{7})\s+(.+?)\s*$",
            line,
        )

        if not match:
            continue

        student_id = normalize_student_id(match.group(1))
        student_name = match.group(2).strip()

        roster[student_id] = student_name

    print(f"Loaded {len(roster)} student(s) from roster.")
    return roster


# ---------------------------------------------------------------------
# Student ID extraction
# ---------------------------------------------------------------------

def extract_student_id(
    selections: dict[str, Any],
) -> str:
    """
    Extract a student ID from common reader output formats.

    Supported examples:

        student_id: "2061993"

        student_id: {
            "selected": "student_id_2061993"
        }

        student_id_1: {
            "selected": "student_id_1_2"
        }

        student_id_2: {
            "selected": "student_id_2_0"
        }
    """

    # First check common complete-ID group names.
    complete_id_groups = (
        "student_id",
        "studentid",
        "student_number",
        "student_num",
    )

    for group_name in complete_id_groups:
        if group_name not in selections:
            continue

        raw_value = clean_value(selections[group_name])

        # If the selected value is something like student_id_2061993,
        # remove the group prefix first.
        value = strip_group_prefix(
            group_name,
            raw_value,
        )

        digits = re.sub(r"\D", "", value)

        if len(digits) >= 7:
            return digits[:7]

        # Also accept a direct seven-digit value.
        if len(re.sub(r"\D", "", raw_value)) >= 7:
            return re.sub(r"\D", "", raw_value)[:7]

    # Next check separate digit groups, such as student_id_1.
    digit_groups: list[tuple[int, str]] = []

    for key, value in selections.items():
        key_text = str(key).casefold()

        match = re.match(
            r"^student[_ ]*id[_ ]*(\d+)$",
            key_text,
        )

        if not match:
            continue

        position = int(match.group(1))
        selected = clean_value(value)

        # Typical selected values:
        # student_id_1_2
        # 2
        digit_match = re.search(r"(\d)$", selected)

        # A position with no digit read (blank/ambiguous bubble) still
        # holds its place with "?" rather than being silently dropped
        # — dropping it would shift every later digit left, producing
        # a plausible-looking but WRONG shorter ID (e.g. "2 _ 0 8 0 2
        # 9" collapsing to "208029") instead of visibly flagging that
        # a digit is missing.
        digit_groups.append(
            (position, digit_match.group(1) if digit_match else "?")
        )

    if digit_groups:
        digit_groups.sort(key=lambda item: item[0])
        return "".join(
            digit
            for _, digit in digit_groups
        )

    return ""


# ---------------------------------------------------------------------
# Report file I/O
# ---------------------------------------------------------------------

def save_student_report(
    output_directory: str | Path,
    page_number: int,
    report_text: str,
) -> Path:
    output_directory = Path(output_directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_directory / (
        f"student_page_{page_number:03d}.txt"
    )

    output_path.write_text(
        report_text,
        encoding="utf-8",
    )

    return output_path


def save_error_report(
    output_directory: str | Path,
    page_number: int,
    error: Exception,
) -> Path:
    output_directory = Path(output_directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_directory / (
        f"student_page_{page_number:03d}_error.txt"
    )

    output_path.write_text(
        f"Error processing page {page_number}\n"
        f"{type(error).__name__}: {error}\n",
        encoding="utf-8",
    )

    return output_path


# ---------------------------------------------------------------------
# Grading arithmetic
# ---------------------------------------------------------------------

def grade_student_answers(
    student_data: dict[str, Any],
    solution: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare student answers to the matching solution key.

    Blank student answers are recorded as incorrect, unless the
    solution key itself has no marked answer for that question, in
    which case every student automatically gets the question correct.
    """

    student_answers = student_data.get("answers", {})
    correct_answers = solution.get("answers", {})

    question_count = int(
        solution.get(
            "question_count",
            student_data.get("question_count", 0),
        )
    )

    question_results: dict[str, dict[str, Any]] = {}
    correct_count = 0
    answered_count = 0

    for question_number in range(1, question_count + 1):
        question_key = str(question_number)

        student_answer = student_answers.get(question_key)
        correct_answer = correct_answers.get(question_key)

        if student_answer is not None:
            student_answer = str(student_answer).strip().upper()

        if correct_answer is not None:
            correct_answer = str(correct_answer).strip().upper()

        if correct_answer is None:
            # No answer marked on the solution key for this question —
            # treat it as a free point for every student.
            is_correct = True
        else:
            is_correct = (
                student_answer is not None
                and student_answer == correct_answer
            )

        if student_answer is not None:
            answered_count += 1

        if is_correct:
            correct_count += 1

        question_results[question_key] = {
            "student_answer": student_answer,
            "correct_answer": correct_answer,
            "correct": is_correct,
        }

    percentage = (
        correct_count / question_count * 100
        if question_count
        else 0.0
    )

    return {
        "question_results": question_results,
        "correct_count": correct_count,
        "answered_count": answered_count,
        "question_count": question_count,
        "percentage": percentage,
        "solution_filename": solution.get("_filename", ""),
    }


# ---------------------------------------------------------------------
# Student data extraction (exam fields: exam + version)
# ---------------------------------------------------------------------

def extract_student_data(
    selections: dict[str, Any],
    question_count: int = DEFAULT_QUESTION_COUNT,
) -> dict[str, Any]:
    """
    Extract student metadata and answers from a combined (front+back)
    exam selections dict -- see exam_reader.process_exam_sheet().
    """

    extracted = extract_solution_data(
        selections=selections,
        question_count=question_count,
    )

    student_id = extract_student_id(selections)

    return {
        "student_id": student_id,
        "exam": extracted["exam"],
        "version": extracted["version"],
        "question_count": extracted["question_count"],
        "answers": extracted["answers"],
        "raw_selections": selections,
        "missing_metadata": extracted["missing_metadata"],
        "detection_diagnostics": selections.get(
            "_detection_diagnostics",
            [],
        ),
    }


# ---------------------------------------------------------------------
# Text report generation
# ---------------------------------------------------------------------

def format_question_table(
    rows: list[tuple[int, str, str, str]],
    headers: tuple[str, ...],
    widths: tuple[int, ...],
    columns: int = 2,
) -> list[str]:
    """
    Lay question rows out in side-by-side columns (filled top to
    bottom, then left to right), so 55 questions fit on one page.
    """

    def cell(values):
        return "  ".join(
            f"{str(value):>{width}}" if index == 0 else f"{str(value):<{width}}"
            for index, (value, width) in enumerate(zip(values, widths))
        )

    blank = " " * len(cell(headers))
    gap = "    |    "
    per_column = -(-len(rows) // columns)  # ceiling division
    chunks = [rows[i:i + per_column] for i in range(0, len(rows), per_column)]

    lines = [gap.join(cell(headers) for _ in chunks)]
    lines.append(gap.join("-" * len(blank) for _ in chunks))

    for index in range(per_column):
        parts = [
            cell(chunk[index]) if index < len(chunk) else blank
            for chunk in chunks
        ]
        lines.append(gap.join(parts).rstrip())

    return lines


def format_student_report(
    sheet_number: int,
    student_data: dict[str, Any],
    student_name: str,
    grading_result: dict[str, Any] | None = None,
) -> str:
    """
    One student's report, laid out to fit a 55-question exam on a
    single page of the combined PDF: summary, instructor notes, then
    the question results in two side-by-side columns.
    """

    rule = "-" * 80
    lines: list[str] = []

    lines.append(f"EXAM REPORT{'':<55}Sheet {sheet_number}")
    lines.append("=" * 80)
    lines.append(
        f"Student:      {student_name or '[UNKNOWN STUDENT]'}"
        f"  ({student_data['student_id'] or 'ID NOT FOUND'})"
    )
    lines.append(
        f"Exam:         {student_data['exam'] or '[blank]'}"
        f"        Version: {student_data['version'] or '[NOT FOUND]'}"
    )
    lines.append("")

    # Everything the instructor should know about this sheet.
    notes: list[str] = []

    if grading_result is not None:
        max_score = grading_result["max_score"]
        extra = grading_result["question_count"] - max_score

        if extra > 0:
            notes.append(
                f"Max score is {format_points(max_score)} "
                f"({grading_result['question_count']} questions, "
                f"{format_points(extra)} extra credit)."
            )

    for issue in student_data.get("issues", []):
        if issue.get("waived"):
            notes.append(f"{issue['reason']} (deduction waived).")
        else:
            notes.append(
                f"{issue['reason']} (-{format_points(issue['deduction'])})."
            )

    notes.extend(student_data.get("warnings", []))

    version_missing = "version" in student_data["missing_metadata"]

    if grading_result is not None and not version_missing:
        lines.append("GRADING SUMMARY")
        lines.append(rule)
        lines.append(f"Raw score:    {grading_result['correct_count']}")
        lines.append(
            f"Deductions:   -{format_points(grading_result['deductions'])}"
        )
        lines.append(
            f"Max cap:      -{format_points(grading_result['cap_deduction'])}"
        )
        lines.append(
            f"Final score:  {format_points(grading_result['final_score'])}"
        )
        lines.append("")
        lines.append(
            f"Answered:     {grading_result['answered_count']}/"
            f"{grading_result['question_count']}"
            + (
                f"        Solution key: {grading_result['solution_filename']}"
                if grading_result.get("solution_filename")
                else ""
            )
        )
        lines.append("")

    elif not version_missing:
        notes.insert(0, "No matching solution key was found; not graded.")

    lines.append("INSTRUCTOR NOTES")
    lines.append(rule)
    lines.extend(f"- {note}" for note in notes)
    lines.append("")
    lines.append("")

    if grading_result is not None and not version_missing:
        lines.append("QUESTION RESULTS")
        lines.append(rule)

        rows = []

        for question_number, result in grading_result["question_results"].items():
            rows.append((
                int(question_number),
                result["student_answer"] or "-",
                result["correct_answer"] or "(any)",
                "correct" if result["correct"] else "WRONG",
            ))

        lines.extend(format_question_table(
            rows,
            headers=("Q", "Answer", "Key", "Result"),
            widths=(3, 6, 5, 7),
        ))
    else:
        lines.append("ANSWERS READ")
        lines.append(rule)

        rows = [
            (int(question), answer or "-")
            for question, answer in sorted(
                student_data["answers"].items(), key=lambda item: int(item[0])
            )
        ]

        lines.extend(format_question_table(
            rows,
            headers=("Q", "Answer"),
            widths=(3, 6),
            columns=3,
        ))

    lines.append("")

    if DEBUG:
        lines.append("CORNER / TIMING MARK DETECTION")
        lines.append(rule)

        detection_diagnostics = student_data.get("detection_diagnostics", [])

        if detection_diagnostics:
            for note in detection_diagnostics:
                lines.append(f"- {note}")
        else:
            lines.append("No corners or timing marks needed interpolation.")

        lines.append("")
        lines.append("RAW READER OUTPUT")
        lines.append(rule)
        lines.append(
            json.dumps(
                {
                    key: value
                    for key, value in student_data["raw_selections"].items()
                    if not key.startswith("_")
                },
                indent=2,
                default=str,
            )
        )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------
# Solution processor integration
# ---------------------------------------------------------------------

def prepare_solution_directory(
    args,
    blank_gray_front=None,
    blank_gray_back=None,
    glyph_baseline=None,
) -> dict:
    """
    Find or build the solution JSON files, in this order:

      1. --solution_pdf given explicitly: (re)generate keys from it.
      2. The solutions directory already holds key files: use them.
      3. Otherwise generate them from SolutionKeys.pdf in the current
         directory.
    """

    solutions_directory = Path(args.solutions)

    solution_pdf = None

    if args.solution_pdf is not None:
        solution_pdf = Path(args.solution_pdf)

        if not solution_pdf.exists():
            raise FileNotFoundError(
                f"Solution PDF does not exist: {solution_pdf}"
            )

    elif solutions_directory.is_dir() and any(
        solutions_directory.glob("*.json")
    ):
        print(
            f"Using existing solution keys in {solutions_directory} "
            f"(pass --solution_pdf to regenerate them)."
        )

    elif args.standard_solution_pdf is not None:
        solution_pdf = args.standard_solution_pdf

    else:
        raise FileNotFoundError(
            f"No solution keys found: {solutions_directory} has no "
            f"key files and there is no {SOLUTION_PDF_NAME}.pdf (or "
            f"<folders>_{SOLUTION_PDF_NAME}.pdf) in {Path.cwd()}."
        )

    if solution_pdf is not None:

        print(
            "Generating solution JSON files using "
            "exam_solution_processor.py..."
        )

        process_solution_pdf(
            input_pdf=solution_pdf,
            output_directory=solutions_directory,
            question_count=args.questions,
            threshold=args.threshold,
            gap=args.gap,
            blank_gray_front=blank_gray_front,
            blank_gray_back=blank_gray_back,
            glyph_baseline=glyph_baseline,
        )

    solution_keys = load_solution_keys(solutions_directory)

    print(
        f"Loaded {len(solution_keys)} solution key(s) "
        f"from {solutions_directory}"
    )

    return solution_keys


def make_solution_lookup_key(version: Any) -> str:
    """
    Create the same lookup key used by exam_solution_processor's
    load_solution_keys(): just the normalized version, since a
    grading run is assumed to be one exam instance (no quiz-number
    equivalent to disambiguate further -- see
    exam_solution_processor.load_solution_keys()).
    """

    return str(version).strip().casefold()



# ---------------------------------------------------------------------
# Instructor prompts (missing student ID / version)
# ---------------------------------------------------------------------

def ask(prompt: str) -> str | None:
    """Read one answer; None if there's no one to ask (end of input)."""

    try:
        return input(prompt).strip()
    except EOFError:
        print()
        return None


def ask_apply_deduction(reason: str) -> dict[str, Any]:
    """
    After a problem is corrected, ask whether the student should lose
    points for it (Enter / y = yes, n = waive -- e.g. not the
    student's fault). Returns the issue entry for the report.
    """

    answer = ask(
        f"  Apply the {format_points(DEDUCTION_PER_ISSUE)}-point deduction "
        f"for: {reason}? [Y/n]: "
    )

    if answer is not None and answer.casefold() in ("n", "no"):
        return {"reason": reason, "deduction": 0.0, "waived": True}

    return {"reason": reason, "deduction": DEDUCTION_PER_ISSUE}


def ask_student_id(
    context: str,
    roster: dict[str, str],
    taken_ids: set[str],
    default: str | None = None,
) -> str | None:
    """
    Ask for a student's 7-digit ID. Must be on the roster and not
    already used by another graded sheet. Returns the ID, or None to
    skip (s, or no one to ask). With `default`, Enter keeps it.
    """

    print(context)
    hint = f" (Enter keeps {default})" if default else ""

    while True:
        answer = ask(f"  Type the 7-digit student ID{hint}, or s to skip: ")

        if answer is None or answer.casefold() == "s":
            return None

        if not answer and default:
            answer = default

        digits = re.sub(r"\D", "", answer)
        normalized = normalize_student_id(digits)

        if len(digits) != 7:
            print("  A student ID has 7 digits.")
            continue

        if normalized not in roster:
            print(f"  {digits} is not on the roster.")
            continue

        if normalized in taken_ids:
            print(f"  {digits} ({roster[normalized]}) is already graded on another sheet.")
            continue

        print(f"  -> {digits}  {roster[normalized]}")
        return digits


def version_matches(
    student_data: dict[str, Any],
    solution_keys: dict[str, Any],
) -> list[tuple[str, int, int]]:
    """
    How many of the student's answers agree with each version's key,
    best first: [(version, agreeing, answered), ...].
    """

    answers = student_data.get("answers", {})
    answered = sum(1 for value in answers.values() if value is not None)
    results = []

    for key, solution in solution_keys.items():
        correct = solution.get("answers", {})
        agree = sum(
            1 for question, value in answers.items()
            if value is not None and correct.get(question) == value
        )
        results.append((str(solution.get("version", key)).upper(), agree, answered))

    return sorted(results, key=lambda item: -item[1])


def ask_version(
    context: str,
    solution_keys: dict[str, Any],
    default: str | None = None,
    student_data: dict[str, Any] | None = None,
) -> str | None:
    """
    Ask for the version letter (must have an answer key); None to skip.

    With `student_data`, first shows how well the student's answers
    match each key -- the right version stands out -- and, if no other
    default was given, Enter accepts the best match.
    """

    letters = sorted(key.upper() for key in solution_keys)
    print(context)

    if student_data is not None:
        matches = version_matches(student_data, solution_keys)

        if matches:
            print(
                "  Answers agree with each key: "
                + ", ".join(f"{v} {agree}/{answered}" for v, agree, answered in matches)
            )

            if default is None and matches[0][1] > 0:
                default = matches[0][0]
    hint = f" (Enter keeps {default})" if default else ""

    while True:
        answer = ask(
            f"  Type the version ({', '.join(letters)}){hint}, or s to skip: "
        )

        if answer is None or answer.casefold() == "s":
            return None

        if not answer and default:
            answer = default

        if make_solution_lookup_key(answer) in solution_keys:
            return answer.strip().upper()

        print(f"  No answer key for version {answer!r}.")


def resolve_metadata(
    sheet_number: int,
    student_data: dict[str, Any],
    roster: dict[str, str],
    solution_keys: dict[str, Any],
    taken_ids: set[str],
    stored: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Settle a sheet's student ID and version, prompting where needed.

    First reading of a sheet (`stored` is None): after each value you
    type in, you're asked whether to apply DEDUCTION_PER_ISSUE. A
    skipped prompt, or a clearly bubbled ID that isn't on the roster,
    sends the sheet to BadSheets with a pending DEDUCTION_PER_ISSUE per
    problem -- you decide on those when the sheet is fixed on a rerun.

    Rerun of a bad sheet (`stored` = its BadSheets record): asks again
    only for what's still needed, with no new deductions; what was
    entered before is reused.

    Returns {"student_id", "version", "issues" (new deductions),
    "bad_reasons", "needs"} -- the sheet is good when "needs" is empty.
    """

    rerun_needs = set(stored.get("needs", [])) if stored else set()
    issues: list[dict[str, Any]] = []
    bad_reasons: list[str] = []
    needs: list[str] = []
    label = f"Sheet {sheet_number}"

    # ---- student ID -----------------------------------------------
    read_id = student_data["student_id"] or ""
    shown_id = read_id.replace("?", "_") or "blank"

    if stored and "student_id" not in rerun_needs:
        student_id = stored.get("student_id")
    elif stored:
        candidate = stored.get("student_id") or read_id
        default = (
            candidate
            if candidate and "?" not in candidate
            and normalize_student_id(candidate) in roster
            and normalize_student_id(candidate) not in taken_ids
            else None
        )
        student_id = ask_student_id(
            f"{label}: student ID was set aside "
            f"({'; '.join(stored.get('reasons', [])) or 'see BadSheets'}).",
            roster, taken_ids, default,
        )

        if student_id is None:
            needs.append("student_id")
            bad_reasons.append("Student ID still needed")
    elif not read_id or "?" in read_id or len(read_id) != 7:
        student_id = ask_student_id(
            f"{label}: student ID read as {shown_id}.",
            roster, taken_ids,
        )

        if student_id is None:
            needs.append("student_id")
            bad_reasons.append(f"Student ID incomplete (read as {shown_id})")
            issues.append({
                "reason": f"Student ID incomplete, sheet set aside (read as {shown_id})",
                "deduction": DEDUCTION_PER_ISSUE,
                "pending": True,
            })
        else:
            issues.append(ask_apply_deduction(
                f"Student ID entered by instructor (read as {shown_id})"
            ))
    elif normalize_student_id(read_id) not in roster:
        print(f"{label}: student ID {read_id} is not on the roster -> BadSheets.")
        student_id = read_id
        needs.append("student_id")
        bad_reasons.append(f"Student ID {read_id} not on roster")
        issues.append({
            "reason": f"Student ID {read_id} not on roster, sheet set aside",
            "deduction": DEDUCTION_PER_ISSUE,
            "pending": True,
        })
    else:
        student_id = read_id

    # ---- version ----------------------------------------------------
    read_version = student_data["version"]
    version_ok = (
        read_version is not None
        and make_solution_lookup_key(read_version) in solution_keys
    )
    shown_version = read_version or "blank"

    if stored and "version" not in rerun_needs:
        version = stored.get("version")
    elif stored:
        version = ask_version(
            f"{label}: version was set aside (read as {shown_version}).",
            solution_keys,
            default=read_version.upper() if version_ok else None,
            student_data=student_data,
        )

        if version is None:
            needs.append("version")
            bad_reasons.append("Version still needed")
    elif version_ok:
        version = read_version
    else:
        problem = (
            "left blank or unreadable"
            if read_version is None
            else f"read as {read_version}, which has no answer key"
        )

        version = None

        if not needs:
            version = ask_version(
                f"{label}: version {problem}.",
                solution_keys,
                student_data=student_data,
            )

        if version is None:
            needs.append("version")
            bad_reasons.append(f"Version {problem}")
            issues.append({
                "reason": f"Version {problem}, sheet set aside",
                "deduction": DEDUCTION_PER_ISSUE,
                "pending": True,
            })
        else:
            issues.append(ask_apply_deduction(
                f"Version entered by instructor ({problem})"
            ))

    return {
        "student_id": student_id,
        "version": version,
        "issues": issues,
        "bad_reasons": bad_reasons,
        "needs": needs,
    }


# ---------------------------------------------------------------------
# BadSheets folder
# ---------------------------------------------------------------------

def copy_sheet_pages(
    source_pdf: Path,
    first_page_index: int,
    output_pdf: Path,
) -> None:
    """Copy one sheet's two pages (front, back) into their own PDF."""

    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    try:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz

        source = fitz.open(str(source_pdf))
        sheet = fitz.open()
        sheet.insert_pdf(
            source,
            from_page=first_page_index,
            to_page=first_page_index + 1,
        )
        sheet.save(str(output_pdf))
        sheet.close()
        source.close()
        return
    except (ImportError, AttributeError):
        pass

    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(source_pdf))
    writer = PdfWriter()
    writer.add_page(reader.pages[first_page_index])
    writer.add_page(reader.pages[first_page_index + 1])

    with output_pdf.open("wb") as file:
        writer.write(file)


def count_pdf_pages(pdf_path: Path) -> int:
    try:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz

        with fitz.open(str(pdf_path)) as document:
            return document.page_count
    except (ImportError, AttributeError):
        from pypdf import PdfReader

        return len(PdfReader(str(pdf_path)).pages)


def append_pdf(main_pdf: Path, extra_pdf: Path) -> None:
    """
    Append extra_pdf's pages to the end of main_pdf. Written to a
    temporary file and checked before it replaces main_pdf, so a
    failure can't damage the original scan.
    """

    before = count_pdf_pages(main_pdf)
    extra = count_pdf_pages(extra_pdf)
    temporary = main_pdf.with_name(main_pdf.stem + ".merging.pdf")

    try:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz

        document = fitz.open(str(main_pdf))
        addition = fitz.open(str(extra_pdf))
        document.insert_pdf(addition)
        document.save(str(temporary))
        addition.close()
        document.close()
    except (ImportError, AttributeError):
        from pypdf import PdfReader, PdfWriter

        writer = PdfWriter()

        for source in (main_pdf, extra_pdf):
            for page in PdfReader(str(source)).pages:
                writer.add_page(page)

        with temporary.open("wb") as file:
            writer.write(file)

    if count_pdf_pages(temporary) != before + extra:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Merging {extra_pdf.name} into {main_pdf.name} didn't come out "
            "right; nothing was changed."
        )

    temporary.replace(main_pdf)


def load_bad_sheets_record() -> dict[str, Any]:
    path = BAD_SHEETS_DIRECTORY / BAD_SHEETS_RECORD

    if not path.exists():
        raise FileNotFoundError(
            f"No {path} here -- nothing to rerun. Run --rerun-bad from "
            "the exam folder that has the BadSheets folder."
        )

    return json.loads(path.read_text(encoding="utf-8"))


def save_bad_sheets_record(record: dict[str, Any]) -> None:
    BAD_SHEETS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    (BAD_SHEETS_DIRECTORY / BAD_SHEETS_RECORD).write_text(
        json.dumps(record, indent=2) + "\n",
        encoding="utf-8",
    )

    # Plain-text summary for a quick look.
    lines = ["BAD SHEETS -- rerun with --rerun-bad once these are sorted out", ""]

    for entry in record["sheets"]:
        lines.append(f"{entry['file']}  (original sheet {entry['sheet']})")
        lines.append(f"  Student ID: {entry.get('student_id') or '?'}"
                     f"   Version: {entry.get('version') or '?'}")

        for reason in entry["reasons"]:
            lines.append(f"  - {reason}")

        for issue in entry["issues"]:
            status = (
                "pending -- you'll be asked on rerun" if issue.get("pending")
                else "waived" if issue.get("waived")
                else f"-{format_points(issue['deduction'])}"
            )
            lines.append(f"  deduction: {issue['reason']} ({status})")

        lines.append("")

    (BAD_SHEETS_DIRECTORY / "BadSheets_summary.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def load_graded_record(output_directory: Path) -> dict[str, Any]:
    path = Path(output_directory) / GRADED_RECORD

    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    return {}


def save_graded_record(output_directory: Path, graded: dict[str, Any]) -> None:
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    (Path(output_directory) / GRADED_RECORD).write_text(
        json.dumps(graded, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


RUN_SETTINGS = "_run_settings.json"


def save_run_settings(output_directory: Path, settings: dict[str, Any]) -> None:
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    (Path(output_directory) / RUN_SETTINGS).write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )


def load_run_settings(output_directory: Path) -> dict[str, Any] | None:
    path = Path(output_directory) / RUN_SETTINGS

    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    return None


def grade_sheet(
    student_data: dict[str, Any],
    solution_keys: dict[str, Any],
    max_score: float,
) -> dict[str, Any]:
    """Grade against the version's key; add extra-credit cap and deductions."""

    solution = solution_keys[make_solution_lookup_key(student_data["version"])]

    warnings = student_data.setdefault("warnings", [])
    key_exam = solution.get("exam")
    student_exam = student_data["exam"]

    if student_exam is None:
        warnings.append("Exam bubble left blank (no deduction).")
    elif (
        key_exam is not None
        and str(student_exam).strip().casefold()
        != str(key_exam).strip().casefold()
    ):
        warnings.append(
            f"Exam bubbled as {student_exam}, but the answer key is "
            f"for exam {key_exam} (no deduction)."
        )

    result = grade_student_answers(student_data=student_data, solution=solution)
    deductions = sum(issue["deduction"] for issue in student_data["issues"])

    result["max_score"] = max_score
    result["deductions"] = deductions
    result["final_score"] = final_score(result["correct_count"], max_score, deductions)
    # Points removed by the extra-credit cap (shown as "Max cap").
    result["cap_deduction"] = max(
        0.0, (result["correct_count"] - deductions) - max_score
    )

    return result


def apply_metadata(
    student_data: dict[str, Any],
    settled: dict[str, Any],
    carried_issues: list[dict[str, Any]] | None = None,
) -> None:
    """Put settled ID/version and all deductions into student_data."""

    student_data["student_id"] = settled["student_id"] or student_data["student_id"]
    student_data["version"] = settled["version"] or student_data["version"]
    student_data["missing_metadata"] = [
        field for field in student_data["missing_metadata"]
        if not (field == "version" and settled["version"])
    ]
    student_data["issues"] = list(carried_issues or []) + settled["issues"]


def grades_csv_path(args) -> Path:
    return args.grades_out or make_output_name(
        Path.cwd() / f"{STUDENT_PDF_NAME}.pdf", "Grades", ".csv"
    )


def combined_reports_path(args) -> Path:
    return args.combined_reports_out or make_output_name(
        Path.cwd() / f"{STUDENT_PDF_NAME}.pdf", "StudentReports", ".pdf"
    )


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract exam student data and grade it from double-"
            "sided scanned answer sheets."
        )
    )

    parser.add_argument(
        "input_pdf",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Student answer-sheet PDF, double-sided scanned (front, "
            "back, front, back, ... -- an even number of pages). "
            f"Default: {STUDENT_PDF_NAME}.pdf or "
            f"<folders>_{STUDENT_PDF_NAME}.pdf in the current "
            "directory (a plain-named file is renamed to the "
            "folder-prefixed name)."
        ),
    )

    parser.add_argument(
        "--solution_pdf",
        type=Path,
        default=None,
        help=(
            "Solution PDF (also double-sided). Given explicitly, keys "
            "are always regenerated from it. Otherwise existing keys "
            "in --solutions are used, or, if there are none, keys are "
            f"generated from {SOLUTION_PDF_NAME}.pdf (or "
            f"<folders>_{SOLUTION_PDF_NAME}.pdf) in the current "
            "directory."
        ),
    )

    parser.add_argument(
        "--solutions",
        type=Path,
        default=DEFAULT_SOLUTIONS_DIRECTORY,
        help=(
            "Directory containing solution JSON files. Also where "
            "newly generated ones (from --solution_pdf) are saved. "
            "Default: ./ExamSolutionJSONs"
        ),
    )

    parser.add_argument(
        "--roster",
        type=Path,
        default=None,
        help=(
            "Student roster: a Canvas CSV export or a plain text file. "
            "Default: the roster file one directory above the current "
            "one."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./StudentReports"),
        help="Directory for extracted student text files.",
    )

    parser.add_argument(
        "--questions",
        type=int,
        default=DEFAULT_QUESTION_COUNT,
        help=(
            "Number of questions on the exam (1-120). "
            f"Default: {DEFAULT_QUESTION_COUNT}"
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help=(
            "Bubble detection threshold. Default: 0.10 (tuned for "
            "the 2026-10 sheet with --glyph-baseline; blank bubbles "
            "score up to ~0.04, light pencil fills 0.10+)."
        ),
    )

    parser.add_argument(
        "--gap",
        type=float,
        default=0.05,
        help="Minimum difference between bubble choices. Default: 0.05",
    )

    parser.add_argument(
        "--blank-sheet",
        dest="blank_sheet",
        type=Path,
        default=None,
        help=(
            "Optional blank reference exam PDF (its first two pages "
            "are used as the front/back reference). When given, each "
            "bubble's printed letter/number is cancelled out by "
            "comparison against the same spot on this blank sheet."
        ),
    )

    parser.add_argument(
        "--glyph-baseline",
        dest="glyph_baseline",
        type=Path,
        default=None,
        help=(
            "Glyph baseline JSON. Default: exam_glyph_baseline.json "
            "next to this script, if present. Like "
            "--blank-sheet, cancels out each bubble's printed letter/"
            "number, but statistically from real scans instead of a "
            "reference sheet. Ignored if --blank-sheet is also given."
        ),
    )

    parser.add_argument(
        "--grades-out",
        dest="grades_out",
        type=Path,
        default=None,
        help=(
            "Only used when --roster is a Canvas CSV export: path "
            "for a copy of that CSV with each matched student's "
            "score filled in. Default: <dir>_<dir>_<dir>_Grades.csv "
            "next to the student answer-sheet PDF, named after the "
            "three folders that contain it."
        ),
    )

    parser.add_argument(
        "--grades-column",
        dest="grades_column",
        type=str,
        default="Exam",
        help=(
            "Name of the column in the Canvas CSV to write scores "
            "into. Default: Exam"
        ),
    )

    parser.add_argument(
        "--no-zero-ungraded",
        dest="zero_fill_ungraded",
        action="store_false",
        help=(
            "By default, a student with no gradable exam (never "
            "scanned, or scanned but unreadable/unmatched) gets a 0 "
            "in the grades CSV. Pass this to leave that cell exactly "
            "as it already was in the source CSV instead (typically "
            "blank)."
        ),
    )

    parser.add_argument(
        "--combined-reports-out",
        dest="combined_reports_out",
        type=Path,
        default=None,
        help=(
            "Path for the single combined PDF of every student "
            "report. Default: <dir>_<dir>_<dir>_StudentReports.pdf "
            "next to the student answer-sheet PDF, named after the "
            "three folders that contain it."
        ),
    )

    parser.add_argument(
        "--max-score",
        dest="max_score",
        type=float,
        default=None,
        help=(
            "Highest score allowed (extra credit can't exceed it). "
            "Default: 5 extra-credit questions per 55 -- 55 questions "
            "-> 50, 110 -> 100, 120 -> 110."
        ),
    )

    parser.add_argument(
        "--rerun-bad",
        dest="rerun_bad",
        action="store_true",
        help=(
            "Regrade only the sheets in BadSheets: asks again for "
            "anything still missing (no new deductions; earlier ones "
            "carry over), adds their reports to StudentReports and "
            "the combined PDF, and updates only their rows in the "
            "Grades CSV."
        ),
    )

    parser.add_argument(
        "--add-late",
        dest="add_late",
        type=Path,
        default=None,
        metavar="PDF",
        help=(
            "Grade a late student sheet (or several, front/back) from "
            "this PDF and add it to the existing reports, combined PDF "
            "and Grades CSV without regrading anyone else."
        ),
    )

    parser.add_argument(
        "--no-combined-reports",
        dest="combine_reports",
        action="store_false",
        help="Skip building the combined student-reports PDF.",
    )

    return parser.parse_args()


def load_scoring_reference(args):
    """Blank reference sheet or glyph baseline, as chosen on the command line."""

    blank_gray_front = None
    blank_gray_back = None
    glyph_baseline = None

    # Use the baseline kept next to this script unless told otherwise.
    if (
        args.blank_sheet is None
        and args.glyph_baseline is None
        and DEFAULT_GLYPH_BASELINE.exists()
    ):
        args.glyph_baseline = DEFAULT_GLYPH_BASELINE

    if args.blank_sheet is not None:
        if not args.blank_sheet.exists():
            raise FileNotFoundError(
                f"Blank reference sheet does not exist: "
                f"{args.blank_sheet}"
            )

        print(f"Preparing blank reference sheet: {args.blank_sheet}")

        blank_pages = pdf_to_images(args.blank_sheet)

        if len(blank_pages) < 2:
            raise ValueError(
                f"Blank reference sheet needs at least 2 pages "
                f"(front, back): {args.blank_sheet}"
            )

        blank_gray_front = prepare_blank_reference(blank_pages[0])
        blank_gray_back = prepare_blank_reference(blank_pages[1])

    elif args.glyph_baseline is not None:
        if not args.glyph_baseline.exists():
            raise FileNotFoundError(
                f"Glyph baseline file does not exist: "
                f"{args.glyph_baseline}"
            )

        print(f"Loading glyph baseline: {args.glyph_baseline}")

        glyph_baseline = load_glyph_baseline(args.glyph_baseline)

        print(f"Loaded baseline for {len(glyph_baseline)} glyph(s).")

    else:
        print(
            "No --blank-sheet or --glyph-baseline given; bubble "
            "scoring will use fixed absolute-darkness thresholds "
            "only. Pass one of those for more reliable detection of "
            "lightly filled bubbles."
        )

    return blank_gray_front, blank_gray_back, glyph_baseline


def load_roster(args) -> tuple[dict[str, str], bool]:
    if args.roster is None:
        args.roster = find_default_roster()
        print(f"Using roster: {args.roster}")

    if not args.roster.exists():
        raise FileNotFoundError(f"Roster file does not exist: {args.roster}")

    roster_is_csv = args.roster.suffix.lower() == ".csv"

    if roster_is_csv:
        return load_roster_from_csv(args.roster), True

    return load_student_roster(args.roster), False


def write_report(args, sheet_number, student_data, roster, grading_result) -> Path:
    report_text = format_student_report(
        sheet_number=sheet_number,
        student_data=student_data,
        student_name=roster.get(normalize_student_id(student_data["student_id"]), ""),
        grading_result=grading_result,
    )

    output_path = save_student_report(
        output_directory=args.output,
        page_number=sheet_number,
        report_text=report_text,
    )

    print(f"Saved: {output_path}")
    return output_path


def build_combined_reports(args) -> None:
    if not args.combine_reports:
        return

    try:
        create_combined_pdf(args.output, combined_reports_path(args))
    except FileNotFoundError as exc:
        print(f"Skipping combined report PDF: {exc}")


def main():
    args = parse_args()

    if args.threshold < 0:
        raise ValueError("--threshold cannot be negative")

    if args.gap < 0:
        raise ValueError("--gap cannot be negative")

    if args.rerun_bad:
        rerun_bad_sheets(args)
        return

    if args.add_late is not None:
        add_late_sheets(args)
        return

    if not 1 <= args.questions <= 120:
        raise ValueError("--questions must be between 1 and 120")

    # Standard input files in the current directory (renamed to the
    # folder-prefixed scheme if they still have their plain names).
    if args.input_pdf is None:
        args.input_pdf = find_standard_input(STUDENT_PDF_NAME)

        if args.input_pdf is None:
            raise FileNotFoundError(
                f"No {STUDENT_PDF_NAME}.pdf (or "
                f"<folders>_{STUDENT_PDF_NAME}.pdf) in {Path.cwd()}."
            )

    args.standard_solution_pdf = (
        find_standard_input(SOLUTION_PDF_NAME)
        if args.solution_pdf is None
        else None
    )

    if not args.input_pdf.exists():
        raise FileNotFoundError(f"Student PDF does not exist: {args.input_pdf}")

    roster, roster_is_csv = load_roster(args)
    blank_gray_front, blank_gray_back, glyph_baseline = load_scoring_reference(args)

    solution_keys = prepare_solution_directory(
        args,
        blank_gray_front=blank_gray_front,
        blank_gray_back=blank_gray_back,
        glyph_baseline=glyph_baseline,
    )

    max_score = args.max_score or max_score_for(args.questions)
    print(
        f"Scoring out of {format_points(max_score)} "
        f"({args.questions} questions; extra credit can't go above the max)."
    )

    print(f"Reading student PDF: {args.input_pdf}")
    pages = pdf_to_images(args.input_pdf)

    if not pages:
        raise ValueError(f"No pages found in PDF: {args.input_pdf}")

    if len(pages) % 2 != 0:
        raise ValueError(
            f"{args.input_pdf} has {len(pages)} page(s) — an odd "
            "number. Exam sheets are always scanned front/back, so "
            "this PDF is missing a page somewhere (a skipped or "
            "misfed sheet during double-sided scanning). Check the "
            "scan before re-running."
        )

    # A full run regrades everything, so start from clean output
    # folders (old reports or bad sheets would otherwise mix in).
    output_directory = Path(args.output)

    if output_directory.exists():
        for old in output_directory.glob("student_page_*.txt"):
            old.unlink()

        (output_directory / GRADED_RECORD).unlink(missing_ok=True)

    if BAD_SHEETS_DIRECTORY.exists():
        print(f"Clearing old {BAD_SHEETS_DIRECTORY}/ (this run rebuilds it).")
        shutil.rmtree(BAD_SHEETS_DIRECTORY)

    sheet_count = len(pages) // 2
    print(f"Found {sheet_count} student exam sheet(s) ({len(pages)} pages).")

    good: list[dict[str, Any]] = []
    bad: list[dict[str, Any]] = []

    for sheet_number, page_index in enumerate(range(0, len(pages), 2), start=1):
        print(
            f"Processing student sheet {sheet_number}/{sheet_count} "
            f"(pages {page_index + 1}-{page_index + 2})..."
        )

        try:
            selections = process_exam_sheet(
                pages[page_index],
                pages[page_index + 1],
                question_count=args.questions,
                threshold=args.threshold,
                gap=args.gap,
                blank_gray_front=blank_gray_front,
                blank_gray_back=blank_gray_back,
                glyph_baseline=glyph_baseline,
            )
        except Exception as exc:
            print(f"Sheet {sheet_number} could not be read ({exc}) -> BadSheets.")
            bad.append({
                "sheet": sheet_number,
                "page_index": page_index,
                "student_id": None,
                "version": None,
                "needs": ["read"],
                "reasons": [f"Sheet could not be read: {type(exc).__name__}: {exc}"],
                "issues": [],
            })
            continue

        student_data = extract_student_data(
            selections=selections,
            question_count=args.questions,
        )

        settled = resolve_metadata(
            sheet_number, student_data, roster, solution_keys, taken_ids=set(),
        )

        if settled["needs"]:
            bad.append({
                "sheet": sheet_number,
                "page_index": page_index,
                "student_id": settled["student_id"],
                "version": settled["version"],
                "needs": settled["needs"],
                "reasons": settled["bad_reasons"],
                "issues": settled["issues"],
            })
            continue

        apply_metadata(student_data, settled)
        good.append({
            "sheet": sheet_number,
            "page_index": page_index,
            "student_data": student_data,
            "settled": settled,
        })

    # Two sheets with the same student ID: both go to BadSheets, so
    # neither contaminates the good reports or the grades file.
    by_id: dict[str, list[dict[str, Any]]] = {}

    for item in good:
        by_id.setdefault(
            normalize_student_id(item["student_data"]["student_id"]), []
        ).append(item)

    for student_id, items in by_id.items():
        if len(items) < 2:
            continue

        sheets = [item["sheet"] for item in items]
        print(
            f"\nSheets {', '.join(map(str, sheets))} all read as student ID "
            f"{items[0]['student_data']['student_id']} "
            f"({roster.get(student_id, 'unknown')}). All of them go to "
            "BadSheets."
        )
        ask("  Press Enter to continue: ")

        for item in items:
            good.remove(item)
            others = [str(n) for n in sheets if n != item["sheet"]]
            bad.append({
                "sheet": item["sheet"],
                "page_index": item["page_index"],
                "student_id": item["student_data"]["student_id"],
                "version": item["student_data"]["version"],
                "needs": ["student_id"],
                "reasons": [
                    f"Duplicate student ID {item['student_data']['student_id']} "
                    f"(also on sheet {', '.join(others)})"
                ],
                "issues": item["student_data"]["issues"] + [{
                    "reason": "Duplicate student ID, sheet set aside",
                    "deduction": DEDUCTION_PER_ISSUE,
                    "pending": True,
                }],
            })

    # Reports and scores for the good sheets only.
    scores: dict[str, float] = {}
    graded: dict[str, Any] = {}

    for item in sorted(good, key=lambda entry: entry["sheet"]):
        student_data = item["student_data"]
        grading_result = grade_sheet(student_data, solution_keys, max_score)

        for warning in student_data["warnings"]:
            print(f"Sheet {item['sheet']}: {warning}")

        write_report(args, item["sheet"], student_data, roster, grading_result)

        normalized_id = normalize_student_id(student_data["student_id"])
        scores[normalized_id] = grading_result["final_score"]
        graded[normalized_id] = {
            "sheet": item["sheet"],
            "score": grading_result["final_score"],
        }

    save_graded_record(output_directory, graded)
    save_run_settings(output_directory, {
        "question_count": args.questions,
        "max_score": max_score,
        "threshold": args.threshold,
        "gap": args.gap,
    })

    # Bad sheets: one 2-page PDF each, plus the record for --rerun-bad.
    if bad:
        record_sheets = []

        for entry in sorted(bad, key=lambda e: e["sheet"]):
            file_name = f"sheet_{entry['sheet']:03d}.pdf"
            copy_sheet_pages(
                args.input_pdf,
                entry.pop("page_index"),
                BAD_SHEETS_DIRECTORY / file_name,
            )
            entry["file"] = file_name
            record_sheets.append(entry)

        save_bad_sheets_record({
            "question_count": args.questions,
            "max_score": max_score,
            "threshold": args.threshold,
            "gap": args.gap,
            "sheets": record_sheets,
        })

        print(
            f"\n{len(bad)} sheet(s) set aside in {BAD_SHEETS_DIRECTORY}/ "
            "(see BadSheets_summary.txt). Fix them later with --rerun-bad."
        )

    print("Student data extraction complete.")

    if roster_is_csv:
        write_grades_csv(
            source_csv_path=args.roster,
            output_csv_path=grades_csv_path(args),
            scores={key: format_points(value) for key, value in scores.items()},
            quiz_column=args.grades_column,
            zero_fill_ungraded=args.zero_fill_ungraded,
        )

    build_combined_reports(args)


def keep_in_bad_sheets(entry: dict[str, Any]) -> dict[str, Any]:
    """
    A late sheet that still needs work gets its own copy in BadSheets
    (like any other bad sheet), so --rerun-bad can find it.
    """

    if entry.get("source"):
        copy_sheet_pages(
            Path(entry["source"]),
            entry["page_index"],
            BAD_SHEETS_DIRECTORY / entry["file"],
        )
        entry = {k: v for k, v in entry.items() if k not in ("source", "page_index")}

    return entry


def add_late_sheets(args) -> None:
    """
    Grade one or more late sheets (a separate PDF, front/back pairs)
    and add them to this exam's existing results, without regrading
    anyone else.

    The late PDF's pages are appended to the end of the main
    StudentAnswers PDF (and the separate file deleted), so each late
    sheet's number is its position there. Each one is read like a
    normal first reading (prompts and deductions as usual, and its ID
    is checked against students already graded), then its report is added to
    StudentReports, the combined PDF is rebuilt, and only that
    student's row changes in the Grades CSV. A late sheet you skip
    stays in BadSheets for a later --rerun-bad.
    """

    late_pdf = Path(args.add_late)

    if not late_pdf.exists():
        raise FileNotFoundError(f"Late sheet PDF does not exist: {late_pdf}")

    output_directory = Path(args.output)

    if not any(output_directory.glob("student_page_*.txt")):
        raise FileNotFoundError(
            f"No earlier results in {output_directory}/ -- run the full "
            "grader for this exam first, then add late sheets."
        )

    record_path = BAD_SHEETS_DIRECTORY / BAD_SHEETS_RECORD
    record = (
        json.loads(record_path.read_text(encoding="utf-8"))
        if record_path.exists()
        else None
    )

    # Same settings as the original run: saved by it, or from the
    # BadSheets record, or (results from before settings were saved)
    # from the command line.
    settings = load_run_settings(output_directory)

    if settings is None and record is not None:
        settings = {key: record[key] for key in ("question_count", "max_score", "threshold", "gap")}

    if settings is None:
        settings = {
            "question_count": args.questions,
            "max_score": args.max_score or max_score_for(args.questions),
            "threshold": args.threshold,
            "gap": args.gap,
        }
        print(
            f"No saved settings from the original run; using "
            f"{args.questions} questions (pass --questions if that's wrong)."
        )

    if record is None:
        record = dict(settings, sheets=[])

    main_pdf = find_standard_input(STUDENT_PDF_NAME)

    if main_pdf is None:
        raise FileNotFoundError(
            f"No {STUDENT_PDF_NAME}.pdf (or <folders>_{STUDENT_PDF_NAME}.pdf) "
            f"in {Path.cwd()} to add the late sheet to."
        )

    if late_pdf.resolve() == main_pdf.resolve():
        raise ValueError("The late sheet PDF can't be the main student PDF itself.")

    late_pages = count_pdf_pages(late_pdf)

    if late_pages == 0 or late_pages % 2:
        raise ValueError(
            f"{late_pdf} has {late_pages} page(s); late sheets must be "
            "scanned front/back (an even number of pages)."
        )

    main_pages = count_pdf_pages(main_pdf)

    # Sheet numbers = position in the main PDF, so a later full run
    # numbers these sheets the same way.
    first_new_sheet = main_pages // 2 + 1
    already_used = {info["sheet"] for info in load_graded_record(output_directory).values()}
    already_used |= {entry["sheet"] for entry in record["sheets"]}

    if first_new_sheet in already_used:
        raise RuntimeError(
            f"Sheet {first_new_sheet} is already in use, so the results "
            f"don't match {main_pdf.name}. Rerun the full grader first."
        )

    append_pdf(main_pdf, late_pdf)
    print(f"Added {late_pdf.name} to the end of {main_pdf.name}.")
    late_pdf.unlink()
    print(f"Deleted {late_pdf.name} (its pages are now in {main_pdf.name}).")

    new_sheets = set()

    for offset in range(0, late_pages, 2):
        sheet_number = first_new_sheet + offset // 2
        record["sheets"].append({
            "sheet": sheet_number,
            "file": f"sheet_{sheet_number:03d}.pdf",
            "source": main_pdf.name,
            "page_index": main_pages + offset,
            "student_id": None,
            "version": None,
            "needs": ["read"],
            "reasons": [f"Late sheet (added from {late_pdf.name})"],
            "issues": [],
        })
        new_sheets.add(sheet_number)

    save_bad_sheets_record(record)
    print(
        f"Adding {len(new_sheets)} late sheet(s) as sheet "
        f"{', '.join(map(str, sorted(new_sheets)))}."
    )

    rerun_bad_sheets(args, only_sheets=new_sheets)


def rerun_bad_sheets(args, only_sheets: set[int] | None = None) -> None:
    """
    Regrade only the sheets in BadSheets/ (see --rerun-bad).

    For each sheet, asks again for whatever is still missing -- no new
    deductions for that -- then asks whether to apply each deduction
    that was held as pending when the sheet was set aside. A fixed
    sheet's report goes into the existing reports folder, the combined
    PDF is rebuilt, and only the fixed students' rows change in the
    existing Grades CSV. Sheets you skip again stay in BadSheets.
    """

    record = load_bad_sheets_record()
    question_count = record["question_count"]
    max_score = args.max_score or record["max_score"]

    roster, roster_is_csv = load_roster(args)
    blank_gray_front, blank_gray_back, glyph_baseline = load_scoring_reference(args)

    solutions_directory = Path(args.solutions)

    if not (solutions_directory.is_dir() and any(solutions_directory.glob("*.json"))):
        raise FileNotFoundError(
            f"No answer keys in {solutions_directory} -- run the full "
            "grader first."
        )

    solution_keys = load_solution_keys(solutions_directory)
    output_directory = Path(args.output)
    graded = load_graded_record(output_directory)
    fixed_dir = BAD_SHEETS_DIRECTORY / "Fixed"

    targets = [
        entry for entry in record["sheets"]
        if only_sheets is None or entry["sheet"] in only_sheets
    ]
    print(f"Grading {len(targets)} sheet(s) from {BAD_SHEETS_DIRECTORY}/.")

    still_bad = []
    fixed_scores: dict[str, float] = {}
    source_pages: dict[str, list] = {}

    for entry in record["sheets"]:
        if only_sheets is not None and entry["sheet"] not in only_sheets:
            still_bad.append(entry)  # not part of this run; leave as is
            continue

        sheet_number = entry["sheet"]
        sheet_pdf = BAD_SHEETS_DIRECTORY / entry["file"]
        print(f"\n--- {entry['file']} (original sheet {sheet_number}) ---")

        for reason in entry["reasons"]:
            print(f"  note: {reason}")

        try:
            if entry.get("source"):
                # Late sheet: read its two pages straight from the main PDF.
                source = entry["source"]

                if source not in source_pages:
                    source_pages[source] = pdf_to_images(Path(source))

                index = entry["page_index"]
                sheet_pages = source_pages[source][index:index + 2]
            else:
                sheet_pages = pdf_to_images(sheet_pdf)

            if len(sheet_pages) != 2:
                raise ValueError(f"expected 2 pages, found {len(sheet_pages)}")

            selections = process_exam_sheet(
                sheet_pages[0],
                sheet_pages[1],
                question_count=question_count,
                threshold=record["threshold"],
                gap=record["gap"],
                blank_gray_front=blank_gray_front,
                blank_gray_back=blank_gray_back,
                glyph_baseline=glyph_baseline,
            )
        except Exception as exc:
            print(f"  Still can't read this sheet ({exc}); leaving it in BadSheets.")
            still_bad.append(keep_in_bad_sheets(entry))
            continue

        student_data = extract_student_data(
            selections=selections,
            question_count=question_count,
        )

        taken = {
            student_id for student_id, info in graded.items()
            if info.get("sheet") != sheet_number
        }

        if "read" in entry["needs"]:
            # Never read before (a late sheet, or one rescanned since):
            # treat as a first reading, with the usual deductions.
            settled = resolve_metadata(
                sheet_number, student_data, roster, solution_keys, taken,
            )

            # Its ID may belong to someone already graded.
            if (
                not settled["needs"]
                and normalize_student_id(settled["student_id"]) in taken
            ):
                other = graded[normalize_student_id(settled["student_id"])]["sheet"]
                new_id = ask_student_id(
                    f"Sheet {sheet_number}: student ID {settled['student_id']} "
                    f"is already graded on sheet {other}.",
                    roster, taken,
                )

                if new_id is None:
                    settled["needs"].append("student_id")
                    settled["bad_reasons"].append(
                        f"Student ID {settled['student_id']} already graded "
                        f"on sheet {other}"
                    )
                    settled["issues"].append({
                        "reason": "Duplicate student ID, sheet set aside",
                        "deduction": DEDUCTION_PER_ISSUE,
                        "pending": True,
                    })
                else:
                    settled["issues"].append(ask_apply_deduction(
                        f"Student ID entered by instructor (bubbled "
                        f"{settled['student_id']}, already graded on sheet {other})"
                    ))
                    settled["student_id"] = new_id
        else:
            settled = resolve_metadata(
                sheet_number, student_data, roster, solution_keys, taken,
                stored=entry,
            )

            # A stored ID that has since been graded elsewhere can't be reused.
            if (
                not settled["needs"]
                and normalize_student_id(settled["student_id"]) in taken
            ):
                print(
                    f"  {settled['student_id']} is already graded on sheet "
                    f"{graded[normalize_student_id(settled['student_id'])]['sheet']}."
                )
                entry = dict(entry, needs=sorted(set(entry["needs"]) | {"student_id"}))
                settled = resolve_metadata(
                    sheet_number, student_data, roster, solution_keys, taken,
                    stored=entry,
                )

        if settled["needs"]:
            entry.update(
                student_id=settled["student_id"] or entry.get("student_id"),
                version=settled["version"] or entry.get("version"),
                needs=settled["needs"],
                reasons=settled["bad_reasons"] or entry["reasons"],
                issues=entry["issues"] + settled["issues"],
            )
            entry["needs"] = [n for n in entry["needs"] if n != "read"] or entry["needs"]
            print("  Skipped; it stays in BadSheets.")
            still_bad.append(keep_in_bad_sheets(entry))
            continue

        # Fixed: decide on each deduction held while it was set aside.
        carried = []

        for issue in entry["issues"]:
            if issue.get("pending"):
                carried.append(ask_apply_deduction(issue["reason"]))
            else:
                carried.append(issue)

        apply_metadata(student_data, settled, carried_issues=carried)
        grading_result = grade_sheet(student_data, solution_keys, max_score)

        for warning in student_data["warnings"]:
            print(f"  {warning}")

        write_report(args, sheet_number, student_data, roster, grading_result)

        normalized_id = normalize_student_id(student_data["student_id"])
        fixed_scores[normalized_id] = grading_result["final_score"]
        graded[normalized_id] = {"sheet": sheet_number, "score": grading_result["final_score"]}

        if not entry.get("source"):
            fixed_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(sheet_pdf), str(fixed_dir / entry["file"]))
        print(
            f"  Graded: {format_points(grading_result['final_score'])}/"
            f"{format_points(max_score)}"
        )

    save_graded_record(output_directory, graded)
    record["sheets"] = still_bad
    save_bad_sheets_record(record)

    print(
        f"\nGraded {len(fixed_scores)} sheet(s); "
        f"{len(still_bad)} in {BAD_SHEETS_DIRECTORY}/."
    )

    if fixed_scores and roster_is_csv:
        grades_path = grades_csv_path(args)

        if grades_path.exists():
            # Update just the fixed students' rows; leave everyone else as is.
            write_grades_csv(
                source_csv_path=grades_path,
                output_csv_path=grades_path,
                scores={k: format_points(v) for k, v in fixed_scores.items()},
                quiz_column=args.grades_column,
                zero_fill_ungraded=False,
            )
        else:
            write_grades_csv(
                source_csv_path=args.roster,
                output_csv_path=grades_path,
                scores={k: format_points(v["score"]) for k, v in graded.items()},
                quiz_column=args.grades_column,
                zero_fill_ungraded=args.zero_fill_ungraded,
            )

    if fixed_scores:
        build_combined_reports(args)


if __name__ == "__main__":
    main()