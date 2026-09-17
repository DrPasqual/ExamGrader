from __future__ import annotations

import argparse
import json
import re
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

DEFAULT_QUESTION_COUNT = 120


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

        term_F          -> F
        year_2025       -> 2025
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
# Student data extraction (field set differs: no quiz number, and
# "version" instead of "quiz_version")
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
        "term": extracted["term"],
        "year": extracted["year"],
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

def format_student_report(
    sheet_number: int,
    student_data: dict[str, Any],
    student_name: str,
    grading_result: dict[str, Any] | None = None,
) -> str:
    lines: list[str] = []

    lines.append("SCANTRON STUDENT DATA AND GRADE")
    lines.append("=" * 80)
    lines.append(f"Sheet:         {sheet_number}")
    lines.append("")

    lines.append("STUDENT")
    lines.append("-" * 80)
    lines.append(
        f"Student ID:    "
        f"{student_data['student_id'] or '[NOT FOUND]'}"
    )

    lines.append(
        f"Student Name:  "
        f"{student_name or '[UNKNOWN STUDENT]'}"
    )
    lines.append("")

    lines.append("EXAM METADATA")
    lines.append("-" * 80)
    lines.append(
        f"Term:          {student_data['term'] or '[NOT FOUND]'}"
    )
    lines.append(
        f"Year:          {student_data['year'] or '[NOT FOUND]'}"
    )
    lines.append(
        f"Version:       {student_data['version'] or '[NOT FOUND]'}"
    )
    lines.append("")

    # Matches the same criterion main() uses to decide whether grading
    # was even attempted: term/year don't block matching (a run is
    # assumed to be one exam instance), so only a missing version
    # means no solution key lookup was possible.
    missing_for_matching = [
        field
        for field in student_data["missing_metadata"]
        if field == "version"
    ]

    if missing_for_matching:
        lines.append("ANSWERS READ")
        lines.append("-" * 80)
        lines.append("Question | Student Response")
        lines.append("-" * 80)

        for question_number in sorted(
            student_data["answers"],
            key=int,
        ):
            answer = student_data["answers"][question_number]

            lines.append(
                f"{int(question_number):8d} | "
                f"{answer if answer is not None else '[BLANK]'}"
            )

        lines.append("")

    elif grading_result is None:
        lines.append("GRADING")
        lines.append("-" * 80)
        lines.append("No matching solution key was found.")
        lines.append("")

    else:
        lines.append("GRADING SUMMARY")
        lines.append("-" * 80)
        lines.append(
            f"Score:         "
            f"{grading_result['correct_count']}/"
            f"{grading_result['question_count']}"
        )
        lines.append(
            f"Percentage:    "
            f"{grading_result['percentage']:.2f}%"
        )
        lines.append(
            f"Answered:      "
            f"{grading_result['answered_count']}/"
            f"{grading_result['question_count']}"
        )

        if grading_result.get("solution_filename"):
            lines.append(
                f"Solution Key:  "
                f"{grading_result['solution_filename']}"
            )

        lines.append("")

        lines.append("QUESTION RESULTS")
        lines.append("-" * 80)
        lines.append(
            "Question | Student Response | Correct Answer | Result"
        )
        lines.append("-" * 80)

        for question_number, result in (
            grading_result["question_results"].items()
        ):
            student_answer = (
                result["student_answer"]
                if result["student_answer"] is not None
                else "[BLANK]"
            )

            correct_answer = (
                result["correct_answer"]
                if result["correct_answer"] is not None
                else "[NO KEY]"
            )

            status = (
                "CORRECT"
                if result["correct"]
                else "INCORRECT"
            )

            lines.append(
                f"{int(question_number):8d} | "
                f"{student_answer:17s} | "
                f"{correct_answer:15s} | "
                f"{status}"
            )

        lines.append("")

        lines.append("INSTRUCTOR NOTES")
        lines.append("-" * 80)

        lines.append("")

    if DEBUG:
        lines.append("CORNER / TIMING MARK DETECTION")
        lines.append("-" * 80)

        detection_diagnostics = student_data.get(
            "detection_diagnostics",
            [],
        )

        if detection_diagnostics:
            for note in detection_diagnostics:
                lines.append(f"- {note}")
        else:
            lines.append("No corners or timing marks needed interpolation.")

        lines.append("")

        lines.append("RAW READER OUTPUT")
        lines.append("-" * 80)
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
    Use exam_solution_processor to generate (if --solution_pdf is
    given) and load solution JSON files.
    """

    solutions_directory = Path(args.solutions)

    if args.solution_pdf is not None:
        solution_pdf = Path(args.solution_pdf)

        if not solution_pdf.exists():
            raise FileNotFoundError(
                f"Solution PDF does not exist: {solution_pdf}"
            )

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

    elif not solutions_directory.exists():
        raise FileNotFoundError(
            "No solution source was provided. Use --solution_pdf "
            "or provide an existing --solutions directory."
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
        help=(
            "Student answer-sheet PDF, double-sided scanned (front, "
            "back, front, back, ... -- an even number of pages)."
        ),
    )

    parser.add_argument(
        "--solution_pdf",
        type=Path,
        default=None,
        help=(
            "Optional solution PDF (also double-sided). If supplied, "
            "exam_solution_processor.py generates JSON files."
        ),
    )

    parser.add_argument(
        "--solutions",
        type=Path,
        default=Path("./ExamSolutionJSONs"),
        help=(
            "Directory containing solution JSON files. Also where "
            "newly generated ones (from --solution_pdf) are saved. "
            "Default: ./ExamSolutionJSONs"
        ),
    )

    parser.add_argument(
        "--roster",
        type=Path,
        required=True,
        help="Student roster: a Canvas CSV export or a plain text file.",
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
        help="Number of questions on the exam (1-120).",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Bubble detection threshold.",
    )

    parser.add_argument(
        "--gap",
        type=float,
        default=0.01,
        help="Minimum difference between bubble choices.",
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
            "Optional glyph baseline JSON, built with "
            "'python exam_reader.py --calibrate-glyphs ...'. Like "
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
            "score filled in. Default: a grades.csv next to the "
            "student answer-sheet PDF."
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
            "report. Default: combined_student_reports.pdf next to "
            "the student answer-sheet PDF."
        ),
    )

    parser.add_argument(
        "--no-combined-reports",
        dest="combine_reports",
        action="store_false",
        help="Skip building the combined student-reports PDF.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input_pdf.exists():
        raise FileNotFoundError(
            f"Student PDF does not exist: {args.input_pdf}"
        )

    if not args.roster.exists():
        raise FileNotFoundError(
            f"Roster file does not exist: {args.roster}"
        )

    if not 1 <= args.questions <= 120:
        raise ValueError("--questions must be between 1 and 120")

    if args.threshold < 0:
        raise ValueError("--threshold cannot be negative")

    if args.gap < 0:
        raise ValueError("--gap cannot be negative")

    blank_gray_front = None
    blank_gray_back = None
    glyph_baseline = None

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

    solution_keys = prepare_solution_directory(
        args,
        blank_gray_front=blank_gray_front,
        blank_gray_back=blank_gray_back,
        glyph_baseline=glyph_baseline,
    )

    roster_is_csv = args.roster.suffix.lower() == ".csv"

    if roster_is_csv:
        roster = load_roster_from_csv(args.roster)
    else:
        roster = load_student_roster(args.roster)

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

    sheet_count = len(pages) // 2

    print(f"Found {sheet_count} student exam sheet(s) ({len(pages)} pages).")

    scores: dict[str, int] = {}

    for sheet_number, page_index in enumerate(
        range(0, len(pages), 2),
        start=1,
    ):
        front_page_number = page_index + 1
        back_page_number = page_index + 2

        print(
            f"Processing student sheet {sheet_number}/{sheet_count} "
            f"(pages {front_page_number}-{back_page_number})..."
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

            student_data = extract_student_data(
                selections=selections,
                question_count=args.questions,
            )

            normalized_id = normalize_student_id(
                student_data["student_id"]
            )

            student_name = roster.get(normalized_id, "")

            # Term/year don't block matching -- see
            # make_solution_lookup_key(). Only a missing version does.
            if "version" in student_data["missing_metadata"]:
                print(
                    f"Sheet {sheet_number}: student left version "
                    "blank; skipping automatic grading for this sheet."
                )
                grading_result = None
            else:
                lookup_key = make_solution_lookup_key(
                    student_data["version"]
                )

                solution = solution_keys.get(lookup_key)

                if solution is None:
                    print(
                        f"No matching solution key found for sheet "
                        f"{sheet_number}: {lookup_key!r}"
                    )
                    grading_result = None
                else:
                    print(
                        f"Using solution key for sheet {sheet_number}: "
                        f"{solution.get('_filename', lookup_key)}"
                    )

                    grading_result = grade_student_answers(
                        student_data=student_data,
                        solution=solution,
                    )

                    if (
                        normalized_id
                        and "?" not in student_data["student_id"]
                    ):
                        if normalized_id in scores:
                            print(
                                f"Warning: sheet {sheet_number} has "
                                "the same student ID as an earlier "
                                f"sheet ({normalized_id}); using this "
                                "sheet's score."
                            )

                        scores[normalized_id] = grading_result[
                            "correct_count"
                        ]

            report_text = format_student_report(
                sheet_number=sheet_number,
                student_data=student_data,
                student_name=student_name,
                grading_result=grading_result,
            )

            output_path = save_student_report(
                output_directory=args.output,
                page_number=sheet_number,
                report_text=report_text,
            )

            print(f"Saved: {output_path}")

        except Exception as exc:
            error_path = save_error_report(
                output_directory=args.output,
                page_number=sheet_number,
                error=exc,
            )

            print(
                f"Sheet {sheet_number} failed. "
                f"Saved error report: {error_path}"
            )

    print("Student data extraction complete.")

    if roster_is_csv:
        grades_out = args.grades_out or (
            args.input_pdf.parent / "grades.csv"
        )

        write_grades_csv(
            source_csv_path=args.roster,
            output_csv_path=grades_out,
            scores=scores,
            quiz_column=args.grades_column,
            zero_fill_ungraded=args.zero_fill_ungraded,
        )

    if args.combine_reports:
        combined_reports_out = args.combined_reports_out or (
            args.input_pdf.parent / "combined_student_reports.pdf"
        )

        try:
            create_combined_pdf(args.output, combined_reports_out)
        except FileNotFoundError as exc:
            print(f"Skipping combined report PDF: {exc}")


if __name__ == "__main__":
    main()
