"""
exam_solution_processor.py

Exam version of the scantron solution processor -- fully standalone,
no dependency on any of the quiz grading files.

Converts exam_reader output into standardized solution-key data, and
matches student sheets to the right solution key. The exam sheet has
an Exam field (1, 2, 3 = midterms, F = final) and a Version field
(A-E, X); the answer range runs to 120. A grading run is assumed to be
exactly ONE exam (never "Exam 1" mixed with "Exam 2" or a final in the
same run), so the solution lookup key is just the version letter. The
exam is still recorded on every key and checked against each student
sheet by the grader, to catch a sheet from the wrong exam.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

VALID_ANSWERS = {"A", "B", "C", "D", "E"}


def selected_value(value: Any) -> Any:
    """Return the selected bubble name from a reader result."""

    if isinstance(value, dict):
        return value.get("selected")

    return value


def suffix_value(group_name: str, value: Any) -> str | None:
    """
    Convert a selected bubble name into its value.

    Examples:
        answer_071_A -> A
        exam_F -> F
        version_B -> B
    """

    value = selected_value(value)

    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    prefix = f"{group_name}_"

    if value.startswith(prefix):
        return value[len(prefix):]

    return value


def get_group_value(
    selections: dict[str, Any],
    *group_names: str,
) -> str | None:
    """Get and simplify the selected value for one of several groups."""

    for group_name in group_names:
        if group_name not in selections:
            continue

        value = suffix_value(
            group_name,
            selections[group_name],
        )

        if value is not None:
            return value

    return None


def get_digit_group_value(
    selections: dict[str, Any],
    group_name: str,
) -> str | None:
    """
    Extract one digit from a metadata group.

    For example:

        group_name = "year_first_digit"
        selected = "year_first_digit_2"
        result = "2"
    """

    value = get_group_value(selections, group_name)

    if value is None:
        return None

    match = re.search(r"\d", value)

    if match:
        return match.group(0)

    return None


def get_metadata_digits(
    selections: dict[str, Any],
    field_name: str,
) -> str | None:
    """
    Assemble metadata from the reader's grouped digit fields.

    Supports groups named:

        <field_name>_first_digit
        <field_name>_second_digit
        <field_name>_third_digit  (etc., if ever needed)
    """

    first = get_digit_group_value(
        selections,
        f"{field_name}_first_digit",
    )

    second = get_digit_group_value(
        selections,
        f"{field_name}_second_digit",
    )

    values = [first, second]

    for ordinal in ("third", "fourth", "fifth", "sixth"):
        value = get_digit_group_value(
            selections,
            f"{field_name}_{ordinal}_digit",
        )

        if value is not None:
            values.append(value)

    digits = "".join(value for value in values if value is not None)

    return digits or None


def normalize_lookup_value(value: Any) -> str:
    """Normalize a metadata value for reliable dictionary lookup."""

    if value is None:
        return ""

    return str(value).strip().casefold()


def extract_solution_data(
    selections: dict[str, dict],
    question_count: int = 120,
) -> dict[str, Any]:
    """
    Convert exam_reader output into standardized solution data.

    Same idea as the quiz's extract_solution_data(), with the exam's
    metadata (exam + version instead of quiz number/version/term/
    year), and the answer loop runs up to 120 questions instead of 20.
    """

    exam = get_group_value(
        selections,
        "exam",
    )

    version = get_group_value(
        selections,
        "version",
    )

    missing = []

    if exam is None:
        missing.append("exam")

    if version is None:
        missing.append("version")

    # Deliberately NOT raising here even if metadata is missing — see
    # the quiz's extract_solution_data() for the full reasoning. The
    # answers below are extracted regardless, and callers that
    # actually require complete metadata (generating a solution key)
    # check "missing_metadata" themselves — see process_solution_pdf().

    answers: dict[str, str | None] = {}

    for question_number in range(1, question_count + 1):
        group_name = f"answer_{question_number:02d}"

        answer = get_group_value(
            selections,
            group_name,
        )

        if answer is not None:
            answer = answer.upper()

        if answer not in VALID_ANSWERS:
            answer = None

        answers[str(question_number)] = answer

    return {
        "exam": exam,
        "version": version,
        "question_count": question_count,
        "answers": answers,
        "missing_metadata": missing,
    }


def make_solution_filename(solution: dict[str, Any]) -> str:
    """Create the standardized answer-key filename."""

    exam = str(solution["exam"]).strip().upper()
    version = str(solution["version"]).strip().upper()

    for label, value in (
        ("exam", exam),
        ("version", version),
    ):
        if not value:
            raise ValueError(f"Empty {label}.")

        if re.search(r"[^A-Za-z0-9_-]", value):
            raise ValueError(
                f"Invalid characters in {label}: {value!r}"
            )

    return (
        f"exam_{exam}_"
        f"version_{version}.json"
    )


def save_solution_json(
    output_directory: Path,
    solution: dict[str, Any],
    source_pdf: str,
    source_pages: tuple[int, int],
) -> Path:
    """
    Save one standardized solution JSON file.

    `source_pages` is (front_page_number, back_page_number) in the
    source PDF, since an exam solution comes from two physical pages.
    """

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_directory / make_solution_filename(solution)

    data = {
        "exam": solution["exam"],
        "version": solution["version"],
        "question_count": solution["question_count"],
        "answers": solution["answers"],
        "source": {
            "pdf": source_pdf,
            "front_page": source_pages[0],
            "back_page": source_pages[1],
        },
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")

    return output_path


def load_solution_keys(
    solution_directory: Path | str,
) -> dict[str, dict[str, Any]]:
    """
    Load all solution JSON files from a directory.

    Returns a dictionary keyed by just `version` (normalized) -- an
    exam grading run is assumed to only ever involve ONE exam
    instance (e.g. "Exam 1", never mixed with "Exam 2" or a final in
    the same run), so unlike the quiz's (quiz_number, quiz_version)
    pair, version alone is enough to identify which solution key a
    student's sheet matches. The exam is stored in each solution's
    data (when present) so the grader can flag a sheet from the wrong
    exam.

    Values are the corresponding solution dictionaries.
    """

    solution_directory = Path(solution_directory)

    if not solution_directory.exists():
        raise FileNotFoundError(
            f"Solution directory does not exist: {solution_directory}"
        )

    if not solution_directory.is_dir():
        raise NotADirectoryError(
            f"Solution path is not a directory: {solution_directory}"
        )

    solution_keys: dict[str, dict[str, Any]] = {}

    # "exam" is deliberately not required, so keys saved before the
    # Exam field existed (term/year era) still load.
    required_fields = {
        "version",
        "question_count",
        "answers",
    }

    for json_path in sorted(solution_directory.glob("*.json")):
        try:
            with json_path.open("r", encoding="utf-8") as file:
                solution = json.load(file)
        except json.JSONDecodeError as exc:
            print(f"Skipping invalid JSON file {json_path}: {exc}")
            continue

        if not isinstance(solution, dict):
            print(f"Skipping {json_path}: top-level JSON value is not an object")
            continue

        missing = sorted(required_fields - solution.keys())

        if missing:
            print(
                f"Skipping {json_path}: "
                f"missing required fields: {', '.join(missing)}"
            )
            continue

        lookup_key = normalize_lookup_value(solution["version"])

        if not lookup_key:
            print(f"Skipping {json_path}: version is empty")
            continue

        if lookup_key in solution_keys:
            previous = solution_keys[lookup_key].get("_filename", "unknown")
            print(
                f"Warning: duplicate version in {json_path.name}; "
                f"replacing {previous}"
            )

        solution_copy = dict(solution)
        solution_copy["_filename"] = json_path.name
        solution_keys[lookup_key] = solution_copy

    return solution_keys


def process_solution_pdf(
    input_pdf: Path,
    output_directory: Path,
    question_count: int,
    threshold: float,
    gap: float,
    blank_gray_front: Any | None = None,
    blank_gray_back: Any | None = None,
    glyph_baseline: dict[str, dict[str, float]] | None = None,
) -> None:
    """
    Process every exam sheet (pairs of pages) in a solution PDF.

    Like the quiz's process_solution_pdf(), but each solution "page"
    is really TWO physical pages (front, back) -- the PDF must have
    an even page count, front/back interleaved, same as a student
    scan.

    `blank_gray_front`/`blank_gray_back` (optional) are blank
    reference pages prepared with exam_reader.prepare_blank_reference().
    `glyph_baseline` (optional; ignored if the blank_gray_* pair is
    given) is the output of exam_reader.calibrate_glyph_baseline().
    """

    from exam_reader import pdf_to_images, process_exam_sheet

    pages = pdf_to_images(input_pdf)

    if not pages:
        raise ValueError(f"No pages found in PDF: {input_pdf}")

    if len(pages) % 2 != 0:
        raise ValueError(
            f"{input_pdf} has {len(pages)} page(s) — an odd number. "
            "Exam sheets are always scanned as front/back pairs, so "
            "this PDF is missing a page somewhere (a skipped or "
            "misfed sheet during double-sided scanning)."
        )

    for sheet_number, page_index in enumerate(
        range(0, len(pages), 2),
        start=1,
    ):
        front_page_number = page_index + 1
        back_page_number = page_index + 2

        print(
            f"Processing solution sheet {sheet_number} "
            f"(pages {front_page_number}-{back_page_number})..."
        )

        selections = process_exam_sheet(
            pages[page_index],
            pages[page_index + 1],
            question_count=question_count,
            threshold=threshold,
            gap=gap,
            blank_gray_front=blank_gray_front,
            blank_gray_back=blank_gray_back,
            glyph_baseline=glyph_baseline,
        )

        solution = extract_solution_data(
            selections,
            question_count,
        )

        if solution["missing_metadata"]:
            raise ValueError(
                f"Sheet {sheet_number} (pages {front_page_number}-"
                f"{back_page_number}): could not determine required "
                "metadata: "
                + ", ".join(solution["missing_metadata"])
                + "\n\nReader returned:\n"
                + json.dumps(selections, indent=2, default=str)
            )

        output_path = save_solution_json(
            output_directory=output_directory,
            solution=solution,
            source_pdf=input_pdf.name,
            source_pages=(front_page_number, back_page_number),
        )

        print(f"Saved answer key: {output_path}")
