"""
canvas_gradebook.py

Reads a Canvas "Grades" CSV export to build a student roster
(student ID -> name), and writes a copy of that same CSV with a quiz
score filled in per student -- ready to re-upload to Canvas.

Canvas's grade-export format (as of this writing) looks like:

    Student,ID,SIS User ID,SIS Login ID,Root Account,Section,<assignment column(s)...>
    ,,,,,,
        Points Possible,,,,,,<points>
    "Last, First",<canvas record id>,<sis id>,<sis id>,<root account>,<section>,<score or blank>
    ...

Two things about this format are easy to get wrong:

  - The "ID" column is Canvas's own internal record ID -- it has
    nothing to do with the school-issued student ID a student bubbles
    in on a scantron sheet. The "SIS User ID" (and identical
    "SIS Login ID") column IS the school-issued ID (e.g. Los Rios
    Community College District's "W1234567" numbers) -- that's the
    one that matches the bubbled ID, once its leading letter is
    stripped.

  - Row 2 (after the header) is a blank spacer row, and row 3 is a
    "Points Possible" row giving each assignment's point value, not a
    student. Both need to be skipped when building a roster or
    matching scores, but preserved as-is when writing the CSV back
    out.

Both entry points here work by column NAME (via csv.DictReader), not
column position, so a reordered or Canvas-updated export still works
as long as the column names are the same.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any


# Canvas has called this column both of these things across versions;
# check both. Whichever is present holds the school-issued ID (e.g.
# "w2010382") that matches what's bubbled on the scantron sheet.
ID_COLUMN_CANDIDATES = ("SIS User ID", "SIS Login ID")

NAME_COLUMN = "Student"

DEFAULT_QUIZ_COLUMN = "Quiz"


def normalize_student_id(value: Any) -> str:
    """
    Keep digits only and remove leading zeroes.

    Same normalization the grader's own normalize_student_id() uses,
    so IDs match regardless of a leading school-ID letter (e.g. Los
    Rios's "W" prefix) or leading zeroes on either side.
    """

    if value is None:
        return ""

    digits = re.sub(r"\D", "", str(value))

    if not digits:
        return ""

    return digits.lstrip("0") or "0"


def _read_csv_rows(
    csv_path: str | Path,
) -> tuple[list[str], list[dict[str, str]]]:
    """Read a Canvas CSV, returning (fieldnames, rows-as-dicts)."""

    csv_path = Path(csv_path)

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    return fieldnames, rows


def _find_id_column(fieldnames: list[str]) -> str:
    for candidate in ID_COLUMN_CANDIDATES:
        if candidate in fieldnames:
            return candidate

    raise ValueError(
        "Could not find a student-ID column in the Canvas CSV "
        f"(looked for {' / '.join(ID_COLUMN_CANDIDATES)}); "
        f"found columns: {', '.join(fieldnames)}"
    )


def _is_student_row(row: dict[str, str], id_column: str) -> bool:
    """
    True for an actual student row -- false for the blank spacer row
    and the "Points Possible" row Canvas puts at the top of the
    export.
    """

    name = (row.get(NAME_COLUMN) or "").strip()

    if not name:
        return False

    if name.casefold().startswith("points possible"):
        return False

    if not (row.get(id_column) or "").strip():
        return False

    return True


def load_roster_from_csv(
    csv_path: str | Path,
) -> dict[str, str]:
    """
    Load a Canvas grade-export CSV as a roster: {normalized_id: name}.

    This is a drop-in replacement for the grader's own
    load_student_roster() -- same return shape -- just reading a
    Canvas CSV export instead of a plain tab/space-delimited text
    file.
    """

    fieldnames, rows = _read_csv_rows(csv_path)
    id_column = _find_id_column(fieldnames)

    roster: dict[str, str] = {}

    for row in rows:
        if not _is_student_row(row, id_column):
            continue

        student_id = normalize_student_id(row[id_column])

        if not student_id:
            continue

        roster[student_id] = (row.get(NAME_COLUMN) or "").strip()

    print(f"Loaded {len(roster)} student(s) from Canvas CSV roster.")

    return roster


def write_grades_csv(
    source_csv_path: str | Path,
    output_csv_path: str | Path,
    scores: dict[str, Any],
    quiz_column: str = DEFAULT_QUIZ_COLUMN,
    zero_fill_ungraded: bool = True,
) -> Path:
    """
    Write a copy of the Canvas CSV with `quiz_column` populated for
    every student (keyed by normalized student ID -- see
    normalize_student_id() / load_roster_from_csv()).

    A student found in `scores` gets that score. A student NOT in
    `scores` -- never scanned, or scanned but not gradable (e.g. an
    unreadable quiz number/version, or an incomplete ID) -- gets "0"
    when `zero_fill_ungraded` is True (the default), so a student who
    didn't turn in the quiz doesn't just look "not yet entered" in
    Canvas. Pass `zero_fill_ungraded=False` to instead leave that
    cell exactly as it already was in the source CSV (typically
    blank).

    Every other row and column -- header, the blank spacer row, and
    the "Points Possible" row -- is always copied through unchanged.
    """

    source_csv_path = Path(source_csv_path)
    output_csv_path = Path(output_csv_path)

    fieldnames, rows = _read_csv_rows(source_csv_path)

    if quiz_column not in fieldnames:
        raise ValueError(
            f"Column {quiz_column!r} not found in {source_csv_path}; "
            f"found columns: {', '.join(fieldnames)}"
        )

    id_column = _find_id_column(fieldnames)

    matched_count = 0
    zero_filled_count = 0

    for row in rows:
        if not _is_student_row(row, id_column):
            continue

        student_id = normalize_student_id(row[id_column])
        score = scores.get(student_id)

        if score is not None:
            row[quiz_column] = str(score)
            matched_count += 1
        elif zero_fill_ungraded:
            row[quiz_column] = "0"
            zero_filled_count += 1

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with output_csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Wrote grades CSV: {output_csv_path} "
        f"({matched_count} of {len(scores)} score(s) matched to a "
        f"row; {zero_filled_count} ungraded student(s) set to 0)"
    )

    return output_csv_path