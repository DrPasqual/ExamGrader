"""
exam_detector.py

Exam version of scantron_detector.py.

Everything about finding the page's registration squares and timing-
mark grid is UNCHANGED from the quiz form -- confirmed empirically on
real scanned exam sheets (front and back alike): same corner-square
detection, same 26-column x 36-row tick grid, same pixel coordinates
after rectification. So this file imports all of that as-is and only
replaces the bubble map, which encodes what's actually printed at
each tick intersection -- and that IS different: no quiz number, the
version bubble moved columns, and the answer grid is split across two
physically different page layouts (two side-by-side blocks on the
front page, three on the back).

Import this the same way calling code already imports
scantron_detector -- e.g. `import exam_detector as detector` -- and
detect_corner_squares(), rectify_image(), detect_timing_marks(),
tick_intersection(), etc. all work identically. Use
build_front_bubble_map() for page 1 and build_back_bubble_map() for
page 2 instead of the quiz's single build_bubble_map().
"""

from scantron_detector import *  # noqa: F401,F403 -- re-export everything generic (see module docstring)


# =====================================================================
# Front page (page 1) -- term/year, student ID, version, questions 1-40
# =====================================================================

# Term and Student ID are printed in the exact same place as on the
# quiz form -- reused verbatim from scantron_detector.TERM_YEAR_MAP /
# STUDENT_ID_MAP (both already re-exported above via `import *`).

# The version bubble ("Test No." on this form) moved from column 9
# (where the quiz's QUIZ_VERSION_MAP put it) to column 6 -- the slot
# the quiz's quiz-number first digit used to occupy, now free since
# exams have no quiz number.
VERSION_MAP = {
    "A": (6, 11),
    "B": (6, 12),
    "C": (6, 13),
    "D": (6, 14),
    "E": (6, 15),
    "X": (6, 16),
}

# Two side-by-side answer blocks on the front page: questions 1-20 at
# columns 12-16, questions 21-40 at columns 19-23. Both blocks share
# the same row range (12-31, i.e. row = 11 + question-within-block),
# matching the quiz's own row convention -- just doubled up
# column-wise instead of the quiz's single block.
ANSWER_VERTICAL_TICKS_LEFT = {
    "A": 12,
    "B": 13,
    "C": 14,
    "D": 15,
    "E": 16,
}

ANSWER_VERTICAL_TICKS_RIGHT = {
    "A": 19,
    "B": 20,
    "C": 21,
    "D": 22,
    "E": 23,
}

QUESTION_MAP_FRONT = {
    question: {
        choice: (
            ANSWER_VERTICAL_TICKS_LEFT[choice],
            11 + question,
        )
        for choice in ANSWER_VERTICAL_TICKS_LEFT
    }
    for question in range(1, 21)
}

QUESTION_MAP_FRONT.update({
    question: {
        choice: (
            ANSWER_VERTICAL_TICKS_RIGHT[choice],
            11 + (question - 20),
        )
        for choice in ANSWER_VERTICAL_TICKS_RIGHT
    }
    for question in range(21, 41)
})


def build_front_bubble_map() -> dict[str, tuple[int, int]]:
    """
    Bubble map for page 1 of an exam sheet: term/year, student ID,
    version, and questions 1-40.

    Each value is (vertical_tick, horizontal_tick), same convention
    as the quiz's build_bubble_map().
    """

    bubbles = {}

    for question, choices in QUESTION_MAP_FRONT.items():
        for choice, location in choices.items():
            bubbles[f"answer_{question:02d}_{choice}"] = location

    for position, digits in STUDENT_ID_MAP.items():
        for digit, location in digits.items():
            bubbles[f"student_id_{position}_{digit}"] = location

    for term, location in TERM_YEAR_MAP["term"].items():
        bubbles[f"term_{term}"] = location

    for digit, location in TERM_YEAR_MAP["year_first_digit"].items():
        bubbles[f"year_first_digit_{digit}"] = location

    for digit, location in TERM_YEAR_MAP["year_second_digit"].items():
        bubbles[f"year_second_digit_{digit}"] = location

    for version, location in VERSION_MAP.items():
        bubbles[f"version_{version}"] = location

    return bubbles


# =====================================================================
# Back page (page 2) -- questions 41-120 only, no metadata at all
# =====================================================================

# Three side-by-side answer blocks: 41-70 at columns 5-9 (rows 2-31),
# 71-100 at columns 12-16 (rows 2-31), 101-120 at columns 19-23
# (rows 2-21 -- only 20 questions, so 20 rows). Row numbering starts
# fresh at 2 on this page (no header text/Name/Date block eating up
# the top rows the way page 1's does), unlike the front page's
# row-11-based questions.
ANSWER_VERTICAL_TICKS_BACK_1 = {
    "A": 5,
    "B": 6,
    "C": 7,
    "D": 8,
    "E": 9,
}

ANSWER_VERTICAL_TICKS_BACK_2 = {
    "A": 12,
    "B": 13,
    "C": 14,
    "D": 15,
    "E": 16,
}

ANSWER_VERTICAL_TICKS_BACK_3 = {
    "A": 19,
    "B": 20,
    "C": 21,
    "D": 22,
    "E": 23,
}

QUESTION_MAP_BACK = {
    question: {
        choice: (
            ANSWER_VERTICAL_TICKS_BACK_1[choice],
            question - 39,
        )
        for choice in ANSWER_VERTICAL_TICKS_BACK_1
    }
    for question in range(41, 71)
}

QUESTION_MAP_BACK.update({
    question: {
        choice: (
            ANSWER_VERTICAL_TICKS_BACK_2[choice],
            question - 69,
        )
        for choice in ANSWER_VERTICAL_TICKS_BACK_2
    }
    for question in range(71, 101)
})

QUESTION_MAP_BACK.update({
    question: {
        choice: (
            ANSWER_VERTICAL_TICKS_BACK_3[choice],
            question - 99,
        )
        for choice in ANSWER_VERTICAL_TICKS_BACK_3
    }
    for question in range(101, 121)
})


def build_back_bubble_map() -> dict[str, tuple[int, int]]:
    """
    Bubble map for page 2 of an exam sheet: questions 41-120 only.
    No term/year/student-ID/version bubbles exist on this page.
    """

    bubbles = {}

    for question, choices in QUESTION_MAP_BACK.items():
        for choice, location in choices.items():
            bubbles[f"answer_{question:02d}_{choice}"] = location

    return bubbles


def build_bubble_map(page: int) -> dict[str, tuple[int, int]]:
    """
    Convenience dispatcher: build_bubble_map(1) or build_bubble_map(2).

    Prefer calling build_front_bubble_map() / build_back_bubble_map()
    directly where the page is already known at the call site; this
    exists for code that wants to select by page number.
    """

    if page == 1:
        return build_front_bubble_map()

    if page == 2:
        return build_back_bubble_map()

    raise ValueError(f"page must be 1 or 2, got {page!r}")
