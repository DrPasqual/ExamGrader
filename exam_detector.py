"""
exam_detector.py

Exam version of the scantron detector -- fully standalone (no longer
imports scantron_detector).

The exam answer sheet was redesigned in October 2026: its corner
squares moved, so after rectification its timing marks sit at slightly
different pixel positions than the quiz form's (up to ~3 px), and its
bubble layout is its own (Exam + Version columns, Student WID lower on
the page, no Term/Year). Keeping the exam's reference ticks and bubble
maps here, instead of borrowing the quiz's, means the quiz and exam
forms can each change without breaking the other.

Page 1 (front): exam, version, student WID, questions 1-40.
Page 2 (back):  questions 41-120.

Install:
    pip install opencv-python numpy pymupdf matplotlib

Examples:
    python exam_detector.py front_page.pdf --page 1
    python exam_detector.py back_page.pdf --page 2 --save debug.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
try:
    import pymupdf as fitz  # PyMuPDF 1.24+
except ImportError:  # older PyMuPDF only has the "fitz" name
    import fitz
import matplotlib.pyplot as plt
import numpy as np


# =====================================================================
# Form mappings
# =====================================================================
#
# Layout of the 2026-10 exam sheet (both pages share the quiz form's
# timing-mark grid; every bubble sits exactly on a tick intersection).
# Each value is (vertical_tick, horizontal_tick): column index counted
# from the left corner square (0) and row index from the top corner
# square (0). Page 1 holds exam, version, student WID and questions
# 1-40; page 2 holds questions 41-120.

STUDENT_ID_MAP = {
    position: {
        str(digit): (position + 2, 22 + digit)
        for digit in range(10)
    }
    for position in range(1, 8)
}


# Exam column (1, 2, 3 = midterms, F = final) at column 3, rows 12-15,
# and Version column (A-E, X) at column 5, rows 12-17 -- both vertical,
# side by side above the Student WID block.
EXAM_MAP = {
    "1": (3, 12),
    "2": (3, 13),
    "3": (3, 14),
    "F": (3, 15),
}

VERSION_MAP = {
    "A": (5, 12),
    "B": (5, 13),
    "C": (5, 14),
    "D": (5, 15),
    "E": (5, 16),
    "X": (5, 17),
}

# Two side-by-side answer blocks on the front page: questions 1-20 at
# columns 12-16, questions 21-40 at columns 19-23. Both blocks share
# the same row range (12-31, i.e. row = 11 + question-within-block).
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


# Three side-by-side answer blocks on the back page: 41-70 at columns
# 5-9 (rows 2-31), 71-100 at columns 12-16 (rows 2-31), 101-120 at
# columns 19-23 (rows 2-21 -- only 20 questions). Row numbering starts
# fresh at 2 on this page (no header text/Name/Date block eating up
# the top rows the way page 1's does).
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


# =====================================================================
# Configuration
# =====================================================================

OUTPUT_WIDTH = 1340
OUTPUT_HEIGHT = 1750
PDF_DPI = 200

MIN_CORNER_SIZE = 0.01
MAX_CORNER_SIZE = 0.10

EDGE_BAND_FRACTION = 0.045
PROFILE_DEPTH_FRACTION = 0.018

THRESHOLD_PERCENTILE = 18

MIN_TICK_WIDTH_FRACTION = 0.002
MAX_TICK_WIDTH_FRACTION = 0.035

USE_FIXED_OUTPUT_SIZE = True

# Bubble overlay appearance.
BUBBLE_RADIUS = 14
BUBBLE_COLOR = (0, 0, 255)          # Red, BGR
BUBBLE_OUTLINE_COLOR = (0, 255, 255)  # Yellow, BGR
BUBBLE_LABEL_COLOR = (255, 255, 255)  # White, BGR

DRAW_BUBBLE_LABELS = True


# =====================================================================
# Loading
# =====================================================================

def load_image(
    path: str | Path,
    dpi: int = PDF_DPI,
) -> np.ndarray:
    """Load the first page of a PDF or a normal image file."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".pdf":
        document = fitz.open(str(path))

        if len(document) == 0:
            raise RuntimeError("The PDF contains no pages.")

        page = document[0]
        scale = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)

        image = np.frombuffer(
            pixmap.samples,
            dtype=np.uint8,
        ).reshape(
            pixmap.height,
            pixmap.width,
            pixmap.n,
        )

        if pixmap.n == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(f"Unable to read image: {path}")

    return image


# =====================================================================
# Corner detection
# =====================================================================


# A candidate contour is only accepted as a given corner if it lies
# within this fraction of the image diagonal from that corner's ideal
# position. This stops the "nearest unused candidate" fallback from
# grabbing some unrelated dark shape when the real corner square is
# missing (e.g. a printing defect), which would otherwise silently
# warp the whole page incorrectly instead of raising or interpolating.
MAX_CORNER_MATCH_DISTANCE_FRACTION = 0.12


def interpolate_missing_corner(
    corners: list[np.ndarray | None],
    missing_index: int,
) -> np.ndarray:
    """
    Estimate one missing corner from the other three.

    This assumes the four corners form (approximately) a
    parallelogram, which holds for a flat page under ordinary
    scanning or photographing conditions (mild rotation/skew, no
    extreme keystone distortion). Opposite corners of a parallelogram
    satisfy:

        corner[0] + corner[2] == corner[1] + corner[3]

    so any missing corner equals the sum of its two neighbors minus
    the corner diagonally opposite it.
    """

    opposite_index = (missing_index + 2) % 4
    neighbor_indices = [
        index for index in range(4)
        if index not in (missing_index, opposite_index)
    ]

    return (
        corners[neighbor_indices[0]]
        + corners[neighbor_indices[1]]
        - corners[opposite_index]
    )


def detect_corner_squares(
    image: np.ndarray,
    diagnostics: list[str] | None = None,
) -> np.ndarray:
    """
    Detect four registration squares.

    Returned order:
        0: top-left
        1: top-right
        2: bottom-right
        3: bottom-left

    If exactly one corner square cannot be confidently located (for
    example, a printing or scanning defect obscures it), its position
    is interpolated from the other three instead of failing outright.

    If `diagnostics` is provided, any interpolation note is appended
    to it (in addition to being printed), so a caller can carry the
    note along with the page it came from — e.g. to include in a
    per-student report.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_height, image_width = gray.shape

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        12,
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    min_width = MIN_CORNER_SIZE * image_width
    max_width = MAX_CORNER_SIZE * image_width
    min_height = MIN_CORNER_SIZE * image_height
    max_height = MAX_CORNER_SIZE * image_height

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)

        if not (
            min_width <= width <= max_width
            and min_height <= height <= max_height
        ):
            continue

        aspect_ratio = width / float(height)

        if not 0.65 <= aspect_ratio <= 1.50:
            continue

        area = cv2.contourArea(contour)

        if area < 0.25 * width * height:
            continue

        center = np.array(
            [
                x + width / 2.0,
                y + height / 2.0,
            ],
            dtype=np.float32,
        )

        candidates.append(
            {
                "center": center,
                "area": area,
                "bbox": (x, y, width, height),
            }
        )

    target_points = np.array(
        [
            [0, 0],
            [image_width, 0],
            [image_width, image_height],
            [0, image_height],
        ],
        dtype=np.float32,
    )

    max_match_distance = MAX_CORNER_MATCH_DISTANCE_FRACTION * float(
        np.hypot(image_width, image_height)
    )

    selected: list[np.ndarray | None] = [None, None, None, None]
    used = set()

    for target_index, target in enumerate(target_points):
        ranked = sorted(
            enumerate(candidates),
            key=lambda item: np.linalg.norm(
                item[1]["center"] - target
            ),
        )

        for index, candidate in ranked:
            if index in used:
                continue

            distance = np.linalg.norm(candidate["center"] - target)

            if distance <= max_match_distance:
                used.add(index)
                selected[target_index] = candidate["center"]

            # Only ever consider the single nearest unused candidate
            # for this corner; if it's too far away, treat the corner
            # as missing rather than grabbing a worse match.
            break

    missing_indices = [
        index for index, corner in enumerate(selected)
        if corner is None
    ]

    if len(missing_indices) >= 2:
        raise RuntimeError(
            f"Only confidently located {4 - len(missing_indices)} of 4 "
            "corner squares. At least 3 are needed to interpolate a "
            "missing one."
        )

    if len(missing_indices) == 1:
        missing_index = missing_indices[0]

        corner_labels = ("top-left", "top-right", "bottom-right", "bottom-left")

        message = (
            f"{corner_labels[missing_index]} corner square was not "
            "detected (likely a printing/scanning defect); "
            "interpolated its position from the other three corners."
        )

        print(f"Warning: {message}")

        if diagnostics is not None:
            diagnostics.append(message)

        selected[missing_index] = interpolate_missing_corner(
            selected,
            missing_index,
        )

    return np.asarray(selected, dtype=np.float32)


# =====================================================================
# Perspective correction
# =====================================================================

def rectify_image(
    image: np.ndarray,
    corners: np.ndarray,
) -> np.ndarray:
    """Perspective-correct the page."""

    if USE_FIXED_OUTPUT_SIZE:
        width = OUTPUT_WIDTH
        height = OUTPUT_HEIGHT
    else:
        top_width = np.linalg.norm(corners[1] - corners[0])
        bottom_width = np.linalg.norm(corners[2] - corners[3])
        left_height = np.linalg.norm(corners[3] - corners[0])
        right_height = np.linalg.norm(corners[2] - corners[1])

        width = int(round(max(top_width, bottom_width)))
        height = int(round(max(left_height, right_height)))

    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )

    transform = cv2.getPerspectiveTransform(
        corners,
        destination,
    )

    return cv2.warpPerspective(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


# =====================================================================
# Timing-mark detection
# =====================================================================

def find_dark_regions(
    profile: np.ndarray,
    minimum_width: int,
    maximum_width: int,
) -> list[float]:
    """Return center coordinates of dark regions in a profile."""

    profile = profile.astype(np.float32)

    # Smooth in the profile direction.
    profile = cv2.GaussianBlur(
        profile.reshape(1, -1),
        (9, 1),
        0,
    ).reshape(-1)

    dark_percentile = np.percentile(
        profile,
        THRESHOLD_PERCENTILE,
    )

    light_percentile = np.percentile(profile, 80)

    threshold = dark_percentile + 0.45 * (
        light_percentile - dark_percentile
    )

    is_dark = profile < threshold

    regions = []
    start = None

    for index, dark in enumerate(is_dark):
        if dark and start is None:
            start = index

        at_end = index == len(is_dark) - 1

        if start is not None and (not dark or at_end):
            end = index if not dark else index + 1
            region_width = end - start

            if minimum_width <= region_width <= maximum_width:
                regions.append((start + end - 1) / 2.0)

            start = None

    return regions


# Reference timing-mark positions, in rectified-image pixel
# coordinates, measured once from a clean scan of the blank reference
# answer sheet. Because every scan is perspective-warped to the same
# fixed OUTPUT_WIDTH x OUTPUT_HEIGHT canvas, a given mark's position is
# nearly identical scan to scan — so these references let us recognize
# and reconstruct marks a printing/scanning defect has obscured, no
# matter where on the edge they fall.
#
# If the printed form itself is ever redesigned, regenerate both
# lists by running detect_timing_marks() on a fresh blank scan and
# reading off ticks["top"]/ticks["left"] (rounded to one decimal).
REFERENCE_COLUMN_TICKS = [  # top / bottom, 26 marks (2026-10 sheet)
    8.5, 93.5, 143.5, 193.5, 243.5, 293.5, 343.5, 393.5, 443.5, 493.5,
    543.5, 594.0, 644.5, 694.5, 744.5, 795.0, 845.0, 895.0, 945.0, 994.5,
    1044.5, 1094.5, 1144.5, 1194.5, 1245.0, 1330.0,
]

REFERENCE_ROW_TICKS = [  # left / right, 36 marks (2026-10 sheet)
    8.0, 55.5, 105.5, 155.0, 205.0, 254.5, 304.5, 354.5, 404.0, 453.5,
    503.5, 553.0, 602.5, 652.0, 701.5, 751.0, 800.5, 850.0, 899.5, 949.0,
    998.5, 1047.5, 1097.5, 1146.5, 1196.5, 1246.0, 1295.5, 1345.0, 1394.5, 1444.5,
    1494.0, 1543.5, 1593.0, 1642.5, 1692.5, 1740.5,
]

# How close (as a fraction of the average tick pitch) a detected tick
# must be to a reference slot to be considered a match for it.
COARSE_MATCH_TOLERANCE_FRACTION = 0.6
FINE_MATCH_TOLERANCE_FRACTION = 0.35


def match_points_to_reference(
    points: list[float],
    reference: list[float],
    tolerance: float,
) -> dict[int, float]:
    """
    Greedily assign each detected point to its nearest not-yet-used
    reference slot, provided it's within `tolerance`.

    Matching against absolute reference positions (rather than
    reasoning about detected points relative to each other) is what
    lets this correctly place a run of detected ticks even when the
    missing ones are all at one end — there's no ambiguity about
    "which end is short" because each point is judged against its own
    known expected position, not against its neighbors.
    """

    assignment: dict[int, float] = {}
    used_slots: set[int] = set()

    for point in sorted(points):
        best_index = None
        best_distance = None

        for index, ref_value in enumerate(reference):
            if index in used_slots:
                continue

            distance = abs(point - ref_value)

            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_index = index

        if best_index is not None and best_distance <= tolerance:
            used_slots.add(best_index)
            assignment[best_index] = point

    return assignment


def reconcile_axis_ticks(
    points: list[float],
    reference: list[float],
) -> list[float]:
    """
    Align detected 1D tick coordinates to the fixed reference grid for
    this edge, filling in any slot a printing/scanning defect kept
    from being detected.

    Without this step, a mark missing on only one edge (e.g. top but
    not bottom) silently shifts every later mark's index out of
    alignment with its opposite edge, throwing off every bubble mapped
    past that point.

    Returns a list the same length as `reference`. Slots that matched
    a detected point use that point's (more precise) actual position;
    unmatched slots fall back to the reference position, optionally
    corrected for a small systematic scale/offset learned from
    whichever ticks did match (to absorb minor scan-to-scan variation
    in corner registration).
    """

    expected_count = len(reference)

    if not points or expected_count < 2:
        return list(reference)

    pitch = (reference[-1] - reference[0]) / (expected_count - 1)

    # Coarse pass: generous tolerance, no correction yet.
    coarse_tolerance = COARSE_MATCH_TOLERANCE_FRACTION * pitch
    assignment = match_points_to_reference(points, reference, coarse_tolerance)

    corrected_reference = reference

    if len(assignment) >= 2:
        # Fit a 1D affine correction (scale + offset) from reference
        # space into this scan's coordinate space, using the matched
        # ticks, so a slight overall stretch/shift (e.g. from
        # interpolating a missing corner) doesn't throw off the fine
        # pass below.
        matched_reference = np.array(
            [reference[index] for index in assignment],
            dtype=np.float64,
        )
        matched_points = np.array(
            [assignment[index] for index in assignment],
            dtype=np.float64,
        )

        design = np.vstack(
            [matched_reference, np.ones_like(matched_reference)]
        ).T

        scale, offset = np.linalg.lstsq(
            design,
            matched_points,
            rcond=None,
        )[0]

        corrected_reference = [
            scale * value + offset for value in reference
        ]

    # Fine pass against the (possibly corrected) reference.
    fine_tolerance = FINE_MATCH_TOLERANCE_FRACTION * pitch
    assignment = match_points_to_reference(
        points,
        corrected_reference,
        fine_tolerance,
    )

    return [
        assignment.get(index, corrected_reference[index])
        for index in range(expected_count)
    ]


def detect_timing_marks(
    warped: np.ndarray,
    diagnostics: list[str] | None = None,
) -> dict[str, list[tuple[float, float]]]:
    """
    Detect timing marks on all four edges.

    If `diagnostics` is provided, a note for each edge with any
    interpolated (undetected) marks is appended to it (in addition to
    being printed), so a caller can carry the notes along with the
    page they came from — e.g. to include in a per-student report.
    """

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    edge_band = max(
        5,
        int(min(width, height) * EDGE_BAND_FRACTION),
    )

    profile_depth = max(
        2,
        int(min(width, height) * PROFILE_DEPTH_FRACTION),
    )

    min_horizontal_width = max(
        2,
        int(width * MIN_TICK_WIDTH_FRACTION),
    )

    max_horizontal_width = max(
        min_horizontal_width + 1,
        int(width * MAX_TICK_WIDTH_FRACTION),
    )

    min_vertical_width = max(
        2,
        int(height * MIN_TICK_WIDTH_FRACTION),
    )

    max_vertical_width = max(
        min_vertical_width + 1,
        int(height * MAX_TICK_WIDTH_FRACTION),
    )

    top_profile = gray[:edge_band, :].mean(axis=0)
    bottom_profile = gray[height - edge_band:, :].mean(axis=0)
    left_profile = gray[:, :edge_band].mean(axis=1)
    right_profile = gray[:, width - edge_band:].mean(axis=1)

    top_x = find_dark_regions(
        top_profile,
        min_horizontal_width,
        max_horizontal_width,
    )

    bottom_x = find_dark_regions(
        bottom_profile,
        min_horizontal_width,
        max_horizontal_width,
    )

    left_y = find_dark_regions(
        left_profile,
        min_vertical_width,
        max_vertical_width,
    )

    right_y = find_dark_regions(
        right_profile,
        min_vertical_width,
        max_vertical_width,
    )

    raw_axis_values = {
        "top": (top_x, REFERENCE_COLUMN_TICKS),
        "bottom": (bottom_x, REFERENCE_COLUMN_TICKS),
        "left": (left_y, REFERENCE_ROW_TICKS),
        "right": (right_y, REFERENCE_ROW_TICKS),
    }

    reconciled_axis_values: dict[str, list[float]] = {}

    for side, (raw_values, reference) in raw_axis_values.items():
        reconciled = reconcile_axis_ticks(raw_values, reference)

        interpolated_count = len(reference) - len(raw_values)

        if interpolated_count > 0:
            message = (
                f"interpolated {interpolated_count} of "
                f"{len(reference)} {side} timing mark(s) not directly "
                "detected (likely a printing/scanning defect)."
            )

            print(f"Note: {message}")

            if diagnostics is not None:
                diagnostics.append(message)

        reconciled_axis_values[side] = reconciled

    return {
        "top": [
            (x, float(profile_depth))
            for x in reconciled_axis_values["top"]
        ],
        "bottom": [
            (x, float(height - profile_depth))
            for x in reconciled_axis_values["bottom"]
        ],
        "left": [
            (float(profile_depth), y)
            for y in reconciled_axis_values["left"]
        ],
        "right": [
            (float(width - profile_depth), y)
            for y in reconciled_axis_values["right"]
        ],
    }


# =====================================================================
# Bubble geometry
# =====================================================================

def line_intersection(
    line1_start: tuple[float, float],
    line1_end: tuple[float, float],
    line2_start: tuple[float, float],
    line2_end: tuple[float, float],
) -> tuple[float, float] | None:
    """Return the intersection of two infinite 2D lines."""

    p = np.asarray(line1_start, dtype=np.float64)
    r = np.asarray(line1_end, dtype=np.float64) - p

    q = np.asarray(line2_start, dtype=np.float64)
    s = np.asarray(line2_end, dtype=np.float64) - q

    cross = r[0] * s[1] - r[1] * s[0]

    if abs(cross) < 1e-9:
        return None

    q_minus_p = q - p

    t = (
        q_minus_p[0] * s[1]
        - q_minus_p[1] * s[0]
    ) / cross

    point = p + t * r

    return float(point[0]), float(point[1])


def get_vertical_line(
    ticks: dict[str, list[tuple[float, float]]],
    tick_index: int,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
] | None:
    """Return the top-to-bottom line for a column tick."""

    top_ticks = sorted(
        ticks["top"],
        key=lambda point: point[0],
    )

    bottom_ticks = sorted(
        ticks["bottom"],
        key=lambda point: point[0],
    )

    if not (
        0 <= tick_index < len(top_ticks)
        and 0 <= tick_index < len(bottom_ticks)
    ):
        return None

    return top_ticks[tick_index], bottom_ticks[tick_index]


def get_horizontal_line(
    ticks: dict[str, list[tuple[float, float]]],
    tick_index: int,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
] | None:
    """Return the left-to-right line for a row tick."""

    left_ticks = sorted(
        ticks["left"],
        key=lambda point: point[1],
    )

    right_ticks = sorted(
        ticks["right"],
        key=lambda point: point[1],
    )

    if not (
        0 <= tick_index < len(left_ticks)
        and 0 <= tick_index < len(right_ticks)
    ):
        return None

    return left_ticks[tick_index], right_ticks[tick_index]


def tick_intersection(
    ticks: dict[str, list[tuple[float, float]]],
    column_tick: int,
    row_tick: int,
) -> tuple[float, float] | None:
    """Find the image coordinate for a mapped bubble."""

    vertical_line = get_vertical_line(ticks, column_tick)
    horizontal_line = get_horizontal_line(ticks, row_tick)

    if vertical_line is None:
        print(
            f"Warning: vertical tick {column_tick} "
            "was not detected on both edges."
        )
        return None

    if horizontal_line is None:
        print(
            f"Warning: horizontal tick {row_tick} "
            "was not detected on both edges."
        )
        return None

    return line_intersection(
        vertical_line[0],
        vertical_line[1],
        horizontal_line[0],
        horizontal_line[1],
    )


# =====================================================================
# Mapping collection
# =====================================================================

def build_front_bubble_map() -> dict[str, tuple[int, int]]:
    """
    Bubble map for page 1 of an exam sheet: exam, version, student
    ID, and questions 1-40.

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

    for exam, location in EXAM_MAP.items():
        bubbles[f"exam_{exam}"] = location

    for version, location in VERSION_MAP.items():
        bubbles[f"version_{version}"] = location

    return bubbles


def build_back_bubble_map() -> dict[str, tuple[int, int]]:
    """
    Bubble map for page 2 of an exam sheet: questions 41-120 only.
    No exam/version/student-ID bubbles exist on this page.
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
    directly where the page is already known at the call site.
    """

    if page == 1:
        return build_front_bubble_map()

    if page == 2:
        return build_back_bubble_map()

    raise ValueError(f"page must be 1 or 2, got {page!r}")


def draw_mapped_bubbles(
    image: np.ndarray,
    ticks: dict[str, list[tuple[float, float]]],
    bubble_map: dict[str, tuple[int, int]],
) -> np.ndarray:
    """Draw a marker at every mapped bubble intersection."""

    output = image.copy()
    height, width = output.shape[:2]

    for bubble_name, location in bubble_map.items():
        column_tick, row_tick = location

        point = tick_intersection(
            ticks,
            column_tick,
            row_tick,
        )

        if point is None:
            continue

        x = int(round(point[0]))
        y = int(round(point[1]))

        if not (
            0 <= x < width
            and 0 <= y < height
        ):
            continue

        cv2.circle(
            output,
            (x, y),
            BUBBLE_RADIUS,
            BUBBLE_COLOR,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        cv2.circle(
            output,
            (x, y),
            BUBBLE_RADIUS,
            BUBBLE_OUTLINE_COLOR,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        if DRAW_BUBBLE_LABELS:
            cv2.putText(
                output,
                bubble_name,
                (x + BUBBLE_RADIUS + 3, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                BUBBLE_LABEL_COLOR,
                1,
                cv2.LINE_AA,
            )

    return output


# =====================================================================
# Drawing
# =====================================================================

def draw_debug_overlay(
    warped: np.ndarray,
    corners: np.ndarray,
    ticks: dict[str, list[tuple[float, float]]],
) -> np.ndarray:
    """Draw corners, timing marks, and corresponding lines."""

    debug = warped.copy()

    for index, point in enumerate(corners, start=1):
        x = int(round(point[0]))
        y = int(round(point[1]))

        cv2.circle(
            debug,
            (x, y),
            18,
            (0, 0, 255),
            3,
        )

        cv2.putText(
            debug,
            f"C{index}",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    tick_colors = {
        "top": (255, 0, 255),
        "bottom": (255, 0, 255),
        "left": (0, 255, 255),
        "right": (0, 255, 255),
    }

    for side, points in ticks.items():
        color = tick_colors[side]

        sorted_points = sorted(
            points,
            key=lambda point: point[0]
            if side in ("top", "bottom")
            else point[1],
        )

        for index, (x, y) in enumerate(sorted_points):
            x_int = int(round(x))
            y_int = int(round(y))

            cv2.circle(
                debug,
                (x_int, y_int),
                7,
                color,
                2,
            )

            cv2.putText(
                debug,
                f"{side[0].upper()}{index}",
                (x_int + 8, y_int - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

    top_ticks = sorted(
        ticks["top"],
        key=lambda point: point[0],
    )

    bottom_ticks = sorted(
        ticks["bottom"],
        key=lambda point: point[0],
    )

    for top, bottom in zip(top_ticks, bottom_ticks):
        cv2.line(
            debug,
            (
                int(round(top[0])),
                int(round(top[1])),
            ),
            (
                int(round(bottom[0])),
                int(round(bottom[1])),
            ),
            (255, 80, 0),
            1,
            cv2.LINE_AA,
        )

    left_ticks = sorted(
        ticks["left"],
        key=lambda point: point[1],
    )

    right_ticks = sorted(
        ticks["right"],
        key=lambda point: point[1],
    )

    for left, right in zip(left_ticks, right_ticks):
        cv2.line(
            debug,
            (
                int(round(left[0])),
                int(round(left[1])),
            ),
            (
                int(round(right[0])),
                int(round(right[1])),
            ),
            (80, 180, 255),
            1,
            cv2.LINE_AA,
        )

    return debug


# =====================================================================
# Console output
# =====================================================================

def print_ticks(
    ticks: dict[str, list[tuple[float, float]]],
) -> None:
    """Print detected tick coordinates."""

    print("\nDetected timing marks:")

    for side in ("top", "bottom", "left", "right"):
        print(f"\n{side.upper()} ({len(ticks[side])} marks)")

        for index, (x, y) in enumerate(ticks[side]):
            print(
                f"  {index:3d}: "
                f"x={x:8.2f}, y={y:8.2f}"
            )


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scantron timing-mark and bubble-location debugger."
    )

    parser.add_argument(
        "input",
        help="Image or PDF to inspect.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=PDF_DPI,
        help=f"PDF rendering DPI. Default: {PDF_DPI}",
    )

    parser.add_argument(
        "--save",
        default=None,
        help="Optional path for saving the annotated image.",
    )

    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Do not draw bubble names beside mapped bubbles.",
    )

    parser.add_argument(
        "--page",
        type=int,
        choices=(1, 2),
        default=1,
        help="Which exam page's bubble map to overlay (1=front, 2=back). Default: 1",
    )

    args = parser.parse_args()

    global DRAW_BUBBLE_LABELS
    DRAW_BUBBLE_LABELS = not args.no_labels

    original = load_image(
        args.input,
        args.dpi,
    )

    original_corners = detect_corner_squares(original)

    warped = rectify_image(
        original,
        original_corners,
    )

    ticks = detect_timing_marks(warped)

    print("\nOriginal corner coordinates:")

    for index, (x, y) in enumerate(
        original_corners,
        start=1,
    ):
        print(
            f"  C{index}: "
            f"x={x:.2f}, y={y:.2f}"
        )

    print_ticks(ticks)

    rectified_corners = np.array(
        [
            [0, 0],
            [warped.shape[1] - 1, 0],
            [
                warped.shape[1] - 1,
                warped.shape[0] - 1,
            ],
            [0, warped.shape[0] - 1],
        ],
        dtype=np.float32,
    )

    annotated = draw_debug_overlay(
        warped,
        rectified_corners,
        ticks,
    )

    bubble_map = build_bubble_map(args.page)

    annotated = draw_mapped_bubbles(
        annotated,
        ticks,
        bubble_map,
    )

    mapped_count = 0

    for location in bubble_map.values():
        if tick_intersection(
            ticks,
            location[0],
            location[1],
        ) is not None:
            mapped_count += 1

    print(
        f"\nMapped bubble locations drawn: "
        f"{mapped_count}/{len(bubble_map)}"
    )

    if args.save is not None:
        save_path = Path(args.save)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.imsave(
            str(save_path),
            cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB,
            ),
        )

        print(f"Saved annotated image to: {save_path}")

    plt.figure(figsize=(14, 18))

    plt.imshow(
        cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB,
        )
    )

    plt.title(
        "Detected timing marks and mapped bubble locations"
    )

    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()