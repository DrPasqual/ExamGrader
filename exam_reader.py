"""
exam_reader.py

Exam version of scantron_reader.py.

Bubble-level scoring (calculate_bubble_features, the glyph-baseline
math, the uniformity signal, etc.) doesn't know or care what form
it's reading -- it just scores whatever pixel patch and baseline it's
handed. All of that is reused unchanged via `from scantron_reader
import *`.

What DOES change, and is redefined below:

  - create_bubble_groups(): the exam has no quiz-number groups, and
    the version field is named "version_" instead of "quiz_version_"
    (see exam_detector.VERSION_MAP).
  - measure_bubbles(), prepare_blank_reference(): both call
    detector.* functions, and need that name bound to exam_detector
    instead of scantron_detector.
  - calibrate_glyph_baseline(): needs to walk (front, back) PAGE
    PAIRS, since each page has its own bubble map, and a glyph like
    "A" gets samples from both.
  - process_page() becomes process_exam_sheet(): takes a front image
    AND a back image (an exam sheet is always two physical pages),
    and merges both pages' results into one combined selections dict
    -- metadata + questions 1-40 from the front, questions 41-120
    from the back.

determine_all_selections(), create_bubble_groups() (redefined),
determine_group_selection(), extract_bubble_glyph(), and
save/load_glyph_baseline() are all still purely dict/string-based
with no detector dependency, so they work unchanged on the exam's
larger, merged bubble map -- no exam-specific version of those is
needed beyond the create_bubble_groups() field-name update.
"""

import cv2
import numpy as np

import exam_detector as detector
from scantron_reader import (  # noqa: F401 -- re-exported for callers
    calculate_bubble_features,
    extract_bubble_glyph,
    determine_group_selection,
    save_glyph_baseline,
    load_glyph_baseline,
    pdf_to_images,
    print_measurements,
    print_selections,
    extract_simple_answers,
    MEASUREMENT_RADIUS,
    RING_INNER_RADIUS,
    RING_OUTER_RADIUS,
    UNIFORMITY_STD_CEILING,
    UNIFORMITY_STD_RANGE,
    DEFAULT_DARKNESS_THRESHOLD,
    DEFAULT_MINIMUM_SCORE_GAP,
    TEMPLATE_DIFF_MIN_DROP,
    GLYPH_BASELINE_FEATURES,
    DEFAULT_GLYPH_BASELINE_PERCENTILE,
    MIN_GLYPH_BASELINE_SAMPLES,
)

# Exams are graded out of 120 questions by default, not the quiz's 20.
DEFAULT_QUESTION_COUNT = 120


# =====================================================================
# Bubble grouping (field set differs from the quiz: no quiz number,
# "version" instead of "quiz_version")
# =====================================================================

def create_bubble_groups(
    bubble_map: dict[str, tuple[int, int]],
) -> dict[str, list[str]]:
    """
    Group bubbles that represent one logical answer.

    Same idea as the quiz's create_bubble_groups(), minus the
    quiz-number groups (exams don't have one) and with "version"
    instead of "quiz_version" as the field name.
    """

    groups: dict[str, list[str]] = {}

    for bubble_name in bubble_map:
        parts = bubble_name.split("_")

        if bubble_name.startswith("answer_"):
            # answer_041_A -> answer_041 (works the same for 1-3
            # digit question numbers -- the format string always
            # zero-pads to at least 2 digits, and 3-digit numbers
            # like 120 need no padding, so this join is unaffected
            # either way).
            group_name = "_".join(parts[:2])

        elif bubble_name.startswith("student_id_"):
            group_name = "_".join(parts[:3])

        elif bubble_name.startswith("term_"):
            group_name = "term"

        elif bubble_name.startswith("year_first_digit_"):
            group_name = "year_first_digit"

        elif bubble_name.startswith("year_second_digit_"):
            group_name = "year_second_digit"

        elif bubble_name.startswith("version_"):
            group_name = "version"

        else:
            group_name = "other"

        groups.setdefault(group_name, []).append(bubble_name)

    return groups


def determine_all_selections(
    measurements: dict[str, dict],
    bubble_map: dict[str, tuple[int, int]],
    threshold: float,
    minimum_score_gap: float,
    question_count: int = DEFAULT_QUESTION_COUNT,
) -> dict[str, dict]:
    """
    Determine selections for metadata and the requested questions.

    Identical logic to the quiz's determine_all_selections() (the
    question_count cap already works generically for any range, exam
    included) -- redefined here only because it calls
    create_bubble_groups() internally, which must resolve to THIS
    module's version (the exam's field set), not the quiz's.
    """

    groups = create_bubble_groups(bubble_map)
    selections = {}

    for group_name, bubble_names in groups.items():
        if group_name.startswith("answer_"):
            try:
                question_number = int(group_name.split("_")[1])
            except (IndexError, ValueError):
                continue

            if question_number > question_count:
                continue

        selections[group_name] = determine_group_selection(
            group_name,
            bubble_names,
            measurements,
            threshold,
            minimum_score_gap,
        )

    return selections


# =====================================================================
# Bubble measurement (needs the exam_detector binding)
# =====================================================================

def measure_bubbles(
    image: np.ndarray,
    ticks: dict,
    bubble_map: dict[str, tuple[int, int]],
    blank_gray: np.ndarray | None = None,
    glyph_baseline: dict[str, dict[str, float]] | None = None,
) -> dict[str, dict]:
    """Same logic as the quiz's measure_bubbles() -- see there for
    the full explanation of blank_gray/glyph_baseline. Redefined here
    only so its detector.tick_intersection() calls resolve to
    exam_detector."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    measurements = {}

    glyph_covered_bubbles: set[str] | None = None

    if glyph_baseline is not None and blank_gray is None:
        glyph_covered_bubbles = set()

        for bubble_names in create_bubble_groups(bubble_map).values():
            if all(
                extract_bubble_glyph(name) in glyph_baseline
                for name in bubble_names
            ):
                glyph_covered_bubbles.update(bubble_names)

    for bubble_name, location in bubble_map.items():
        column_tick, row_tick = location

        point = detector.tick_intersection(
            ticks,
            column_tick,
            row_tick,
        )

        if point is None:
            measurements[bubble_name] = {
                "point": None,
                "column_tick": column_tick,
                "row_tick": row_tick,
                "mean_intensity": 255.0,
                "darkness_score": 0.0,
                "dark_pixel_fraction": 0.0,
                "medium_dark_fraction": 0.0,
                "center_darkness": 0.0,
                "filled_area_score": 0.0,
                "ring_difference": 0.0,
                "baseline_dark_pixel_fraction": 0.0,
                "baseline_medium_dark_fraction": 0.0,
                "baseline_center_darkness": 0.0,
                "baseline_darkness_score": 0.0,
                "template_diff_fraction": 0.0,
                "interior_std": 0.0,
                "uniformity_fill_score": 0.0,
            }
            continue

        bubble_glyph_baseline = None

        if (
            glyph_baseline is not None
            and glyph_covered_bubbles is not None
            and bubble_name in glyph_covered_bubbles
        ):
            bubble_glyph_baseline = glyph_baseline.get(
                extract_bubble_glyph(bubble_name)
            )

        features = calculate_bubble_features(
            gray,
            point,
            radius=MEASUREMENT_RADIUS,
            blank_gray=blank_gray,
            glyph_baseline=bubble_glyph_baseline,
        )

        measurements[bubble_name] = {
            "point": point,
            "column_tick": column_tick,
            "row_tick": row_tick,
            **features,
        }

    return measurements


def prepare_blank_reference(
    blank_image: np.ndarray,
) -> np.ndarray:
    """
    Rectify a blank reference page to the same fixed canvas every scan
    is warped to, and return it as grayscale. Call once per page
    (front and back are different physical layouts, so calibrate each
    separately) -- see prepare_blank_reference() in scantron_reader.py
    for the full explanation.
    """

    corners = detector.detect_corner_squares(blank_image)
    warped = detector.rectify_image(blank_image, corners)
    return cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)


# =====================================================================
# Glyph baseline calibration (walks front/back PAGE PAIRS)
# =====================================================================

def calibrate_glyph_baseline(
    page_pairs: list[tuple[np.ndarray, np.ndarray]],
    question_count: int = DEFAULT_QUESTION_COUNT,
    threshold: float = DEFAULT_DARKNESS_THRESHOLD,
    gap: float = DEFAULT_MINIMUM_SCORE_GAP,
    percentile: float = DEFAULT_GLYPH_BASELINE_PERCENTILE,
    min_samples: int = MIN_GLYPH_BASELINE_SAMPLES,
) -> dict[str, dict[str, float]]:
    """
    Build an empirical per-glyph fill baseline from a batch of
    already-scanned exam sheets.

    Same method as the quiz's calibrate_glyph_baseline() -- see there
    for the full explanation of the "exclude the selected bubble,
    keep the rest as blank samples" logic -- just applied once per
    page of each (front, back) pair, using that page's own bubble
    map, and pooled into the same glyph-keyed sample store (a glyph
    like "A" appears on both pages and gets samples from each).

    `page_pairs` is a list of (front_image, back_image) tuples --
    each one already-loaded OpenCV BGR image, not yet rectified.
    """

    samples: dict[str, dict[str, list[float]]] = {}

    def accumulate(image, bubble_map, page_question_count):
        corners = detector.detect_corner_squares(image)
        warped = detector.rectify_image(image, corners)
        ticks = detector.detect_timing_marks(warped)

        measurements = measure_bubbles(warped, ticks, bubble_map)

        selections = determine_all_selections(
            measurements,
            bubble_map,
            threshold=threshold,
            minimum_score_gap=gap,
            question_count=page_question_count,
        )

        groups = create_bubble_groups(bubble_map)

        for group_name, bubble_names in groups.items():
            if group_name not in selections:
                continue

            selected = selections[group_name]["selected"]

            for bubble_name in bubble_names:
                if bubble_name == selected:
                    continue

                measurement = measurements.get(bubble_name)

                if measurement is None or measurement.get("point") is None:
                    continue

                glyph = extract_bubble_glyph(bubble_name)
                glyph_samples = samples.setdefault(
                    glyph,
                    {feature: [] for feature in GLYPH_BASELINE_FEATURES},
                )

                for feature in GLYPH_BASELINE_FEATURES:
                    glyph_samples[feature].append(measurement[feature])

    front_map = detector.build_front_bubble_map()
    back_map = detector.build_back_bubble_map()

    for front_image, back_image in page_pairs:
        # The front page's questions only go up to 40 regardless of
        # the exam's total question_count; the cap only matters for
        # the back page's upper bound.
        accumulate(front_image, front_map, min(question_count, 40))
        accumulate(back_image, back_map, question_count)

    baseline: dict[str, dict[str, float]] = {}

    for glyph, glyph_samples in samples.items():
        sample_count = len(glyph_samples[GLYPH_BASELINE_FEATURES[0]])

        if sample_count < min_samples:
            print(
                f"Note: skipping baseline for glyph {glyph!r} — only "
                f"{sample_count} sample(s), need at least "
                f"{min_samples}."
            )
            continue

        baseline[glyph] = {
            feature: float(
                np.percentile(glyph_samples[feature], percentile)
            )
            for feature in GLYPH_BASELINE_FEATURES
        }

        baseline[glyph]["sample_count"] = sample_count

    return baseline


# =====================================================================
# Two-page exam processing
# =====================================================================

def process_exam_sheet(
    front_image: np.ndarray,
    back_image: np.ndarray,
    question_count: int = DEFAULT_QUESTION_COUNT,
    threshold: float = DEFAULT_DARKNESS_THRESHOLD,
    gap: float = DEFAULT_MINIMUM_SCORE_GAP,
    blank_gray_front: np.ndarray | None = None,
    blank_gray_back: np.ndarray | None = None,
    glyph_baseline: dict[str, dict[str, float]] | None = None,
) -> dict[str, dict]:
    """
    Process one exam sheet's two already-loaded page images.

    Both images must already be loaded as OpenCV BGR images, in
    order: `front_image` is page 1 (name/term/year/student
    ID/version/questions 1-40), `back_image` is page 2 (questions
    41-120 only).

    Returns one combined grouped-selections dict covering both pages
    -- exactly like the quiz's process_page(), just assembled from
    two physical sheets instead of one. The reserved key
    "_detection_diagnostics" holds interpolation notes from BOTH
    pages combined.

    `blank_gray_front`/`blank_gray_back` (optional) are blank
    reference pages prepared with prepare_blank_reference() -- pass
    both or neither. `glyph_baseline` (optional; ignored if the
    blank_gray_* pair is given) is the output of
    calibrate_glyph_baseline().
    """

    if not 1 <= question_count <= 120:
        raise ValueError("question_count must be between 1 and 120")

    diagnostics: list[str] = []

    front_corners = detector.detect_corner_squares(
        front_image,
        diagnostics=diagnostics,
    )
    front_warped = detector.rectify_image(front_image, front_corners)
    front_ticks = detector.detect_timing_marks(
        front_warped,
        diagnostics=diagnostics,
    )
    front_map = detector.build_front_bubble_map()

    front_measurements = measure_bubbles(
        front_warped,
        front_ticks,
        front_map,
        blank_gray=blank_gray_front,
        glyph_baseline=glyph_baseline,
    )

    back_corners = detector.detect_corner_squares(
        back_image,
        diagnostics=diagnostics,
    )
    back_warped = detector.rectify_image(back_image, back_corners)
    back_ticks = detector.detect_timing_marks(
        back_warped,
        diagnostics=diagnostics,
    )
    back_map = detector.build_back_bubble_map()

    back_measurements = measure_bubbles(
        back_warped,
        back_ticks,
        back_map,
        blank_gray=blank_gray_back,
        glyph_baseline=glyph_baseline,
    )

    # Bubble names are unique across the two pages (metadata/1-40 only
    # exist on the front, 41-120 only on the back), so the merged
    # measurements and bubble maps can be classified together in one
    # pass -- this is what lets determine_all_selections() /
    # create_bubble_groups() stay completely unchanged for the exam.
    combined_measurements = {**front_measurements, **back_measurements}
    combined_map = {**front_map, **back_map}

    selections = determine_all_selections(
        combined_measurements,
        combined_map,
        threshold=threshold,
        minimum_score_gap=gap,
        question_count=question_count,
    )

    selections["_detection_diagnostics"] = diagnostics

    return selections
