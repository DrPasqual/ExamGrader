"""
exam_reader.py

Exam version of the scantron reader -- fully standalone, no
dependency on any of the quiz grading files.

Reads bubble darkness from an exam sheet's two pages using
exam_detector.py, merges the two pages' results into one combined
per-student record, and builds/uses the glyph baseline (walking
front/back page pairs). Needs no quiz files (scantron_reader.py).

Expected files:
    exam_detector.py
    exam_reader.py

Install:
    pip install opencv-python numpy pymupdf matplotlib

Examples:
    python exam_reader.py front.pdf --page 1
    python exam_reader.py back.pdf --page 2 --threshold 0.25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

try:
    import pymupdf as fitz  # PyMuPDF 1.24+
except ImportError:  # older PyMuPDF only has the "fitz" name
    import fitz

import exam_detector as detector


# =====================================================================
# Configuration
# =====================================================================

# Radius used to measure the inside of each bubble.
# This should be smaller than the visible bubble outline.
MEASUREMENT_RADIUS = 10

# Ignore the outer edge of each bubble when calculating darkness.
# This prevents the printed circle outline from affecting the score.
INNER_RADIUS_FRACTION = 0.65

# Absolute pixel radii (on the fixed rectified canvas) for the ring
# used to estimate local background/scan brightness. Measured
# directly from the printed form: the bubble outline's true radius is
# about 16-17px (larger than MEASUREMENT_RADIUS), and the nearest
# table border line — in the tightest case, the outermost Student ID
# columns — can be as close as ~22-25px out. This ring sits in the
# gap between the two.
RING_INNER_RADIUS = 19
RING_OUTER_RADIUS = 22

# A bubble is considered selected if its darkness score is greater than
# this threshold. The score ranges approximately from 0.0 to 1.0.
DEFAULT_DARKNESS_THRESHOLD = 0.25

# A selected bubble must also be sufficiently darker than the next
# darkest bubble in the same group.
DEFAULT_MINIMUM_SCORE_GAP = 0.05

# When comparing against a blank reference sheet, a pixel counts as
# "newly filled" if it's at least this many intensity levels (0-255)
# darker than the SAME pixel on the blank sheet. Set comfortably above
# normal scan/print noise so faint scanner artifacts don't register as
# fill, but well below the contrast a genuine pencil/pen mark produces.
TEMPLATE_DIFF_MIN_DROP = 35

# Internal-uniformity signal (see calculate_bubble_features): a
# bubble's own interior standard deviation, independent of any
# calibration or reference sheet. Measured from real scans, blank
# bubbles cluster around std 26-35 and filled ones around std 9-25.
# uniformity_fill_score = clip((CEILING - std) / RANGE, 0, 1), so a
# bubble at or above the ceiling scores 0 and one RANGE below it
# scores 1. Re-measure if the printed form's glyph weight or a
# different pencil/pen style shifts these clusters.
UNIFORMITY_STD_CEILING = 27.0
UNIFORMITY_STD_RANGE = 20.0

# Draw measurement circles on the output image.
DRAW_MEASUREMENT_OVERLAY = True

MEASUREMENT_COLOR = (210, 190, 150)  # Muted blue-gray
SELECTED_COLOR = (0, 0, 220)         # Darker red for selected bubbles
UNSELECTED_COLOR = (190, 185, 175)   # Warm gray


# =====================================================================
# Bubble measurement
# =====================================================================

def make_circular_mask(
    image_shape: tuple[int, int],
    center: tuple[int, int],
    radius: int,
) -> np.ndarray:
    """Create a circular Boolean mask around a bubble center."""

    height, width = image_shape[:2]

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    x, y = center

    cv2.circle(
        mask,
        (int(x), int(y)),
        int(radius),
        255,
        thickness=-1,
    )

    return mask.astype(bool)


def calculate_bubble_features(
    gray: np.ndarray,
    center: tuple[float, float],
    radius: int = MEASUREMENT_RADIUS,
    blank_gray: np.ndarray | None = None,
    glyph_baseline: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Extract features that help distinguish printed letters from filled bubbles.

    A filled bubble should have:
        - high average darkness
        - many dark pixels
        - a dark center
        - a large dark-area fraction

    A printed letter usually has:
        - moderate average darkness
        - fewer dark pixels
        - a relatively small dark-area fraction

    Two independent ways to cancel out the printed letter/number's own
    ink (use at most one; `blank_gray` takes priority if both given):

    `blank_gray` — the same-size, rectified grayscale image of a blank
    reference sheet. Every bubble is compared pixel-by-pixel against
    the same location on it. Most precise, but the reference needs to
    come from the same kind of scan (not a computer-rendered PDF) or
    scanner noise/print differences leave residual mismatch right at
    the glyph's edges.

    `glyph_baseline` — a dict with "dark_pixel_fraction",
    "medium_dark_fraction", and "center_darkness" keys, the typical
    (already-scanned) unfilled level for THIS bubble's specific letter
    or digit, from calibrate_glyph_baseline(). Less precise than a
    pixel-aligned template (it's a statistical average, not this exact
    bubble), but needs no reference sheet, isn't thrown off by small
    registration differences, and is built from real scans on your own
    printer/scanner — including their actual noise characteristics.

    Either way, this is what lets a lightly filled bubble (which only
    needs to look darker than an unfilled one, not darker than some
    fixed brightness level) score clearly higher than an unfilled
    bubble printed with a naturally heavy glyph like "8" or "B".
    """

    x = int(round(center[0]))
    y = int(round(center[1]))

    height, width = gray.shape[:2]

    empty_result = {
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

    # Avoid invalid regions near image boundaries.
    if (
        x - radius < 0
        or x + radius >= width
        or y - radius < 0
        or y + radius >= height
    ):
        return empty_result

    yy, xx = np.ogrid[:height, :width]

    distance_squared = (xx - x) ** 2 + (yy - y) ** 2

    # Main bubble interior.
    interior_radius = int(radius * 0.70)
    interior_mask = distance_squared <= interior_radius ** 2

    # Smaller center region. This is useful because a fill usually
    # darkens the center, while a printed letter may not.
    center_radius = int(radius * 0.35)
    center_mask = distance_squared <= center_radius ** 2

    # Ring outside the bubble interior. This estimates local scan
    # brightness and compensates for shadows or uneven illumination.
    #
    # These are ABSOLUTE pixel radii, not scaled from `radius`
    # (MEASUREMENT_RADIUS). The printed bubble outline on this form's
    # fixed-size rectified canvas has a true radius of about 16-17px —
    # noticeably larger than MEASUREMENT_RADIUS (10) — so a ring
    # scaled from that (as this used to be: 0.9x-1.2x) sits INSIDE the
    # printed circle rather than outside it. For an evenly, heavily
    # filled bubble that's harmless (inside is uniformly dark either
    # way), but for a light or uneven pencil mark it means the "local
    # background" sample can land on the mark itself rather than
    # clean surrounding paper, corrupting the estimate. RING_INNER
    # clears the true outline with a couple pixels of margin;
    # RING_OUTER stays short of the nearest table border line, which
    # in the densest part of the grid (the outermost Student ID
    # columns) can be as close as ~22-25px out.
    ring_inner_radius = RING_INNER_RADIUS
    ring_outer_radius = RING_OUTER_RADIUS

    ring_mask = (
        (distance_squared >= ring_inner_radius ** 2)
        & (distance_squared <= ring_outer_radius ** 2)
    )

    interior_pixels = gray[interior_mask].astype(np.float32)
    center_pixels = gray[center_mask].astype(np.float32)
    ring_pixels = gray[ring_mask].astype(np.float32)

    if interior_pixels.size == 0:
        return empty_result

    mean_intensity = float(np.mean(interior_pixels))

    # Local brightness correction.
    if ring_pixels.size:
        local_background = float(np.median(ring_pixels))
    else:
        local_background = 255.0

    # Difference from the surrounding paper.
    local_darkness = max(
        0.0,
        (local_background - mean_intensity) / 255.0,
    )

    # Dark-pixel fractions.
    dark_pixel_fraction = float(
        np.mean(interior_pixels < 125)
    )

    medium_dark_fraction = float(
        np.mean(interior_pixels < 175)
    )

    # Internal-uniformity signal: independent of any baseline or
    # reference sheet, computed purely from THIS bubble's own pixels.
    # A pencil/pen fill tends to cover the interior fairly evenly,
    # pulling its standard deviation down; an unfilled bubble's
    # interior is mostly bare white paper plus a few bold, high-
    # contrast printed glyph strokes, which pushes its standard
    # deviation up. Measured across 1,111 real blank and 167 real
    # filled bubbles: blanks cluster around std 26-35, fills around
    # std 9-25 — the two barely overlap. This is what lets an
    # otherwise-uncalibrated glyph (no glyph_baseline entry yet) still
    # get a meaningful, independent signal instead of falling back to
    # the old absolute-threshold formula alone.
    interior_std = float(np.std(interior_pixels))

    uniformity_fill_score = float(
        np.clip(
            (UNIFORMITY_STD_CEILING - interior_std)
            / UNIFORMITY_STD_RANGE,
            0.0,
            1.0,
        )
    )

    # Center darkness, also normalized to the local background.
    center_mean = float(np.mean(center_pixels))

    center_darkness = max(
        0.0,
        (local_background - center_mean) / 255.0,
    )

    baseline_dark_pixel_fraction = 0.0
    baseline_medium_dark_fraction = 0.0
    baseline_center_darkness = 0.0
    baseline_darkness_score = 0.0
    template_diff_fraction = 0.0

    if blank_gray is not None:
        blank_interior_pixels = blank_gray[interior_mask].astype(np.float32)
        blank_center_pixels = blank_gray[center_mask].astype(np.float32)
        blank_ring_pixels = blank_gray[ring_mask].astype(np.float32)

        if blank_interior_pixels.size and blank_interior_pixels.shape == interior_pixels.shape:
            blank_local_background = (
                float(np.median(blank_ring_pixels))
                if blank_ring_pixels.size else 255.0
            )

            baseline_dark_pixel_fraction = float(
                np.mean(blank_interior_pixels < 125)
            )

            baseline_medium_dark_fraction = float(
                np.mean(blank_interior_pixels < 175)
            )

            blank_center_mean = float(np.mean(blank_center_pixels))

            baseline_center_darkness = max(
                0.0,
                (blank_local_background - blank_center_mean) / 255.0,
            )

            # This scan's local background may be globally lighter or
            # darker than the blank reference's (different scanner,
            # exposure, lighting) even though nothing is actually
            # filled. Correct for that shift — using the SAME
            # ring-based background estimate already computed above —
            # before comparing pixel-by-pixel, the same way the
            # existing darkness_score is normalized to its own local
            # background rather than an absolute level.
            background_offset = local_background - blank_local_background
            corrected_interior_pixels = interior_pixels - background_offset

            # Pixel-level template diff: how much darker is this
            # scan's interior than the SAME pixels on the blank
            # sheet, after correcting for any overall scan brightness
            # difference? This cancels the printed glyph almost
            # exactly (it's the same ink in the same spot on an
            # unfilled bubble) and picks out genuinely new fill,
            # however light, since every fill pixel darkens relative
            # to its own unfilled counterpart.
            pixel_diff = blank_interior_pixels - corrected_interior_pixels

            template_diff_fraction = float(
                np.mean(pixel_diff > TEMPLATE_DIFF_MIN_DROP)
            )

    elif glyph_baseline is not None:
        # No reference image — use the pre-computed statistical
        # baseline for this bubble's specific letter/digit instead.
        # These values already come from real scans (same printer and
        # scanner), so no per-image background correction is applied
        # here the way it is for blank_gray above; calibrate against a
        # representative batch and re-calibrate if the scanner setup
        # ever changes.
        baseline_dark_pixel_fraction = glyph_baseline.get(
            "dark_pixel_fraction", 0.0
        )
        baseline_medium_dark_fraction = glyph_baseline.get(
            "medium_dark_fraction", 0.0
        )
        baseline_center_darkness = glyph_baseline.get(
            "center_darkness", 0.0
        )
        baseline_darkness_score = glyph_baseline.get(
            "darkness_score", 0.0
        )

    # Baseline-adjusted scalar features (clipped at zero): how much
    # MORE ink is present than the printed glyph alone accounts for.
    adjusted_dark_pixel_fraction = max(
        0.0,
        dark_pixel_fraction - baseline_dark_pixel_fraction,
    )

    adjusted_medium_dark_fraction = max(
        0.0,
        medium_dark_fraction - baseline_medium_dark_fraction,
    )

    adjusted_center_darkness = max(
        0.0,
        center_darkness - baseline_center_darkness,
    )

    adjusted_darkness_score = max(
        0.0,
        local_darkness - baseline_darkness_score,
    )

    if blank_gray is not None:
        # Template diff is the primary, most discriminative signal
        # when a pixel-aligned blank reference is available: it
        # directly measures added ink at the pixel level rather than
        # a scalar average, so it separates a light fill from a heavy
        # printed glyph much better than absolute-intensity
        # thresholds can.
        filled_area_score = (
            0.60 * template_diff_fraction
            + 0.25 * adjusted_medium_dark_fraction
            + 0.15 * adjusted_center_darkness
        )
    elif glyph_baseline is not None:
        # No pixel-level diff available in this mode, so the
        # continuous adjusted_darkness_score (not a hard threshold
        # crossing) carries most of the weight — it's far more
        # sensitive to a light, even fill than the two threshold-based
        # fractions are.
        filled_area_score = (
            0.55 * adjusted_darkness_score
            + 0.25 * adjusted_dark_pixel_fraction
            + 0.15 * adjusted_medium_dark_fraction
            + 0.05 * adjusted_center_darkness
        )
    else:
        # No baseline available at all — including a glyph this
        # page's group wasn't fully calibrated for (e.g. "F" if it's
        # always the selected term and never appears as a blank
        # sample). uniformity_fill_score needs no calibration at all
        # and, measured across 1,111 real blank and 167 real filled
        # bubbles, is dramatically more discriminative here than the
        # original absolute-threshold features alone — so it carries
        # the most weight, with the rest as a secondary contribution.
        #
        # It is NOT blended into the blank_gray/glyph_baseline modes
        # above: it has its own glyph-dependent bias (a thin glyph
        # like "1" naturally has lower interior variance even
        # unfilled, versus a bold one like "8"), and unlike the other
        # features that bias isn't corrected by the existing
        # calibration data — so adding it there would introduce a new
        # uncorrected error rather than a genuine improvement.
        filled_area_score = (
            0.40 * uniformity_fill_score
            + 0.30 * dark_pixel_fraction
            + 0.20 * medium_dark_fraction
            + 0.10 * center_darkness
        )

    return {
        "mean_intensity": mean_intensity,
        "darkness_score": local_darkness,
        "dark_pixel_fraction": dark_pixel_fraction,
        "medium_dark_fraction": medium_dark_fraction,
        "center_darkness": center_darkness,
        "filled_area_score": filled_area_score,
        "ring_difference": local_darkness,
        "baseline_dark_pixel_fraction": baseline_dark_pixel_fraction,
        "baseline_medium_dark_fraction": baseline_medium_dark_fraction,
        "baseline_center_darkness": baseline_center_darkness,
        "baseline_darkness_score": baseline_darkness_score,
        "template_diff_fraction": template_diff_fraction,
        "interior_std": interior_std,
        "uniformity_fill_score": uniformity_fill_score,
    }


def extract_bubble_glyph(bubble_name: str) -> str:
    """
    Get the printed letter or digit inside a bubble from its name.

    The same glyph appears in many different bubbles across the sheet
    (e.g. the letter "C" is printed inside answer_01_C, answer_02_C,
    ..., version_C), and every bubble name's final "_"-separated
    part is exactly that glyph:

        answer_071_C            -> "C"
        student_id_4_8          -> "8"
        exam_F                  -> "F"
        version_X                -> "X"
    """

    return bubble_name.rsplit("_", 1)[-1]


def measure_bubbles(
    image: np.ndarray,
    ticks: dict,
    bubble_map: dict[str, tuple[int, int]],
    blank_gray: np.ndarray | None = None,
    glyph_baseline: dict[str, dict[str, float]] | None = None,
) -> dict[str, dict]:
    """
    Calculate multiple features for every mapped bubble.

    `blank_gray` (optional) is the rectified grayscale image of a
    blank reference answer sheet, same size as `image`'s rectified
    canvas. When given, each bubble's printed letter/number is
    cancelled out by comparison against that same location on the
    blank sheet — see calculate_bubble_features().

    `glyph_baseline` (optional; ignored if `blank_gray` is given) is
    the output of calibrate_glyph_baseline(): a statistical baseline
    per printed letter/digit, built from real scans, needing no
    reference sheet at all.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    measurements = {}

    # A competing group (e.g. one question's 5 choices, or one
    # Student ID digit's 10 choices) must not mix a calibrated glyph
    # against an uncalibrated one: the uncalibrated bubble falls back
    # to the old scoring, which sits on a different (much higher)
    # floor than the calibrated scale, and can spuriously outscore a
    # real, lightly-filled, calibrated bubble sitting right next to
    # it. So the baseline is only used for a bubble if EVERY bubble in
    # its group has one; otherwise the whole group scores uniformly on
    # the old scale instead, preserving a fair, consistent comparison.
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



# =====================================================================
# Bubble grouping
# =====================================================================

def create_bubble_groups(
    bubble_map: dict[str, tuple[int, int]],
) -> dict[str, list[str]]:
    """
    Group bubbles that represent one logical answer.

    Each group contains mutually exclusive bubbles.

    Examples:
        answer_01:
            answer_01_A through answer_01_E

        student_id_1:
            student_id_1_0 through student_id_1_9
    """

    groups: dict[str, list[str]] = {}

    for bubble_name in bubble_map:
        parts = bubble_name.split("_")

        if bubble_name.startswith("answer_"):
            # answer_041_A -> answer_041 (works the same for 1-3
            # digit question numbers).
            group_name = "_".join(parts[:2])

        elif bubble_name.startswith("student_id_"):
            # student_id_1_0 -> student_id_1
            group_name = "_".join(parts[:3])

        elif bubble_name.startswith("version_"):
            group_name = "version"

        elif bubble_name.startswith("exam_"):
            group_name = "exam"

        else:
            group_name = "other"

        groups.setdefault(group_name, []).append(bubble_name)

    return groups


def determine_group_selection(
    group_name: str,
    bubble_names: list[str],
    measurements: dict[str, dict],
    threshold: float,
    minimum_score_gap: float,
) -> dict:
    """Select a bubble using its filled-area score."""

    ranked = sorted(
        bubble_names,
        key=lambda name: measurements[name]["filled_area_score"],
        reverse=True,
    )

    if not ranked:
        return {
            "group": group_name,
            "selected": None,
            "status": "empty",
            "scores": {},
        }

    best_name = ranked[0]

    best_score = measurements[
        best_name
    ]["filled_area_score"]

    second_score = 0.0

    if len(ranked) > 1:
        second_score = measurements[
            ranked[1]
        ]["filled_area_score"]

    score_gap = best_score - second_score

    if best_score < threshold:
        status = "blank"
        selected = None

    elif score_gap < minimum_score_gap:
        status = "ambiguous"
        selected = None

    else:
        status = "selected"
        selected = best_name

    return {
        "group": group_name,
        "selected": selected,
        "status": status,
        "best_score": best_score,
        "second_score": second_score,
        "score_gap": score_gap,
        "scores": {
            name: measurements[name]["filled_area_score"]
            for name in ranked
        },
    }



def determine_all_selections(
    measurements: dict[str, dict],
    bubble_map: dict[str, tuple[int, int]],
    threshold: float,
    minimum_score_gap: float,
    question_count: int = 120,
) -> dict[str, dict]:
    """Determine selections for metadata and the requested questions."""

    groups = create_bubble_groups(bubble_map)
    selections = {}

    for group_name, bubble_names in groups.items():
        # Only include answer groups for questions 1 through question_count.
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
# Reporting
# =====================================================================

def print_measurements(measurements: dict[str, dict]) -> None:
    print("\nBubble measurements:")

    for name in sorted(measurements):
        measurement = measurements[name]

        print(
            f"{name:40s} "
            f"area={measurement['filled_area_score']:.3f} "
            f"dark={measurement['darkness_score']:.3f} "
            f"dark_px={measurement['dark_pixel_fraction']:.3f} "
            f"medium_px={measurement['medium_dark_fraction']:.3f} "
            f"center={measurement['center_darkness']:.3f}"
        )



def print_selections(
    selections: dict[str, dict],
) -> None:
    """Print selected values grouped by category."""

    print("\nDetected selections:")

    for group_name in sorted(selections):
        if group_name.startswith("_"):
            # Reserved metadata (e.g. "_detection_diagnostics"), not a
            # real bubble group.
            continue

        result = selections[group_name]

        selected = result["selected"]
        status = result["status"]

        if selected is None:
            selected_text = "NONE"
        else:
            selected_text = selected

        print(
            f"{group_name:30s} "
            f"{status:10s} "
            f"{selected_text}"
        )


def extract_simple_answers(
    selections: dict[str, dict],
) -> dict[str, str | None]:
    """
    Convert internal bubble names into simpler values.

    For example:
        answer_01_A -> answer_01 = A
        student_id_1_7 -> student_id_1 = 7
    """

    output = {}

    for group_name, result in selections.items():
        if group_name.startswith("_"):
            # Reserved metadata (e.g. "_detection_diagnostics"), not a
            # real bubble group.
            continue

        selected = result["selected"]

        if selected is None:
            output[group_name] = None
            continue

        prefix = group_name + "_"

        if selected.startswith(prefix):
            output[group_name] = selected[len(prefix):]
        else:
            output[group_name] = selected

    return output


# =====================================================================
# Visualization
# =====================================================================

def draw_measurement_overlay(
    image: np.ndarray,
    measurements: dict[str, dict],
    selections: dict[str, dict],
) -> np.ndarray:
    """Draw measured bubble centers and selected bubbles."""

    output = image.copy()

    selected_names = {
        result["selected"]
        for result in selections.values()
        if result["selected"] is not None
    }

    for bubble_name, measurement in measurements.items():
        point = measurement["point"]

        if point is None:
            continue

        x = int(round(point[0]))
        y = int(round(point[1]))

        if bubble_name in selected_names:
            color = SELECTED_COLOR
            thickness = 4
        else:
            color = MEASUREMENT_COLOR
            thickness = 2

        cv2.circle(
            output,
            (x, y),
            MEASUREMENT_RADIUS,
            color,
            thickness,
            cv2.LINE_AA,
        )

        score_text = (
            f"{measurement['darkness_score']:.2f}"
        )

        cv2.putText(
            output,
            score_text,
            (x - 12, y - MEASUREMENT_RADIUS - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )

    return output

def pdf_to_images(pdf_path):
    """Convert every PDF page into an OpenCV BGR image."""

    document = fitz.open(str(pdf_path))
    images = []

    for page in document:
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)

        image = np.frombuffer(
            pixmap.samples,
            dtype=np.uint8,
        ).reshape(
            pixmap.height,
            pixmap.width,
            pixmap.n,
        )

        if pixmap.n == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        images.append(image)

    document.close()
    return images

def prepare_blank_reference(
    blank_image: np.ndarray,
) -> np.ndarray:
    """
    Rectify a blank reference answer sheet to the same fixed canvas
    every scan is warped to, and return it as grayscale.

    Compute this once per grading run (not once per page) and pass
    the result as `blank_gray` to process_page()/measure_bubbles() for
    every student page, so each bubble's printed letter/number can be
    cancelled out by comparison against the same pixel location here.
    """

    corners = detector.detect_corner_squares(blank_image)
    warped = detector.rectify_image(blank_image, corners)
    return cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)


# Features aggregated per glyph during calibration, and later
# subtracted per bubble during scoring. "darkness_score" is included
# because, unlike the two threshold-based fractions, it's continuous
# rather than a hard cutoff — much more sensitive to a subtle, even
# fill that doesn't push many pixels across an absolute brightness
# line.
GLYPH_BASELINE_FEATURES = (
    "dark_pixel_fraction",
    "medium_dark_fraction",
    "center_darkness",
    "darkness_score",
)

# Conservative (high) percentile used to summarize each glyph's
# "blank" sample distribution — safer than the mean/median, since a
# few accidentally-included faint real fills shouldn't drag the whole
# glyph's baseline down and make genuine light fills harder to catch.
DEFAULT_GLYPH_BASELINE_PERCENTILE = 90.0

# Skip building a baseline for a glyph with too few real samples to
# be a trustworthy statistic.
MIN_GLYPH_BASELINE_SAMPLES = 20


def calibrate_glyph_baseline(
    page_pairs: list[tuple[np.ndarray, np.ndarray]],
    question_count: int = 120,
    threshold: float = DEFAULT_DARKNESS_THRESHOLD,
    gap: float = DEFAULT_MINIMUM_SCORE_GAP,
    percentile: float = DEFAULT_GLYPH_BASELINE_PERCENTILE,
    min_samples: int = MIN_GLYPH_BASELINE_SAMPLES,
) -> dict[str, dict[str, float]]:
    """
    Build an empirical per-glyph fill baseline from a batch of
    already-scanned exam sheets — no blank reference sheet needed.

    Same method as the quiz reader's calibrate_glyph_baseline() (see
    there for the full "exclude the selected bubble, keep the rest as
    blank samples" reasoning), just applied once per page of each
    (front, back) pair, using that page's own bubble map, and pooled
    into the same glyph-keyed sample store — a glyph like "A" appears
    on both pages and gets samples from each.

    `page_pairs` is a list of (front_image, back_image) tuples — each
    one an already-loaded OpenCV BGR image, not yet rectified.

    Returns {glyph: {"dark_pixel_fraction": ..., "medium_dark_fraction":
    ..., "center_darkness": ..., "darkness_score": ..., "sample_count":
    N}}. A glyph with fewer than `min_samples` samples is omitted.
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


def save_glyph_baseline(
    path: str | Path,
    baseline: dict[str, dict[str, float]],
) -> Path:
    """Save a calibrated glyph baseline to a JSON file."""

    path = Path(path)

    with path.open("w", encoding="utf-8") as file:
        json.dump(baseline, file, indent=2, sort_keys=True)
        file.write("\n")

    return path


def load_glyph_baseline(
    path: str | Path,
) -> dict[str, dict[str, float]]:
    """Load a glyph baseline previously saved with save_glyph_baseline()."""

    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def process_exam_sheet(
    front_image: np.ndarray,
    back_image: np.ndarray,
    question_count: int = 120,
    threshold: float = DEFAULT_DARKNESS_THRESHOLD,
    gap: float = DEFAULT_MINIMUM_SCORE_GAP,
    blank_gray_front: np.ndarray | None = None,
    blank_gray_back: np.ndarray | None = None,
    glyph_baseline: dict[str, dict[str, float]] | None = None,
) -> dict[str, dict]:
    """
    Process one exam sheet's two already-loaded page images.

    Both images must already be loaded as OpenCV BGR images, in
    order: `front_image` is page 1 (exam/version/student
    ID/questions 1-40), `back_image` is page 2 (questions
    41-120 only).

    Returns one combined grouped-selections dict covering both pages.
    The reserved key "_detection_diagnostics" holds interpolation
    notes from BOTH pages combined.

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
    # pass.
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



# =====================================================================
# Main
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure and classify exam bubbles."
    )

    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help=(
            "Image or PDF to process (a single page, inspected with "
            "--page). Not used with --calibrate-glyphs."
        ),
    )

    parser.add_argument(
        "--page",
        type=int,
        choices=(1, 2),
        default=1,
        help="Which exam page `input` is (1=front, 2=back). Default: 1",
    )

    parser.add_argument(
        "--calibrate-glyphs",
        dest="calibrate_glyphs",
        nargs="+",
        default=None,
        metavar="PDF",
        help=(
            "Build a per-glyph fill baseline from one or more "
            "already-scanned, double-sided exam-sheet PDFs (each PDF "
            "must have an even page count: front, back, front, "
            "back, ... -- same convention as exam_grader.py's input) "
            "and exit. Use with --calibrate-out."
        ),
    )

    parser.add_argument(
        "--calibrate-out",
        dest="calibrate_out",
        type=str,
        default="exam_glyph_baseline.json",
        help=(
            "Output path for the calibrated baseline JSON. "
            "Default: exam_glyph_baseline.json"
        ),
    )

    parser.add_argument(
        "--calibrate-percentile",
        dest="calibrate_percentile",
        type=float,
        default=DEFAULT_GLYPH_BASELINE_PERCENTILE,
        help=(
            "Percentile of each glyph's blank-sample distribution "
            "used as its baseline (higher = more conservative). "
            f"Default: {DEFAULT_GLYPH_BASELINE_PERCENTILE}"
        ),
    )

    parser.add_argument(
        "--calibrate-min-samples",
        dest="calibrate_min_samples",
        type=int,
        default=MIN_GLYPH_BASELINE_SAMPLES,
        help=(
            "Minimum samples required to calibrate a glyph; lower "
            "this to accept a less-certain baseline for a glyph "
            "that's inherently hard to get many samples for (e.g. a "
            "term letter every page in your batches happens to "
            f"share). Default: {MIN_GLYPH_BASELINE_SAMPLES}"
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=detector.PDF_DPI,
        help=f"PDF rendering DPI. Default: {detector.PDF_DPI}",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_DARKNESS_THRESHOLD,
        help=(
            "Minimum darkness score for a selected bubble. "
            f"Default: {DEFAULT_DARKNESS_THRESHOLD}"
        ),
    )

    parser.add_argument(
        "--gap",
        type=float,
        default=DEFAULT_MINIMUM_SCORE_GAP,
        help=(
            "Minimum score gap between first and second bubbles. "
            f"Default: {DEFAULT_MINIMUM_SCORE_GAP}"
        ),
    )

    parser.add_argument(
        "--save",
        default=None,
        help="Optional path for saving the measurement overlay.",
    )

    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Do not display or save the measurement overlay.",
    )

    parser.add_argument(
        "--questions",
        type=int,
        default=120,
        help="Number of questions to analyze, starting with question 1 "
            "(default: 120)"
    )

    args = parser.parse_args()

    if not 1 <= args.questions <= 120:
        parser.error("--questions must be between 1 and 120")

    if args.calibrate_glyphs is not None:
        page_pairs = []

        for pdf_path in args.calibrate_glyphs:
            pdf_path = Path(pdf_path)

            if not pdf_path.exists():
                parser.error(f"PDF does not exist: {pdf_path}")

            pdf_pages = pdf_to_images(pdf_path)

            if len(pdf_pages) % 2 != 0:
                parser.error(
                    f"{pdf_path} has {len(pdf_pages)} page(s) — an "
                    "odd number. Exam calibration PDFs must be "
                    "double-sided (front, back, front, back, ...)."
                )

            for i in range(0, len(pdf_pages), 2):
                page_pairs.append((pdf_pages[i], pdf_pages[i + 1]))

        print(
            f"Calibrating from {len(page_pairs)} exam sheet(s) "
            f"across {len(args.calibrate_glyphs)} PDF(s)..."
        )

        baseline = calibrate_glyph_baseline(
            page_pairs,
            question_count=args.questions,
            threshold=args.threshold,
            gap=args.gap,
            percentile=args.calibrate_percentile,
            min_samples=args.calibrate_min_samples,
        )

        output_path = save_glyph_baseline(args.calibrate_out, baseline)

        print(
            f"\nCalibrated {len(baseline)} glyph(s), saved to "
            f"{output_path}"
        )

        for glyph in sorted(baseline):
            print(
                f"  {glyph:4s} "
                f"(n={baseline[glyph]['sample_count']})"
            )

        return

    if args.input is None:
        parser.error("input is required unless using --calibrate-glyphs")

    # ---------------------------------------------------------------
    # Registration and timing marks
    # ---------------------------------------------------------------

    original = detector.load_image(
        args.input,
        args.dpi,
    )

    corners = detector.detect_corner_squares(
        original,
    )

    warped = detector.rectify_image(
        original,
        corners,
    )

    ticks = detector.detect_timing_marks(
        warped,
    )

    # ---------------------------------------------------------------
    # Build mapped bubble locations for the specified page
    # ---------------------------------------------------------------

    bubble_map = detector.build_bubble_map(args.page)

    # ---------------------------------------------------------------
    # Measure darkness
    # ---------------------------------------------------------------

    measurements = measure_bubbles(
        warped,
        ticks,
        bubble_map,
    )

    # ---------------------------------------------------------------
    # Classify selections
    # ---------------------------------------------------------------

    selections = determine_all_selections(
        measurements,
        bubble_map,
        threshold=args.threshold,
        minimum_score_gap=args.gap,
        question_count=args.questions,
    )


    # ---------------------------------------------------------------
    # Print results
    # ---------------------------------------------------------------

    print_measurements(measurements)
    print_selections(selections)

    simple_answers = extract_simple_answers(
        selections,
    )

    print("\nSimplified result:")
    for field, value in sorted(simple_answers.items()):
        print(f"  {field}: {value}")

    # ---------------------------------------------------------------
    # Draw optional overlay
    # ---------------------------------------------------------------

    if not args.no_overlay:
        annotated = draw_measurement_overlay(
            warped,
            measurements,
            selections,
        )

        if args.save is not None:
            save_path = Path(args.save)
            save_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            cv2.imwrite(
                str(save_path),
                annotated,
            )

            print(
                f"\nSaved measurement overlay to: "
                f"{save_path}"
            )

        cv2.imshow(
            "Scantron Bubble Measurements",
            annotated,
        )

        print("\nPress any key in the image window to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()