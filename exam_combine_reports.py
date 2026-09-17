from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch


REPORT_DIRECTORY = Path("./")
OUTPUT_PDF = Path("../combined_student_reports.pdf")


def create_combined_pdf(report_directory: Path, output_pdf: Path):
    """
    Same idea as the quiz's combine_reports.py, but each report can
    now span MULTIPLE PDF pages instead of being forced onto exactly
    one: a 120-question report runs well past what fits on a single
    letter page at this font size, where the quiz's 15-20 question
    reports comfortably fit on one.
    """

    report_files = sorted(
        report_directory.glob("*.txt"),
        key=lambda path: path.name.lower()
    )

    if not report_files:
        raise FileNotFoundError(
            f"No text reports were found in {report_directory}"
        )

    page_width, page_height = letter

    pdf = canvas.Canvas(
        str(output_pdf),
        pagesize=letter
    )

    # Courier is useful because it preserves text alignment.
    font_name = "Courier"
    font_size = 8.5
    line_height = 10

    margin_left = 0.45 * inch
    margin_right = 0.45 * inch
    margin_top = 0.45 * inch
    margin_bottom = 0.45 * inch

    usable_width = page_width - margin_left - margin_right
    usable_height = page_height - margin_top - margin_bottom

    lines_per_page = int(usable_height // line_height)

    max_characters = int(usable_width / (font_size * 0.6))

    total_pdf_pages = 0

    for report_file in report_files:
        text = report_file.read_text(
            encoding="utf-8",
            errors="replace"
        )

        lines = text.splitlines()

        # Split this one report across as many PDF pages as it needs,
        # instead of truncating it to fit on one.
        for page_start in range(0, len(lines), lines_per_page):
            page_lines = lines[page_start:page_start + lines_per_page]

            pdf.setFont(font_name, font_size)

            y = page_height - margin_top

            for line in page_lines:
                # Prevent extremely long lines from running off the page.
                if len(line) > max_characters:
                    line = line[:max_characters - 3] + "..."

                pdf.drawString(
                    margin_left,
                    y,
                    line
                )

                y -= line_height

            pdf.showPage()
            total_pdf_pages += 1

    pdf.save()

    print(f"Created: {output_pdf}")
    print(f"Reports combined: {len(report_files)}")
    print(f"Total PDF pages: {total_pdf_pages}")


if __name__ == "__main__":
    create_combined_pdf(
        REPORT_DIRECTORY,
        OUTPUT_PDF
    )
