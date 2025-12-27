"""
Reports module for generating exportable financial reports.
"""

try:
    from .pdf_generator import generate_pdf
    print("Reports module loaded...")
except OSError as e:
    print(f"Reports module: PDF generation disabled (missing system libraries)")
    print("  → Install with: brew install pango")

    def generate_pdf(*args, **kwargs):
        """Placeholder when weasyprint dependencies are missing."""
        print("PDF generation skipped - install pango first: brew install pango")
        return None
