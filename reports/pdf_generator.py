"""
PDF Report Generator for Stock Financial Analysis.

Uses weasyprint to convert HTML templates to PDF.
"""

import os
from datetime import datetime
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

# Try to import weasyprint, provide helpful error if missing
try:
    from weasyprint import HTML
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    print("Warning: weasyprint not installed. PDF generation disabled.")

# Paths
REPORTS_DIR = Path(__file__).parent
TEMPLATES_DIR = REPORTS_DIR / "templates"
OUTPUT_DIR = REPORTS_DIR / "generated"


def generate_pdf(
    ticker: str,
    harmonize_result: dict,
    original_analyses: list,
    debate_result: dict = None,
) -> str:
    """
    Generate a PDF financial report for a ticker.

    Args:
        ticker: Stock symbol
        harmonize_result: Output from harmonize_and_check_debates()
        original_analyses: List of LLM analyses with metrics + reasons
        debate_result: Optional output from run_debate()

    Returns:
        Path to the generated PDF file, or None if generation failed
    """
    if not WEASYPRINT_AVAILABLE:
        print("Error: weasyprint not available. Install with: uv add weasyprint")
        return None

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Prepare report data
    report_data = _prepare_report_data(
        ticker, harmonize_result, original_analyses, debate_result
    )

    # Load and render template
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
    template = env.get_template("financial_report.html")
    html_content = template.render(**report_data)

    # Generate PDF
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{ticker}_report_{timestamp}.pdf"
    output_path = OUTPUT_DIR / output_filename

    HTML(string=html_content).write_pdf(output_path)

    print(f"PDF report generated: {output_path}")
    return str(output_path)


def _prepare_report_data(
    ticker: str,
    harmonize_result: dict,
    original_analyses: list,
    debate_result: dict = None,
) -> dict:
    """
    Prepare all data needed for the report template.
    """
    harmonization_log = harmonize_result.get('harmonization_log', [])

    # Separate clear and complex metrics
    clear_entries = [e for e in harmonization_log if e['action'] in ('already_aligned', 'harmonized')]
    complex_entries = [e for e in harmonization_log if e['action'] == 'debate']

    # Get debate results if available
    debate_results = debate_result.get('debate_results', {}) if debate_result else {}

    # Build clear metrics with reasons
    clear_metrics = []
    for entry in clear_entries:
        metric = entry['metric']
        rating = entry['result']
        reason = _find_matching_reason(metric, rating, original_analyses)
        clear_metrics.append({
            'name': metric,
            'rating': rating,
            'reason': reason[:200] + '...' if len(reason) > 200 else reason,
            'rating_class': _get_rating_class(rating),
        })

    # Build complex metrics with debate info
    complex_metrics = []
    for entry in complex_entries:
        metric = entry['metric']
        original_ratings = entry.get('ratings', [])
        final_rating = debate_results.get(metric, 'Pending')
        reason = _find_matching_reason(metric, final_rating, original_analyses) if final_rating not in ('COMPLEX', 'Pending') else ''

        complex_metrics.append({
            'name': metric,
            'before_ratings': original_ratings,
            'final_rating': final_rating,
            'reason': reason[:200] + '...' if len(reason) > 200 else reason,
            'is_complex': final_rating == 'COMPLEX',
            'rating_class': _get_rating_class(final_rating),
        })

    # Collect all final ratings for scoring
    all_ratings = {}
    for entry in clear_entries:
        all_ratings[entry['metric']] = entry['result']
    if debate_result:
        for metric, rating in debate_results.items():
            all_ratings[metric] = rating

    # Calculate score and verdict
    score = _calculate_score(all_ratings)
    verdict = _get_verdict(score)

    # Categorize for summary
    strengths = [m for m in clear_metrics + complex_metrics if m['rating_class'] in ('excellent', 'good')]
    watch = [m for m in clear_metrics + complex_metrics if m['rating_class'] == 'neutral']
    concerns = [m for m in clear_metrics + complex_metrics if m['rating_class'] in ('bad', 'horrible')]
    unresolved = [m for m in complex_metrics if m.get('is_complex')]

    return {
        'ticker': ticker,
        'generated_date': datetime.now().strftime('%B %d, %Y'),
        'generated_time': datetime.now().strftime('%H:%M'),
        'score': score,
        'score_display': f"+{score}" if score > 0 else str(score),
        'verdict': verdict,
        'verdict_class': verdict.lower().replace(' ', '-'),
        'clear_metrics': clear_metrics,
        'complex_metrics': complex_metrics,
        'strengths': strengths,
        'watch': watch,
        'concerns': concerns,
        'unresolved': unresolved,
        'num_experts': len(original_analyses),
        'num_debates': len(complex_entries),
        'num_resolved': len(complex_entries) - len(unresolved),
    }


def _find_matching_reason(metric: str, final_rating: str, analyses: list) -> str:
    """Find reason from first LLM whose rating matches."""
    reason_key = f"{metric}_reason"

    for analysis in analyses:
        llm_rating = analysis.get(metric, "").lower()
        if llm_rating == final_rating.lower():
            return analysis.get(reason_key, "")

    # Fallback: return first available reason
    for analysis in analyses:
        if reason_key in analysis and analysis[reason_key]:
            return analysis[reason_key]

    return ""


def _calculate_score(ratings: dict) -> int:
    """Calculate financial score from -16 to +16."""
    score_map = {
        'excellent': 2,
        'good': 1,
        'neutral': 0,
        'bad': -1,
        'horrible': -2,
        'complex': 0,
    }

    total = 0
    for metric, rating in ratings.items():
        rating_lower = rating.lower() if rating else ""
        total += score_map.get(rating_lower, 0)

    return total


def _get_verdict(score: int) -> str:
    """Get verdict label from score."""
    if score <= -11:
        return "Extremely Risky"
    elif score <= -4:
        return "Risky"
    elif score <= 3:
        return "Neutral"
    elif score <= 10:
        return "Safe"
    else:
        return "Extremely Safe"


def _get_rating_class(rating: str) -> str:
    """Get CSS class for rating."""
    if not rating:
        return "neutral"
    rating_lower = rating.lower()
    if rating_lower in ('excellent', 'good', 'neutral', 'bad', 'horrible', 'complex'):
        return rating_lower
    return "neutral"
