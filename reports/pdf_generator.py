"""
PDF Report Generator for Stock Financial Analysis.

Uses weasyprint to convert HTML templates to PDF.
"""

import os
import math
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

        # Gather unique expert reasons grouped by rating
        expert_reasons = _get_expert_reasons_by_rating(metric, original_analyses)

        complex_metrics.append({
            'name': metric,
            'before_ratings': [r for r in original_ratings if r],  # Filter out None
            'final_rating': final_rating,
            'reason': reason[:200] + '...' if len(reason) > 200 else reason,
            'is_complex': final_rating == 'COMPLEX',
            'rating_class': _get_rating_class(final_rating),
            'expert_reasons': expert_reasons,
            'debate_rounds': 3,  # Default debate rounds
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

    # Generate gauge SVG
    gauge_svg = _generate_gauge_svg(score, verdict)

    # Get overall summary from first analysis that has it
    overall_summary = ""
    for analysis in original_analyses:
        if analysis.get('overall_summary'):
            overall_summary = analysis['overall_summary']
            break

    return {
        'ticker': ticker,
        'generated_date': datetime.now().strftime('%B %d, %Y'),
        'generated_time': datetime.now().strftime('%H:%M'),
        'score': score,
        'score_display': f"+{score}" if score > 0 else str(score),
        'verdict': verdict,
        'verdict_class': verdict.lower().replace(' ', '-'),
        'gauge_svg': gauge_svg,
        'overall_summary': overall_summary,
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


def _get_expert_reasons_by_rating(metric: str, analyses: list) -> list:
    """
    Get unique expert reasons grouped by rating for a metric.
    Returns list of {'rating': 'Good', 'reason': '...'} dicts.
    Only includes unique ratings to avoid repetition.
    """
    reason_key = f"{metric}_reason"
    seen_ratings = set()
    expert_reasons = []

    for analysis in analyses:
        raw_rating = analysis.get(metric, '')
        if not raw_rating or "not enough information" in raw_rating.lower():
            continue

        # Extract first word as rating
        rating = raw_rating.split()[0].capitalize() if raw_rating else ''
        if rating and rating not in seen_ratings:
            seen_ratings.add(rating)
            reason = analysis.get(reason_key, '')
            if reason and "not enough information" not in reason.lower():
                # Truncate long reasons
                truncated = reason[:150] + '...' if len(reason) > 150 else reason
                expert_reasons.append({
                    'rating': rating,
                    'reason': truncated
                })

    return expert_reasons


def _generate_gauge_svg(score: int, verdict: str) -> str:
    """Generate horizontal gradient bar with arrow marker."""
    # Normalize score from -16..+16 to 0..100% position
    # -16 = 0% (left), 0 = 50% (center), +16 = 100% (right)
    position_percent = ((score + 16) / 32) * 100
    arrow_x = 10 + (position_percent / 100) * 180  # 10 to 190 range

    return f'''
    <svg width="220" height="65" viewBox="0 0 220 65">
        <defs>
            <linearGradient id="barGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" style="stop-color:#dc2626"/>
                <stop offset="25%" style="stop-color:#f59e0b"/>
                <stop offset="50%" style="stop-color:#9ca3af"/>
                <stop offset="75%" style="stop-color:#34d399"/>
                <stop offset="100%" style="stop-color:#059669"/>
            </linearGradient>
        </defs>

        <!-- Verdict text above arrow -->
        <text x="{arrow_x}" y="14" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1a1a">{verdict}</text>

        <!-- Arrow marker -->
        <polygon points="{arrow_x},28 {arrow_x - 6},18 {arrow_x + 6},18" fill="#1a1a1a"/>

        <!-- Horizontal gradient bar -->
        <rect x="10" y="30" width="180" height="10" rx="5" fill="url(#barGradient)"/>

        <!-- Labels -->
        <text x="10" y="55" font-size="7" fill="#991b1b" font-weight="600">Extreme Risk</text>
        <text x="210" y="55" text-anchor="end" font-size="7" fill="#059669" font-weight="600">Extreme Safety</text>
    </svg>
    '''
