"""
Debate logic for the Counsel of LLMs.
Handles agreement calculation and debate orchestration.
"""

import re
from typing import Optional

RATING_MAP = {
    "excellent": 5,
    "good": 4,
    "neutral": 3,
    "bad": 2,
    "horrible": 1
}

METRICS = [
    'revenue', 'net_income', 'gross_margin', 'operational_costs',
    'cash_flow', 'quaterly_growth', 'total_assets', 'total_debt'
]


def parse_metric_rating(rating_str: str) -> Optional[int]:
    """
    Convert rating string to numeric value.
    "Excellent" -> 5, "Good" -> 4, etc.
    Returns None if rating is missing or invalid.
    """
    if not rating_str or "not enough information" in rating_str.lower():
        return None
    first_word = rating_str.lower().split()[0]
    return RATING_MAP.get(first_word)


def parse_strength_score(strength_str: str) -> Optional[int]:
    """
    Extract X from "X/8" format.
    "7/8" -> 7, "4/8" -> 4
    Returns None if format not found or invalid.
    """
    if not strength_str or "not enough information" in strength_str.lower():
        return None
    match = re.search(r'(\d)/8', strength_str)
    return int(match.group(1)) if match else None


def calculate_agreement(analyses: list[dict]) -> dict:
    """
    Analyze agreement level between 3 LLM analyses.

    Args:
        analyses: List of 3 FinancialInformation dicts from LLMs

    Returns:
        {
            'debate_level': 'none' | 'small' | 'large',
            'score_spread': int,
            'scores': list[int],
            'metric_disagreements': list[str],
            'missing_data': bool
        }
    """
    # Parse financial_strenght scores
    scores = []
    missing_data = False

    for analysis in analyses:
        score = parse_strength_score(analysis.get('financial_strenght', ''))
        if score is not None:
            scores.append(score)
        else:
            missing_data = True

    # Calculate spread (if we have at least 2 scores)
    if len(scores) >= 2:
        score_spread = max(scores) - min(scores)
    else:
        score_spread = 0
        missing_data = True

    # Find metric disagreements (ratings differ by 2+ points)
    metric_disagreements = []
    for metric in METRICS:
        ratings = []
        for analysis in analyses:
            rating = parse_metric_rating(analysis.get(metric, ''))
            if rating is not None:
                ratings.append(rating)

        if len(ratings) >= 2:
            metric_spread = max(ratings) - min(ratings)
            if metric_spread >= 2:
                metric_disagreements.append(metric)

    # Determine debate level
    if score_spread < 2:
        debate_level = 'none'
    elif score_spread == 2:
        debate_level = 'small'
    else:
        debate_level = 'large'

    return {
        'debate_level': debate_level,
        'score_spread': score_spread,
        'scores': scores,
        'metric_disagreements': metric_disagreements,
        'missing_data': missing_data
    }
