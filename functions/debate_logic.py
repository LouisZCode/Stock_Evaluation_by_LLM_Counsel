"""
Debate logic for the Counsel of LLMs.
Handles agreement calculation and debate orchestration.
"""

import re
import copy
from typing import Optional

RATING_MAP = {
    "excellent": 5,
    "good": 4,
    "neutral": 3,
    "bad": 2,
    "horrible": 1
}

# Tier definitions for harmonization logic
POSITIVE_TIER = {'excellent', 'good'}   # Bullish ratings
NEGATIVE_TIER = {'bad', 'horrible'}     # Bearish ratings
NEUTRAL_TIER = {'neutral'}              # Ambiguous - always triggers debate

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


def get_metric_comparison(analyses: list[dict]) -> dict:
    """
    Generate comparison table of all metrics across N LLMs.

    Args:
        analyses: List of FinancialInformation dicts from LLMs

    Returns:
        {
            'rows': [
                {'metric': 'revenue', 'ratings': ['Excellent', 'Excellent', 'Excellent'], 'spread': 0},
                {'metric': 'cash_flow', 'ratings': [None, 'Good', 'Good'], 'spread': 0},
                ...
            ],
            'missing_counts': [3, 0, 2],  # per LLM
            'llm_count': 3
        }
    """
    rows = []
    missing_counts = [0] * len(analyses)

    for metric in METRICS:
        ratings = []
        numeric_ratings = []

        for i, analysis in enumerate(analyses):
            raw_rating = analysis.get(metric, '')

            # Check if missing
            if not raw_rating or "not enough information" in raw_rating.lower():
                ratings.append(None)
                missing_counts[i] += 1
            else:
                # Extract first word (Excellent, Good, etc.)
                first_word = raw_rating.split()[0] if raw_rating else ''
                ratings.append(first_word)
                numeric = parse_metric_rating(raw_rating)
                if numeric is not None:
                    numeric_ratings.append(numeric)

        # Calculate spread (None if less than 2 valid ratings)
        if len(numeric_ratings) >= 2:
            spread = max(numeric_ratings) - min(numeric_ratings)
        else:
            spread = None

        rows.append({
            'metric': metric,
            'ratings': ratings,
            'spread': spread
        })

    return {
        'rows': rows,
        'missing_counts': missing_counts,
        'llm_count': len(analyses)
    }


def fill_missing_with_consensus(analyses: list[dict]) -> list[dict]:
    """
    Fill missing metrics when other LLMs have consensus.

    Rules:
    - Only fill if exactly 1 LLM is missing for that metric
    - Remaining LLMs must all agree on the rating
    - Returns new list (doesn't mutate original)

    Args:
        analyses: List of FinancialInformation dicts from LLMs

    Returns:
        New list with missing values filled where consensus exists
    """
    filled = copy.deepcopy(analyses)
    llm_count = len(analyses)

    for metric in METRICS:
        ratings = []
        missing_indices = []

        # Collect ratings and track missing
        for i, analysis in enumerate(analyses):
            raw_rating = analysis.get(metric, '')
            if not raw_rating or "not enough information" in raw_rating.lower():
                missing_indices.append(i)
                ratings.append(None)
            else:
                first_word = raw_rating.split()[0].capitalize() if raw_rating else None
                ratings.append(first_word)

        # Only fill if exactly 1 missing
        if len(missing_indices) != 1:
            continue

        # Check if remaining ratings all agree
        non_missing = [r for r in ratings if r is not None]
        if len(set(non_missing)) == 1:
            # All agree - fill the missing one
            consensus_value = non_missing[0]
            missing_idx = missing_indices[0]
            filled[missing_idx][metric] = consensus_value
            # Also fill the reason field
            reason_key = f"{metric}_reason"
            filled[missing_idx][reason_key] = f"Filled by consensus ({consensus_value})"

    return filled


def recalculate_strength_scores(analyses: list[dict]) -> list[int]:
    """
    Recalculate X/8 scores by counting Good/Excellent ratings.

    Args:
        analyses: List of FinancialInformation dicts (after filling)

    Returns:
        List of scores [6, 7, 6] for each LLM
    """
    scores = []

    for analysis in analyses:
        positive_count = 0

        for metric in METRICS:
            rating_str = analysis.get(metric, '')

            # Skip missing
            if not rating_str or "not enough information" in rating_str.lower():
                continue

            # Check if Good or Excellent
            first_word = rating_str.lower().split()[0] if rating_str else ''
            if first_word in ('good', 'excellent'):
                positive_count += 1

        scores.append(positive_count)

    return scores


def calculate_agreement_from_scores(scores: list[int]) -> dict:
    """
    Calculate agreement info from a list of scores.

    Args:
        scores: List of recalculated scores [6, 7, 6]

    Returns:
        Same format as calculate_agreement()
    """
    if len(scores) >= 2:
        score_spread = max(scores) - min(scores)
    else:
        score_spread = 0

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
        'scores': scores
    }


def _get_tier(rating: str) -> str:
    """Get the tier for a rating."""
    rating_lower = rating.lower()
    if rating_lower in POSITIVE_TIER:
        return 'positive'
    elif rating_lower in NEGATIVE_TIER:
        return 'negative'
    elif rating_lower in NEUTRAL_TIER:
        return 'neutral'
    return 'unknown'


def _get_majority(ratings: list[str]) -> str:
    """Get the majority rating from a list."""
    from collections import Counter
    # Count occurrences, return most common
    counts = Counter(r.lower() for r in ratings if r)
    if counts:
        most_common = counts.most_common(1)[0][0]
        return most_common.capitalize()
    return ratings[0] if ratings else None


def harmonize_and_check_debates(analyses: list[dict]) -> dict:
    """
    For each metric:
    1. Get ratings (skip missing)
    2. Determine tiers present
    3. If all same tier → harmonize to majority
    4. If Neutral OR cross-tier → flag for debate

    Args:
        analyses: List of FinancialInformation dicts (after fill_missing)

    Returns:
        {
            'harmonized_analyses': [...],
            'metrics_to_debate': ['cash_flow', 'total_debt'],
            'harmonization_log': [
                {'metric': 'revenue', 'action': 'harmonized', 'original': ['E','E','G'], 'result': 'Excellent'},
                {'metric': 'cash_flow', 'action': 'debate', 'reason': 'neutral_present', 'ratings': ['G','N','G']},
            ]
        }
    """
    harmonized = copy.deepcopy(analyses)
    metrics_to_debate = []
    harmonization_log = []

    for metric in METRICS:
        # Collect ratings for this metric
        ratings = []
        has_missing = False

        for analysis in analyses:
            raw_rating = analysis.get(metric, '')
            if not raw_rating or "not enough information" in raw_rating.lower():
                has_missing = True
                ratings.append(None)
            else:
                first_word = raw_rating.split()[0].capitalize() if raw_rating else None
                ratings.append(first_word)

        # Skip if has missing - already handled by fill_missing
        valid_ratings = [r for r in ratings if r is not None]
        if len(valid_ratings) < 2:
            harmonization_log.append({
                'metric': metric,
                'action': 'skipped',
                'reason': 'insufficient_data',
                'ratings': ratings
            })
            continue

        # Get tiers for each rating
        tiers = set(_get_tier(r) for r in valid_ratings)

        # Check for unanimous agreement FIRST (even if all Neutral)
        if len(set(r.lower() for r in valid_ratings)) == 1:
            # All same rating - no debate needed
            harmonization_log.append({
                'metric': metric,
                'action': 'already_aligned',
                'ratings': ratings,
                'result': valid_ratings[0].capitalize()
            })
            continue

        # Check for debate triggers
        if 'neutral' in tiers:
            # Neutral present - needs debate
            metrics_to_debate.append(metric)
            harmonization_log.append({
                'metric': metric,
                'action': 'debate',
                'reason': 'neutral_present',
                'ratings': ratings
            })
        elif 'positive' in tiers and 'negative' in tiers:
            # Cross-tier conflict - needs debate
            metrics_to_debate.append(metric)
            harmonization_log.append({
                'metric': metric,
                'action': 'debate',
                'reason': 'cross_tier_conflict',
                'ratings': ratings
            })
        else:
            # All same tier but different values - harmonize to majority
            majority = _get_majority(valid_ratings)
            original_ratings = ratings.copy()

            # Apply harmonization
            for i, analysis in enumerate(harmonized):
                if ratings[i] is not None:
                    harmonized[i][metric] = majority
                    # Update reason to note harmonization
                    reason_key = f"{metric}_reason"
                    original_reason = analysis.get(reason_key, '')
                    if original_reason and 'Harmonized' not in original_reason:
                        harmonized[i][reason_key] = f"{original_reason} [Harmonized to {majority}]"

            harmonization_log.append({
                'metric': metric,
                'action': 'harmonized',
                'original': original_ratings,
                'result': majority
            })

    return {
        'harmonized_analyses': harmonized,
        'metrics_to_debate': metrics_to_debate,
        'harmonization_log': harmonization_log
    }
