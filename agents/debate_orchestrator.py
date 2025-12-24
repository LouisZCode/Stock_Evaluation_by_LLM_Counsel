"""
Multi-round debate orchestrator for resolving metric disagreements.

Runs structured debates between the 3 debate agents (Socrates, Pythagoras, Diogenes)
when metrics_to_debate is non-empty after harmonization.
"""

import asyncio
import re
import uuid
from typing import List, Dict
from collections import Counter

from .agents import openai_socrates, anthropic_pythagoras, mistral_diogenes
from .prompt_loader import load_prompts

# Load debate prompts
prompts = load_prompts()
DEBATE_ROUND1 = prompts["DEBATE_ROUND1"]
DEBATE_REVIEW = prompts["DEBATE_REVIEW"]
DEBATE_FINAL = prompts["DEBATE_FINAL"]

# Debate agents configuration
DEBATE_AGENTS = [
    ("Socrates", openai_socrates),
    ("Pythagoras", anthropic_pythagoras),
    ("Diogenes", mistral_diogenes),
]

# Valid ratings for extraction
VALID_RATINGS = {'excellent', 'good', 'neutral', 'bad', 'horrible'}


async def run_debate(
    ticker: str,
    metrics_to_debate: List[str],
    original_analyses: List[dict],
    max_rounds: int = 3
) -> dict:
    """
    Run multi-round debate on disputed metrics (sequentially per metric).

    Args:
        ticker: Stock symbol
        metrics_to_debate: List of metric names needing debate
        original_analyses: Original LLM analyses with ratings and reasons
        max_rounds: Number of debate rounds (default 3)

    Returns:
        {
            'debate_results': {metric: final_rating},
            'position_changes': [{llm, metric, from, to}],
            'transcript': [{round, metric, llm, content}]
        }
    """
    debate_results = {}
    all_transcripts = []
    position_changes = []

    print(f"\n{'='*50}")
    print(f"DEBATE STARTING: {len(metrics_to_debate)} metrics to debate")
    print(f"Metrics: {', '.join(metrics_to_debate)}")
    print(f"{'='*50}")

    # Debate each metric sequentially
    for metric in metrics_to_debate:
        result = await _debate_single_metric(
            ticker, metric, original_analyses, max_rounds
        )
        debate_results[metric] = result['final_rating']
        all_transcripts.extend(result['transcript'])
        position_changes.extend(result['changes'])

    print(f"\n{'='*50}")
    print(f"DEBATE COMPLETE")
    for metric, rating in debate_results.items():
        status = "COMPLEX (no consensus)" if rating == "COMPLEX" else rating
        print(f"  {metric}: {status}")
    print(f"{'='*50}\n")

    return {
        'debate_results': debate_results,
        'position_changes': position_changes,
        'transcript': all_transcripts
    }


async def _debate_single_metric(
    ticker: str,
    metric: str,
    analyses: List[dict],
    max_rounds: int
) -> dict:
    """Debate a single metric through multiple rounds."""
    positions = {}  # {agent_name: {rating, reason, history}}
    transcript = []
    changes = []

    # Generate unique thread_id for this metric's debate
    debate_thread_id = f"debate_{ticker}_{metric}_{uuid.uuid4().hex[:8]}"
    print(f"\n[Debate] Starting debate on '{metric}' (thread: {debate_thread_id})")

    # Initialize positions from original analyses
    for i, (name, agent) in enumerate(DEBATE_AGENTS):
        positions[name] = {
            'rating': analyses[i].get(metric, 'Unknown'),
            'reason': analyses[i].get(f'{metric}_reason', 'No reason provided'),
            'history': [],
            'thread_id': f"{debate_thread_id}_{name}"  # Unique per agent
        }
        print(f"  {name}: {positions[name]['rating']}")

    # Round 1: State positions (parallel)
    print(f"[Debate] Round 1 - Stating positions...")
    round1_tasks = []
    for name, agent in DEBATE_AGENTS:
        prompt = _build_round1_prompt(ticker, metric, positions[name])
        thread_id = positions[name]['thread_id']
        round1_tasks.append(_invoke_agent(agent, prompt, name, thread_id))

    round1_results = await asyncio.gather(*round1_tasks)
    for name, response in round1_results:
        positions[name]['history'].append(response)
        transcript.append({
            'round': 1,
            'metric': metric,
            'llm': name,
            'content': response
        })

    # Rounds 2 to N-1: Review and respond (parallel per round)
    for round_num in range(2, max_rounds):
        print(f"[Debate] Round {round_num} - Reviewing positions...")
        review_tasks = []
        for name, agent in DEBATE_AGENTS:
            other_positions = {k: v for k, v in positions.items() if k != name}
            prompt = _build_review_prompt(ticker, metric, positions[name], other_positions)
            thread_id = positions[name]['thread_id']
            review_tasks.append(_invoke_agent(agent, prompt, name, thread_id))

        review_results = await asyncio.gather(*review_tasks)
        for name, response in review_results:
            old_rating = positions[name]['rating']
            new_rating = _extract_updated_rating(response)
            if new_rating and new_rating.lower() != old_rating.lower():
                print(f"  {name} changed: {old_rating} → {new_rating}")
                changes.append({
                    'llm': name,
                    'metric': metric,
                    'from': old_rating,
                    'to': new_rating
                })
                positions[name]['rating'] = new_rating
            positions[name]['history'].append(response)
            transcript.append({
                'round': round_num,
                'metric': metric,
                'llm': name,
                'content': response
            })

    # Final round: Take final stance (parallel)
    print(f"[Debate] Final round - Taking final stances...")
    final_tasks = []
    for name, agent in DEBATE_AGENTS:
        prompt = _build_final_prompt(ticker, metric, positions[name])
        thread_id = positions[name]['thread_id']
        final_tasks.append(_invoke_agent(agent, prompt, name, thread_id))

    final_results = await asyncio.gather(*final_tasks)
    final_ratings = []
    for name, response in final_results:
        final_rating = _extract_final_rating(response)
        if final_rating:
            positions[name]['final'] = final_rating
            final_ratings.append(final_rating)
            print(f"  {name} final: {final_rating}")
        else:
            # Fallback to last known rating
            final_ratings.append(positions[name]['rating'])
            print(f"  {name} final: {positions[name]['rating']} (fallback)")
        transcript.append({
            'round': 'final',
            'metric': metric,
            'llm': name,
            'content': response
        })

    # Determine winning rating (majority or COMPLEX)
    consensus = _get_majority_rating(final_ratings)
    print(f"[Debate] Consensus on '{metric}': {consensus}")

    return {
        'final_rating': consensus,
        'transcript': transcript,
        'changes': changes
    }


async def _invoke_agent(agent, prompt: str, name: str, thread_id: str) -> tuple:
    """Invoke a debate agent and return (name, response)."""
    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]},
            {"configurable": {"thread_id": thread_id}}
        )
        return (name, result["messages"][-1].content)
    except Exception as e:
        print(f"  [Debate Error] {name}: {str(e)}")
        return (name, f"[Error: {str(e)}]")


def _build_round1_prompt(ticker: str, metric: str, position: dict) -> str:
    """Build Round 1 prompt - state position."""
    return DEBATE_ROUND1.format(
        ticker=ticker,
        metric=metric,
        rating=position['rating'],
        reason=position['reason']
    )


def _build_review_prompt(
    ticker: str,
    metric: str,
    my_position: dict,
    other_positions: dict
) -> str:
    """Build Review round prompt - review others' positions."""
    # Format other positions
    others_text = ""
    for name, pos in other_positions.items():
        others_text += f"- {name}: {pos['rating']} - {pos['reason']}\n"
        if pos.get('history'):
            # Include last argument (truncated)
            last_arg = pos['history'][-1][:300]
            others_text += f"  Their latest argument: {last_arg}...\n"

    return DEBATE_REVIEW.format(
        ticker=ticker,
        metric=metric,
        my_rating=my_position['rating'],
        my_reason=my_position['reason'],
        other_positions=others_text
    )


def _build_final_prompt(ticker: str, metric: str, position: dict) -> str:
    """Build Final round prompt - commit to final stance."""
    # Summarize debate history
    history_summary = ""
    for i, entry in enumerate(position.get('history', [])[-2:], 1):
        history_summary += f"Round {i}: {entry[:200]}...\n"

    return DEBATE_FINAL.format(
        ticker=ticker,
        metric=metric,
        my_rating=position['rating'],
        history_summary=history_summary if history_summary else "No prior debate rounds."
    )


def _extract_updated_rating(response: str) -> str:
    """Extract rating from 'UPDATED RATING: X' in response."""
    match = re.search(r'UPDATED\s+RATING:\s*(\w+)', response, re.IGNORECASE)
    if match:
        rating = match.group(1).lower()
        if rating in VALID_RATINGS:
            return rating.capitalize()
    return None


def _extract_final_rating(response: str) -> str:
    """Extract rating from 'FINAL RATING: X' in response."""
    match = re.search(r'FINAL\s+RATING:\s*(\w+)', response, re.IGNORECASE)
    if match:
        rating = match.group(1).lower()
        if rating in VALID_RATINGS:
            return rating.capitalize()
    return None


def _get_majority_rating(ratings: List[str]) -> str:
    """
    Return majority rating, or 'COMPLEX' if no majority (3-way tie).

    For user attention: COMPLEX means the debate couldn't resolve the disagreement
    and the user should review the transcript.
    """
    if not ratings:
        return None

    counts = Counter(r.lower() for r in ratings if r)
    if not counts:
        return None

    most_common = counts.most_common()

    # Check if there's a clear majority (count > 1 for 3 LLMs)
    if most_common[0][1] > 1:
        return most_common[0][0].capitalize()
    else:
        # 3-way tie - no resolution, mark as complex
        return "COMPLEX"
