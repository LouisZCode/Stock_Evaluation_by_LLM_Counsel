"""
Logger for capturing full LLM conversations including tool calls and responses.

You can find:
start_new_log, log_llm_conversation, log_debate_check, log_llm_timing
"""

import os
from datetime import datetime

LOGS_FOLDER = "logs/conversations"

# Track current log file for a session
_current_log_file = None


def start_new_log(ticker: str) -> str:
    """
    Creates a new log file for a ticker research session.
    Returns the log file path.
    """
    global _current_log_file

    os.makedirs(LOGS_FOLDER, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f"{timestamp}_{ticker}.log"
    filepath = os.path.join(LOGS_FOLDER, filename)

    _current_log_file = filepath

    # Write header
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n")
        f.write(f"  {ticker} Research - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")

    return filepath


def log_llm_timing(elapsed_time: float, log_file: str = None):
    """
    Logs the time taken for all LLM calls to complete.

    Args:
        elapsed_time: Time in seconds
        log_file: Optional specific log file path. Uses current session log if not provided.
    """
    filepath = log_file or _current_log_file

    if not filepath:
        print(f"Warning: No log file set. Call start_new_log() first.")
        return

    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"LLM Response Time: {elapsed_time:.2f}s (parallel)\n\n")


def log_llm_conversation(llm_name: str, response: dict, log_file: str = None):
    """
    Logs the full conversation from an LLM response to the log file.

    Args:
        llm_name: Name of the LLM (e.g., "OpenAI", "Claude", "Mistral")
        response: The full response dict from agent.invoke()
        log_file: Optional specific log file path. Uses current session log if not provided.
    """
    filepath = log_file or _current_log_file

    if not filepath:
        print(f"Warning: No log file set. Call start_new_log() first.")
        return

    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"\n{'-'*60}\n")
        f.write(f"  {llm_name.upper()}\n")
        f.write(f"{'-'*60}\n\n")

        messages = response.get("messages", [])

        for msg in messages:
            msg_type = type(msg).__name__

            if msg_type == "HumanMessage":
                f.write(f"[Human]\n{msg.content}\n\n")

            elif msg_type == "AIMessage":
                # Check for tool calls
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.get("name", "unknown")
                        tool_args = tool_call.get("args", {})
                        f.write(f"[AI - Tool Call: {tool_name}]\n")
                        f.write(f"Args: {tool_args}\n\n")

                # Log content if present
                if msg.content:
                    f.write(f"[AI - Response]\n{msg.content}\n\n")

            elif msg_type == "ToolMessage":
                tool_name = getattr(msg, "name", "unknown")
                content = msg.content

                # Special handling for retriever_tool: truncate per chunk
                if tool_name == "retriever_tool" and "---" in content:
                    chunks = content.split("---")
                    truncated_chunks = []
                    for chunk in chunks:
                        chunk = chunk.strip()
                        if not chunk:
                            continue
                        # Find metadata line and content
                        lines = chunk.split("\n", 1)
                        if len(lines) == 2:
                            metadata, body = lines
                            body = body[:150] + "..." if len(body) > 150 else body
                            truncated_chunks.append(f"{metadata}\n{body}")
                        else:
                            truncated_chunks.append(chunk[:150] + "...")
                    content = "\n---\n".join(truncated_chunks)
                elif len(content) > 1000:
                    # Default truncation for other tools
                    content = content[:1000] + "\n... (truncated)"

                f.write(f"[Tool: {tool_name}]\n{content}\n\n")

            else:
                # Fallback for other message types
                f.write(f"[{msg_type}]\n{getattr(msg, 'content', str(msg))}\n\n")

        f.write(f"\n")


def log_harmonization(harmonize_result: dict, final_scores: list[int] = None, log_file: str = None):
    """
    Logs harmonization results and debate decisions.

    Args:
        harmonize_result: Output from harmonize_and_check_debates()
        final_scores: Optional recalculated scores after harmonization
        log_file: Optional specific log file path. Uses current session log if not provided.
    """
    filepath = log_file or _current_log_file

    if not filepath:
        print(f"Warning: No log file set. Call start_new_log() first.")
        return

    harmonization_log = harmonize_result.get('harmonization_log', [])
    metrics_to_debate = harmonize_result.get('metrics_to_debate', [])

    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"  HARMONIZATION & DEBATE CHECK\n")
        f.write(f"{'='*60}\n\n")

        # Group log entries by action
        harmonized = [e for e in harmonization_log if e['action'] == 'harmonized']
        aligned = [e for e in harmonization_log if e['action'] == 'already_aligned']
        debates = [e for e in harmonization_log if e['action'] == 'debate']
        skipped = [e for e in harmonization_log if e['action'] == 'skipped']

        # Harmonized metrics
        if harmonized or aligned:
            f.write("Harmonized Metrics:\n")
            for entry in harmonized:
                original = entry.get('original', [])
                original_str = ', '.join(str(r) for r in original)
                f.write(f"  {entry['metric']:20}: [{original_str}] → {entry['result']}\n")
            for entry in aligned:
                ratings = entry.get('ratings', [])
                ratings_str = ', '.join(str(r) for r in ratings)
                f.write(f"  {entry['metric']:20}: [{ratings_str}] → {entry['result']} (no change)\n")
            f.write("\n")

        # Metrics flagged for debate
        if debates:
            f.write("Metrics Flagged for Debate:\n")
            for entry in debates:
                ratings = entry.get('ratings', [])
                ratings_str = ', '.join(str(r) if r else 'missing' for r in ratings)
                reason = entry.get('reason', 'unknown')
                f.write(f"  {entry['metric']:20}: [{ratings_str}] → {reason}\n")
            f.write("\n")

        # Skipped metrics (insufficient data)
        if skipped:
            f.write("Skipped (insufficient data):\n")
            for entry in skipped:
                ratings = entry.get('ratings', [])
                ratings_str = ', '.join(str(r) if r else 'missing' for r in ratings)
                f.write(f"  {entry['metric']:20}: [{ratings_str}]\n")
            f.write("\n")

        # Final scores if provided
        if final_scores:
            f.write(f"Final Scores: {final_scores}\n\n")

        # Summary
        if metrics_to_debate:
            f.write(f"→ Debate needed on {len(metrics_to_debate)} metrics: {', '.join(metrics_to_debate)}\n")
        else:
            f.write(f"→ No debate needed (all metrics aligned or harmonized)\n")

        f.write("\n")


def log_final_report(
    ticker: str,
    harmonize_result: dict,
    original_analyses: list,
    debate_result: dict = None,
    log_file: str = None
):
    """
    Log the final metrics report with clear and complex metrics.

    Args:
        ticker: Stock symbol
        harmonize_result: Output from harmonize_and_check_debates()
        original_analyses: List of 3 LLM analyses (filled_analyses) with reasons
        debate_result: Optional output from run_debate() if debate occurred
        log_file: Optional specific log file path
    """
    filepath = log_file or _current_log_file

    if not filepath:
        print("Warning: No log file set. Call start_new_log() first.")
        return

    harmonization_log = harmonize_result.get('harmonization_log', [])
    metrics_to_debate = harmonize_result.get('metrics_to_debate', [])

    # Separate clear metrics from complex metrics
    clear_entries = [e for e in harmonization_log if e['action'] in ('already_aligned', 'harmonized')]
    complex_entries = [e for e in harmonization_log if e['action'] == 'debate']

    # Expert names (generic to support future LLM changes)
    expert_names = ["Expert 1", "Expert 2", "Expert 3"]

    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"  FINAL METRICS REPORT\n")
        f.write(f"{'='*60}\n\n")

        # --- CLEAR METRICS ---
        if clear_entries:
            f.write(f"CLEAR METRICS ({len(clear_entries)}/8)\n")
            f.write(f"{'─'*20}\n")

            for entry in clear_entries:
                metric = entry['metric']
                final_rating = entry['result']

                # Find first LLM whose rating matches final rating
                reason = _find_matching_reason(metric, final_rating, original_analyses)

                f.write(f"  ✓ {metric}: {final_rating}\n")
                if reason:
                    # Truncate reason if too long
                    reason_short = reason[:250] + "..." if len(reason) > 250 else reason
                    f.write(f"    {reason_short}\n")
                f.write("\n")

        # --- COMPLEX METRICS ---
        if complex_entries or (debate_result and debate_result.get('debate_results')):
            debate_results = debate_result.get('debate_results', {}) if debate_result else {}
            complex_count = len(debate_results) if debate_results else len(complex_entries)

            f.write(f"COMPLEX METRICS ({complex_count}/8)\n")
            f.write(f"{'─'*20}\n")

            for entry in complex_entries:
                metric = entry['metric']
                original_ratings = entry.get('ratings', [])

                # Format original ratings with expert names
                ratings_str = ", ".join(
                    f"{expert_names[i]}:{r}" if i < len(expert_names) else str(r)
                    for i, r in enumerate(original_ratings)
                )

                f.write(f"  ⚡ {metric}\n")
                f.write(f"    Before: [{ratings_str}]\n")

                # Add debate result if available
                if debate_results and metric in debate_results:
                    final_rating = debate_results[metric]
                    if final_rating == "COMPLEX":
                        f.write(f"    Debate: 3 rounds (no majority)\n")
                        f.write(f"    Result: ⚠️ COMPLEX (requires user review)\n")
                    else:
                        f.write(f"    Debate: 3 rounds\n")
                        f.write(f"    Result: {final_rating} (consensus)\n")
                        # Add reason from matching LLM
                        reason = _find_matching_reason(metric, final_rating, original_analyses)
                        if reason:
                            reason_short = reason[:250] + "..." if len(reason) > 250 else reason
                            f.write(f"    {reason_short}\n")
                else:
                    f.write(f"    Debate: pending\n")

                f.write("\n")

        # --- OVERALL SUMMARY ---
        f.write(f"{'='*60}\n")
        f.write(f"  OVERALL SUMMARY\n")
        f.write(f"{'='*60}\n\n")

        # Collect all final ratings
        all_ratings = {}
        for entry in clear_entries:
            all_ratings[entry['metric']] = entry['result']

        # Add debate results (overwrite if debated)
        if debate_result:
            for metric, rating in debate_result.get('debate_results', {}).items():
                all_ratings[metric] = rating

        # Categorize metrics by rating
        strengths = []  # Excellent, Good
        watch = []      # Neutral
        concerns = []   # Bad, Horrible
        unresolved = [] # COMPLEX

        for metric, rating in all_ratings.items():
            rating_lower = rating.lower() if rating else ""
            if rating_lower in ('excellent', 'good'):
                strengths.append(f"{metric} ({rating})")
            elif rating_lower == 'neutral':
                watch.append(f"{metric}")
            elif rating_lower in ('bad', 'horrible'):
                concerns.append(f"{metric} ({rating})")
            elif rating_lower == 'complex' or rating == 'COMPLEX':
                unresolved.append(f"{metric}")

        # Calculate score (-16 to +16)
        score = _calculate_score(all_ratings)

        # Determine verdict label from score
        if score <= -11:
            verdict = "Extremely Risky"
        elif score <= -4:
            verdict = "Risky"
        elif score <= 3:
            verdict = "Neutral"
        elif score <= 10:
            verdict = "Safe"
        else:
            verdict = "Extremely Safe"

        # Calculate counts for display
        num_experts = len(original_analyses)
        total_metrics = len(all_ratings)
        num_positive = len(strengths)
        num_neutral = len(watch)
        num_negative = len(concerns)
        num_debates = len(complex_entries)
        num_resolved = num_debates - len(unresolved)

        # Write summary
        f.write(f"{ticker} Financial Summary ({num_experts} Experts, {total_metrics} Metrics)\n")
        f.write(f"{'─'*45}\n")

        # Score line with sign
        score_display = f"+{score}" if score > 0 else str(score)
        f.write(f"Score: {score_display}/16 → {verdict}\n\n")

        # Build metrics breakdown - only show counts that exist
        breakdown_parts = []
        if num_positive > 0:
            breakdown_parts.append(f"{num_positive} positive")
        if num_neutral > 0:
            breakdown_parts.append(f"{num_neutral} neutral")
        if num_negative > 0:
            breakdown_parts.append(f"{num_negative} negative")

        if breakdown_parts:
            f.write(f"Breakdown: {', '.join(breakdown_parts)}\n")

        if strengths:
            f.write(f"Strengths: {', '.join(strengths)}\n")
        if watch:
            f.write(f"Watch: {', '.join(watch)}\n")
        if concerns:
            f.write(f"Concerns: {', '.join(concerns)}\n")
        if unresolved:
            f.write(f"Unresolved: {', '.join(unresolved)} (COMPLEX - requires review)\n")

        f.write("\n")

        # Debate summary line
        if num_debates > 0:
            f.write(f"{num_debates} metric(s) debated, {num_resolved} resolved.\n")
        else:
            f.write(f"No debates needed. All experts aligned.\n")

        f.write("\n")


def _find_matching_reason(metric: str, final_rating: str, analyses: list) -> str:
    """
    Find the reason from the first LLM whose rating matches the final rating.

    Args:
        metric: The metric name (e.g., 'revenue')
        final_rating: The consensus rating (e.g., 'Good')
        analyses: List of LLM analyses with {metric} and {metric}_reason fields

    Returns:
        The reason string, or empty string if not found
    """
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
    """
    Calculate financial score from -16 to +16.

    Scoring:
        Excellent: +2
        Good: +1
        Neutral: 0
        Bad: -1
        Horrible: -2
        COMPLEX: 0 (unresolved)

    Args:
        ratings: Dict of {metric: rating}

    Returns:
        Score from -16 to +16
    """
    score_map = {
        'excellent': 2,
        'good': 1,
        'neutral': 0,
        'bad': -1,
        'horrible': -2,
        'complex': 0,  # Unresolved counts as neutral
    }

    total = 0
    for metric, rating in ratings.items():
        rating_lower = rating.lower() if rating else ""
        total += score_map.get(rating_lower, 0)

    return total


def log_debate_transcript(debate_result: dict, log_file: str = None):
    """
    Log the full debate transcript.

    Args:
        debate_result: Output from run_debate() containing transcript, results, and changes
        log_file: Optional specific log file path. Uses current session log if not provided.
    """
    filepath = log_file or _current_log_file

    if not filepath:
        print(f"Warning: No log file set. Call start_new_log() first.")
        return

    transcript = debate_result.get('transcript', [])
    debate_results = debate_result.get('debate_results', {})
    changes = debate_result.get('position_changes', [])

    # Get unique metrics from transcript
    metrics = []
    seen = set()
    for t in transcript:
        if t['metric'] not in seen:
            metrics.append(t['metric'])
            seen.add(t['metric'])

    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"  DEBATE TRANSCRIPT\n")
        f.write(f"{'='*60}\n\n")

        # Log each metric's debate
        for metric in metrics:
            f.write(f"--- {metric.upper()} ---\n\n")

            # Get entries for this metric
            metric_entries = [t for t in transcript if t['metric'] == metric]

            # Group by round
            rounds = {}
            for entry in metric_entries:
                round_key = entry['round']
                if round_key not in rounds:
                    rounds[round_key] = []
                rounds[round_key].append(entry)

            # Write each round
            for round_key in sorted(rounds.keys(), key=lambda x: (0, x) if isinstance(x, int) else (1, x)):
                round_label = f"Round {round_key}" if isinstance(round_key, int) else "FINAL"
                f.write(f"[{round_label}]\n")

                for entry in rounds[round_key]:
                    f.write(f"  {entry['llm']}:\n")
                    # Truncate long content for readability
                    content = entry['content']
                    if len(content) > 500:
                        content = content[:500] + "..."
                    # Indent content
                    indented = '\n    '.join(content.split('\n'))
                    f.write(f"    {indented}\n\n")

            # Final result for this metric
            final_rating = debate_results.get(metric, 'Unknown')
            if final_rating == "COMPLEX":
                f.write(f"→ RESULT: COMPLEX (no majority - requires user attention)\n\n")
            else:
                f.write(f"→ CONSENSUS: {final_rating}\n\n")

        # Position changes summary
        if changes:
            f.write(f"{'-'*40}\n")
            f.write("Position Changes During Debate:\n")
            for change in changes:
                f.write(f"  {change['llm']}: {change['metric']} ({change['from']} → {change['to']})\n")
            f.write("\n")

        # Overall summary
        f.write(f"{'-'*40}\n")
        f.write("Debate Results Summary:\n")
        for metric, rating in debate_results.items():
            status = "⚠️ COMPLEX" if rating == "COMPLEX" else f"✓ {rating}"
            f.write(f"  {metric}: {status}\n")

        f.write("\n")
