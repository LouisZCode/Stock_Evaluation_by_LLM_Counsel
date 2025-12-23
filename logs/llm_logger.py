"""
Logger for capturing full LLM conversations including tool calls and responses.

You can find:
start_new_log, log_llm_conversation, log_debate_check
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


def _write_comparison_table(f, comparison: dict, label: str):
    """Helper to write a comparison table to log file."""
    f.write(f"\n{label}:\n")
    for row in comparison.get('rows', []):
        metric = row['metric']
        ratings = row['ratings']
        spread = row['spread']

        # Format ratings (replace None with 'missing')
        formatted_ratings = [r if r else 'missing' for r in ratings]
        ratings_str = ' | '.join(f"{r:10}" for r in formatted_ratings)

        spread_str = f"(spread: {spread})" if spread is not None else "(-)"
        f.write(f"  {metric:20}: {ratings_str} {spread_str}\n")

    missing_counts = comparison.get('missing_counts', [])
    if missing_counts:
        f.write(f"  Missing counts: {missing_counts}\n")


def log_debate_check(original_agreement: dict, recalc_agreement: dict = None,
                     original_comparison: dict = None, filled_comparison: dict = None,
                     log_file: str = None):
    """
    Logs the debate check results after analyzing LLM agreement.

    Args:
        original_agreement: Output from calculate_agreement() on original LLM responses
        recalc_agreement: Output from calculate_agreement_from_scores() after filling (optional)
        original_comparison: Output from get_metric_comparison() on original data (optional)
        filled_comparison: Output from get_metric_comparison() after filling (optional)
        log_file: Optional specific log file path. Uses current session log if not provided.
    """
    filepath = log_file or _current_log_file

    if not filepath:
        print(f"Warning: No log file set. Call start_new_log() first.")
        return

    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"  DEBATE CHECK\n")
        f.write(f"{'='*60}\n\n")

        # Original scores
        orig_scores = original_agreement.get('scores', [])
        orig_spread = original_agreement.get('score_spread', 'N/A')
        orig_level = original_agreement.get('debate_level', 'none').upper()
        f.write(f"Original Scores: {orig_scores} (spread: {orig_spread}) → {orig_level}\n")

        # Original comparison table
        if original_comparison:
            _write_comparison_table(f, original_comparison, "Metric Comparison (original)")

        f.write(f"\n{'-'*40}\n")

        # Recalculated scores (if provided)
        if recalc_agreement:
            recalc_scores = recalc_agreement.get('scores', [])
            recalc_spread = recalc_agreement.get('score_spread', 'N/A')
            recalc_level = recalc_agreement.get('debate_level', 'none').upper()
            f.write(f"\nRecalculated:    {recalc_scores} (spread: {recalc_spread}) → {recalc_level}\n")

        # Filled comparison table
        if filled_comparison:
            _write_comparison_table(f, filled_comparison, "Metric Comparison (after filling)")

        disagreements = original_agreement.get('metric_disagreements', [])
        if disagreements:
            f.write(f"\nMetric Disagreements: {', '.join(disagreements)}\n")

        # Final decision (use recalculated if available, otherwise original)
        final_agreement = recalc_agreement if recalc_agreement else original_agreement
        level = final_agreement.get('debate_level', 'none')

        if level == 'none':
            f.write(f"\n→ Final decision: NO debate (LLMs aligned after recalculation)\n")
        elif level == 'small':
            f.write(f"\n→ Final decision: SMALL debate (1-2 rounds)\n")
        else:
            f.write(f"\n→ Final decision: LARGE debate (3 rounds)\n")

        f.write(f"\n")
