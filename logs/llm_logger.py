"""
Logger for capturing full LLM conversations including tool calls and responses.

You can find:
start_new_log, log_llm_conversation
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
