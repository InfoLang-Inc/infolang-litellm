"""Pure helpers to read messages/responses and shape memory text.

No I/O, no LiteLLM/InfoLang imports beyond the SDK ``Chunk`` type — this keeps
the message-handling logic trivially testable and stable across LiteLLM's
str-or-parts content shapes.
"""

from __future__ import annotations

from typing import Any

from infolang import Chunk


def _get(obj: Any, key: str) -> Any:
    """Read ``key`` from a dict or an attribute from an object."""

    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def message_text(message: Any) -> str | None:
    """Extract plain text from a chat message (str content or content parts)."""

    content = _get(message, "content")
    if isinstance(content, str):
        return content or None
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
            elif isinstance(part, str) and part:
                parts.append(part)
        return "\n".join(parts) if parts else None
    return None


def last_message_text(messages: Any, role: str = "user") -> str | None:
    """Return the text of the last message with ``role`` (default: user)."""

    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if _get(message, "role") == role:
            text = message_text(message)
            if text and text.strip():
                return text
    return None


def response_text(response_obj: Any) -> str | None:
    """Extract the assistant text from a LiteLLM completion response."""

    choices = _get(response_obj, "choices")
    if not choices:
        return None
    first = choices[0]
    message = _get(first, "message")
    if message is not None:
        content = _get(message, "content")
        if isinstance(content, str):
            return content or None
        return message_text(message)
    text = _get(first, "text")  # text-completion shape
    return text if isinstance(text, str) and text else None


def build_context_block(
    chunks: list[Chunk],
    *,
    header: str,
    min_score: float = 0.0,
    max_chunks: int | None = None,
) -> str | None:
    """Render recalled chunks into a single injectable context block."""

    lines: list[str] = []
    for chunk in chunks:
        if chunk.score is not None and chunk.score < min_score:
            continue
        text = (chunk.text or "").strip()
        if not text:
            continue
        lines.append(f"- {text}")
        if max_chunks is not None and len(lines) >= max_chunks:
            break
    if not lines:
        return None
    return header + "\n" + "\n".join(lines)


def build_turn(
    query: str | None,
    answer: str | None,
    *,
    user_label: str = "User",
    assistant_label: str = "Assistant",
) -> str | None:
    """Combine a user query and assistant answer into one memory record."""

    parts: list[str] = []
    if query and query.strip():
        parts.append(f"{user_label}: {query.strip()}")
    if answer and answer.strip():
        parts.append(f"{assistant_label}: {answer.strip()}")
    return "\n".join(parts) if parts else None
