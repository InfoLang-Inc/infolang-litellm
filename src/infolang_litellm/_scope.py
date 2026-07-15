"""Resolve the InfoLang namespace/session to use for a given request.

LiteLLM surfaces caller-supplied metadata in a few shapes depending on whether
we're in the proxy pre-call path (``data``) or the logging path (``kwargs``).
These helpers look in all the usual places so a single ``metadata={...}`` on the
request keys memory consistently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Scope:
    """Resolved memory scope for a request."""

    namespace: str | None
    session: str | None


def _as_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def candidate_metadata(source: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect the metadata dicts LiteLLM may attach, in priority order."""

    metas: list[dict[str, Any]] = []
    for key in ("metadata", "litellm_metadata"):
        value = source.get(key)
        if isinstance(value, dict):
            metas.append(value)
    params = source.get("litellm_params")
    if isinstance(params, dict):
        nested = params.get("metadata")
        if isinstance(nested, dict):
            metas.append(nested)
    return metas


def meta_value(source: dict[str, Any], key: str) -> str | None:
    """First string value for ``key`` across metadata dicts, then top level."""

    for meta in candidate_metadata(source):
        found = _as_str(meta.get(key))
        if found is not None:
            return found
    return _as_str(source.get(key))


def resolve_scope(
    source: dict[str, Any],
    *,
    namespace_key: str,
    session_key: str,
    default_namespace: str | None,
    use_session_as_namespace: bool = False,
    session_namespace_prefix: str = "",
) -> Scope:
    """Resolve the namespace + session for a request.

    Precedence for namespace:

    1. explicit ``namespace_key`` in metadata (or top level);
    2. the session id (optionally prefixed) when ``use_session_as_namespace``;
    3. the logger's ``default_namespace``.
    """

    if not isinstance(source, dict):
        return Scope(namespace=default_namespace, session=None)
    session = meta_value(source, session_key) or _as_str(source.get("user"))
    explicit_ns = meta_value(source, namespace_key)
    if explicit_ns is not None:
        namespace: str | None = explicit_ns
    elif use_session_as_namespace and session is not None:
        namespace = f"{session_namespace_prefix}{session}"
    else:
        namespace = default_namespace
    return Scope(namespace=namespace, session=session)
