"""InfoLang integration for LiteLLM.

A ``CustomLogger`` that recalls context before a call and remembers the turn
after, turning InfoLang into model-agnostic memory for the LiteLLM proxy or SDK
through the published ``infolang`` Python SDK.

Quickstart (SDK)::

    import litellm
    from infolang_litellm import InfoLangLogger

    litellm.callbacks = [InfoLangLogger(namespace="my-app")]
"""

from __future__ import annotations

from ._scope import Scope, resolve_scope
from ._version import __version__
from .logger import InfoLangLogger

__all__ = [
    "__version__",
    "InfoLangLogger",
    "Scope",
    "resolve_scope",
]
