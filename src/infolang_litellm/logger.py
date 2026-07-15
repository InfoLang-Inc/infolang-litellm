"""``InfoLangLogger`` — a LiteLLM ``CustomLogger`` that adds memory to any model.

- **Pre-call** (`async_pre_call_hook`): recall context for the latest user
  message and inject it as a system message before the LLM call.
- **Post-call** (`async_log_success_event` / `log_success_event`): remember the
  user turn + assistant response so the next call can recall it.

Everything is built on the published InfoLang SDK (``infolang>=0.2,<0.3``) — no
HTTP, no engine internals. Memory failures never break the LLM call: hooks fail
open and return the request unchanged unless ``raise_on_error=True``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from typing import Any

from infolang import AsyncInfoLang, InfoLang
from litellm.integrations.custom_logger import CustomLogger

from ._format import (
    _get,
    build_context_block,
    build_turn,
    last_message_text,
    response_text,
)
from ._scope import Scope, resolve_scope

_log = logging.getLogger("infolang_litellm")

DEFAULT_HEADER = "Relevant context from memory (InfoLang):"
DEFAULT_CALL_TYPES = ("completion", "acompletion")


class InfoLangLogger(CustomLogger):
    """Recall-before / remember-after memory callback for LiteLLM.

    ### Proxy usage (``config.yaml``)

    ```yaml
    litellm_settings:
      callbacks: ["infolang_callbacks.infolang_memory"]
    ```

    with an ``infolang_callbacks.py`` on the path:

    ```python
    from infolang_litellm import InfoLangLogger
    infolang_memory = InfoLangLogger(namespace="my-app")
    ```

    ### SDK usage

    ```python
    import litellm
    from infolang_litellm import InfoLangLogger

    litellm.callbacks = [InfoLangLogger(namespace="my-app")]
    ```
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        namespace: str | None = None,
        workspace: str | None = None,
        base_url: str | None = None,
        client: AsyncInfoLang | None = None,
        sync_client: InfoLang | None = None,
        recall: bool = True,
        retain: bool = True,
        top_k: int = 5,
        min_score: float = 0.0,
        max_context_chunks: int | None = None,
        context_header: str = DEFAULT_HEADER,
        inject_role: str = "system",
        inject_position: str = "end_of_system",
        source: str = "litellm",
        tags: str | None = None,
        retain_user: bool = True,
        retain_assistant: bool = True,
        namespace_metadata_key: str = "infolang_namespace",
        session_metadata_key: str = "infolang_session",
        use_session_as_namespace: bool = False,
        session_namespace_prefix: str = "",
        call_types: Iterable[str] = DEFAULT_CALL_TYPES,
        raise_on_error: bool = False,
    ) -> None:
        super().__init__()
        if top_k <= 0:
            raise ValueError(f"top_k must be greater than 0, got {top_k}")
        if inject_position not in ("start", "end_of_system", "before_last_user"):
            raise ValueError(f"invalid inject_position: {inject_position!r}")
        self.api_key = api_key
        self.namespace = namespace
        self.workspace = workspace
        self.base_url = base_url
        self._aclient = client
        self._sync_client = sync_client
        self.recall = recall
        self.retain = retain
        self.top_k = top_k
        self.min_score = min_score
        self.max_context_chunks = max_context_chunks
        self.context_header = context_header
        self.inject_role = inject_role
        self.inject_position = inject_position
        self.source = source
        self.tags = tags
        self.retain_user = retain_user
        self.retain_assistant = retain_assistant
        self.namespace_metadata_key = namespace_metadata_key
        self.session_metadata_key = session_metadata_key
        self.use_session_as_namespace = use_session_as_namespace
        self.session_namespace_prefix = session_namespace_prefix
        self.call_types = tuple(call_types)
        self.raise_on_error = raise_on_error
        # Observability: last measured hook latencies (seconds). See README.
        self.last_recall_seconds: float | None = None
        self.last_retain_seconds: float | None = None

    # --- clients --------------------------------------------------------

    def _async(self) -> AsyncInfoLang:
        if self._aclient is None:
            if self.api_key:
                self._aclient = AsyncInfoLang.from_api_key(
                    self.api_key,
                    namespace=self.namespace,
                    workspace=self.workspace,
                    base_url=self.base_url,
                )
            else:
                self._aclient = AsyncInfoLang(
                    namespace=self.namespace,
                    workspace=self.workspace,
                    base_url=self.base_url,
                )
        return self._aclient

    def _sync(self) -> InfoLang:
        if self._sync_client is None:
            if self.api_key:
                self._sync_client = InfoLang.from_api_key(
                    self.api_key,
                    namespace=self.namespace,
                    workspace=self.workspace,
                    base_url=self.base_url,
                )
            else:
                self._sync_client = InfoLang(
                    namespace=self.namespace,
                    workspace=self.workspace,
                    base_url=self.base_url,
                )
        return self._sync_client

    # --- scope + injection ---------------------------------------------

    def _resolve_scope(self, source: dict[str, Any]) -> Scope:
        return resolve_scope(
            source,
            namespace_key=self.namespace_metadata_key,
            session_key=self.session_metadata_key,
            default_namespace=self.namespace,
            use_session_as_namespace=self.use_session_as_namespace,
            session_namespace_prefix=self.session_namespace_prefix,
        )

    def _inject(self, messages: list[Any], block: str) -> list[Any]:
        context_message = {"role": self.inject_role, "content": block}
        if self.inject_position == "start":
            return [context_message, *messages]
        if self.inject_position == "before_last_user":
            index = len(messages)
            for i in range(len(messages) - 1, -1, -1):
                if _get(messages[i], "role") == "user":
                    index = i
                    break
            return [*messages[:index], context_message, *messages[index:]]
        # end_of_system: after any leading system messages
        index = 0
        for message in messages:
            if _get(message, "role") == "system":
                index += 1
            else:
                break
        return [*messages[:index], context_message, *messages[index:]]

    # --- pre-call recall ------------------------------------------------

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict[str, Any],
        call_type: str,
    ) -> dict[str, Any]:
        """Recall memory and inject it into ``data['messages']`` before the call."""

        if not self.recall or call_type not in self.call_types:
            return data
        try:
            messages = data.get("messages")
            query = last_message_text(messages)
            if not query or not isinstance(messages, list):
                return data
            scope = self._resolve_scope(data)
            started = time.perf_counter()
            result = await self._async().recall(
                query, namespace=scope.namespace, top_k=self.top_k
            )
            self.last_recall_seconds = time.perf_counter() - started
            block = build_context_block(
                result.chunks,
                header=self.context_header,
                min_score=self.min_score,
                max_chunks=self.max_context_chunks,
            )
            if block:
                data["messages"] = self._inject(messages, block)
        except Exception as exc:  # fail open: memory must never break the call
            if self.raise_on_error:
                raise
            _log.warning("InfoLang recall skipped: %s", exc)
        return data

    # --- post-call retain -----------------------------------------------

    def _turn_to_remember(
        self, kwargs: dict[str, Any], response_obj: Any
    ) -> str | None:
        query = last_message_text(kwargs.get("messages")) if self.retain_user else None
        answer = response_text(response_obj) if self.retain_assistant else None
        return build_turn(query, answer)

    async def async_log_success_event(
        self, kwargs: Any, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        """Remember the completed turn (async)."""

        if not self.retain or not isinstance(kwargs, dict):
            return
        try:
            text = self._turn_to_remember(kwargs, response_obj)
            if not text:
                return
            scope = self._resolve_scope(kwargs)
            started = time.perf_counter()
            await self._async().remember(
                text, namespace=scope.namespace, source=self.source, tags=self.tags
            )
            self.last_retain_seconds = time.perf_counter() - started
        except Exception as exc:  # fail open
            if self.raise_on_error:
                raise
            _log.warning("InfoLang retain skipped: %s", exc)

    def log_success_event(
        self, kwargs: Any, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        """Remember the completed turn (sync mirror for non-proxy SDK usage)."""

        if not self.retain or not isinstance(kwargs, dict):
            return
        try:
            text = self._turn_to_remember(kwargs, response_obj)
            if not text:
                return
            scope = self._resolve_scope(kwargs)
            started = time.perf_counter()
            self._sync().remember(
                text, namespace=scope.namespace, source=self.source, tags=self.tags
            )
            self.last_retain_seconds = time.perf_counter() - started
        except Exception as exc:  # fail open
            if self.raise_on_error:
                raise
            _log.warning("InfoLang retain skipped: %s", exc)
