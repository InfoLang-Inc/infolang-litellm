"""Shared fixtures: offline fake InfoLang clients and response builders.

The suite runs offline by default (no network, no credentials) by injecting
these fakes into ``InfoLangLogger``. Live tests carry the ``live`` marker and
are deselected unless explicitly requested.
"""

from __future__ import annotations

from typing import Any

import pytest
from infolang import Chunk, RecallResult, RememberResult


def make_chunk(
    id: str = "c1",
    text: str = "remembered fact",
    score: float | None = 0.9,
    tags: str | None = None,
) -> Chunk:
    return Chunk(id=id, text=text, score=score, tags=tags)


def make_recall_result(
    chunks: list[Chunk] | None = None, namespace: str | None = "ns"
) -> RecallResult:
    return RecallResult(chunks=chunks or [], namespace=namespace)


def make_remember_result(memory_id: str | None = "m1") -> RememberResult:
    return RememberResult.model_validate({"id": memory_id, "namespace": "ns"})


def make_response(content: str | None = "assistant answer") -> dict[str, Any]:
    """A minimal OpenAI-style completion response (dict form)."""

    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def user_messages(text: str = "what is the plan?") -> list[dict[str, Any]]:
    return [{"role": "user", "content": text}]


class RecordingAsyncClient:
    """Stand-in for :class:`infolang.AsyncInfoLang` that records calls."""

    def __init__(
        self,
        recall_result: RecallResult | None = None,
        remember_result: RememberResult | None = None,
    ) -> None:
        self.recall_result = recall_result or make_recall_result([make_chunk()])
        self.remember_result = remember_result or make_remember_result()
        self.calls: list[tuple[str, Any, dict[str, Any]]] = []

    async def recall(
        self,
        query: str,
        *,
        namespace: str | None = None,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        verbose: bool | None = None,
    ) -> RecallResult:
        self.calls.append(("recall", query, {"namespace": namespace, "top_k": top_k}))
        return self.recall_result

    async def remember(
        self,
        text: str,
        *,
        namespace: str | None = None,
        source: str | None = None,
        tags: str | None = None,
    ) -> RememberResult:
        self.calls.append(
            ("remember", text, {"namespace": namespace, "source": source, "tags": tags})
        )
        return self.remember_result


class RecordingSyncClient:
    """Stand-in for :class:`infolang.InfoLang` that records calls."""

    def __init__(self, remember_result: RememberResult | None = None) -> None:
        self.remember_result = remember_result or make_remember_result()
        self.calls: list[tuple[str, Any, dict[str, Any]]] = []

    def remember(
        self,
        text: str,
        *,
        namespace: str | None = None,
        source: str | None = None,
        tags: str | None = None,
    ) -> RememberResult:
        self.calls.append(
            ("remember", text, {"namespace": namespace, "source": source, "tags": tags})
        )
        return self.remember_result


class FailingAsyncClient:
    async def recall(self, *args: Any, **kwargs: Any) -> RecallResult:
        raise RuntimeError("boom")

    async def remember(self, *args: Any, **kwargs: Any) -> RememberResult:
        raise RuntimeError("boom")


class FailingSyncClient:
    def remember(self, *args: Any, **kwargs: Any) -> RememberResult:
        raise RuntimeError("boom")


@pytest.fixture
def async_client() -> RecordingAsyncClient:
    return RecordingAsyncClient()


@pytest.fixture
def sync_client() -> RecordingSyncClient:
    return RecordingSyncClient()
