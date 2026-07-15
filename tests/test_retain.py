"""Tests for the post-call retain hooks (async + sync)."""

from __future__ import annotations

from infolang_litellm import InfoLangLogger

from .conftest import (
    RecordingAsyncClient,
    RecordingSyncClient,
    make_response,
    user_messages,
)


async def test_async_remembers_full_turn(async_client: RecordingAsyncClient) -> None:
    logger = InfoLangLogger(namespace="ns", source="litellm", client=async_client)
    kwargs = {"messages": user_messages("the question")}
    await logger.async_log_success_event(kwargs, make_response("the answer"), 0, 1)
    op, text, meta = async_client.calls[0]
    assert op == "remember"
    assert "User: the question" in text
    assert "Assistant: the answer" in text
    assert meta == {"namespace": "ns", "source": "litellm", "tags": None}


async def test_async_retain_disabled(async_client: RecordingAsyncClient) -> None:
    logger = InfoLangLogger(retain=False, client=async_client)
    await logger.async_log_success_event({"messages": user_messages()}, make_response(), 0, 1)
    assert async_client.calls == []


async def test_retain_user_only(async_client: RecordingAsyncClient) -> None:
    logger = InfoLangLogger(retain_assistant=False, client=async_client)
    await logger.async_log_success_event(
        {"messages": user_messages("q")}, make_response("a"), 0, 1
    )
    _, text, _ = async_client.calls[0]
    assert "User: q" in text
    assert "Assistant" not in text


async def test_retain_assistant_only(async_client: RecordingAsyncClient) -> None:
    logger = InfoLangLogger(retain_user=False, client=async_client)
    await logger.async_log_success_event(
        {"messages": user_messages("q")}, make_response("a"), 0, 1
    )
    _, text, _ = async_client.calls[0]
    assert "Assistant: a" in text
    assert "User" not in text


async def test_no_content_no_remember(async_client: RecordingAsyncClient) -> None:
    logger = InfoLangLogger(client=async_client)
    await logger.async_log_success_event({"messages": []}, make_response(None), 0, 1)
    assert async_client.calls == []


async def test_namespace_from_metadata(async_client: RecordingAsyncClient) -> None:
    logger = InfoLangLogger(namespace="def", client=async_client)
    kwargs = {
        "messages": user_messages(),
        "litellm_params": {"metadata": {"infolang_namespace": "from-meta"}},
    }
    await logger.async_log_success_event(kwargs, make_response(), 0, 1)
    assert async_client.calls[0][2]["namespace"] == "from-meta"


async def test_tags_passed(async_client: RecordingAsyncClient) -> None:
    logger = InfoLangLogger(tags="chat", client=async_client)
    await logger.async_log_success_event({"messages": user_messages()}, make_response(), 0, 1)
    assert async_client.calls[0][2]["tags"] == "chat"


async def test_async_kwargs_not_dict_is_noop(async_client: RecordingAsyncClient) -> None:
    logger = InfoLangLogger(client=async_client)
    await logger.async_log_success_event(None, make_response(), 0, 1)
    assert async_client.calls == []


async def test_records_retain_latency(async_client: RecordingAsyncClient) -> None:
    logger = InfoLangLogger(client=async_client)
    await logger.async_log_success_event({"messages": user_messages()}, make_response(), 0, 1)
    assert logger.last_retain_seconds is not None


def test_sync_remembers_turn(sync_client: RecordingSyncClient) -> None:
    logger = InfoLangLogger(namespace="ns", sync_client=sync_client)
    logger.log_success_event({"messages": user_messages("q")}, make_response("a"), 0, 1)
    op, text, meta = sync_client.calls[0]
    assert op == "remember"
    assert "User: q" in text and "Assistant: a" in text
    assert meta["namespace"] == "ns"


def test_sync_retain_disabled(sync_client: RecordingSyncClient) -> None:
    logger = InfoLangLogger(retain=False, sync_client=sync_client)
    logger.log_success_event({"messages": user_messages()}, make_response(), 0, 1)
    assert sync_client.calls == []


def test_sync_kwargs_not_dict_is_noop(sync_client: RecordingSyncClient) -> None:
    logger = InfoLangLogger(sync_client=sync_client)
    logger.log_success_event("nope", make_response(), 0, 1)
    assert sync_client.calls == []


def test_sync_no_content_no_remember(sync_client: RecordingSyncClient) -> None:
    logger = InfoLangLogger(sync_client=sync_client)
    logger.log_success_event({"messages": []}, make_response(None), 0, 1)
    assert sync_client.calls == []
