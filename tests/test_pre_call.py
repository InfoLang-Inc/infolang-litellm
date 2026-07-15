"""Tests for the pre-call recall/injection hook."""

from __future__ import annotations

from infolang_litellm import InfoLangLogger

from .conftest import RecordingAsyncClient, make_chunk, make_recall_result, user_messages


async def test_injects_context_after_leading_system(async_client: RecordingAsyncClient) -> None:
    async_client.recall_result = make_recall_result([make_chunk(text="fact")])
    logger = InfoLangLogger(namespace="ns", retain=False, client=async_client)
    data = {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]}
    out = await logger.async_pre_call_hook({}, None, data, "completion")
    msgs = out["messages"]
    assert len(msgs) == 3
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == "sys"
    assert msgs[1]["role"] == "system" and "fact" in msgs[1]["content"]
    assert msgs[2]["role"] == "user"


async def test_inject_position_start(async_client: RecordingAsyncClient) -> None:
    logger = InfoLangLogger(inject_position="start", retain=False, client=async_client)
    data = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]}
    out = await logger.async_pre_call_hook({}, None, data, "completion")
    assert out["messages"][0]["role"] == "system"
    assert "remembered fact" in out["messages"][0]["content"]
    assert out["messages"][1]["content"] == "s"


async def test_inject_position_before_last_user(async_client: RecordingAsyncClient) -> None:
    logger = InfoLangLogger(inject_position="before_last_user", retain=False, client=async_client)
    data = {
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "new"},
        ]
    }
    out = await logger.async_pre_call_hook({}, None, data, "completion")
    msgs = out["messages"]
    assert msgs[2]["role"] == "system"
    assert msgs[3]["content"] == "new"


async def test_no_messages_returns_unchanged(async_client: RecordingAsyncClient) -> None:
    logger = InfoLangLogger(retain=False, client=async_client)
    data: dict = {"model": "gpt-4o"}
    out = await logger.async_pre_call_hook({}, None, data, "completion")
    assert out is data
    assert async_client.calls == []


async def test_no_user_message_skips_recall(async_client: RecordingAsyncClient) -> None:
    logger = InfoLangLogger(retain=False, client=async_client)
    data = {"messages": [{"role": "system", "content": "s"}]}
    out = await logger.async_pre_call_hook({}, None, data, "completion")
    assert out["messages"] == data["messages"]
    assert async_client.calls == []


async def test_recall_disabled(async_client: RecordingAsyncClient) -> None:
    logger = InfoLangLogger(recall=False, client=async_client)
    data = {"messages": user_messages()}
    out = await logger.async_pre_call_hook({}, None, data, "completion")
    assert len(out["messages"]) == 1
    assert async_client.calls == []


async def test_call_type_not_in_set_skips(async_client: RecordingAsyncClient) -> None:
    logger = InfoLangLogger(retain=False, client=async_client)
    data = {"messages": user_messages()}
    out = await logger.async_pre_call_hook({}, None, data, "embedding")
    assert len(out["messages"]) == 1
    assert async_client.calls == []


async def test_empty_chunks_no_injection(async_client: RecordingAsyncClient) -> None:
    async_client.recall_result = make_recall_result([])
    logger = InfoLangLogger(retain=False, client=async_client)
    data = {"messages": user_messages()}
    out = await logger.async_pre_call_hook({}, None, data, "completion")
    assert len(out["messages"]) == 1
    assert async_client.calls[0][0] == "recall"


async def test_min_score_filters_all(async_client: RecordingAsyncClient) -> None:
    async_client.recall_result = make_recall_result([make_chunk(score=0.1)])
    logger = InfoLangLogger(min_score=0.5, retain=False, client=async_client)
    data = {"messages": user_messages()}
    out = await logger.async_pre_call_hook({}, None, data, "completion")
    assert len(out["messages"]) == 1


async def test_namespace_from_metadata_passed_to_recall(async_client: RecordingAsyncClient) -> None:
    logger = InfoLangLogger(namespace="default-ns", retain=False, client=async_client)
    data = {
        "messages": user_messages(),
        "metadata": {"infolang_namespace": "req-ns"},
    }
    await logger.async_pre_call_hook({}, None, data, "completion")
    assert async_client.calls[0][2]["namespace"] == "req-ns"
    assert async_client.calls[0][2]["top_k"] == 5


async def test_records_recall_latency(async_client: RecordingAsyncClient) -> None:
    logger = InfoLangLogger(retain=False, client=async_client)
    await logger.async_pre_call_hook({}, None, {"messages": user_messages()}, "completion")
    assert logger.last_recall_seconds is not None
    assert logger.last_recall_seconds >= 0


def test_inject_before_last_user_without_user_appends() -> None:
    logger = InfoLangLogger(inject_position="before_last_user")
    messages = [{"role": "system", "content": "s"}, {"role": "assistant", "content": "a"}]
    out = logger._inject(messages, "CTX")
    assert out[-1] == {"role": "system", "content": "CTX"}


def test_inject_end_of_system_all_system_appends() -> None:
    logger = InfoLangLogger(inject_position="end_of_system")
    messages = [{"role": "system", "content": "s1"}, {"role": "system", "content": "s2"}]
    out = logger._inject(messages, "CTX")
    assert out[-1] == {"role": "system", "content": "CTX"}
    assert len(out) == 3
