"""Hooks must fail open: memory errors never break the LLM call."""

from __future__ import annotations

import pytest

from infolang_litellm import InfoLangLogger

from .conftest import FailingAsyncClient, FailingSyncClient, make_response, user_messages


async def test_pre_call_fails_open() -> None:
    logger = InfoLangLogger(retain=False, client=FailingAsyncClient())
    data = {"messages": user_messages()}
    out = await logger.async_pre_call_hook({}, None, data, "completion")
    assert out is data
    assert len(out["messages"]) == 1  # unchanged


async def test_pre_call_raise_on_error() -> None:
    logger = InfoLangLogger(retain=False, raise_on_error=True, client=FailingAsyncClient())
    with pytest.raises(RuntimeError):
        await logger.async_pre_call_hook({}, None, {"messages": user_messages()}, "completion")


async def test_async_retain_fails_open() -> None:
    logger = InfoLangLogger(client=FailingAsyncClient())
    # Should not raise.
    await logger.async_log_success_event({"messages": user_messages()}, make_response(), 0, 1)


async def test_async_retain_raise_on_error() -> None:
    logger = InfoLangLogger(raise_on_error=True, client=FailingAsyncClient())
    with pytest.raises(RuntimeError):
        await logger.async_log_success_event(
            {"messages": user_messages()}, make_response(), 0, 1
        )


def test_sync_retain_fails_open() -> None:
    logger = InfoLangLogger(sync_client=FailingSyncClient())
    logger.log_success_event({"messages": user_messages()}, make_response(), 0, 1)


def test_sync_retain_raise_on_error() -> None:
    logger = InfoLangLogger(raise_on_error=True, sync_client=FailingSyncClient())
    with pytest.raises(RuntimeError):
        logger.log_success_event({"messages": user_messages()}, make_response(), 0, 1)
