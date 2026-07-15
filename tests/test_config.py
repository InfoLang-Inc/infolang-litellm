"""Tests for construction, validation, and client building."""

from __future__ import annotations

import pytest
from infolang import AsyncInfoLang, InfoLang
from litellm.integrations.custom_logger import CustomLogger

from infolang_litellm import InfoLangLogger

from .conftest import RecordingAsyncClient, RecordingSyncClient


def test_is_a_litellm_custom_logger() -> None:
    assert isinstance(InfoLangLogger(), CustomLogger)


def test_invalid_top_k_raises() -> None:
    with pytest.raises(ValueError):
        InfoLangLogger(top_k=0)


def test_invalid_inject_position_raises() -> None:
    with pytest.raises(ValueError):
        InfoLangLogger(inject_position="middle")


def test_defaults() -> None:
    logger = InfoLangLogger()
    assert logger.recall is True
    assert logger.retain is True
    assert logger.top_k == 5
    assert logger.call_types == ("completion", "acompletion")
    assert logger.last_recall_seconds is None


def test_call_types_coerced_to_tuple() -> None:
    logger = InfoLangLogger(call_types=["completion"])
    assert logger.call_types == ("completion",)


def test_injected_clients_returned() -> None:
    async_client = RecordingAsyncClient()
    sync_client = RecordingSyncClient()
    logger = InfoLangLogger(client=async_client, sync_client=sync_client)
    assert logger._async() is async_client
    assert logger._sync() is sync_client


def test_async_client_built_from_api_key() -> None:
    logger = InfoLangLogger(api_key="il_live_x", namespace="ns", workspace="ws")
    client = logger._async()
    assert isinstance(client, AsyncInfoLang)
    assert client.namespace == "ns"
    assert client.workspace == "ws"


def test_sync_client_built_from_api_key() -> None:
    logger = InfoLangLogger(api_key="il_live_x", namespace="ns")
    client = logger._sync()
    assert isinstance(client, InfoLang)
    assert client.namespace == "ns"


def test_async_client_env_fallback(monkeypatch) -> None:
    monkeypatch.setenv("INFOLANG_API_KEY", "il_live_env")
    logger = InfoLangLogger(namespace="envns")
    client = logger._async()
    assert isinstance(client, AsyncInfoLang)
    assert client.namespace == "envns"


def test_sync_client_env_fallback(monkeypatch) -> None:
    monkeypatch.setenv("INFOLANG_API_KEY", "il_live_env")
    logger = InfoLangLogger()
    assert isinstance(logger._sync(), InfoLang)


def test_client_built_once() -> None:
    logger = InfoLangLogger(api_key="il_live_x")
    first = logger._async()
    assert logger._async() is first
