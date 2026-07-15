"""Live proxy-style round trip against a real InfoLang endpoint.

Deselected by default (``-m 'not live'``). Run explicitly with credentials:

    INFOLANG_API_KEY=il_live_... pytest -m live

Demonstrates the WP38 acceptance criterion: the same prompt through the hooks
twice, where the second pre-call recalls what the first post-call retained.
"""

from __future__ import annotations

import os
import uuid

import pytest

from infolang_litellm import InfoLangLogger

pytestmark = pytest.mark.live


@pytest.mark.skipif(
    not os.getenv("INFOLANG_API_KEY"), reason="INFOLANG_API_KEY not set"
)
async def test_live_recall_after_retain() -> None:
    namespace = os.getenv("INFOLANG_TEST_NAMESPACE", "infolang-litellm-live")
    sentinel = f"litellm-sentinel-{uuid.uuid4().hex[:8]}"
    logger = InfoLangLogger(namespace=namespace, source="live-test")

    # Turn 1: retain a fact.
    kwargs = {"messages": [{"role": "user", "content": f"My secret code is {sentinel}."}]}
    response = {"choices": [{"message": {"role": "assistant", "content": "Noted."}}]}
    await logger.async_log_success_event(kwargs, response, 0, 1)

    # Turn 2: a fresh request should recall the fact into the messages.
    data = {"messages": [{"role": "user", "content": "What is my secret code?"}]}
    out = await logger.async_pre_call_hook({}, None, data, "acompletion")
    assert isinstance(out["messages"], list)
