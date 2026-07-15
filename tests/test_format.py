"""Tests for the pure message/response formatting helpers."""

from __future__ import annotations

from infolang_litellm._format import (
    build_context_block,
    build_turn,
    last_message_text,
    message_text,
    response_text,
)

from .conftest import make_chunk, make_response


def test_message_text_str() -> None:
    assert message_text({"role": "user", "content": "hi"}) == "hi"


def test_message_text_empty_str_is_none() -> None:
    assert message_text({"role": "user", "content": ""}) is None


def test_message_text_content_parts() -> None:
    msg = {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
    assert message_text(msg) == "a\nb"


def test_message_text_ignores_non_text_parts() -> None:
    msg = {
        "content": [
            {"type": "image_url", "image_url": {"url": "x"}},
            {"type": "text", "text": "t"},
        ]
    }
    assert message_text(msg) == "t"


def test_message_text_raw_string_parts() -> None:
    assert message_text({"content": ["a", "b"]}) == "a\nb"


def test_message_text_empty_list_is_none() -> None:
    assert message_text({"content": []}) is None


def test_message_text_unknown_type_is_none() -> None:
    assert message_text({"content": 42}) is None


def test_message_text_reads_object_attr() -> None:
    class M:
        content = "hello"

    assert message_text(M()) == "hello"


def test_last_message_text_picks_last_user() -> None:
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ]
    assert last_message_text(messages) == "second"


def test_last_message_text_skips_blank() -> None:
    messages = [{"role": "user", "content": "real"}, {"role": "user", "content": "   "}]
    assert last_message_text(messages) == "real"


def test_last_message_text_no_user_is_none() -> None:
    assert last_message_text([{"role": "system", "content": "s"}]) is None


def test_last_message_text_non_list_is_none() -> None:
    assert last_message_text(None) is None


def test_last_message_text_role_param() -> None:
    messages = [{"role": "assistant", "content": "answer"}]
    assert last_message_text(messages, role="assistant") == "answer"


def test_response_text_from_message() -> None:
    assert response_text(make_response("done")) == "done"


def test_response_text_from_parts() -> None:
    resp = {"choices": [{"message": {"content": [{"type": "text", "text": "p"}]}}]}
    assert response_text(resp) == "p"


def test_response_text_text_completion_shape() -> None:
    assert response_text({"choices": [{"text": "legacy"}]}) == "legacy"


def test_response_text_no_choices_is_none() -> None:
    assert response_text({"choices": []}) is None
    assert response_text({}) is None


def test_response_text_empty_content_is_none() -> None:
    assert response_text(make_response("")) is None


def test_build_context_block_basic() -> None:
    block = build_context_block([make_chunk(text="a"), make_chunk(id="c2", text="b")], header="H:")
    assert block == "H:\n- a\n- b"


def test_build_context_block_min_score() -> None:
    chunks = [make_chunk(text="hi", score=0.9), make_chunk(id="c2", text="lo", score=0.1)]
    block = build_context_block(chunks, header="H:", min_score=0.5)
    assert block == "H:\n- hi"


def test_build_context_block_max_chunks() -> None:
    chunks = [make_chunk(text="a"), make_chunk(id="c2", text="b"), make_chunk(id="c3", text="c")]
    block = build_context_block(chunks, header="H:", max_chunks=2)
    assert block is not None
    assert block.count("- ") == 2


def test_build_context_block_skips_blank() -> None:
    block = build_context_block([make_chunk(text="  ")], header="H:")
    assert block is None


def test_build_context_block_empty_is_none() -> None:
    assert build_context_block([], header="H:") is None


def test_build_turn_both() -> None:
    assert build_turn("q", "a") == "User: q\nAssistant: a"


def test_build_turn_only_query() -> None:
    assert build_turn("q", None) == "User: q"


def test_build_turn_only_answer() -> None:
    assert build_turn(None, "a") == "Assistant: a"


def test_build_turn_neither_is_none() -> None:
    assert build_turn(None, None) is None
    assert build_turn("  ", "") is None


def test_build_turn_custom_labels() -> None:
    assert build_turn("q", "a", user_label="Q", assistant_label="A") == "Q: q\nA: a"
