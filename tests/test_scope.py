"""Tests for namespace/session scope resolution."""

from __future__ import annotations

from infolang_litellm._scope import candidate_metadata, meta_value, resolve_scope


def test_candidate_metadata_collects_all_shapes() -> None:
    source = {
        "metadata": {"a": 1},
        "litellm_metadata": {"b": 2},
        "litellm_params": {"metadata": {"c": 3}},
    }
    metas = candidate_metadata(source)
    assert {"a": 1} in metas
    assert {"b": 2} in metas
    assert {"c": 3} in metas


def test_meta_value_finds_in_metadata() -> None:
    source = {"metadata": {"infolang_namespace": "ns1"}}
    assert meta_value(source, "infolang_namespace") == "ns1"


def test_meta_value_top_level_fallback() -> None:
    assert meta_value({"infolang_namespace": "ns2"}, "infolang_namespace") == "ns2"


def test_meta_value_none_when_missing() -> None:
    assert meta_value({"metadata": {}}, "infolang_namespace") is None


def test_meta_value_ignores_blank() -> None:
    assert meta_value({"metadata": {"k": "  "}}, "k") is None


def test_resolve_scope_explicit_namespace() -> None:
    scope = resolve_scope(
        {"metadata": {"infolang_namespace": "explicit"}},
        namespace_key="infolang_namespace",
        session_key="infolang_session",
        default_namespace="fallback",
    )
    assert scope.namespace == "explicit"


def test_resolve_scope_session_from_user_field() -> None:
    scope = resolve_scope(
        {"user": "user-123"},
        namespace_key="infolang_namespace",
        session_key="infolang_session",
        default_namespace="fallback",
    )
    assert scope.session == "user-123"
    assert scope.namespace == "fallback"


def test_resolve_scope_session_as_namespace_with_prefix() -> None:
    scope = resolve_scope(
        {"metadata": {"infolang_session": "s9"}},
        namespace_key="infolang_namespace",
        session_key="infolang_session",
        default_namespace=None,
        use_session_as_namespace=True,
        session_namespace_prefix="chat:",
    )
    assert scope.session == "s9"
    assert scope.namespace == "chat:s9"


def test_resolve_scope_default_namespace() -> None:
    scope = resolve_scope(
        {},
        namespace_key="infolang_namespace",
        session_key="infolang_session",
        default_namespace="def",
    )
    assert scope.namespace == "def"
    assert scope.session is None


def test_resolve_scope_non_dict_source() -> None:
    scope = resolve_scope(
        "not-a-dict",  # type: ignore[arg-type]
        namespace_key="infolang_namespace",
        session_key="infolang_session",
        default_namespace="def",
    )
    assert scope.namespace == "def"
    assert scope.session is None


def test_resolve_scope_explicit_beats_session_as_namespace() -> None:
    scope = resolve_scope(
        {"metadata": {"infolang_namespace": "explicit", "infolang_session": "s"}},
        namespace_key="infolang_namespace",
        session_key="infolang_session",
        default_namespace=None,
        use_session_as_namespace=True,
    )
    assert scope.namespace == "explicit"
