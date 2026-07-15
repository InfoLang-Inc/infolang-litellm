# infolang-litellm — agent instructions

InfoLang semantic-memory callback for **LiteLLM** (proxy + SDK). Package import
name: `infolang_litellm`; PyPI name: `infolang-litellm`.

## Frozen contract

- Depend **only** on the published public SDK: `infolang>=0.2,<0.3` (PyPI) and
  `litellm>=1.72,<2`. Never reimplement HTTP, import runtime/engine internals,
  or reference core-ip.
- SDK surface used: `from infolang import InfoLang, AsyncInfoLang`;
  `recall(query, namespace=, top_k=)`, `remember(text, namespace=, source=, tags=)`.
- Scoping: `workspace` = tenant, `namespace` = bank; per-request namespace/session
  resolved from LiteLLM metadata.

## Architecture

- `src/infolang_litellm/logger.py` — `InfoLangLogger(CustomLogger)`:
  - `async_pre_call_hook` — recall + inject a system message into `data["messages"]`.
  - `async_log_success_event` / `log_success_event` — remember the turn (async/sync).
- `src/infolang_litellm/_scope.py` — resolve namespace/session from `data`/`kwargs`
  metadata (`metadata`, `litellm_metadata`, `litellm_params.metadata`, `user`).
- `src/infolang_litellm/_format.py` — pure message/response text extraction and
  context/turn shaping (handles str and content-parts message shapes).

## Rules

- Hooks **fail open**: a memory error must never break the LLM call (log + return
  the request unchanged) unless `raise_on_error=True`.
- Verify hook signatures against the installed LiteLLM `CustomLogger` — do not
  assume. This package targets LiteLLM 1.72+ (verified against 1.92.0).
- Tests mock the InfoLang client (offline default). Live tests carry the `live`
  marker and are deselected unless explicitly run.

## Commands

```bash
pip install -e ".[dev]"
ruff check .
mypy
pytest
```
