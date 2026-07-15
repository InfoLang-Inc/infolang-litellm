# infolang-litellm

First-party [InfoLang](https://infolang.ai) semantic-memory callback for
[LiteLLM](https://docs.litellm.ai) — add long-term memory to **any** model with
one config block.

`InfoLangLogger` is a LiteLLM `CustomLogger` that:

- **recalls** relevant context before the call and injects it as a system
  message (`async_pre_call_hook`), and
- **remembers** the user turn + assistant response after the call
  (`async_log_success_event` / `log_success_event`).

Built entirely on the published `infolang` Python SDK — no HTTP, no engine
internals. Memory failures **never** break the LLM call: hooks fail open.

## Install

```bash
pip install infolang-litellm
```

Pulls the published SDK (`infolang>=0.2,<0.3`) and `litellm>=1.72`.

## Configure

```bash
export INFOLANG_API_KEY="il_live_..."
# optional
export INFOLANG_NAMESPACE="my-app"     # default bank
export INFOLANG_WORKSPACE="acme"       # tenant
```

## Quickstart — LiteLLM Proxy (zero code changes)

Create `infolang_callbacks.py` on the proxy's Python path:

```python
from infolang_litellm import InfoLangLogger

infolang_memory = InfoLangLogger(namespace="my-app")
```

Reference it in `config.yaml`:

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o

litellm_settings:
  callbacks: ["infolang_callbacks.infolang_memory"]
```

Now every request through the proxy recalls context before the model call and
retains the turn after. Key memory per user/session by passing metadata:

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "what did we decide about auth?"}],
    "metadata": {"infolang_namespace": "team-acme", "infolang_session": "user-42"}
  }'
```

## Quickstart — LiteLLM SDK

```python
import litellm
from infolang_litellm import InfoLangLogger

litellm.callbacks = [InfoLangLogger(namespace="my-app")]

litellm.completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "remember: deploys run on Fridays"}],
)
# a later call recalls it:
litellm.completion(model="gpt-4o", messages=[{"role": "user", "content": "when do we deploy?"}])
```

## Scoping (namespace / session)

Per-request scope is resolved from LiteLLM metadata, checked in
`metadata`, `litellm_metadata`, `litellm_params.metadata`, then top level:

| Config | Meaning |
|---|---|
| `infolang_namespace` metadata | memory bank for this request (overrides default) |
| `infolang_session` metadata / `user` field | session id |
| `use_session_as_namespace=True` | derive namespace from the session id (`session_namespace_prefix` + session) for per-user isolation |

Scoping follows the InfoLang contract: `workspace` = tenant, `namespace` = bank;
managed keys honor `namespace` on both reads and writes.

## Configuration

`InfoLangLogger(...)` options (all optional):

| Option | Default | Description |
|---|---|---|
| `namespace` / `workspace` / `base_url` | `None` | default scope + endpoint (falls back to `INFOLANG_*` env) |
| `api_key` | `None` | overrides `INFOLANG_API_KEY` |
| `recall` / `retain` | `True` | toggle each side |
| `top_k` | `5` | chunks to recall |
| `min_score` | `0.0` | drop recalled chunks below this score |
| `max_context_chunks` | `None` | cap injected chunks |
| `context_header` | `"Relevant context from memory (InfoLang):"` | injected block header |
| `inject_role` | `"system"` | role of the injected message |
| `inject_position` | `"end_of_system"` | `start`, `end_of_system`, or `before_last_user` |
| `retain_user` / `retain_assistant` | `True` | which sides of the turn to store |
| `source` / `tags` | `"litellm"` / `None` | metadata attached to remembered turns |
| `raise_on_error` | `False` | if `True`, surface memory errors instead of failing open |

## Latency budget

Recall injection sits on the request hot path. Following WP5 discipline, this
package **does not** claim "zero overhead." The added latency is one
`recall` round-trip to `api.infolang.ai` before the model call; retain happens in
the post-call logging path and is off the response's critical path.

Every hook records its measured wall time so you can publish honest numbers:

```python
logger = InfoLangLogger(namespace="my-app")
# after a call:
print(logger.last_recall_seconds, logger.last_retain_seconds)
```

Measure p50/p95 against your endpoint before publishing overhead figures; do not
assume the local dev numbers hold for the managed edge.

## Acceptance demo

The same prompt through the hooks twice — the second run recalls what the first
retained — is exercised by `tests/test_live.py::test_live_recall_after_retain`.

## Development

```bash
pip install -e ".[dev]"
ruff check .
mypy
pytest            # offline; live tests are deselected by default
```

Live end-to-end tests hit a real endpoint and are opt-in:

```bash
INFOLANG_API_KEY=il_live_... pytest -m live
```

## License

Apache-2.0. See [LICENSE](LICENSE).
