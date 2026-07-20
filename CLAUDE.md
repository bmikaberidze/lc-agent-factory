# lc-agent-factory

Config-driven wrapper around LangChain v1's `create_agent`, packaged for PyPI.

## Stack

- **LangChain v1** (`langchain>=1.2`) — `create_agent`, `init_chat_model`, middleware
- **LangGraph v1** (`langgraph>=1.1`) — underpins `create_agent` under the hood
- **Python 3.10+**, build via **hatchling**, linting via **ruff**, tests via **pytest**

## Project layout

```
src/lc_agent_factory/
  __init__.py              # exports create_agent
  factory.py               # create_agent.create() — main entry point
  tools/__init__.py        # get_tools(config) — tool loading
  middleware/
    __init__.py            # re-exports get_prebuilt_middleware, get_custom_middleware
    _prebuilt.py           # wraps LangChain built-in middleware from config
    _custom.py             # custom middleware list (currently: handle_tool_errors)
    tool_error_handler.py  # @wrap_tool_call middleware that catches tool exceptions
  utils/
    enums.py               # LLMProviderSE, ComplexPrebuiltMiddlewareSE
```

## Key design decisions

- `create_agent.create(config, **kwargs)` accepts a `SimpleNamespace` config (typically parsed from YAML/JSON) and builds a fully-configured agent.
- Provider packages (`langchain-openai`, `langchain-anthropic`, etc.) are **not** listed as dependencies — callers install the ones they need.
- `langchain-core` is pinned (`>=0.3`) as an explicit dependency even though it's a transitive of `langchain`.
- Complex prebuilt middleware (`ModelFallbackMiddleware`, `ContextEditingMiddleware`) are handled separately in `_prebuilt.py` because they require nested init logic; simple flat middleware is handled generically.

## Dev setup

```bash
pip install -e ".[dev]"
```

## Common commands

```bash
ruff check src/           # lint
ruff format src/          # format
pytest                    # run tests
hatch build               # build wheel + sdist
hatch publish             # upload to PyPI
```

## Adding a new prebuilt middleware

1. Add the class name to `ComplexPrebuiltMiddlewareSE` in `utils/enums.py` if it needs nested init; otherwise it's handled automatically by the generic loop in `_prebuilt.py`.
2. Import from `langchain.agents.middleware` and wire config in `_prebuilt.py`.

## Adding tools

Return tool instances from `get_tools(config)` in `tools/__init__.py`. Extra tools can also be passed at call time via `create_agent.create(config, tools=[...])`.

## Roadmap

Workspace-wide deferred features live in `../docs/roadmap.md` (see the "lc-agent-factory/" section). Consult when a task touches a listed item. **Mutations (mark-done, remove, add) require explicit operator approval** — propose the change first, edit only after a "yes". Never silently mutate.
