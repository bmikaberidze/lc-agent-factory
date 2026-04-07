# Contributing

## Setup

Requires Python 3.10+. Uses [`uv`](https://docs.astral.sh/uv/) for environment management.

```bash
# create virtual environment
uv venv --python 3.11

# activate
source .venv/bin/activate

# install package + dev dependencies
uv pip install -e ".[dev]"

# install a provider for running examples
uv pip install langchain-google-genai
```

## Project layout

```
src/lc_agent_factory/
  __init__.py              # public exports: create_agent, LCAgentFactory, __version__
  factory.py               # create_agent() facade + LCAgentFactory implementation
  config.py                # Pydantic config validation (FactoryConfig)
  tools/__init__.py        # get_tools(config)
  middleware/
    __init__.py
    _prebuilt.py           # LangChain built-in middleware wiring
    _custom.py             # custom middleware list
    tool_error_handler.py  # ToolErrorHandler (sync + async)
  utils/
    enums.py               # LLMProviderSE, ComplexPrebuiltMiddlewareSE

examples/
  agent.py                 # minimal runnable example
  config.yaml              # full example config

tests/
  conftest.py              # FakeChatModel, fake_model, invoke_fn fixture
  test_tool_error_handler.py
  test_call_limits.py
  test_retry.py
  test_context_editing.py
  test_model_fallback.py
```

## Running tests

```bash
pytest tests/ -v
```

Tests use `GenericFakeChatModel` — no API key required.

## Linting

```bash
ruff check src/
ruff format src/
```

## Adding a prebuilt middleware

1. If the middleware requires nested init logic (e.g. resolving another model by name), add it to `ComplexPrebuiltMiddlewareSE` in `utils/enums.py` and handle it explicitly in `middleware/_prebuilt.py`.
2. For simple flat middleware, no code changes needed — it's handled generically by `_init_middleware_list`.

## Adding custom middleware

1. Create a subclass of `AgentMiddleware` in `middleware/`.
2. Implement both `wrap_tool_call` (sync) and `awrap_tool_call` (async) if the middleware touches tool execution.
3. Add it to the list returned by `get_custom_middleware` in `middleware/_custom.py`.

## Build

```bash
uv pip install hatch
hatch build          # produces dist/*.whl and dist/*.tar.gz
```

## Publish

```bash
hatch publish        # prompts for PyPI credentials
```

Or with an explicit token:

```bash
hatch publish --user __token__ --auth <your-pypi-token>
```

## Versioning

Update `version` in `pyproject.toml` and `__version__` in `src/lc_agent_factory/__init__.py` together before each release.
