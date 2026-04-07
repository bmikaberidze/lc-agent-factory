# lc-agent-factory

A thin, config-driven wrapper around [LangChain v1's `create_agent`](https://docs.langchain.com/oss/python/langchain/agents).

`create_agent(config, **kwargs)` accepts the same `**kwargs` as `create_agent`, plus a `config` dict — the factory's contribution: declare your agent's model, middleware, and tools in a plain dict or external file instead of wiring them in code. The returned agent is exactly what `create_agent` returns.

> **Note:** Not all `create_agent` [contract](https://reference.langchain.com/python/langchain/agents/factory/create_agent) is config-definable yet — pass unsupported ones directly as kwargs. When the same parameter appears in both, kwargs take priority for scalar values; for lists (e.g. `tools`, `middleware`) config and kwargs are merged with kwargs appended last.

```python
from lc_agent_factory import create_agent

agent = create_agent(config)  # config is just a dict
```

`create_agent()` accepts a plain `dict`. Load it however fits your stack — YAML file, JSON, environment variable, database, or constructed directly in code:

```python
import yaml, json

# from a YAML file (recommended for most projects)
config = yaml.safe_load(open("config.yaml"))

# from a JSON file
config = json.load(open("config.json"))

# from an environment variable
config = json.loads(os.environ["AGENT_CONFIG"])

# constructed in code
config = {
    "models": {"primary": {"model_provider": "openai", "model": "gpt-4o"}},
}
```

---

## Install

```bash
pip install lc-agent-factory
```

Then install your LLM provider:

```bash
pip install langchain-openai        # OpenAI
pip install langchain-anthropic     # Anthropic
pip install langchain-google-genai  # Google GenAI
# see https://python.langchain.com/docs/integrations/chat/ for all providers
```

---

## Quickstart

**1. Create a config file:**

```yaml
# config.yaml
models:
    primary:
        model_provider: google_genai
        model: gemini-2.5-flash
        temperature: 0

prebuilt_middleware:
    - ModelCallLimitMiddleware:
          run_limit: 10
          exit_behavior: 'end'
```

**2. Set your API key:**

```bash
export GOOGLE_API_KEY="..."
```

**3. Run:**

```python
import yaml
from lc_agent_factory import create_agent

agent = create_agent(config)

res = agent.invoke({"messages": [
    {"role": "user", "content": "Hello!"},
]})
print(res["messages"][-1].content)
```

---

## Configuration reference

### `globals` _(optional)_

```yaml
globals:
    set_debug: false # default: false
    set_verbose: false # default: false
```

### `models`

Named model configurations. `primary` is required. Any additional models (e.g. `fallback`) are referenced by name in middleware.

```yaml
models:
    primary:
        model_provider: openai # any init_chat_model provider
        model: gpt-4o
        temperature: 0
        timeout: 60

    fallback:
        model_provider: openai
        model: gpt-4o-mini
        temperature: 0
```

All keys under a model entry are passed as-is to LangChain's [`init_chat_model`](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html) — refer to its docs for available options per provider.

### `prebuilt_middleware` _(optional)_

A list of LangChain built-in middleware. Each entry is a single-key dict: `{MiddlewareClassName: {kwargs}}`.

```yaml
prebuilt_middleware:
    # Stop after N model calls
    - ModelCallLimitMiddleware:
          run_limit: 10
          exit_behavior: 'end'

    # Stop after N total tool calls
    - ToolCallLimitMiddleware:
          run_limit: 20

    # Per-tool call limit
    - ToolCallLimitMiddleware:
          tool_name: 'web_search'
          run_limit: 5

    # Retry failed model calls
    - ModelRetryMiddleware:
          max_retries: 3
          backoff_factor: 2.0
          initial_delay: 1.0

    # Retry failed tool calls
    - ToolRetryMiddleware:
          max_retries: 2
          backoff_factor: 2.0
          initial_delay: 1.0

    # Automatically switch to fallback model on failure
    - ModelFallbackMiddleware:
          first_model: 'fallback'

    # Trim old tool messages when context grows too large
    - ContextEditingMiddleware:
          token_count_method: 'approximate'
          edits:
              - ClearToolUsesEdit:
                    trigger: 50000
                    keep: 3
                    clear_tool_inputs: false
                    placeholder: '[cleared]'
```

Kwargs are passed as-is to LangChain middleware constructors — refer to [LangChain middleware docs](https://python.langchain.com/docs/concepts/agents/#middleware) for the full list and options.

### `custom_middleware` _(optional)_

Same structure as `prebuilt_middleware`, for your own `AgentMiddleware` subclasses.

---

## Adding tools

Pass tools at call time — they are merged and deduplicated with any tools returned by internal loaders:

```python
from langchain.tools import tool

@tool
def web_search(query: str) -> str:
    """Search the web."""
    ...

agent = create_agent(config, tools=[web_search])
```

---

## Async

The returned agent is a compiled LangGraph and supports all four invocation modes out of the box:

```python
# sync
agent.invoke({"messages": [...]})

# async
await agent.ainvoke({"messages": [...]})

# streaming
for chunk in agent.stream({"messages": [...]}):
    print(chunk)

# async streaming
async for chunk in agent.astream({"messages": [...])):
    print(chunk)
```

---

## License

MIT — free to use for any purpose, personal or commercial. See [LICENSE](LICENSE).
