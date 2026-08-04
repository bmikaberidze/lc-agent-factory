from langchain.agents.middleware import (
    ModelFallbackMiddleware,
    SummarizationMiddleware,
    ContextEditingMiddleware,
)

from ..utils.enums import ComplexPrebuiltMiddlewareSE

# Middleware              Description
#
# Summarization           Automatically summarize conversation history when approaching token limits.
# Human-in-the-loop       Pause execution for human approval of tool calls.
# Model call limit        Limit the number of model calls to prevent excessive costs.
# Tool call limit         Control tool execution by limiting call counts.
# Model fallback          Automatically fallback to alternative models when primary fails.
# PII detection           Detect and handle Personally Identifiable Information (PII).
# To-do list              Equip agents with task planning and tracking capabilities.
# LLM tool selector       Use an LLM to select relevant tools before calling main model.
# Tool retry              Automatically retry failed tool calls with exponential backoff.
# Model retry             Automatically retry failed model calls with exponential backoff.
# LLM tool emulator       Emulate tool execution using an LLM for testing purposes.
# Context editing         Manage conversation context by trimming or clearing tool uses.
# Shell tool              Expose a persistent shell session to agents for command execution.
# File search             Provide Glob and Grep search tools over filesystem files.
# Filesystem              Provide agents with a filesystem for storing context and long-term memories.
# Subagent                Add the ability to spawn subagents.


_NOT_FOUND = object()


def _find_midd_cfg(midd_cfgs: list[dict], name: str):
    """Return the config dict for the first entry matching name, {} if entry is bare, or None if missing."""
    found = next((m[name] for m in midd_cfgs if name in m), _NOT_FOUND)
    if found is _NOT_FOUND:
        return None
    return found or {}


def _validate_summary_prompt(prompt: str) -> None:
    """Reject a summary_prompt that str.format cannot render.

    LangChain calls `summary_prompt.format(messages=...)` and wraps it in a bare
    `except Exception` that returns the string "Error generating summary: ..."
    — which it then writes into state *in place of the whole conversation*.
    A single stray brace (an operator pasting a JSON example into the prompt)
    therefore does not raise: it silently and permanently destroys the thread's
    history, one checkpoint at a time, with no log and no failed request.

    Two ways to render nothing, both fatal, both checked here:
      - the template omits {messages}, so history is never interpolated;
      - the template has a brace .format() chokes on (KeyError/IndexError/ValueError).
    """
    if "{messages}" not in prompt:
        raise ValueError(
            "SummarizationMiddleware.summary_prompt must contain the "
            "'{messages}' placeholder — without it the conversation history "
            "is never interpolated into the prompt."
        )
    try:
        prompt.format(messages="")
    except (KeyError, IndexError, ValueError) as e:
        raise ValueError(
            "SummarizationMiddleware.summary_prompt is not a valid str.format "
            f"template ({type(e).__name__}: {e}). Every literal brace must be "
            "doubled ('{{' and '}}') — LangChain swallows the format error and "
            "replaces the conversation history with the error text."
        ) from e


def get_prebuilt_middleware(config: dict):
    """Get built-in middleware for the agent."""

    from ..factory import LCAgentFactory

    middleware = []
    midd_cfgs: list[dict] = config.get("middleware", {}).get("prebuilt", [])

    # --- complex middleware (requires nested init logic) ---

    fallback_cfg = _find_midd_cfg(midd_cfgs, ComplexPrebuiltMiddlewareSE.MODEL_FALLBACK)
    if fallback_cfg is not None:
        first_model_key = fallback_cfg.get("first_model") or (
            "fallback" if "fallback" in config.get("models", {}) else None
        )
        fallback_model = LCAgentFactory._init_model(config, first_model_key)
        if fallback_model:
            middleware.append(ModelFallbackMiddleware(first_model=fallback_model))

    sum_cfg = _find_midd_cfg(midd_cfgs, ComplexPrebuiltMiddlewareSE.SUMMARIZATION)
    if sum_cfg:
        sum_model = LCAgentFactory._init_model(config, sum_cfg.get("model"))
        trigger = sum_cfg.get("trigger")
        if isinstance(trigger[0], list):
            trigger = [tuple(t) for t in trigger]
        else:
            trigger = tuple(trigger)
        keep = tuple(sum_cfg.get("keep"))
        if sum_model and trigger and keep:
            # Both keys are omitted rather than passed as None when absent:
            # LangChain's defaults are module constants, and forwarding None
            # would override them with a null instead.
            #
            # The two differ on what an *explicit* null means. `None` is a
            # meaningful value for trim_tokens_to_summarize — it disables
            # trimming — so it is forwarded. There is no useful null prompt, so
            # `summary_prompt: null` is treated as "unset" rather than crashing
            # the build on a config that plainly meant "use the default".
            optional = {}
            if "trim_tokens_to_summarize" in sum_cfg:
                optional["trim_tokens_to_summarize"] = sum_cfg["trim_tokens_to_summarize"]
            if sum_cfg.get("summary_prompt") is not None:
                _validate_summary_prompt(sum_cfg["summary_prompt"])
                optional["summary_prompt"] = sum_cfg["summary_prompt"]
            middleware.append(
                SummarizationMiddleware(
                    model=sum_model,
                    trigger=trigger,
                    keep=keep,
                    **optional,
                )
            )

    context_editing_cfg = _find_midd_cfg(
        midd_cfgs, ComplexPrebuiltMiddlewareSE.CONTEXT_EDITING
    )
    if context_editing_cfg:
        middleware.append(
            ContextEditingMiddleware(
                token_count_method=context_editing_cfg.get(
                    "token_count_method", "approximate"
                ),
                edits=LCAgentFactory._init_middleware_list(
                    midd_cfgs=context_editing_cfg.get("edits", []),
                    submodule="context_editing",
                ),
            )
        )

    # --- simple (flat) middleware ---

    simple_midd_cfgs = [
        m for m in midd_cfgs if not ComplexPrebuiltMiddlewareSE.has(next(iter(m)))
    ]
    middleware.extend(LCAgentFactory._init_middleware_list(simple_midd_cfgs))

    return middleware
