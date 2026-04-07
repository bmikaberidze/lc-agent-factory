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


def _find_midd_cfg(midd_cfgs: list[dict], name: str) -> dict | None:
    """Return the config dict for the first entry matching name, or None."""
    return next((m[name] for m in midd_cfgs if name in m), None)


def get_prebuilt_middleware(config: dict):
    """Get built-in middleware for the agent."""

    from ..factory import LCAgentFactory

    middleware = []
    midd_cfgs: list[dict] = config.get("middleware", {}).get("prebuilt", [])

    # --- complex middleware (requires nested init logic) ---

    fallback_cfg = _find_midd_cfg(midd_cfgs, ComplexPrebuiltMiddlewareSE.MODEL_FALLBACK)
    if fallback_cfg:
        fallback_model = LCAgentFactory._init_model(
            config, fallback_cfg.get("first_model")
        )
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
            middleware.append(
                SummarizationMiddleware(
                    model=sum_model,
                    trigger=trigger,
                    keep=keep,
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
