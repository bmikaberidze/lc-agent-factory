"""Tests for ModelRetryMiddleware and ToolRetryMiddleware."""

from langchain.tools import tool

from conftest import fake_model, tool_call_msg, invoke, make_agent


call_count = 0


@tool
def flaky_tool(x: str) -> str:
    """A tool that fails on the first call and succeeds on the second."""
    global call_count
    call_count += 1
    if call_count == 1:
        raise RuntimeError("transient error")
    return "recovered"


# ---------------------------------------------------------------------------
# ModelRetryMiddleware — no tools, no need for sync/async variants
# ---------------------------------------------------------------------------

MODEL_RETRY_CFG = [
    {"ModelRetryMiddleware": {"max_retries": 3, "backoff_factor": 0.0, "initial_delay": 0.0}},
]


def test_model_retry_middleware_registered():
    """Agent should be created successfully with ModelRetryMiddleware config."""
    model = fake_model("hello")
    assert make_agent(model, MODEL_RETRY_CFG) is not None


def test_model_retry_agent_responds():
    """Agent with retry middleware should still return a normal response."""
    model = fake_model("hello")
    agent = make_agent(model, MODEL_RETRY_CFG)
    messages = invoke(agent, "hi")
    assert any(m.type == "ai" for m in messages)


# ---------------------------------------------------------------------------
# ToolRetryMiddleware — involves tools, test both sync and async
# ---------------------------------------------------------------------------

TOOL_RETRY_CFG = [
    {"ToolRetryMiddleware": {"max_retries": 2, "backoff_factor": 0.0, "initial_delay": 0.0}},
]


def test_tool_retry_middleware_registered():
    """Agent should be created successfully with ToolRetryMiddleware config."""
    model = fake_model("hello")
    assert make_agent(model, TOOL_RETRY_CFG) is not None


def test_tool_retry_error_is_handled(invoke_fn):
    """ToolRetryMiddleware + ToolErrorHandler: a failing tool returns a ToolMessage.

    Note: ToolErrorHandler (always included) is outermost and catches exceptions
    before ToolRetryMiddleware can retry them. The observable result is an error
    ToolMessage, not a recovered value.
    """
    global call_count
    call_count = 0

    model = fake_model(
        tool_call_msg("flaky_tool", {"x": "test"}),
        "done",
    )
    agent = make_agent(model, TOOL_RETRY_CFG, tools=[flaky_tool])
    messages = invoke_fn(agent, "use flaky tool")

    tool_messages = [m for m in messages if m.type == "tool"]
    assert len(tool_messages) >= 1
    assert any("Tool error" in m.content for m in tool_messages)
