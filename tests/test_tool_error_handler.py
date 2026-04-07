"""Tests for ToolErrorHandler middleware — both sync and async paths."""

from langchain.tools import tool

from conftest import fake_model, tool_call_msg, make_agent


@tool
def always_fails(x: str) -> str:
    """A tool that always raises."""
    raise ValueError("boom")


@tool
def always_succeeds(x: str) -> str:
    """A tool that always returns a value."""
    return f"ok: {x}"


def test_tool_error_is_caught_and_returned_as_message(invoke_fn):
    """wrap_tool_call and awrap_tool_call both catch exceptions and return a ToolMessage."""
    model = fake_model(
        tool_call_msg("always_fails", {"x": "test"}),
        "done",
    )
    agent = make_agent(model, middleware_cfg=[], tools=[always_fails])
    messages = invoke_fn(agent, "trigger a failing tool")

    tool_messages = [m for m in messages if m.type == "tool"]
    assert any("Tool error" in m.content for m in tool_messages)


def test_successful_tool_is_not_affected(invoke_fn):
    """wrap_tool_call and awrap_tool_call both pass through successful tool calls."""
    model = fake_model(
        tool_call_msg("always_succeeds", {"x": "hello"}),
        "done",
    )
    agent = make_agent(model, middleware_cfg=[], tools=[always_succeeds])
    messages = invoke_fn(agent, "use the succeeding tool")

    tool_messages = [m for m in messages if m.type == "tool"]
    assert any("ok: hello" in m.content for m in tool_messages)
