"""Tests for ModelCallLimitMiddleware and ToolCallLimitMiddleware."""

import pytest
from langchain.tools import tool

from conftest import fake_model, tool_call_msg, invoke, make_agent


@tool
def dummy_tool(x: str) -> str:
    """A simple tool for testing call limits."""
    return f"result: {x}"


# ---------------------------------------------------------------------------
# ModelCallLimitMiddleware — no tools, no need for sync/async variants
# ---------------------------------------------------------------------------

MODEL_CALL_LIMIT_CFG = [
    {"ModelCallLimitMiddleware": {"run_limit": 2, "exit_behavior": "end"}},
]


def test_model_call_limit_stops_agent():
    """Agent should stop after run_limit model calls."""
    model = fake_model("turn 1", "turn 2", "turn 3", "turn 4")
    agent = make_agent(model, MODEL_CALL_LIMIT_CFG)
    messages = invoke(agent, "keep going")

    ai_messages = [m for m in messages if m.type == "ai"]
    assert len(ai_messages) <= 2


def test_model_call_limit_agent_still_responds():
    """Agent should return a response even when limit is hit."""
    model = fake_model("only response")
    agent = make_agent(model, [
        {"ModelCallLimitMiddleware": {"run_limit": 1, "exit_behavior": "end"}},
    ])
    messages = invoke(agent, "hello")
    assert len(messages) >= 2  # user + at least one ai message


# ---------------------------------------------------------------------------
# ToolCallLimitMiddleware — involves tools, test both sync and async
# ---------------------------------------------------------------------------

TOOL_CALL_LIMIT_CFG = [
    {"ToolCallLimitMiddleware": {"run_limit": 1}},
]


def test_tool_call_limit_stops_after_limit(invoke_fn):
    """Agent should stop calling tools after run_limit is reached."""
    model = fake_model(
        tool_call_msg("dummy_tool", {"x": "a"}),
        tool_call_msg("dummy_tool", {"x": "b"}),
        "done",
    )
    agent = make_agent(model, TOOL_CALL_LIMIT_CFG, tools=[dummy_tool])
    messages = invoke_fn(agent, "use dummy_tool twice")

    tool_messages = [m for m in messages if m.type == "tool"]
    # First call succeeds; second is blocked with an error ToolMessage
    assert len(tool_messages) == 2
    assert tool_messages[0].status != "error"
    assert tool_messages[1].status == "error"
    assert "limit" in tool_messages[1].content.lower()


PER_TOOL_LIMIT_CFG = [
    {"ToolCallLimitMiddleware": {"tool_name": "dummy_tool", "run_limit": 1}},
]


def test_per_tool_call_limit(invoke_fn):
    """Per-tool limit should apply only to the named tool."""
    model = fake_model(
        tool_call_msg("dummy_tool", {"x": "first"}),
        tool_call_msg("dummy_tool", {"x": "second"}),
        "done",
    )
    agent = make_agent(model, PER_TOOL_LIMIT_CFG, tools=[dummy_tool])
    messages = invoke_fn(agent, "call dummy_tool twice")

    tool_messages = [m for m in messages if m.type == "tool"]
    assert len(tool_messages) == 2
    assert tool_messages[0].status != "error"
    assert tool_messages[1].status == "error"
    assert "dummy_tool" in tool_messages[1].content
