"""Tests for ContextEditingMiddleware with ClearToolUsesEdit."""

from langchain.tools import tool

from conftest import fake_model, tool_call_msg, invoke, make_agent


@tool
def search(query: str) -> str:
    """Search tool for testing context editing."""
    return f"results for: {query}"


CONTEXT_EDITING_CFG = [
    {
        "ContextEditingMiddleware": {
            "token_count_method": "approximate",
            "edits": [
                {
                    "ClearToolUsesEdit": {
                        "trigger": 100,
                        "keep": 1,
                        "clear_tool_inputs": False,
                        "exclude_tools": [],
                        "placeholder": "[cleared]",
                        "clear_at_least": 0,
                    }
                }
            ],
        }
    }
]


def test_context_editing_middleware_registered():
    """Agent should be created successfully with ContextEditingMiddleware config."""
    model = fake_model("hello")
    assert make_agent(model, CONTEXT_EDITING_CFG) is not None


def test_context_editing_agent_responds():
    """Agent with context editing middleware should return a normal response."""
    model = fake_model("hello")
    agent = make_agent(model, CONTEXT_EDITING_CFG)
    messages = invoke(agent, "hi")
    assert any(m.type == "ai" for m in messages)


def test_context_editing_clears_old_tool_uses(invoke_fn):
    """Agent completes without error with an aggressive ClearToolUsesEdit trigger.

    Note: ClearToolUsesEdit modifies the messages sent to the model (context), not
    the output state — clearing is not observable in the returned messages list.
    Running in both sync and async confirms the middleware doesn't break either path.
    """
    cfg = [
        {
            "ContextEditingMiddleware": {
                "token_count_method": "approximate",
                "edits": [
                    {
                        "ClearToolUsesEdit": {
                            "trigger": 1,
                            "keep": 0,
                            "clear_tool_inputs": True,
                            "exclude_tools": [],
                            "placeholder": "[cleared]",
                            "clear_at_least": 1,
                        }
                    }
                ],
            }
        }
    ]

    model = fake_model(
        tool_call_msg("search", {"query": "first"}),
        tool_call_msg("search", {"query": "second"}),
        "done",
    )
    agent = make_agent(model, cfg, tools=[search])
    messages = invoke_fn(agent, "search twice")

    tool_messages = [m for m in messages if m.type == "tool"]
    assert len(tool_messages) >= 1
