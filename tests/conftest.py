"""Shared fixtures for all middleware tests."""

import asyncio
from unittest.mock import patch

import pytest

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolCall

from lc_agent_factory import LCAgentFactory


class FakeChatModel(GenericFakeChatModel):
    """GenericFakeChatModel extended to support bind_tools (required by create_agent)."""

    def bind_tools(self, _tools, *, _tool_choice=None, **_kwargs):
        return self


def fake_model(*responses):
    """Fake chat model that replays responses in order (strings or AIMessages)."""
    messages = [
        AIMessage(content=r) if isinstance(r, str) else r
        for r in responses
    ]
    return FakeChatModel(messages=iter(messages))


def tool_call_msg(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    """AIMessage with a single tool call — use as a fake model response."""
    return AIMessage(
        content="",
        tool_calls=[ToolCall(name=name, args=args, id=call_id)],
    )


def invoke(agent, content: str) -> list:
    """Async invoke — use in tests that don't need to exercise both sync and async paths."""
    res = asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": content}]}))
    return res["messages"]


@pytest.fixture(params=["sync", "async"], ids=["sync", "async"])
def invoke_fn(request):
    """Parametrized fixture that runs each test in both sync and async invoke modes.

    Use this fixture (instead of invoke()) in tests that cover middleware with
    both wrap_tool_call and awrap_tool_call implementations.
    """
    def _sync(agent, content):
        res = agent.invoke({"messages": [{"role": "user", "content": content}]})
        return res["messages"]

    def _async(agent, content):
        res = asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": content}]}))
        return res["messages"]

    return _sync if request.param == "sync" else _async


def make_agent(model, middleware_cfg: list[dict], tools: list = None) -> object:
    """Build an agent with a fake model and the given middleware.prebuilt config."""
    config = {
        "globals": {"set_debug": False, "set_verbose": False},
        "models": {"primary": {"model_provider": "fake", "model": "fake"}},
        "middleware": {"prebuilt": middleware_cfg},
    }
    with patch.object(LCAgentFactory, "_init_model", return_value=model):
        return LCAgentFactory.create(config, tools=tools or [])
