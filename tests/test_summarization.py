"""Tests for SummarizationMiddleware config pass-through."""

from unittest.mock import patch

import pytest

from langchain.agents.middleware.summarization import (
    DEFAULT_SUMMARY_PROMPT,
    SummarizationMiddleware,
)

from lc_agent_factory import LCAgentFactory
from lc_agent_factory.middleware import get_prebuilt_middleware

from conftest import fake_model, invoke, make_agent


BASE_CFG = {
    "model": "primary",
    "trigger": [["tokens", 55000], ["messages", 60]],
    "keep": ["messages", 20],
}


def _cfg(**overrides):
    return [{"SummarizationMiddleware": {**BASE_CFG, **overrides}}]


def _summarizer(middleware_cfg: list[dict]) -> SummarizationMiddleware:
    """Build the prebuilt middleware list and return its lone summarizer.

    Goes through get_prebuilt_middleware rather than a built agent: the compiled
    graph does not expose its middleware, and this is the unit actually under test.
    """
    config = {
        "models": {"primary": {"model_provider": "fake", "model": "fake"}},
        "middleware": {"prebuilt": middleware_cfg},
    }
    with patch.object(LCAgentFactory, "_init_model", return_value=fake_model("x")):
        built = get_prebuilt_middleware(config)

    found = [m for m in built if isinstance(m, SummarizationMiddleware)]
    assert len(found) == 1, f"expected exactly one summarizer, got {len(found)}"
    return found[0]


def test_defaults_used_when_keys_absent():
    """Omitting the optional keys must leave LangChain's own defaults in place."""
    mw = _summarizer(_cfg())

    assert mw.summary_prompt == DEFAULT_SUMMARY_PROMPT
    assert mw.trim_tokens_to_summarize == 4000


def test_summary_prompt_passed_through():
    """A custom prompt in config must reach the middleware verbatim."""
    custom = "Summarize in Georgian.\n<messages>\n{messages}\n</messages>"
    mw = _summarizer(_cfg(summary_prompt=custom))

    assert mw.summary_prompt == custom
    assert mw.summary_prompt != DEFAULT_SUMMARY_PROMPT


def test_trim_tokens_passed_through():
    """trim_tokens_to_summarize is what produced the empty-summary failure."""
    mw = _summarizer(_cfg(trim_tokens_to_summarize=12000))

    assert mw.trim_tokens_to_summarize == 12000


def test_trim_tokens_none_disables_trimming():
    """None is a meaningful value here (no trimming), not 'use the default'."""
    mw = _summarizer(_cfg(trim_tokens_to_summarize=None))

    assert mw.trim_tokens_to_summarize is None


def test_prompt_without_placeholder_raises():
    """A prompt lacking {messages} would silently summarize an empty history."""
    with pytest.raises(ValueError, match="must contain the '{messages}' placeholder"):
        make_agent(fake_model("hello"), _cfg(summary_prompt="Summarize in Georgian."))


def test_prompt_with_stray_brace_raises():
    """A brace .format() chokes on is worse than a crash — it eats the thread.

    LangChain wraps .format() in a bare `except Exception` and returns
    "Error generating summary: ...", which replaces the entire conversation.
    Nothing raises, nothing logs, and the damage is written to the checkpoint.
    """
    pasted_json = 'Example: {"code_id": "28216"}\n{messages}'
    with pytest.raises(ValueError, match="not a valid str.format template"):
        make_agent(fake_model("hello"), _cfg(summary_prompt=pasted_json))


def test_prompt_with_doubled_braces_is_accepted():
    """Correctly escaped literal braces must still pass."""
    escaped = 'Example: {{"code_id": "28216"}}\n{messages}'
    mw = _summarizer(_cfg(summary_prompt=escaped))

    assert mw.summary_prompt == escaped
    assert '{"code_id": "28216"}' in mw.summary_prompt.format(messages="H")


def test_null_summary_prompt_falls_back_to_default():
    """There is no useful null prompt, so a null means 'unset', not a crash."""
    mw = _summarizer(_cfg(summary_prompt=None))

    assert mw.summary_prompt == DEFAULT_SUMMARY_PROMPT


def test_agent_still_responds_with_custom_prompt():
    """Pass-through must not disturb normal agent operation."""
    custom = "Summarize.\n{messages}"
    agent = make_agent(fake_model("hello"), _cfg(summary_prompt=custom))

    assert any(m.type == "ai" for m in invoke(agent, "hi"))
