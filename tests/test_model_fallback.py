"""Tests for ModelFallbackMiddleware."""

import pytest
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from lc_agent_factory import LCAgentFactory
from conftest import fake_model, invoke, make_agent


MIDDLEWARE_CFG = [
    {"ModelFallbackMiddleware": {"first_model": "fallback"}},
]


def test_fallback_model_is_registered():
    """Agent should be created successfully with ModelFallbackMiddleware config."""
    primary = fake_model("hello from primary")
    fallback = fake_model("hello from fallback")

    config = {
        "globals": {"set_debug": False, "set_verbose": False},
        "models": {
            "primary": {"model_provider": "fake", "model": "fake-primary"},
            "fallback": {"model_provider": "fake", "model": "fake-fallback"},
        },
        "middleware": {"prebuilt": MIDDLEWARE_CFG},
    }

    def mock_init_model(cfg, key):
        return primary if key == "primary" else fallback

    with patch.object(LCAgentFactory, "_init_model", side_effect=mock_init_model):
        agent = LCAgentFactory.create(config)

    assert agent is not None


def test_fallback_auto_wires_to_models_fallback_when_first_model_omitted():
    """Bare ModelFallbackMiddleware: {} defaults first_model to models.fallback."""
    primary = fake_model("primary")
    fallback = fake_model("fallback")

    config = {
        "models": {
            "primary": {"model_provider": "fake", "model": "fake-primary"},
            "fallback": {"model_provider": "fake", "model": "fake-fallback"},
        },
        "middleware": {"prebuilt": [{"ModelFallbackMiddleware": {}}]},
    }

    seen_keys = []

    def mock_init_model(_cfg, key):
        seen_keys.append(key)
        return primary if key == "primary" else fallback

    with patch.object(LCAgentFactory, "_init_model", side_effect=mock_init_model):
        agent = LCAgentFactory.create(config)

    assert agent is not None
    assert "fallback" in seen_keys


def test_fallback_skipped_when_no_models_fallback_and_no_first_model():
    """Bare ModelFallbackMiddleware: {} with no models.fallback is silently skipped."""
    primary = fake_model("primary")

    config = {
        "models": {"primary": {"model_provider": "fake", "model": "fake"}},
        "middleware": {"prebuilt": [{"ModelFallbackMiddleware": {}}]},
    }

    def mock_init_model(_cfg, key):
        return primary if key == "primary" else None

    with patch.object(LCAgentFactory, "_init_model", side_effect=mock_init_model):
        agent = LCAgentFactory.create(config)

    assert agent is not None


def test_fallback_missing_model_key_is_skipped():
    """If first_model key doesn't resolve, fallback middleware is silently skipped."""
    primary = fake_model("hello")

    config = {
        "globals": {"set_debug": False, "set_verbose": False},
        "models": {"primary": {"model_provider": "fake", "model": "fake"}},
        "middleware": {"prebuilt": [
            {"ModelFallbackMiddleware": {"first_model": "nonexistent"}},
        ]},
    }

    def mock_init_model(cfg, key):
        return primary if key == "primary" else None

    with patch.object(LCAgentFactory, "_init_model", side_effect=mock_init_model):
        agent = LCAgentFactory.create(config)

    assert agent is not None
