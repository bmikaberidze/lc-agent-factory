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


# --- a chain of backups: first_model, then additional_models -----------------


class _FailingModel(GenericFakeChatModel):
    """A chat model whose every call raises — a provider that is down."""

    def bind_tools(self, _tools, *, _tool_choice=None, **_kwargs):
        return self

    def _generate(self, *args, **kwargs):
        raise RuntimeError("503 UNAVAILABLE")

    async def _agenerate(self, *args, **kwargs):
        raise RuntimeError("503 UNAVAILABLE")


def _down():
    return _FailingModel(messages=iter([]))


def _chain_config(fallback_cfg, model_keys=("primary", "fallback", "backup_2", "backup_3")):
    return {
        "models": {k: {"model_provider": "fake", "model": f"fake-{k}"} for k in model_keys},
        "middleware": {"prebuilt": [{"ModelFallbackMiddleware": fallback_cfg}]},
    }


def _create(config, models):
    seen = []

    def mock_init_model(_cfg, key):
        seen.append(key)
        return models.get(key)

    with patch.object(LCAgentFactory, "_init_model", side_effect=mock_init_model):
        agent = LCAgentFactory.create(config)
    return agent, seen


def _last_text(result):
    return result[-1].content


def test_chain_tries_backups_in_order_until_one_answers():
    """Primary and the first two backups are down: the third answers."""
    config = _chain_config({"first_model": "fallback", "additional_models": ["backup_2", "backup_3"]})
    agent, seen = _create(config, {
        "primary": _down(), "fallback": _down(), "backup_2": _down(),
        "backup_3": fake_model("hello from backup 3"),
    })

    assert seen.index("fallback") < seen.index("backup_2") < seen.index("backup_3")
    assert _last_text(invoke(agent, "hi")) == "hello from backup 3"


def test_chain_stops_at_the_first_backup_that_answers():
    config = _chain_config({"additional_models": ["backup_2"]})  # first_model defaults to models.fallback
    agent, _ = _create(config, {
        "primary": _down(), "fallback": fake_model("hello from fallback"),
        "backup_2": fake_model("never reached"),
    })

    assert _last_text(invoke(agent, "hi")) == "hello from fallback"


def test_chain_raises_when_every_model_is_down():
    config = _chain_config({"first_model": "fallback", "additional_models": ["backup_2"]})
    agent, _ = _create(config, {"primary": _down(), "fallback": _down(), "backup_2": _down()})

    with pytest.raises(RuntimeError, match="503"):
        invoke(agent, "hi")


def test_additional_models_unknown_key_raises():
    config = _chain_config({"first_model": "fallback", "additional_models": ["backup_9"]})
    with pytest.raises(ValueError, match="unknown models \\['backup_9'\\]"):
        _create(config, {"primary": fake_model("x"), "fallback": fake_model("y")})


def test_additional_models_must_be_a_list():
    config = _chain_config({"first_model": "fallback", "additional_models": "backup_2"})
    with pytest.raises(ValueError, match="must be a list"):
        _create(config, {"primary": fake_model("x"), "fallback": fake_model("y")})


def test_additional_models_without_a_first_model_raises():
    config = _chain_config({"additional_models": ["backup_2"]}, model_keys=("primary", "backup_2"))
    with pytest.raises(ValueError, match="needs a first_model"):
        _create(config, {"primary": fake_model("x"), "backup_2": fake_model("y")})
