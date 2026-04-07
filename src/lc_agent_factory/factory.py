import importlib

from langchain.agents import create_agent as _lc_create_agent
from langchain.chat_models import init_chat_model
from langchain_core.globals import set_verbose, set_debug

from .config import FactoryConfig
from .tools import get_tools
from .middleware import get_custom_middleware, get_prebuilt_middleware

# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
class LCAgentFactory:

    MIDDLEWARE_MODULE = "langchain.agents.middleware"

    @staticmethod
    def create(config: dict, **kwargs):

        config = FactoryConfig(**config).model_dump()
        globals_cfg = config.get("globals", {})
        set_debug(globals_cfg.get("set_debug", False))
        set_verbose(globals_cfg.get("set_verbose", False))

        model = LCAgentFactory._init_model(config, "primary")

        extra_tools = kwargs.pop("tools", [])
        tools = LCAgentFactory._dedupe(
            *get_tools(config),
            *extra_tools,
        )

        extra_middleware = kwargs.pop("middleware", [])
        middleware = LCAgentFactory._dedupe(
            *(get_prebuilt_middleware(config) or []),
            *(get_custom_middleware(config) or []),
            *extra_middleware,
        )

        agent_args = {
            "model": model,
            "tools": tools or None,
            "middleware": middleware or None,
        }
        agent_args = {k: v for k, v in agent_args.items() if v is not None}
        agent_args.update(kwargs)

        return _lc_create_agent(**agent_args)

    @staticmethod
    def _init_model(config: dict, key: str):
        mc = config.get("models", {}).get(key)
        return init_chat_model(**mc) if mc else None

    @staticmethod
    def _init_middleware(name: str, cfg: dict, submodule: str = None):
        module = LCAgentFactory.MIDDLEWARE_MODULE
        if submodule:
            module += f".{submodule}"
        mod = importlib.import_module(module)
        middleware_class = getattr(mod, name)
        return middleware_class(**cfg)

    @staticmethod
    def _init_middleware_list(midd_cfgs: list[dict], submodule: str = None):
        middleware = []
        for midd_cfg in midd_cfgs:
            name, cfg = next(iter(midd_cfg.items()))
            middleware.append(LCAgentFactory._init_middleware(name, cfg or {}, submodule))
        return middleware

    @staticmethod
    def _dedupe(*items):
        seen = set()
        result = []
        for item in items:
            key = getattr(item, "name", None) or type(item).__name__
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_agent(config: dict, **kwargs):
    """Config-driven facade for LangChain's create_agent.

    See LCAgentFactory.create() for full documentation.
    """
    return LCAgentFactory.create(config, **kwargs)
