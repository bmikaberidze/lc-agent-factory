from typing import Any

from pydantic import BaseModel, field_validator


class GlobalsConfig(BaseModel):
    set_debug: bool = False
    set_verbose: bool = False


def _validate_single_key_list(
    field_name: str, value: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    for i, entry in enumerate(value):
        if len(entry) != 1:
            raise ValueError(
                f"{field_name}[{i}] must have exactly one key "
                f"(the middleware class name), got: {list(entry.keys())}"
            )
    return value


class FactoryConfig(BaseModel):
    """Validates the parts of the config consumed by lc-agent-factory.

    - globals:                  fully validated (known shape), optional
    - models:                   structure validated (named dict of dicts); contents
                                passed as-is to init_chat_model — LangChain validates
    - middleware.prebuilt:      optional, structure validated (list of single-key dicts)
    - middleware.custom:        optional, structure validated (list of single-key dicts)

    Any extra keys are silently ignored.
    """

    model_config = {"extra": "ignore"}

    globals: GlobalsConfig = GlobalsConfig()
    models: dict[str, dict[str, Any]] = {}
    middleware: dict[str, list[dict[str, Any]]] = {"prebuilt": [], "custom": []}

    @field_validator("models")
    @classmethod
    def primary_model_required(cls, v):
        if not v.get("primary"):
            raise ValueError("config.models.primary is required")
        return v

    @field_validator("middleware")
    @classmethod
    def middleware_entries_are_single_key(cls, v):
        for key in ("prebuilt", "custom"):
            _validate_single_key_list(f"middleware.{key}", v.get(key, []))
        return v
