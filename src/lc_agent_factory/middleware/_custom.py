from .tool_error_handler import ToolErrorHandler


def get_custom_middleware(config: dict):
    """Get custom middleware for the agent."""
    return [
        ToolErrorHandler(config),
    ]
