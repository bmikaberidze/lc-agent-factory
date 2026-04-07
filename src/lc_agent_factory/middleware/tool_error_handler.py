from langchain.messages import ToolMessage
from langchain.agents.middleware import AgentMiddleware

class ToolErrorHandler(AgentMiddleware):
    """Return a ToolMessage with an error description when a tool raises."""

    def __init__(self, config: dict):
        self.config = config

    def wrap_tool_call(self, request, handler):
        try:
            return handler(request)
        except Exception as e:
            return self._error_tool_message(request, e)

    async def awrap_tool_call(self, request, handler):
        try:
            return await handler(request)
        except Exception as e:
            return self._error_tool_message(request, e)

    def _error_tool_message(self, request, error):
        return ToolMessage(
            content=f"Tool error: {error}. Please check your arguments and try again.",
            tool_call_id=request.tool_call["id"],
        )