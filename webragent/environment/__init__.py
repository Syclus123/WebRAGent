"""Browser environment abstractions for WebRAGent."""
from .base import BaseEnvironment
# Re-export for convenience
from agent.Environment import ActionExecutionError, create_action
from agent.Environment.html_env.async_env import AsyncHTMLEnvironment

__all__ = [
    "BaseEnvironment",
    "ActionExecutionError",
    "create_action",
    "AsyncHTMLEnvironment",
]
