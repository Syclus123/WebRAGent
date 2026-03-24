"""Abstract base class for browser environments."""
from __future__ import annotations

import abc
from typing import Any


class BaseEnvironment(abc.ABC):
    """Abstract base class for all browser environments.

    Subclasses must implement the core navigation, action execution,
    and observation methods.
    """

    @abc.abstractmethod
    async def reset(self, *args, **kwargs) -> None:
        """Reset the environment to initial state."""

    @abc.abstractmethod
    async def execute_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Execute a single action and return result."""

    @abc.abstractmethod
    async def get_observation(self) -> str:
        """Get current DOM/visual observation."""

    @abc.abstractmethod
    async def get_screenshot(self) -> str | None:
        """Get current screenshot as base64 string, or None if not supported."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Clean up resources."""

    async def __aenter__(self) -> "BaseEnvironment":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
