"""Utility functions for environment management."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def create_environment(mode: str, **kwargs: Any):
    """Factory function to create the appropriate environment.

    Args:
        mode: Environment mode ('dom', 'operator', 'vision', etc.)
        **kwargs: Additional arguments passed to the environment constructor.

    Returns:
        An environment instance appropriate for the given mode.
    """
    from agent.Environment.html_env.async_env import AsyncHTMLEnvironment
    # All modes currently use AsyncHTMLEnvironment as the base
    return AsyncHTMLEnvironment(**kwargs)
