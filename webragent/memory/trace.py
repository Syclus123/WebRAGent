"""Trace and long-term memory components for WebRAGent.

Merges base_trace.py, long_memory/reference_trace.py, and
long_memory/website_knowledge.py into a single module.
All original source files were empty stubs; this module provides
the public BaseTrace class that the package __init__ re-exports.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class BaseTrace:
    """Base class for agent execution traces.

    Subclasses can extend this to record reference trajectories
    (ReferenceTrace) or website-specific knowledge (WebsiteKnowledge).
    """

    def __init__(self) -> None:
        self._steps: list[dict[str, Any]] = []

    def add_step(self, step: dict[str, Any]) -> None:
        """Append a single step record to the trace."""
        self._steps.append(step)
        logger.debug(f"Trace step added: {step}")

    def get_steps(self) -> list[dict[str, Any]]:
        """Return all recorded steps."""
        return list(self._steps)

    def clear(self) -> None:
        """Clear all recorded steps."""
        self._steps.clear()
        logger.debug("Trace cleared.")

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(steps={len(self._steps)})"


class ReferenceTrace(BaseTrace):
    """Long-term memory trace that stores reference trajectories.

    Corresponds to agent/Memory/long_memory/reference_trace.py.
    """


class WebsiteKnowledge(BaseTrace):
    """Long-term memory that stores website-specific knowledge.

    Corresponds to agent/Memory/long_memory/website_knowledge.py.
    """
