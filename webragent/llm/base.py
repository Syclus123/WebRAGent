"""Abstract base class for all LLM backends in WebRAGent."""
from __future__ import annotations

import abc
from typing import Any


class BaseLLM(abc.ABC):
    """Abstract base class for all LLM backend integrations.

    Every concrete backend (OpenAI, Claude, Gemini, TogetherAI, Operator …)
    must inherit from this class and implement the :meth:`request` coroutine.

    Attributes
    ----------
    model : str
        The model identifier string (e.g. ``"gpt-4o"``, ``"claude-3-opus-20240229"``).
        Subclasses are expected to set this in their ``__init__``.

    Example
    -------
    ::

        class MyLLM(BaseLLM):
            def __init__(self, model: str):
                self.model = model

            async def request(self, messages, max_tokens=4096, temperature=0.7):
                # Call your backend here
                return response_text, ""
    """

    model: str  # declared as a class-level annotation; concrete classes set it in __init__

    @abc.abstractmethod
    async def request(
        self,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> tuple[str, str]:
        """Send a chat request to the LLM backend.

        Parameters
        ----------
        messages : list[dict[str, Any]] | None
            A list of chat messages in OpenAI-compatible format
            (``[{"role": "...", "content": "..."}]``).
        max_tokens : int
            Maximum number of tokens to generate in the response.
        temperature : float
            Sampling temperature (0 = deterministic, 1 = creative).

        Returns
        -------
        tuple[str, str]
            A two-tuple of ``(response_text, error_message)``.
            On success ``error_message`` is an empty string.
            On failure ``response_text`` is an empty string and
            ``error_message`` contains a human-readable description.
        """
