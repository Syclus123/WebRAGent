"""Anthropic Claude backend for WebRAGent."""
import os
import asyncio
import logging
import concurrent.futures
import multiprocessing
from functools import partial

from anthropic import AsyncAnthropic

from ..base import BaseLLM

logger = logging.getLogger(__name__)


class ClaudeGenerator(BaseLLM):

    def __init__(self, model=None):
        self.model = model
        self.client = AsyncAnthropic(
            api_key=os.environ.get('ANTHROPIC_API_KEY')
        )
        self.pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=multiprocessing.cpu_count() * 2)

    async def request(self, messages: list = None, max_tokens: int = 4096, temperature: float = 0.7) -> tuple[str, str]:
        """Send a request to the Claude API.

        Parameters
        ----------
        messages : list | None
            Chat messages in OpenAI-compatible format.
        max_tokens : int
            Maximum tokens to generate (default raised from 500 → 4096).
        temperature : float
            Sampling temperature.

        Returns
        -------
        tuple[str, str]
            ``(response_text, error_message)``
        """
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(self.pool, partial(self.chat, messages, max_tokens, temperature))
            return await response, ""
        except Exception as e:
            logger.error(f"Error in ClaudeGenerator.request: {e}")
            return "", str(e)

    async def chat(self, message, max_tokens=4096, temperature=0.7):
        messages = [
            {"role": "user", "content": "Please follow the instructions"},
            {"role": "assistant", "content": message[0].get("content")},
            {"role": "user", "content": message[1].get("content")},
        ]
        data = {
            'model': self.model,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'messages': messages,
        }
        response = await self.client.messages.create(**data)
        return response.content[0].text
