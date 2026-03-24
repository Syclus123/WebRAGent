"""LLM factory with registry pattern for easy extensibility.

Usage
-----
::

    from webragent.llm import create_llm_instance

    llm = create_llm_instance("gpt-4o")
    response, error = await llm.request(messages)
"""
from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseLLM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry: model-prefix → (module_path, class_name)
#
# Keys are matched via str.startswith() in the order they appear (first match
# wins), EXCEPT that "operator" / "computer-use" are checked first regardless
# of order because they share the "o" prefix with OpenAI o-series models.
# ---------------------------------------------------------------------------
_LLM_REGISTRY: dict[str, tuple[str, str]] = {
    "gpt": ("webragent.llm.backends.openai_llm", "GPTGenerator"),
    "o1": ("webragent.llm.backends.openai_llm", "GPTGenerator"),
    "o3": ("webragent.llm.backends.openai_llm", "GPTGenerator"),
    "o4": ("webragent.llm.backends.openai_llm", "GPTGenerator"),
    "operator": ("webragent.llm.backends.operator_llm", "OperatorGenerator"),
    "computer-use": ("webragent.llm.backends.operator_llm", "OperatorGenerator"),
    "claude": ("webragent.llm.backends.claude_llm", "ClaudeGenerator"),
    "gemini": ("webragent.llm.backends.gemini_llm", "GeminiGenerator"),
}

# Prefixes that must be tested before the general registry scan so that
# e.g. "operator" isn't accidentally matched as "o1/o3/o4".
_PRIORITY_PREFIXES: tuple[str, ...] = ("operator", "computer-use", "computer-use-preview")


def _load_class(module_path: str, class_name: str):
    """Lazily import *class_name* from *module_path*.

    Parameters
    ----------
    module_path : str
        Dotted Python module path, e.g. ``"webragent.llm.backends.openai_llm"``.
    class_name : str
        Name of the class to retrieve from the module.

    Returns
    -------
    type
        The requested class object.
    """
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def create_llm_instance(
    model: str,
    json_mode: bool = False,
    all_json_models: list[str] | None = None,
) -> "BaseLLM":
    """Create and return the appropriate LLM backend instance for *model*.

    The factory uses a registry-based lookup (matching the start of *model*)
    to select the correct backend class.  This makes adding new backends as
    simple as inserting a single entry into ``_LLM_REGISTRY``.

    Parameters
    ----------
    model : str
        Model identifier string (e.g. ``"gpt-4o"``, ``"claude-3-opus-20240229"``).
    json_mode : bool
        When ``True``, return a JSON-mode variant of the backend if one exists.
        Raises ``ValueError`` for backends that do not support JSON mode.
    all_json_models : list[str] | None
        Whitelist of model names that are allowed to use JSON mode.  When
        provided, *model* must appear in this list for JSON mode to be
        activated; otherwise a ``ValueError`` is raised.

    Returns
    -------
    BaseLLM
        A concrete ``BaseLLM`` subclass instance ready to call.

    Raises
    ------
    ValueError
        If *json_mode* is requested but not supported by the selected backend.
    """
    # ------------------------------------------------------------------
    # 1. Operator / computer-use always wins — checked before the general
    #    registry to avoid false matches against "o1", "o3", "o4" etc.
    # ------------------------------------------------------------------
    if "operator" in model or "computer-use" in model:
        if json_mode:
            if all_json_models and model in all_json_models:
                OperatorGeneratorWithJSON = _load_class(
                    "webragent.llm.backends.operator_llm", "OperatorGeneratorWithJSON"
                )
                logger.debug("Creating OperatorGeneratorWithJSON for model=%s", model)
                return OperatorGeneratorWithJSON(model)
            else:
                raise ValueError("The operator model does not support JSON mode.")
        else:
            OperatorGenerator = _load_class(
                "webragent.llm.backends.operator_llm", "OperatorGenerator"
            )
            logger.debug("Creating OperatorGenerator for model=%s", model)
            return OperatorGenerator(model)

    # ------------------------------------------------------------------
    # 2. General registry scan (first prefix match wins)
    # ------------------------------------------------------------------
    for prefix, (module_path, class_name) in _LLM_REGISTRY.items():
        # Skip operator entries — already handled above
        if prefix in ("operator", "computer-use"):
            continue
        if model.startswith(prefix):
            if json_mode:
                # Only OpenAI GPT/o-series backends have a JSON variant
                if prefix in ("gpt", "o1", "o3", "o4"):
                    if all_json_models and model in all_json_models:
                        GPTGeneratorWithJSON = _load_class(
                            "webragent.llm.backends.openai_llm", "GPTGeneratorWithJSON"
                        )
                        logger.debug("Creating GPTGeneratorWithJSON for model=%s", model)
                        return GPTGeneratorWithJSON(model)
                    else:
                        raise ValueError("The text model does not support JSON mode.")
                else:
                    backend_label = class_name.replace("Generator", "")
                    raise ValueError(f"{backend_label} does not support JSON mode.")
            else:
                cls = _load_class(module_path, class_name)
                logger.debug("Creating %s for model=%s", class_name, model)
                return cls(model)

    # ------------------------------------------------------------------
    # 3. Fallback → TogetherAI (preserves original behaviour)
    # ------------------------------------------------------------------
    if json_mode:
        raise ValueError("TogetherAI does not support JSON mode.")
    TogetherAIGenerator = _load_class(
        "webragent.llm.backends.togetherai_llm", "TogetherAIGenerator"
    )
    logger.debug("Falling back to TogetherAIGenerator for model=%s", model)
    return TogetherAIGenerator(model)


async def semantic_match_llm_request(messages: list | None = None):
    """Send *messages* to GPT-3.5-Turbo for lightweight semantic matching.

    Parameters
    ----------
    messages : list | None
        Chat messages in OpenAI format.

    Returns
    -------
    tuple[str, str]
        ``(response_text, error_message)`` as returned by :meth:`BaseLLM.request`.
    """
    GPTGenerator = _load_class("webragent.llm.backends.openai_llm", "GPTGenerator")
    gpt35 = GPTGenerator(model="gpt-3.5-turbo")
    return await gpt35.request(messages)
