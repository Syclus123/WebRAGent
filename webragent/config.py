"""
webragent.config
~~~~~~~~~~~~~~~~
Shared experiment configuration dataclasses.

Extracted from ``eval.py`` (DOM mode) and ``eval_op.py`` (Operator mode) to
provide a single, authoritative definition that both evaluation scripts can
import.

Usage::

    from webragent.config import DOMExperimentConfig, OperatorExperimentConfig
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# DOM-specific config  (eval.py)
# ---------------------------------------------------------------------------


@dataclass
class DOMExperimentConfig:
    """Configuration for DOM-based evaluation (``eval.py``).

    Matches the original ``ExperimentConfig`` dataclass in ``eval.py``
    exactly, preserving field order (required fields before defaulted ones).
    """

    # --- Required fields (no defaults) ---
    mode: str
    """Evaluation mode identifier (e.g. ``"dom"``)."""

    global_reward_mode: str
    """Global reward evaluation mode (e.g. ``"no_global_reward"``)."""

    planning_text_model: str
    """Name/ID of the model used for planning / observation."""

    global_reward_text_model: str
    """Name/ID of the model used for global reward scoring."""

    ground_truth_mode: bool
    """Whether ground-truth annotations are used."""

    single_task_name: str
    """Task name when running in ``single_task`` mode."""

    config: Dict[str, Any]
    """Parsed TOML/dict configuration object."""

    ground_truth_data: Optional[Dict[str, Any]]
    """Loaded ground-truth annotation data (may be None)."""

    write_result_file_path: str
    """Directory or file path where JSON results are written."""

    record_time: str
    """Timestamp string used to label this experiment run."""

    file: Optional[List[Any]]
    """Task file list (batch mode) or ``None`` (single-task mode)."""

    rag_enabled: bool
    """Whether retrieval-augmented generation is active."""

    rag_path: str
    """Path to the RAG vector store / index directory."""

    # --- Optional / defaulted fields ---
    rag_log_dir: Optional[str] = None
    """Directory for RAG query/result logs."""

    end_judge_mode: str = "disabled"
    """End-condition judge mode. One of ``"disabled"``, ``"llm"``, etc."""

    end_judge_confidence_threshold: float = 0.8
    """Confidence threshold above which the end judge fires."""

    end_judge_min_steps: int = 2
    """Minimum number of steps before the end judge is consulted."""

    consecutive_error_threshold: int = 2
    """Number of consecutive errors before the run is aborted."""

    rag_mode: str = "description"
    """RAG retrieval mode (e.g. ``"description"``, ``"screenshot"``)."""

    rag_cache_dir: Optional[str] = None
    """Directory for caching RAG embeddings/results."""


# ---------------------------------------------------------------------------
# Operator-specific config  (eval_op.py)
# ---------------------------------------------------------------------------


@dataclass
class OperatorExperimentConfig:
    """Configuration for Operator-based evaluation (``eval_op.py``).

    Matches the original ``ExperimentConfig`` dataclass in ``eval_op.py``
    exactly, preserving field order (all fields declared without defaults in
    the original are kept without defaults here).
    """

    # --- Required fields (no defaults — matches eval_op.py declaration order) ---
    mode: str
    global_reward_mode: str
    planning_text_model: str
    global_reward_text_model: str
    ground_truth_mode: bool
    single_task_name: str
    single_task_id: str
    """Unique task identifier used in operator mode."""
    config: dict
    ground_truth_data: dict
    write_result_file_path: str
    record_time: str
    file: list
    rag_enabled: bool
    rag_path: str
    screenshot_base_dir: str
    """Base directory where per-step screenshots are saved."""
    rag_logging_enabled: bool
    """Whether to log RAG queries and retrieved results."""
    rag_log_dir: str
    rag_mode: str
    prompt_logging_enabled: bool
    """Whether to log full prompts sent to the model."""
    prompt_log_dir: str
    """Directory for prompt log files."""
    end_judge_mode: str
    end_judge_confidence_threshold: float
    end_judge_min_steps: int
    rag_cache_dir: str
    consecutive_error_threshold: int


# ---------------------------------------------------------------------------
# Convenience alias — use when code is mode-agnostic
# ---------------------------------------------------------------------------

#: Union type hint for functions that accept either config flavour.
ExperimentConfigT = DOMExperimentConfig | OperatorExperimentConfig
