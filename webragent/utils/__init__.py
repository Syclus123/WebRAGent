"""Utility helpers for WebRAGent."""
from .logging_config import setup_logging, get_logger, logger
from .io import download_data, upload_result, save_json, read_json_file, save_screenshot
from .text import print_info, print_limited_json, is_valid_base64, extract_longest_substring
from .rag_logger import RAGLogger, VisionRAGLogger

__all__ = [
    # logging
    "setup_logging",
    "get_logger",
    "logger",
    # I/O
    "download_data",
    "upload_result",
    "save_json",
    "read_json_file",
    "save_screenshot",
    # text
    "print_info",
    "print_limited_json",
    "is_valid_base64",
    "extract_longest_substring",
    # RAG logging
    "RAGLogger",
    "VisionRAGLogger",
]
