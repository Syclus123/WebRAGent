"""Token counting and message truncation utilities for WebRAGent.

This module merges the functionality of the original
``agent/LLM/token_calculator.py`` and ``agent/LLM/token_utils.py`` into a
single, framework-independent file.

Public API
----------
- :func:`calculation_of_token`        – count tokens in messages/strings
- :func:`save_token_count_to_file`    – persist token usage to JSON
- :func:`truncate_messages_based_on_estimated_tokens` – trim messages to fit a budget
- :func:`read_config`                 – read the project's TOML config
- :func:`is_model_supported`          – check pricing-config coverage
- :func:`estimate_tokens`             – fast character-level token estimate
- :func:`truncate_text`               – truncate a string by character length
- :func:`process_content`             – truncate a message's content field
"""

import json
import logging
from typing import Union, List, Dict, Tuple, Optional

import tiktoken
import toml

logger = logging.getLogger(__name__)

# ===========================================================================
# token_utils.py (original) — merged below
# ===========================================================================

def read_config(toml_path: Optional[str] = None) -> Dict:
    """Read configuration from TOML file.

    Args:
        toml_path: Path to the TOML config file. Defaults to 'configs/setting.toml'

    Returns:
        Dict containing configuration data
    """
    if toml_path is None:
        toml_path = 'configs/setting.toml'
    with open(toml_path, 'r') as f:
        config = toml.load(f)
    return config


def is_model_supported(model_name: str) -> bool:
    """Check if the model is supported in the configuration.

    Args:
        model_name: Name of the model to check

    Returns:
        bool indicating whether the model is supported
    """
    try:
        config = read_config()
        return model_name in config["token_pricing"]["pricing_models"]
    except Exception:
        return False


def estimate_tokens(text: str) -> float:
    """Estimate the number of tokens for a given text.

    Args:
        text: Input text to estimate tokens for

    Returns:
        Estimated number of tokens
    """
    return len(text) / 4.8


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to fit within the maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length allowed

    Returns:
        Truncated text
    """
    return text[:max_length]


def process_content(
    content: Union[str, List[Dict]],
    remaining_tokens: float
) -> Tuple[Union[str, List[Dict]], float]:
    """Process and possibly truncate content based on remaining token allowance.

    Args:
        content: Content to process (either string or list of content items)
        remaining_tokens: Number of tokens remaining

    Returns:
        Tuple of (processed content, tokens used)
    """
    if isinstance(content, list):
        truncated_content = []
        used_tokens = 0

        for item in content:
            if item['type'] == 'text':
                item_text = item['text']
                item_tokens = estimate_tokens(item_text)

                if used_tokens + item_tokens > remaining_tokens:
                    max_length = int((remaining_tokens - used_tokens) * 4.8)
                    truncated_t = truncate_text(item_text, max_length)
                    truncated_content.append({'type': 'text', 'text': truncated_t})
                    used_tokens += estimate_tokens(truncated_t)
                    break

                truncated_content.append(item)
                used_tokens += item_tokens

        return truncated_content, used_tokens
    else:
        # Simple text content
        tokens = estimate_tokens(content)
        if tokens > remaining_tokens:
            truncated_content = truncate_text(content, int(remaining_tokens * 4.8))
            return truncated_content, estimate_tokens(truncated_content)
        return content, tokens


def truncate_messages_based_on_estimated_tokens(
    messages: List[Dict],
    max_tokens: int
) -> List[Dict]:
    """Truncate a list of messages based on an estimated token limit.

    Args:
        messages: List of message dictionaries to process
        max_tokens: Maximum number of tokens allowed

    Returns:
        List of truncated messages
    """
    current_tokens = 0
    truncated_messages = []

    for message in messages:
        content = message['content']
        processed_content, used_tokens = process_content(content, max_tokens - current_tokens)

        if used_tokens > 0:
            truncated_messages.append({
                'role': message['role'],
                'content': processed_content
            })
            current_tokens += used_tokens

        if current_tokens >= max_tokens:
            break

    return truncated_messages


# ===========================================================================
# token_calculator.py (original) — merged below
# ===========================================================================

def calculation_of_token(
    messages: Union[str, List[Dict]],
    model: str = 'gpt-3.5-turbo',
    max_tokens: int = 4096
) -> int:
    """Calculate the number of tokens in the messages.

    Args:
        messages: List of messages or string to calculate tokens for
        model: Model to use for tokenization
        max_tokens: Maximum number of tokens allowed

    Returns:
        int: Number of tokens in the messages
    """
    if not is_model_supported(model):
        logger.debug(f"Message: Model {model} not in pricing configuration. Skipping token calculation.")
        return 0

    try:
        # Handle special model names that tiktoken doesn't recognise
        if "computer-use-preview" in model or "operator" in model:
            # Computer-use-preview models use the same encoding as GPT-4
            encoding = tiktoken.encoding_for_model("gpt-4")
        elif "gpt-4.1" in model:
            # GPT-4.1 uses the same encoding as GPT-4
            encoding = tiktoken.encoding_for_model("gpt-4")
        else:
            encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.debug(f"Warning: Model '{model}' not found in tiktoken. Using default encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    current_tokens = 0

    if isinstance(messages, str):
        tokens = encoding.encode(messages)
        current_tokens += len(tokens)
        return current_tokens

    for message in messages:
        content = message.get('content')
        if content is None:
            logger.debug("Warning: Message content is None. Skipping.")
            break

        if isinstance(content, list):
            # Process list of prompt elements
            for element in content:
                element_type = element.get('type', '')
                if 'text' in element_type:
                    tokens = encoding.encode(element['text'])
                    current_tokens += len(tokens)
                elif 'image' in element_type or element_type == 'image_url':
                    # Calculate image tokens based on OpenAI's official vision model pricing
                    # Formula: 85 base tokens + 170 * (number of 512x512 tiles)
                    image_url = element.get('image_url', {})
                    if isinstance(image_url, dict):
                        detail = image_url.get('detail', 'auto')
                        if detail == 'low':
                            current_tokens += 85
                            logger.debug(f"📸 Image tokens calculated (low detail): 85")
                        else:
                            # High detail calculation using official OpenAI method
                            # Assume typical screenshot dimensions (1280x720)
                            width, height = 1280, 720

                            # Step 1: Scale down to fit within 2048x2048 if necessary
                            if width > 2048 or height > 2048:
                                aspect_ratio = width / height
                                if aspect_ratio > 1:
                                    width = 2048
                                    height = int(2048 / aspect_ratio)
                                else:
                                    height = 2048
                                    width = int(2048 * aspect_ratio)

                            # Step 2: Scale so shortest side is 768px if both dimensions > 768
                            if width > 768 and height > 768:
                                aspect_ratio = width / height
                                if aspect_ratio > 1:
                                    height = 768
                                    width = int(768 * aspect_ratio)
                                else:
                                    width = 768
                                    height = int(768 / aspect_ratio)

                            # Step 3: Calculate tiles (512x512 each)
                            tiles_width = -(-width // 512)   # Ceiling division
                            tiles_height = -(-height // 512)
                            image_tokens = 85 + 170 * (tiles_width * tiles_height)

                            current_tokens += image_tokens
                            logger.debug(f"📸 Image tokens calculated (high detail): {image_tokens} (tiles: {tiles_width}x{tiles_height})")
                    else:
                        # Fallback for direct base64 images - use conservative estimate
                        current_tokens += 765  # Typical for 1280x720 screenshots
                        logger.debug(f"📸 Image tokens calculated (fallback): 765")
        else:
            # Process direct text content
            tokens = encoding.encode(content)
            current_tokens += len(tokens)

    return current_tokens


def save_token_count_to_file(
    filename: str,
    step_tokens: Dict,
    task_name: str,
    global_reward_text_model: str,
    planning_text_model: str,
    token_pricing: Dict
) -> None:
    """Save token count to a file in JSON format.

    Args:
        filename: Name of the file to save the token count
        step_tokens: Number of tokens used in steps
        task_name: Name of the task associated with the token count
        global_reward_text_model: Model used for reward modeling
        planning_text_model: Model used for planning
        token_pricing: Pricing information for models
    """
    if not is_model_supported(planning_text_model) or not is_model_supported(global_reward_text_model):
        logger.debug(
            f"Message: One or both models ({planning_text_model}, {global_reward_text_model}) "
            "not in pricing configuration. Skipping token saving."
        )
        return

    # Initialize or load existing data
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {
            "calls": [],
            "total_planning_input_tokens": 0,
            "total_planning_output_tokens": 0,
            "total_reward_input_tokens": 0,
            "total_reward_output_tokens": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
        }

    # Update call records
    call_record = {
        "task_name": task_name,
        "step_tokens": step_tokens
    }
    data["calls"].append(call_record)

    # Update token counts
    data["total_planning_input_tokens"] += step_tokens["steps_planning_input_token_counts"]
    data["total_planning_output_tokens"] += step_tokens["steps_planning_output_token_counts"]
    data["total_reward_input_tokens"] += step_tokens["steps_reward_input_token_counts"]
    data["total_reward_output_tokens"] += step_tokens["steps_reward_output_token_counts"]
    data["total_input_tokens"] += step_tokens["steps_input_token_counts"]
    data["total_output_tokens"] += step_tokens["steps_output_token_counts"]
    data["total_tokens"] += step_tokens["steps_token_counts"]

    # Update planning model costs
    if planning_text_model in token_pricing["pricing_models"]:
        if "total_planning_input_token_cost" not in data:
            data["total_planning_input_token_cost"] = 0
        if "total_planning_output_token_cost" not in data:
            data["total_planning_output_token_cost"] = 0

        data["total_planning_input_token_cost"] += (
            step_tokens["steps_planning_input_token_counts"] *
            token_pricing[f"{planning_text_model}_input_price"]
        )
        data["total_planning_output_token_cost"] += (
            step_tokens["steps_planning_output_token_counts"] *
            token_pricing[f"{planning_text_model}_output_price"]
        )

    # Update reward model costs
    if global_reward_text_model in token_pricing["pricing_models"]:
        if "total_reward_input_token_cost" not in data:
            data["total_reward_input_token_cost"] = 0
        if "total_reward_output_token_cost" not in data:
            data["total_reward_output_token_cost"] = 0

        data["total_reward_input_token_cost"] += (
            step_tokens["steps_reward_input_token_counts"] *
            token_pricing[f"{global_reward_text_model}_input_price"]
        )
        data["total_reward_output_token_cost"] += (
            step_tokens["steps_reward_output_token_counts"] *
            token_pricing[f"{global_reward_text_model}_output_price"]
        )

    # Update total costs
    if (planning_text_model in token_pricing["pricing_models"] and
            global_reward_text_model in token_pricing["pricing_models"]):

        if "total_input_token_cost" not in data:
            data["total_input_token_cost"] = 0
        if "total_output_token_cost" not in data:
            data["total_output_token_cost"] = 0
        if "total_token_cost" not in data:
            data["total_token_cost"] = 0

        data["total_input_token_cost"] += (
            data["total_planning_input_token_cost"] +
            data["total_reward_input_token_cost"]
        )
        data["total_output_token_cost"] += (
            data["total_planning_output_token_cost"] +
            data["total_reward_output_token_cost"]
        )
        data["total_token_cost"] += (
            data["total_input_token_cost"] +
            data["total_output_token_cost"]
        )

    # Save updated data
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
