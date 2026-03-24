"""File I/O utility functions for WebRAGent."""
import base64
import json
import os
from datetime import datetime
from io import BytesIO

import json5
import requests
from PIL import Image

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Network I/O
# ---------------------------------------------------------------------------

def download_data(url, dest_path):
    response = requests.get(url)
    with open(dest_path, 'wb') as file:
        file.write(response.content)


def upload_result(url, data):
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    return response.status_code, response.json()


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def save_json(data, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def read_json_file(file_path):
    """
    Read and parse a JSON file.

    Args:
    - file_path: str, the path of the JSON file.

    Returns:
    - Returns the parsed data on success.
    - Returns an error message on failure.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json5.load(file)
            return data
    except FileNotFoundError:
        return f"File not found: {file_path}"


# ---------------------------------------------------------------------------
# Screenshot helpers
# ---------------------------------------------------------------------------

def save_screenshot(
    mode: str,
    record_time: str,
    task_name: str,
    step_number: int,
    description: str,
    screenshot_base64: str,
    task_name_id: str = None,
    task_uuid: str = None,
    file_path: str = None,
):
    """Save a base64-encoded screenshot to disk.

    Prior use task_uuid, else task_name_id.
    """
    identifier = task_uuid if task_uuid is not None else task_name_id

    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        task_name = task_name.replace(char, '_')

    # Use file_path if provided, otherwise use the old hardcoded path for backward compatibility
    if file_path:
        if identifier is None:
            task_folder = os.path.join(file_path, "img_screenshots", task_name)
        else:
            task_folder = os.path.join(file_path, "img_screenshots", f"{identifier}_{task_name}")
    else:
        # old hardcoded path
        if identifier is None:
            task_folder = f'results/screenshots/screenshots_{mode}_{record_time}/{task_name}'
        else:
            task_folder = f'results/screenshots/screenshots_{mode}_{record_time}/{identifier}_{task_name}'

    if not os.path.exists(task_folder):
        os.makedirs(task_folder)

    image_data = base64.b64decode(screenshot_base64)
    image = Image.open(BytesIO(image_data))

    screenshot_filename = f'{task_folder}/{step_number}.png'
    image.save(screenshot_filename)
