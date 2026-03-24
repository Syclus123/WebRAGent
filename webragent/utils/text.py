"""Text processing utility functions for WebRAGent."""
import base64
import json5
import logging

logger = logging.getLogger(__name__)


def print_info(info, color):
    """Print coloured text to stdout using ANSI escape codes."""
    if color == 'yellow':
        print(f"\033[33m{info}\033[0m")
    elif color == 'red':
        print(f"\033[31m{info}\033[0m")
    elif color == 'green':
        print(f"\033[32m{info}\033[0m")
    elif color == 'cyan':
        print(f"\033[36m{info}\033[0m")
    elif color == 'blue':
        print(f"\033[34m{info}\033[0m")
    elif color == 'purple':
        print(f"\033[35m{info}\033[0m")
    elif color == 'white':
        print(f"\033[37m{info}\033[0m")
    elif color == 'black':
        print(f"\033[30m{info}\033[0m")
    elif color == 'bold':
        print(f"\033[1m{info}\033[0m")
    elif color == 'underline':
        print(f"\033[4m{info}\033[0m")
    else:
        print(f"{color}{info}\033[0m")  # \033[0m


def print_limited_json(obj, limit=500, indent=0):
    """Return a pretty-printed, length-limited JSON-like string representation."""
    spaces = ' ' * indent
    if isinstance(obj, dict):
        items = []
        for k, v in obj.items():
            formatted_value = print_limited_json(v, limit, indent + 4)
            items.append(f'{spaces}    "{k}": {formatted_value}')
        return f'{spaces}{{\n' + ',\n'.join(items) + '\n' + spaces + '}'
    elif isinstance(obj, list):
        elements = [print_limited_json(element, limit, indent + 4) for element in obj]
        return f'{spaces}[\n' + ',\n'.join(elements) + '\n' + spaces + ']'
    else:
        truncated_str = str(obj)[:limit] + "..." if len(str(obj)) > limit else str(obj)
        return json5.dumps(truncated_str)


def is_valid_base64(s):
    """
    Validate if a given string is a valid Base64 encoded string.

    :param s: String to be checked.
    :return: A tuple (bool, str) where the first element is True if the string is a valid
             Base64 encoded string, and the second element is a message indicating the result
             or the type of error.

    Usage:  is_valid, message = is_valid_base64(s)
    This function is only used to determine whether the picture is base64 encoded.
    """
    if s is None:
        return False, "The string is None."

    if not isinstance(s, str):
        return False, "The input is not a string."

    if len(s) == 0:
        return False, "The string is empty."

    try:
        base64.b64decode(s, validate=True)
        return True, "The string is a valid Base64 encoded string."
    except ValueError:
        return False, "The string is NOT a valid Base64 encoded string."


def extract_longest_substring(s):
    """Extract the longest JSON-object substring from *s*."""
    start = s.find('{')   # Find the first occurrence of '{'
    end = s.rfind('}')    # Find the last occurrence of '}'
    # Check if '{' and '}' were found and if they are in the right order
    if start != -1 and end != -1 and end > start:
        return s[start:end + 1]  # Return the longest substring
    else:
        return None  # Return None if no valid substring was found
