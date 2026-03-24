"""
webragent.constants
~~~~~~~~~~~~~~~~~~~
Project-wide constants extracted from scattered hard-coded values.

Import this module instead of sprinkling magic numbers / strings throughout
the codebase::

    from webragent.constants import VIEWPORT_WIDTH, DEFAULT_MAX_STEPS
"""

# ---------------------------------------------------------------------------
# Browser / viewport
# ---------------------------------------------------------------------------

#: Domains that are known to load slowly and may need extended wait times.
SLOW_LOAD_DOMAINS: frozenset = frozenset(
    {
        "flightaware",
        "student.com",
        "booking.com",
        "expedia",
        "kayak",
        "airbnb",
    }
)

VIEWPORT_WIDTH: int = 1280
VIEWPORT_HEIGHT: int = 720

# ---------------------------------------------------------------------------
# Agent execution limits
# ---------------------------------------------------------------------------

DEFAULT_MAX_STEPS: int = 80
DEFAULT_LOOP_THRESHOLD: int = 5
DEFAULT_TIMEOUT_SECONDS: int = 600

# ---------------------------------------------------------------------------
# Screenshot / media
# ---------------------------------------------------------------------------

DEFAULT_SCREENSHOT_QUALITY: int = 85

# ---------------------------------------------------------------------------
# RAG / retrieval
# ---------------------------------------------------------------------------

#: Number of results returned per RAG query by default.
RAG_N_RESULTS: int = 1
