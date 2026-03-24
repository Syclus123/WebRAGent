"""Backward compatibility shim. Import from webragent.planning instead."""
from webragent.planning.planner import Planner as Planning
from webragent.planning.modes import *

__all__ = ["Planning"]
