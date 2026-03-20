"""Backward-compatible exports for the renamed app factory module."""

from .app_factory import *  # noqa: F401,F403
from .app_factory import reload_app_context as reload_shell_context
