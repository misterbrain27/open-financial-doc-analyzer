"""Pytest configuration and shared fixtures for all tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure 'app' module is importable from tests
sys.path.insert(0, str(Path(__file__).parent.parent))
