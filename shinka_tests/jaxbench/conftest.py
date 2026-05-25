"""Pytest hooks for the JaxBench-aligned tests.

Shared constants live in `_util.py` (the jax repo has its own root conftest.py, so
these tests import config from _util, not from conftest). GPU cases auto-skip when no
GPU is present. Mirrors the JaxBench correctness gate
(https://github.com/Tyronita/JaxBench).
"""
import os
import sys

import pytest

os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, os.path.dirname(__file__))  # make _util importable

from _util import HAS_GPU  # noqa: E402


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(reason="no GPU device")
    for it in items:
        if "gpu" in it.nodeid and not HAS_GPU:
            it.add_marker(skip)
