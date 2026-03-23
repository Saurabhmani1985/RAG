"""
tests/conftest.py
──────────────────
Shared pytest fixtures and configuration.
"""

import os
import pytest

# Set test environment before any imports
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-placeholder"
os.environ["CHROMA_PERSIST_DIR"] = "/tmp/test_chroma_pytest"
os.environ["LOG_LEVEL"] = "WARNING"


def pytest_configure(config):
    """Register custom marks."""
    config.addinivalue_line("markers", "integration: requires real API keys")
    config.addinivalue_line("markers", "slow: long-running tests (e.g., full ingestion)")
