"""Smoke tests: the package is installed and importable, and its metadata is consistent."""

import re
from importlib.metadata import version

import agentcheck


def test_version_matches_installed_metadata() -> None:
    assert agentcheck.__version__ == version("agentcheck")


def test_version_is_semver_like() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", agentcheck.__version__)
