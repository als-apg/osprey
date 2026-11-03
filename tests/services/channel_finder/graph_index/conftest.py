"""Fixtures shared by the graph-index suites."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests._builds import BuiltProject
from tests._graph_index import demo_corpus_path


@pytest.fixture(scope="module")
def demo_path(built_control_assistant: BuiltProject) -> Iterator[Path]:
    """The graph view the control-assistant build writes."""
    with demo_corpus_path(built_control_assistant) as path:
        yield path
