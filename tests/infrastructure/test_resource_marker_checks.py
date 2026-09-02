"""Every ``requires_<resource>`` marker has a skip-gate behind it.

``tests/conftest.py`` turns ``@pytest.mark.requires_<resource>`` into a real
skip when the resource is missing, but only for markers listed in its
``_RESOURCE_CHECKS`` table. A marker registered in ``pyproject.toml`` and
absent from that table passes ``--strict-markers`` and then fails *open*: the
test runs with no credential and dies on the provider call instead of
skipping. This pins the two lists to each other in both directions.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from tests.conftest import _RESOURCE_CHECKS

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _declared_requires_markers() -> set[str]:
    markers = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["pytest"]["ini_options"][
        "markers"
    ]
    names = {entry.split(":", 1)[0].strip() for entry in markers}
    return {name for name in names if name.startswith("requires_")}


def test_declared_markers_have_a_floor() -> None:
    declared = _declared_requires_markers()
    assert {"requires_api", "requires_anthropic", "requires_ollama"} <= declared, declared


def test_every_requires_marker_has_a_resource_check() -> None:
    missing = sorted(_declared_requires_markers() - set(_RESOURCE_CHECKS))
    assert missing == [], (
        f"requires_* markers registered in pyproject.toml with no entry in "
        f"tests/conftest.py's _RESOURCE_CHECKS — they fail open instead of skipping: {missing}"
    )


def test_every_resource_check_is_a_declared_marker() -> None:
    unknown = sorted(set(_RESOURCE_CHECKS) - _declared_requires_markers())
    assert unknown == [], f"_RESOURCE_CHECKS entries with no registered marker: {unknown}"


def test_resource_checks_are_predicate_reason_pairs() -> None:
    for name, entry in _RESOURCE_CHECKS.items():
        predicate, reason = entry
        assert callable(predicate), name
        assert isinstance(reason, str) and reason, name
