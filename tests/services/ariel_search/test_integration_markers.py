"""Guard: every ARIEL integration test module carries the docker markers.

Each ``integration/test_*.py`` must set ``pytestmark`` with
``pytest.mark.xdist_group("docker")``: one testcontainers session per run, and
the shared ``ariel_test`` database serialized so parallel workers do not collide
on migrations, seed and truncate (``integration/test_migrations.py`` spells the
rule out). A module with coroutine tests also carries ``pytest.mark.asyncio``;
an all-sync module must not, because the mark on a sync test only warns and
pyproject's ``asyncio_mode = "auto"`` covers the rest.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

INTEGRATION_DIR = Path(__file__).parent / "integration"
MODULES = sorted(p.stem for p in INTEGRATION_DIR.glob("test_*.py"))


def _marks(module) -> list[pytest.Mark]:
    marks = getattr(module, "pytestmark", [])
    if not isinstance(marks, list | tuple):
        marks = [marks]
    return [getattr(m, "mark", m) for m in marks]


def _has_coroutine_test(module) -> bool:
    for name, member in vars(module).items():
        if not name.startswith("test_") and not (
            inspect.isclass(member) and name.startswith("Test")
        ):
            continue
        if inspect.isclass(member):
            if member.__module__ != module.__name__:
                continue
            if any(
                attr.startswith("test_") and inspect.iscoroutinefunction(fn)
                for attr, fn in vars(member).items()
            ):
                return True
        elif inspect.iscoroutinefunction(member):
            return True
    return False


def test_integration_modules_are_found() -> None:
    """The glob sees the package, so an empty parametrization cannot pass silently."""
    assert "test_migrations" in MODULES


@pytest.mark.parametrize("stem", MODULES)
def test_module_carries_the_docker_markers(stem: str) -> None:
    module = importlib.import_module(f"tests.services.ariel_search.integration.{stem}")
    marks = _marks(module)

    assert any(m.name == "xdist_group" and m.args == ("docker",) for m in marks), (
        f'{stem}.pytestmark lacks pytest.mark.xdist_group("docker")'
    )
    has_asyncio = any(m.name == "asyncio" for m in marks)
    if _has_coroutine_test(module):
        assert has_asyncio, f"{stem} has coroutine tests but no pytest.mark.asyncio"
    else:
        assert not has_asyncio, f"{stem} has no coroutine tests but carries pytest.mark.asyncio"
