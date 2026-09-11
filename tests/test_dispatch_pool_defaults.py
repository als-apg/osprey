"""Unit tests for the dispatcher pool limits and the leaf that holds them.

:mod:`osprey.dispatch_pool_defaults` is the one place the dispatcher's
concurrency and queue-depth defaults are spelled. Three consumers read it — the
runtime dataclass, the ``triggers.yml`` loader, and the CLI's build-profile
schema — so the tests here pin the numbers and both structural properties that
let those three share them:

* The module is a stdlib-only leaf. The profile schema is imported by every
  ``osprey build`` / ``osprey config`` / ``osprey init`` invocation, and the
  numbers must reach it without dragging :mod:`osprey.dispatch` into the
  profile import graph at all.
* :mod:`osprey.dispatch.trigger_config` re-exports them, so code reading the
  defaults from the runtime package sees the same pair.

Both are checked on a fresh interpreter, not on the already-populated
``sys.modules`` of the test session.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from osprey.cli.build_profile_schema import DispatchConfig
from osprey.dispatch_pool_defaults import DEFAULT_MAX_CONCURRENT_RUNS, DEFAULT_MAX_QUEUE_DEPTH

_SRC = str(Path(__file__).resolve().parents[1] / "src")


def _fresh_import_modules(module: str, allowed: set[str]) -> list[str]:
    """Return the non-stdlib ``osprey`` modules a fresh import of ``module`` adds.

    Args:
        module: Dotted name to import in a child interpreter.
        allowed: Module names the import may add without being reported.

    Returns:
        The offending module names, sorted — empty when the import stays
        inside ``allowed``.

    Raises:
        AssertionError: If the child interpreter fails to import the module.
    """
    code = (
        "import json, sys;"
        "before = set(sys.modules);"
        f"import {module};"
        f"allowed = {sorted(allowed)!r};"
        "delta = set(sys.modules) - before - set(allowed);"
        "print(json.dumps(sorted("
        "m for m in delta if m.split('.')[0] not in sys.stdlib_module_names)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=_SRC),
        check=False,
    )
    assert result.returncode == 0, f"fresh import of {module} failed:\n{result.stderr}"
    return json.loads(result.stdout)


def test_defaults_are_the_documented_pair():
    """The hand-run dispatcher starts at the posture the documentation describes."""
    assert (DEFAULT_MAX_CONCURRENT_RUNS, DEFAULT_MAX_QUEUE_DEPTH) == (2, 50)


def test_the_leaf_imports_nothing_from_osprey():
    """Importing the defaults costs the package ``__init__`` and nothing more.

    The parent package ``__init__`` runs for any submodule import and pulls in
    ``osprey`` and ``osprey.version``; those two and the module itself are the
    only ``osprey`` names the import may add.
    """
    offenders = _fresh_import_modules(
        "osprey.dispatch_pool_defaults",
        {"osprey", "osprey.version", "osprey.dispatch_pool_defaults"},
    )
    assert offenders == []


def test_the_profile_schema_keeps_the_dispatch_package_out_of_its_import_graph():
    """The CLI reads the numbers from the leaf, not through ``osprey.dispatch``.

    Every ``build``, ``config`` and ``init`` invocation imports the schema, and
    the dispatcher package has no place in that import graph at all: the
    profile's own trigger check imports it on demand, and the pool defaults
    must not undo that. This is a stronger guarantee than the package's own
    lazy exports, which say only that importing the package is cheap, and it
    must not be relaxed into it.
    """
    fresh = _fresh_import_modules("osprey.cli.build_profile_schema", set())
    assert not [name for name in fresh if name.split(".")[:2] == ["osprey", "dispatch"]]


def test_trigger_config_re_exports_the_leaf_values():
    """The runtime package's spelling of the defaults agrees with the leaf's."""
    from osprey.dispatch import trigger_config

    assert trigger_config.DEFAULT_MAX_CONCURRENT_RUNS == DEFAULT_MAX_CONCURRENT_RUNS
    assert trigger_config.DEFAULT_MAX_QUEUE_DEPTH == DEFAULT_MAX_QUEUE_DEPTH


def test_every_consumer_starts_at_the_leaf_values():
    """Runtime dataclass and build-profile block agree by construction."""
    from osprey.dispatch.trigger_config import DispatcherConfig

    assert DispatcherConfig(dispatch_target="").max_concurrent_runs == DEFAULT_MAX_CONCURRENT_RUNS
    assert DispatcherConfig(dispatch_target="").max_queue_depth == DEFAULT_MAX_QUEUE_DEPTH
    assert DispatchConfig(triggers="x").max_concurrent_runs == DEFAULT_MAX_CONCURRENT_RUNS
    assert DispatchConfig(triggers="x").max_queue_depth == DEFAULT_MAX_QUEUE_DEPTH
