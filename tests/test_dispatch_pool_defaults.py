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

from osprey.cli.build_profile_schema import DispatchConfig
from osprey.dispatch_pool_defaults import DEFAULT_MAX_CONCURRENT_RUNS, DEFAULT_MAX_QUEUE_DEPTH
from tests._fresh_import import non_stdlib_added_by_import


def test_defaults_are_the_documented_pair():
    """The hand-run dispatcher starts at the posture the documentation describes."""
    assert (DEFAULT_MAX_CONCURRENT_RUNS, DEFAULT_MAX_QUEUE_DEPTH) == (2, 50)


def test_the_leaf_imports_nothing_from_osprey():
    """Importing the defaults costs the package ``__init__`` and nothing more.

    The parent package ``__init__`` runs for any submodule import, and what it
    imports is the package's own cost. The child imports ``osprey`` before the
    baseline, so the module itself is the only name the import may add.
    """
    offenders = non_stdlib_added_by_import(
        "osprey.dispatch_pool_defaults", {"osprey.dispatch_pool_defaults"}, preload=("osprey",)
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
    fresh = non_stdlib_added_by_import("osprey.cli.build_profile_schema")
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
