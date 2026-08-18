"""A compose render is a function of its inputs and nothing else.

The golden suites beside this one pin *what* every bundled template renders.
This one pins something they structurally cannot: that rendering the same
deployment twice produces the same bytes. A golden suite compares one render
against a committed file, so a value that is fresh on every render is invisible
to it the moment the harness pins that value into the context — which is
exactly how a wall-clock deploy timestamp survived in the rendered documents
while every golden stayed green.

Why the property matters beyond tidiness: compose decides whether to recreate a
container by diffing its service definition, so any per-run value in a rendered
document makes ``osprey up`` tear down and rebuild the whole stack on a build
that changed nothing. A build directory whose bytes move on their own is also
undiffable — an operator cannot tell a real configuration change from noise.

The context is built by the production injection
(``_inject_project_metadata``), twice, from one unchanged config — not built
once and rendered twice, which would only prove that Jinja is deterministic.
The two injections are deliberately separated by a real clock tick, so a
timestamp reintroduced into the context is caught here rather than passing by
landing inside the same microsecond.
"""

from __future__ import annotations

import datetime
import tempfile
from pathlib import Path

# Imported by bare module name for the same reason the axis-shape suite does it:
# this directory carries no ``__init__.py``, so pytest names the module by its
# basename.
from test_render_defaults_golden import (
    TEMPLATES,
    _raw_default_config,
    _render_templates,
)


def _await_clock_tick() -> None:
    """Block until the system clock reports a different instant.

    Cheap — the resolution of ``datetime.now()`` is microseconds — and it is
    what gives the comparison below its teeth: two renders that happened to
    share a timestamp would agree even with a timestamp in the document.
    """
    start = datetime.datetime.now()
    while datetime.datetime.now() == start:
        pass


def _render_everything(config: dict) -> dict[str, str]:
    """Inject *config* the way production does, then render every template."""
    from osprey.deployment.compose_generator import _inject_project_metadata

    return _render_templates(_inject_project_metadata(dict(config)), TEMPLATES)


def test_rendering_the_same_deployment_twice_is_byte_identical() -> None:
    """Every bundled template renders the same bytes on a second pass.

    Asserted per document rather than on the two mappings at once, so a failure
    names the template that moved instead of dumping every rendered file.
    """
    with tempfile.TemporaryDirectory() as directory:
        repo_root = Path(directory)
        (repo_root / ".env").write_text("OSPREY_GOLDEN_FIXTURE=1\n", encoding="utf-8")
        config = _raw_default_config(repo_root)

        first = _render_everything(config)
        _await_clock_tick()
        second = _render_everything(config)

    assert first.keys() == second.keys()
    assert first, "no bundled templates were rendered — the discovery glob found nothing"

    for name in sorted(first):
        assert first[name] == second[name], (
            f"{name} renders differently on a second pass from an unchanged "
            "deployment. Something in the render context varies per run — a "
            "clock, a random value, or an unordered iteration. A compose "
            "document must be a function of its inputs alone."
        )


def test_the_injected_context_carries_no_per_run_values() -> None:
    """The context itself is stable, not merely the templates' use of it.

    The render check above only catches a volatile value that some template
    happens to emit. This catches one sitting in the context unused, waiting for
    the next template author to reach for it.
    """
    from osprey.deployment.compose_generator import _inject_project_metadata

    with tempfile.TemporaryDirectory() as directory:
        repo_root = Path(directory)
        (repo_root / ".env").write_text("OSPREY_GOLDEN_FIXTURE=1\n", encoding="utf-8")
        config = _raw_default_config(repo_root)

        first = _inject_project_metadata(dict(config))
        _await_clock_tick()
        second = _inject_project_metadata(dict(config))

    assert first["osprey_labels"] == second["osprey_labels"]
    assert first == second
