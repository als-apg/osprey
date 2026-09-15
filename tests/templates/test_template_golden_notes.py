"""Every bundled service template names the suites that pin its render.

Each template under the packaged ``templates/services/`` tree opens with a
Jinja comment saying its render is pinned byte for byte and listing one
regeneration command per suite that pins it. The note exists so an editor
learns what a reworded comment costs *in the file being edited*, rather than
meeting it later as a red full-tree gate.

A note is a copy of a mapping that lives elsewhere, so it can go stale. These
tests derive the same mapping from its sources — the defaults suite's glob and
the axis-shapes suite's ``SCENARIOS`` — and fail when a template names a suite
that does not pin it, or omits one that does.
"""

from __future__ import annotations

from pathlib import PurePosixPath

# Imported by bare module name, not as ``tests.templates.…``: this directory
# carries no ``__init__.py``, so pytest puts it on ``sys.path`` and names each
# module by its basename.
from test_render_axis_shapes import SCENARIOS
from test_render_defaults_golden import TEMPLATES, _templates_root

#: The regeneration entry point that pins EVERY bundled template: it discovers
#: them by glob, so a template it does not pin cannot exist.
_DEFAULTS_ENTRY_POINT = "tests/templates/test_render_defaults_golden.py"

#: The entry point that pins whatever some ``Scenario.templates`` names.
_AXIS_SHAPES_ENTRY_POINT = "tests/templates/test_render_axis_shapes.py"

#: The deployment suites that each pin exactly one template, by service key.
#: A literal map rather than a derivation: each of these renders one service on
#: a path of its own, and there is no glob or table to read it back off.
_DEPLOYMENT_SUITES = {
    "tests/deployment/test_lane_compose.py": "bluesky",
    "tests/deployment/test_va_compose_instances.py": "virtual_accelerator",
    "tests/deployment/test_recorder_standin_compose.py": "archiver_recorder",
}


def _service_key(rel_path: str) -> str:
    """The service a bundled template renders, from its templates-root path.

    The one template at the services root renders no single service; it is
    keyed ``services-root``, the name its golden already carries.
    """
    parent = PurePosixPath(rel_path).parent.name
    return "services-root" if parent == "services" else parent


def _template_text(rel_path: str) -> str:
    """The source of the bundled template at *rel_path*."""
    return (_templates_root() / rel_path).read_text(encoding="utf-8")


def _scenario_named() -> set[str]:
    """Service keys named by at least one axis-shapes scenario."""
    return {key for scenario in SCENARIOS for key in scenario.templates}


def test_every_template_names_the_defaults_entry_point() -> None:
    """The suite that pins all of them is listed in all of them."""
    missing = [rel for rel in TEMPLATES if _DEFAULTS_ENTRY_POINT not in _template_text(rel)]
    assert not missing, (
        "these templates omit the defaults regeneration command "
        f"({_DEFAULTS_ENTRY_POINT}), which pins every bundled template: "
        f"{sorted(_service_key(rel) for rel in missing)}"
    )


def test_the_axis_shapes_entry_point_is_named_by_exactly_the_scenario_templates() -> None:
    """A template a scenario renders names that suite; one no scenario renders does not."""
    named = _scenario_named()
    omits = sorted(
        _service_key(rel)
        for rel in TEMPLATES
        if _service_key(rel) in named and _AXIS_SHAPES_ENTRY_POINT not in _template_text(rel)
    )
    claims = sorted(
        _service_key(rel)
        for rel in TEMPLATES
        if _service_key(rel) not in named and _AXIS_SHAPES_ENTRY_POINT in _template_text(rel)
    )
    assert not omits, (
        f"these templates are rendered by some Scenario.templates but omit "
        f"{_AXIS_SHAPES_ENTRY_POINT}: {omits}"
    )
    assert not claims, (
        f"these templates name {_AXIS_SHAPES_ENTRY_POINT} but no scenario "
        f"renders them, so that suite pins nothing of theirs: {claims}"
    )


def test_each_deployment_suite_is_named_only_by_the_template_it_pins() -> None:
    """One suite, one template — and no other template claims it."""
    for entry_point, key in _DEPLOYMENT_SUITES.items():
        by_key = {_service_key(rel): rel for rel in TEMPLATES}
        assert key in by_key, f"{entry_point} pins {key}, which is not a bundled template"
        assert entry_point in _template_text(by_key[key]), (
            f"the {key} template omits {entry_point}, the suite that pins its render"
        )
        claims = sorted(
            other
            for other, rel in by_key.items()
            if other != key and entry_point in _template_text(rel)
        )
        assert not claims, f"these templates name {entry_point}, which pins {key} alone: {claims}"
