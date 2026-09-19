"""No two e2e modules deploy under the same compose project name.

A compose project is addressed BY NAME. Two deploy e2es that share one name do
not merely collide on container names when they run together: compose adopts
whichever containers already carry the name, and a teardown from either side —
``osprey down``, or the project-scoped volume sweep — removes the other
module's running stack. The failure lands on the innocent module, minutes or
hours into its own deploy, as a service that refuses to start or a port that is
suddenly free.

The suite's own lane makes that reachable rather than theoretical: CI runs
``tests/e2e/`` with ``-n 4 --dist loadfile``, so four modules deploy at once on
one daemon, and a developer running two modules by name gets the same.

This reads the SOURCE of every e2e module with ``ast`` instead of importing it.
Importing an e2e module executes its import-time guards — reserving host ports,
resolving a runtime, skipping on a missing binary — none of which a check about
names has any business doing.

The constants it reads are the ones whose name carries ``PROJECT_NAME``, with a
plain string literal for a value. That is what every deploy e2e calls the name
it passes to ``osprey init`` and derives its container, image, network and
volume targets from. A value built from an f-string is derived from one of
these and is unique exactly when its source is.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

#: The suite whose names must not collide, and the marker in a constant's NAME
#: that says the value is a compose project name.
E2E_ROOT = Path(__file__).resolve().parents[1] / "e2e"
PROJECT_NAME_MARKER = "PROJECT_NAME"

#: What compose takes as a project name: lowercase ASCII letters, digits,
#: hyphen and underscore, starting with a letter or a digit.
LEGAL_PROJECT_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")


def _module_project_names(path: Path) -> dict[str, str]:
    """Every module-level compose project name *path* declares, by constant name.

    ``AnnAssign`` as well as ``Assign``: ``PROJECT_NAME: str = "..."`` is a
    spelling this codebase already uses, and reading only the bare form would
    let an annotated constant slip past the check unseen.

    Only plain string literals are collected. An f-string value is derived from
    another constant in the same module, so it carries that one's uniqueness and
    has none of its own to check.
    """
    found: dict[str, str] = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        elif isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        else:
            continue
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        for target in targets:
            if isinstance(target, ast.Name) and PROJECT_NAME_MARKER in target.id:
                found[target.id] = value.value
    return found


def _declared_project_names() -> dict[str, list[str]]:
    """Every compose project name the e2e suite declares, mapped to its sites.

    A site reads ``<module>.py:<CONSTANT>`` so a collision report names both
    halves of the pair without a reader having to grep for either.
    """
    by_name: dict[str, list[str]] = {}
    for path in sorted(E2E_ROOT.rglob("*.py")):
        for constant, value in _module_project_names(path).items():
            site = f"{path.relative_to(E2E_ROOT)}:{constant}"
            by_name.setdefault(value, []).append(site)
    return by_name


def test_the_e2e_suite_declares_project_names_the_uniqueness_check_can_compare() -> None:
    """Non-vacuity: a reader that finds nothing would pass every check below.

    The whole guard rests on the constants being found in source. A rename of
    the convention, a move of the suite, or a parse that quietly returns nothing
    would leave the uniqueness check passing over an empty set. Two is the bar
    rather than one, because a single name collides with nothing either.
    """
    declared = _declared_project_names()
    assert len(declared) >= 2, (
        f"{E2E_ROOT} yielded {len(declared)} compose project name(s), so the "
        f"uniqueness check below has nothing to compare. Either the suite moved "
        f"or deploy e2es stopped spelling the constant with {PROJECT_NAME_MARKER!r}."
    )


def test_the_reader_takes_project_names_and_leaves_everything_else(tmp_path: Path) -> None:
    """The reader, against a source that carries one of each shape.

    A guard over the real suite passes whether the reader works or reads
    nothing, and the day it starts reading nothing is the day the suite is
    unprotected. So the reader is exercised on a file whose answer is known:
    the annotated and the bare constant are both taken, a derived name and a
    path are both left, and a value that is not a plain string is left.
    """
    source = tmp_path / "test_shapes.py"
    source.write_text(
        'PROJECT_NAME: str = "osprey-e2e-shapes"\n'
        'HETERO_PROJECT_NAME = "osprey-e2e-shapes-hetero"\n'
        'BRIDGE_CONTAINER = f"{PROJECT_NAME}-bluesky-bridge"\n'
        'WORKER_PROJECT_DIR = "/app/osprey-e2e-shapes"\n'
        'PERSONA_PROJECT = "shapes-operator"\n'
        'SETTING_SOURCES_PROJECT = ["--setting-sources", "project"]\n',
        encoding="utf-8",
    )
    assert _module_project_names(source) == {
        "PROJECT_NAME": "osprey-e2e-shapes",
        "HETERO_PROJECT_NAME": "osprey-e2e-shapes-hetero",
    }


def test_no_two_e2e_sites_declare_the_same_compose_project_name() -> None:
    """Every declared compose project name is claimed exactly once.

    A shared name is a teardown that reaches into another stack, so this fails
    on the collision itself rather than on the symptom it produces in whichever
    module happens to lose the race.

    By SITE rather than by module, which is the stricter reading and the right
    one: one module declaring the same name under two constants would deploy two
    stacks onto one project just as surely as two modules would.
    """
    shared = {
        name: sorted(sites) for name, sites in _declared_project_names().items() if len(sites) > 1
    }
    assert not shared, (
        "these compose project names are claimed by more than one site:\n"
        + "\n".join(f"  {name!r}: {', '.join(sites)}" for name, sites in sorted(shared.items()))
        + "\nA compose project is addressed by name: the two stacks adopt each "
        "other's containers, and a teardown from either side removes the other's. "
        "Give each module its own descriptive name (the convention is "
        "'osprey-e2e-<what the module deploys>')."
    )


@pytest.mark.parametrize("path", sorted(E2E_ROOT.rglob("test_*.py")), ids=lambda p: p.name)
def test_every_e2e_project_name_is_a_legal_compose_project(path: Path) -> None:
    """A project name compose cannot take aborts the deploy before any container.

    Compose accepts a lowercase ASCII name built from letters, digits, hyphen
    and underscore, starting with a letter or a digit. A name outside that is
    rejected by the CLI, which is a deploy that never happens rather than a test
    that fails on its subject — so it is caught here, where the name is written.

    Matched against :data:`LEGAL_PROJECT_NAME` rather than tested character by
    character with ``str.isalnum``, which answers true for letters and digits
    compose does not take ('ä', '½').
    """
    for constant, value in _module_project_names(path).items():
        assert LEGAL_PROJECT_NAME.fullmatch(value), (
            f"{path.name}:{constant} is {value!r}, which compose cannot take as a "
            f"project name: it must match {LEGAL_PROJECT_NAME.pattern} — lowercase "
            "ASCII letters, digits, hyphen and underscore, starting with a letter "
            "or a digit. A deploy under such a name never happens, and the "
            "containers this module tears down by name are not the ones it created"
        )
