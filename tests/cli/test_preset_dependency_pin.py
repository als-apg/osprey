"""Pin: a bundled preset must not need a profile ``dependencies:`` entry.

A profile's ``dependencies:`` list reaches exactly three places, all of them the
*deployed project's*: the project venv at ``build/.venv``, the ``pyproject.toml``
recorded beside it, and the install line of the project image. It never reaches
the interpreter running the ``osprey`` CLI — and some of what a deploy does runs
there, in-process, notably the archiver store's seeding (see
``container_lifecycle._stage_archiver_store``).

That gap is why the ``control-assistant`` preset once carried
``dependencies: [pymongo>=4.0]`` and ``osprey up`` still refused: the entry was
real, it did install pymongo into all three project environments, and none of
them was the one doing the seeding. The declaration looked like the fix, which
is what kept the actual problem — a framework dependency filed as an optional
extra — out of sight.

So the rule pinned here is the one that was violated, not the symptom: a preset
OSPREY ships must be deployable from a plain ``pip install osprey-framework``. A
non-empty entry below is a claim that a package OSPREY does not ship is
nonetheless required by a preset OSPREY does ship. That is a legitimate thing to
want — a facility toolkit the agent imports inside the Python executor is
project business, not framework business — but it is worth stopping over,
because the same line is indistinguishable from the mistake above. Add the entry
knowingly, with the reason beside it.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from osprey.cli.build_profile import _load_preset_raw, list_presets

#: preset name -> the ``dependencies:`` it declares in its own file.
#:
#: Raw rather than resolved, deliberately: this pins what an author edits. A
#: value inherited through ``extends:`` shows up on the base that declares it,
#: which is where the decision was made.
PINNED_PRESET_DEPENDENCIES: dict[str, list[str]] = {
    "ariel-standalone": [],
    "channel-finder-standalone": [],
    # Emptied when pymongo became a core OSPREY dependency. Before that this
    # read ``[pymongo>=4.0]`` — see the module docstring.
    "control-assistant": [],
    "control-assistant-ariel": [],
    "control-assistant-readonly": [],
    "control-assistant-readwrite": [],
    "hello-world": [],
}

#: Packages a deploy imports in the CLI's own process, which therefore cannot be
#: supplied by any profile. Each must be a core dependency of the framework.
HOST_SIDE_PACKAGES = {
    # container_lifecycle._preflight_archiver_pymongo and, behind it,
    # _stage_archiver_store -> simulation.apply.archiver_collection.
    "pymongo",
}


def _distribution_name(spec: str) -> str:
    """The bare distribution name from a requirement string."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    return canonicalize_name(Requirement(spec).name)


def _framework_pyproject() -> dict:
    root = Path(__file__).resolve().parents[2]
    return tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))


def test_bundled_preset_set_is_pinned():
    """A new preset must be classified here before it ships.

    Without this the comparison below would silently skip an unpinned preset,
    and the pin would degrade as presets are added.
    """
    assert list_presets() == sorted(PINNED_PRESET_DEPENDENCIES)


def test_every_bundled_preset_declares_the_dependencies_it_is_pinned_to():
    """No bundled preset asks for a package OSPREY does not already ship."""
    actual = {
        name: list(_load_preset_raw(name)[0].get("dependencies", []) or [])
        for name in list_presets()
    }
    assert actual == PINNED_PRESET_DEPENDENCIES


def test_host_side_packages_are_core_dependencies():
    """A package the CLI imports during a deploy cannot live behind an extra.

    This is the check that would have caught the original mistake at the commit
    that made it, rather than at a deploy months later: pymongo was reachable
    only through the ``archiver-mongodb`` extra while a shipped preset's deploy
    path imported it unconditionally.
    """
    core = {_distribution_name(spec) for spec in _framework_pyproject()["project"]["dependencies"]}
    missing = sorted(HOST_SIDE_PACKAGES - core)
    assert not missing, (
        f"{missing} are imported by the deploy path in the CLI's own process, "
        "so they must be core dependencies — no profile can supply them"
    )


def test_host_side_packages_are_not_also_optional():
    """An extra that re-declares a core dependency invites the floor to drift.

    It also leaves an install command in circulation that reads as necessary and
    is not: pip only *warns* about an extra that does not exist, so a stale hint
    sends the reader to install nothing and hit the same failure again.
    """
    extras = _framework_pyproject()["project"].get("optional-dependencies", {})
    offenders = {
        f"{extra}: {spec}"
        for extra, specs in extras.items()
        for spec in specs
        if _distribution_name(spec) in HOST_SIDE_PACKAGES
    }
    assert not offenders, sorted(offenders)
