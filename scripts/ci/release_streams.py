"""The component release streams: one record per independently released package.

Every component package (osprey-connectors, osprey-facility, osprey-panels)
releases on its own calendar tag stream, ``<tag_prefix><version>``. The release
workflow, the lean import walk and the compatibility lanes read what differs
between the streams from this table rather than from per-package literals, so a
fourth stream is one record here.

Commands, each taking ``<ref>`` as the full ``GITHUB_REF``
(``refs/tags/osprey-facility-v2026.12.0``; a bare tag is refused):

``resolve <ref>``
    The stream's scalar outputs, one ``name=value`` line each, appended to
    ``$GITHUB_OUTPUT`` when it is set and printed on stdout.
``lanes <ref>``
    The stream's required CI lanes as ``<job id>=<rendered name prefix>`` lines,
    the form ``require_green_lanes.py --lanes`` reads.
``dists [--joined]``
    The wheel/sdist file prefix of every stream, one per line, or on one
    space-separated line with ``--joined`` (for a job output).

A ref outside the three tag globs, and a version not spelled in canonical
PEP 440 form, exit non-zero. A non-canonical spelling is refused rather than
normalised, so this stdlib-only table never names a version the built wheel
does not carry.

Stdlib only: the lean walk loads this file by path under ``python -I``.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass

TAG_REF_PREFIX = "refs/tags/"

#: The agent and web stack a lean wheel must never load, whatever the stream.
_AGENT_STACK = ("litellm", "openai", "anthropic", "fastapi", "playwright")

#: Canonical PEP 440 release: ``X.Y.Z``, optionally ``aN``/``bN``/``rcN`` and
#: ``.devN``, with no leading zero in any number.
_N = r"(?:0|[1-9][0-9]*)"
CANONICAL_VERSION = re.compile(rf"^{_N}\.{_N}\.{_N}(?:(?:a|b|rc){_N})?(?:\.dev{_N})?$")
#: The final-release whitelist ``release.yml`` uses; anything else is a prerelease.
FINAL_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")

#: Lanes every stream requires: (ci.yml job id, rendered name prefix).
_COMMON_LANES = (("test", "Test Python"), ("lint", "Lint and Code Quality"))


@dataclass(frozen=True)
class Stream:
    """One component release stream.

    Attributes:
        key: Short stream name (``connectors``, ``facility``, ``panels``).
        dist: The distribution name on PyPI.
        import_name: The top-level import package.
        package_dir: The package directory, relative to the repository root.
        tag_prefix: The release tag is ``<tag_prefix><version>``.
        environment: The GitHub environment the publish job runs in.
        changelog: The package changelog, relative to the repository root.
        required_lanes: (ci.yml job id, rendered name prefix) pairs that must be
            green on the tagged commit.
        lean_forbidden: Top-level modules the installed wheel must never load.
        lean_extras: Extras installed for the lean walk.
        typed: Whether the wheel ships ``py.typed``.
        floor_consumers: Import names of the framework trees that consume this
            stream; its floor scan reads them.
        lean_entry_modules: Modules the lean walk must find loaded.
        smoke: The command verify-install runs, relative to the bare venv's
            ``bin/`` parent; empty for none.
        compat: Whether the stream has compatibility lanes.
    """

    key: str
    dist: str
    import_name: str
    package_dir: str
    tag_prefix: str
    environment: str
    changelog: str
    required_lanes: tuple[tuple[str, str], ...]
    lean_forbidden: frozenset[str]
    lean_extras: tuple[str, ...]
    typed: bool
    floor_consumers: tuple[str, ...]
    lean_entry_modules: tuple[str, ...]
    smoke: str
    compat: bool

    @property
    def tag_glob(self) -> str:
        """The release-tag glob, calendar versions only (``<tag_prefix>2*``)."""
        return f"{self.tag_prefix}2*"

    @property
    def dist_prefix(self) -> str:
        """The wheel/sdist file-name prefix (``osprey_connectors-``)."""
        return f"{self.dist.replace('-', '_')}-"


STREAMS: tuple[Stream, ...] = (
    Stream(
        key="connectors",
        dist="osprey-connectors",
        import_name="osprey_connectors",
        package_dir="packages/osprey-connectors",
        tag_prefix="osprey-connectors-v",
        environment="pypi-connectors",
        changelog="packages/osprey-connectors/CHANGELOG.md",
        required_lanes=(
            *_COMMON_LANES,
            ("package", "Package Build"),
            ("connectors-standalone", "Connectors Standalone Tests"),
            ("connectors-compat", "Connectors Compat"),
            ("framework-at-connectors-floor", "Framework at Connectors Floor Tests"),
        ),
        lean_forbidden=frozenset({*_AGENT_STACK, "osprey"}),
        lean_extras=(),
        typed=True,
        floor_consumers=("osprey", "pyaml_cs_osprey"),
        lean_entry_modules=(
            "osprey_connectors",
            "osprey_connectors.config",
            "osprey_connectors.control_system",
            "osprey_connectors.control_system._epics_connector",
            "osprey_connectors.errors",
            "osprey_connectors.logger",
            "osprey_connectors.simulation",
        ),
        smoke="",
        compat=True,
    ),
    Stream(
        key="facility",
        dist="osprey-facility",
        import_name="osprey_facility",
        package_dir="packages/osprey-facility",
        tag_prefix="osprey-facility-v",
        environment="pypi-facility",
        changelog="packages/osprey-facility/CHANGELOG.md",
        required_lanes=(
            *_COMMON_LANES,
            ("package", "Package Build"),
            ("osprey-facility-standalone", "Facility Standalone Tests"),
            ("facility-compat", "Facility Compat"),
            ("framework-at-facility-floor", "Framework at Facility Floor Tests"),
            # HEAD facility runs at its own connectors floor in this lane.
            ("framework-at-connectors-floor", "Framework at Connectors Floor Tests"),
        ),
        lean_forbidden=frozenset({*_AGENT_STACK, "osprey"}),
        lean_extras=("mml",),
        typed=False,
        floor_consumers=("osprey",),
        lean_entry_modules=("osprey_facility",),
        smoke="",
        compat=True,
    ),
    Stream(
        key="panels",
        dist="osprey-panels",
        import_name="osprey_panels",
        package_dir="packages/osprey-panels",
        tag_prefix="osprey-panels-v",
        environment="pypi-panels",
        changelog="packages/osprey-panels/CHANGELOG.md",
        # `package` does not build the panels wheel; the standalone lane does,
        # and runs the walk against it.
        required_lanes=(
            *_COMMON_LANES,
            ("osprey-panels-standalone", "Panels Standalone Tests"),
        ),
        # Panels reads the facility document as JSON; it imports neither the
        # framework nor the other component packages.
        lean_forbidden=frozenset({*_AGENT_STACK, "osprey", "osprey_facility", "osprey_connectors"}),
        lean_extras=(),
        typed=False,
        floor_consumers=(),
        lean_entry_modules=("osprey_panels", "osprey_panels.cli"),
        smoke="bin/osprey-panels --help",
        compat=False,
    ),
)


class StreamError(ValueError):
    """A ref or version this table refuses."""


def _globs() -> str:
    return ", ".join(TAG_REF_PREFIX + stream.tag_glob for stream in STREAMS)


def stream_for(ref: str) -> tuple[Stream, str]:
    """The stream a full tag ref belongs to, and its canonical version.

    Raises:
        StreamError: *ref* is not ``refs/tags/<one of the tag globs>``, or its
            version is not canonical PEP 440.
    """
    if not ref.startswith(TAG_REF_PREFIX):
        raise StreamError(f"{ref!r} is not a full tag ref; expected one of: {_globs()}")
    tag = ref[len(TAG_REF_PREFIX) :]
    for stream in STREAMS:
        if fnmatch.fnmatchcase(tag, stream.tag_glob):
            version = tag[len(stream.tag_prefix) :]
            if not CANONICAL_VERSION.match(version):
                raise StreamError(
                    f"{ref!r}: version {version!r} is not canonical PEP 440 "
                    "(X.Y.Z, optionally aN/bN/rcN and .devN, no leading zeros)"
                )
            return stream, version
    raise StreamError(f"{ref!r} is not a component release tag; expected one of: {_globs()}")


def resolve_outputs(ref: str) -> list[tuple[str, str]]:
    """The ``resolve`` outputs for *ref*, in order."""
    stream, version = stream_for(ref)
    prerelease = "false" if FINAL_VERSION.match(version) else "true"
    outputs = [
        ("key", stream.key),
        ("dist", stream.dist),
        ("import_name", stream.import_name),
        ("package_dir", stream.package_dir),
        ("environment", stream.environment),
        ("changelog", stream.changelog),
        ("tag_prefix", stream.tag_prefix),
        ("lean_extras", ",".join(stream.lean_extras)),
        ("smoke", stream.smoke),
        ("version", version),
        ("prerelease", prerelease),
    ]
    for name, value in outputs:
        # GitHub's single-line form: a newline would need the delimiter form.
        assert "\n" not in value and "\r" not in value, f"{name} value spans lines"
    return outputs


def lane_lines(ref: str) -> list[str]:
    """The ``<job id>=<rendered name prefix>`` lines of *ref*'s stream."""
    stream, _ = stream_for(ref)
    return [f"{job}={name}" for job, name in stream.required_lanes]


def dist_prefixes() -> list[str]:
    """The wheel/sdist file prefix of every stream."""
    return [stream.dist_prefix for stream in STREAMS]


def _write_outputs(lines: Sequence[str]) -> None:
    text = "\n".join(lines)
    print(text)
    target = os.environ.get("GITHUB_OUTPUT")
    if target:
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(text + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    resolve = commands.add_parser("resolve", help="write the stream outputs for a tag ref")
    resolve.add_argument("ref")
    lanes = commands.add_parser("lanes", help="print the stream's required lanes")
    lanes.add_argument("ref")
    dists = commands.add_parser("dists", help="print every stream's wheel prefix")
    dists.add_argument("--joined", action="store_true", help="on one line")
    args = parser.parse_args(argv)

    try:
        if args.command == "resolve":
            _write_outputs([f"{name}={value}" for name, value in resolve_outputs(args.ref)])
        elif args.command == "lanes":
            print("\n".join(lane_lines(args.ref)))
        else:
            prefixes = dist_prefixes()
            print(" ".join(prefixes) if args.joined else "\n".join(prefixes))
    except StreamError as exc:
        print(f"release_streams: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
