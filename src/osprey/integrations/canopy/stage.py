# ruff: noqa: T201 -- the pre_build stage command reports on stdout
"""Stage the CANOPY payload into a deployment repo: the ``pre_build`` step.

Run as ``"{python}" -m osprey.integrations.canopy.stage`` with the repo root as
the working directory. It builds nothing. The payload — the ``canopy`` and
``pydmconverter`` wheels, the viewer (whose ``ir-map.json`` is the widget
catalog), Trace, the staged spec when there is one, and the ``STAGED.json``
that names what they were built from — is made in the canopy repo by
``scripts/build-payload.sh``; this step checks that directory and copies it
in:

* the two wheels go into the repo root, where the image's wheel-drop layer
  installs every ``*.whl`` it finds;
* everything else goes under ``canopy_payload/``, ``spec/`` included: CANOPY
  finds the IR and prop schemas there, beside the viewer, and answers
  ``spec_unavailable`` without them.

``STAGED.json`` records the spec as ``"spec"`` (the commit it came from) and
``"spec-digest"`` (a digest of the tree), both ``null`` when the payload has
none. The two must agree with the directory: a payload whose ``STAGED.json``
names a spec it does not hold, or holds a ``spec/`` its ``STAGED.json`` does
not name, is refused.

``STAGED.json`` also names the commits the wheels were built from, as
``"canopy"`` and ``"pydm"``. Each must be the whole commit
:data:`~osprey.integrations.canopy.PROVEN_COMMITS` names for that
distribution: write refusal through CANOPY is proven only against that pair,
so a payload built from any other commit, or one that names none, is refused.

The payload directory is named by ``OSPREY_CANOPY_PAYLOAD``: the process
environment first, then ``.env.shared`` and ``.env`` in the working directory
(``.env`` wins). The lifecycle runner does not pass the repo's env chain to a
``pre_build`` step, which is why the files are read here. A directory that is
missing, or that has no ``STAGED.json`` — the builder writes it last, so its
absence means a half-built payload — is refused with one line on stderr and a
non-zero exit, and nothing in the repo is touched.

Staging is idempotent. Every ``canopy-*.whl`` and ``pydmconverter-*.whl`` in
the repo root (matched case-insensitively) is removed before the new pair is
copied, so a version bump leaves no stale wheel for the image to install, and
``canopy_payload/`` is replaced whole: copied beside the old one, then renamed
into place. When the staged ``STAGED.json`` already equals the payload's and
both wheels are in place, nothing is rewritten; so a payload rebuilt with
another spec (a changed ``spec`` or ``spec-digest``) is staged anew, and so is
one whose staged copy has lost the ``spec/`` its ``STAGED.json`` names.

With ``OSPREY_CANOPY_PAYLOAD`` unset nothing is staged and the step passes:
the build leaves the CANOPY panel out of the deployment
(:func:`panel_off`) and says so in one line. Set but missing, half-built or
naming other commits is refused as above.

``OSPREY_CANOPY_PAYLOAD_CHECK=0`` exits 0 without staging anything, and so
without the pin check. It is a development switch for building an image with no
CANOPY payload; a deployment that serves the CANOPY panel leaves it unset.

Before any of that, and whatever that switch says, the step checks the
deployment's screen library (``data/canopy/library``) against its write-binding
report with :func:`~osprey.integrations.canopy.library.library_is_fresh`. A
library whose files changed since the report was written stops the build
here, with one line per difference and the command that regenerates the
report; a deployment with no library, or an empty one, passes.

This module imports nothing from CANOPY: it runs under the interpreter that
runs ``osprey build``, where CANOPY is not installed.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from osprey.integrations.canopy import LIBRARY_SUBPATH, PAYLOAD_DIRNAME, PROVEN_COMMITS
from osprey.integrations.canopy import proven as proven_module
from osprey.integrations.canopy.library import library_is_fresh
from osprey_connectors.dotenv import merge_chain
from osprey_connectors.safety_file import atomic_replace

#: The mode a staged wheel is written with: read by the image build, never executed.
WHEEL_MODE = 0o644

#: Names the payload directory ``scripts/build-payload.sh`` wrote.
PAYLOAD_ENV = "OSPREY_CANOPY_PAYLOAD"
#: ``0`` skips the step entirely.
CHECK_ENV = "OSPREY_CANOPY_PAYLOAD_CHECK"

#: The file that makes a directory a payload, and names what it was built from.
STAGED_FILENAME = "STAGED.json"
#: The new copy while it is being written, and the old one while it is being
#: replaced. Dot-prefixed, so neither is ever taken for a profile directory.
SCRATCH_DIRNAME = f".{PAYLOAD_DIRNAME}.staging"
PREVIOUS_DIRNAME = f".{PAYLOAD_DIRNAME}.previous"

#: The staged spec directory inside a payload, and the ``STAGED.json`` keys
#: that name it.
SPEC_DIRNAME = "spec"
SPEC_KEYS = ("spec", "spec-digest")

#: The ``STAGED.json`` key that names each proven distribution's commit, keyed by that key.
COMMIT_KEYS = {key: distribution for distribution, key in proven_module.STAGED_KEYS.items()}
#: How much of a commit a refusal shows.
_SHOWN_HEX = 12

#: Distribution-name prefixes of the two staged wheels. ``canopy-`` does not
#: match ``canopy_extras-…``: a wheel's name and version are joined by ``-``.
WHEEL_PREFIXES = ("canopy-", "pydmconverter-")
_WHEEL_SUFFIX = ".whl"

_HINT = (
    "build the payload in the canopy repo with scripts/build-payload.sh and name "
    f"its output directory in {PAYLOAD_ENV} (the deployment's .env), "
    f"or set {CHECK_ENV}=0 to skip staging"
)


class StageError(Exception):
    """The payload cannot be staged; the message is the one line the user sees."""


class StaleLibrary(StageError):
    """The screen library is not covered by its report; ``lines`` is what the user sees."""

    def __init__(self, lines: tuple[str, ...]):
        super().__init__("; ".join(lines))
        self.lines = lines


@dataclass(frozen=True)
class Payload:
    """A checked payload directory."""

    root: Path
    staged: bytes
    wheels: tuple[Path, ...]
    #: Whether the payload carries a staged spec (``spec/``).
    has_spec: bool = False


def wheel_prefix(name: str) -> str | None:
    """Return which staged wheel ``name`` is, or ``None`` when it is not one."""
    folded = name.casefold()
    if not folded.endswith(_WHEEL_SUFFIX):
        return None
    return next((prefix for prefix in WHEEL_PREFIXES if folded.startswith(prefix)), None)


def _setting(name: str, repo_root: Path) -> str:
    """Process environment first, then the repo's env chain; empty means unset."""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    return merge_chain(repo_root).get(name, "").strip()


def panel_off(repo_root: Path) -> bool:
    """Whether the build leaves the CANOPY panel out: no payload named, staging not skipped."""
    return _setting(CHECK_ENV, repo_root) != "0" and not _setting(PAYLOAD_ENV, repo_root)


#: The one line a build prints when :func:`panel_off` holds.
PANEL_OFF_NOTICE = (
    f"CANOPY panel off: {PAYLOAD_ENV} names no payload. Build one with the canopy repo's "
    f"scripts/build-payload.sh and name its directory in {PAYLOAD_ENV} (the deployment's .env)."
)


def _root_wheels(repo_root: Path) -> list[Path]:
    return sorted(p for p in repo_root.iterdir() if wheel_prefix(p.name) and not p.is_dir())


def read_payload(source: Path) -> Payload:
    """Check ``source`` is a complete payload and return what staging needs from it.

    :raises StageError: the directory is missing, has no readable
        ``STAGED.json``, was built from commits other than the proven ones, or
        does not hold exactly one wheel of each package.
    """
    if not source.is_dir():
        raise StageError(f"payload directory {source} does not exist; {_HINT}")
    staged_path = source / STAGED_FILENAME
    if not staged_path.is_file():
        raise StageError(
            f"{source} has no {STAGED_FILENAME}, so it is not a finished payload; {_HINT}"
        )
    staged = staged_path.read_bytes()
    try:
        parsed = json.loads(staged)
    except ValueError as exc:
        raise StageError(f"{staged_path} is not JSON ({exc}); {_HINT}") from exc
    if not isinstance(parsed, dict):
        raise StageError(f"{staged_path} is not a JSON object; {_HINT}")
    has_spec = _check_spec(source, staged_path, parsed)
    _check_proven(staged_path, parsed)

    wheels = []
    for prefix in WHEEL_PREFIXES:
        found = sorted(p for p in source.iterdir() if wheel_prefix(p.name) == prefix)
        if len(found) != 1:
            names = ", ".join(p.name for p in found) or "none"
            raise StageError(
                f"{source} must hold exactly one {prefix}*.whl (found: {names}); {_HINT}"
            )
        wheels.append(found[0])
    return Payload(root=source, staged=staged, wheels=tuple(wheels), has_spec=has_spec)


def _check_spec(source: Path, staged_path: Path, parsed: dict) -> bool:
    """Whether ``source`` carries a spec, refusing one ``STAGED.json`` disagrees with.

    A payload without the two keys predates the staged spec and must hold no
    ``spec/``; one with them names the spec it holds (both strings) or names
    none (both ``null``).
    """
    spec_dir = source / SPEC_DIRNAME
    present = spec_dir.is_dir()
    values = [parsed.get(key) for key in SPEC_KEYS]
    named = all(isinstance(value, str) and value for value in values)
    unnamed = all(value is None for value in values)
    if not (named or unnamed):
        raise StageError(
            f"{staged_path} must name the spec with both {' and '.join(SPEC_KEYS)} "
            f"or with neither; {_HINT}"
        )
    if named and not present:
        raise StageError(
            f"{staged_path} names spec {values[0]} but {spec_dir} is not a directory; {_HINT}"
        )
    if unnamed and present:
        raise StageError(f"{source} holds {SPEC_DIRNAME}/ but {staged_path} names no spec; {_HINT}")
    return present


def _check_proven(staged_path: Path, parsed: dict) -> None:
    """Refuse a payload whose ``STAGED.json`` names commits other than the proven ones.

    The record holds whole commits, so each is compared whole: a missing key, a
    value that is not a string and a short form are all another commit.
    """
    for key, distribution in COMMIT_KEYS.items():
        proven = PROVEN_COMMITS[distribution]
        held = parsed.get(key)
        if isinstance(held, str) and proven_module.is_pin(distribution, held):
            continue
        if isinstance(held, str) and held:
            found = f"holds {distribution} {held[:_SHOWN_HEX]}"
        else:
            found = f"names no {distribution} commit"
        raise StageError(
            f"payload {staged_path.parent} {found}; this OSPREY is proven against "
            f"{proven[:_SHOWN_HEX]}: build the payload at that commit"
        )


def is_current(repo_root: Path, payload: Payload) -> bool:
    """Whether the repo already holds exactly this payload.

    ``STAGED.json`` is the last thing a staging puts in place, so an equal one
    means the directory beside it is whole. The wheels live outside that
    directory and are checked by name and size.
    """
    staged_path = repo_root / PAYLOAD_DIRNAME / STAGED_FILENAME
    if not staged_path.is_file() or staged_path.read_bytes() != payload.staged:
        return False
    if payload.has_spec != (repo_root / PAYLOAD_DIRNAME / SPEC_DIRNAME).is_dir():
        return False
    present = {p.name: p.stat().st_size for p in _root_wheels(repo_root)}
    return present == {p.name: p.stat().st_size for p in payload.wheels}


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def stage(repo_root: Path, payload: Payload) -> None:
    """Replace whatever was staged in ``repo_root`` with ``payload``."""
    scratch = repo_root / SCRATCH_DIRNAME
    previous = repo_root / PREVIOUS_DIRNAME
    target = repo_root / PAYLOAD_DIRNAME
    _remove(scratch)
    _remove(previous)

    def _skip(directory: str, names: list[str]) -> list[str]:
        if Path(directory) != payload.root:
            return []
        return [name for name in names if wheel_prefix(name)]

    # Copy first: a payload that cannot be read fails before anything staged
    # earlier has been removed.
    shutil.copytree(payload.root, scratch, ignore=_skip)

    for stale in _root_wheels(repo_root):
        stale.unlink()
    for wheel in payload.wheels:
        atomic_replace(repo_root / wheel.name, wheel.read_bytes(), mode=WHEEL_MODE)

    # The directory goes last, and with it the STAGED.json that marks the
    # staging complete: an interrupted run is redone, not mistaken for current.
    if target.is_dir() and not target.is_symlink():
        os.rename(target, previous)
    else:
        _remove(target)
    os.rename(scratch, target)
    _remove(previous)


def check_library(repo_root: Path) -> None:
    """Refuse a screen library whose report no longer covers its files.

    :raises StaleLibrary: with every line of the refusal.
    """
    verdict = library_is_fresh(repo_root / LIBRARY_SUBPATH)
    if not verdict:
        raise StaleLibrary(verdict.lines)


def run(repo_root: Path) -> str:
    """Check the screen library, then stage the payload; return a one-line account.

    The library check comes first and no switch turns it off: a build that
    skips staging still bakes the library into the image.

    :raises StaleLibrary: the library's report does not cover its files.
    :raises StageError: the named payload is refused.
    """
    check_library(repo_root)
    if _setting(CHECK_ENV, repo_root) == "0":
        return f"skipped ({CHECK_ENV}=0)"

    named = _setting(PAYLOAD_ENV, repo_root)
    if not named:
        return f"nothing staged ({PAYLOAD_ENV} is not set)"
    source = (repo_root / Path(named).expanduser()).resolve()
    if source == (repo_root / PAYLOAD_DIRNAME).resolve():
        raise StageError(
            f"{PAYLOAD_ENV} names the staging directory itself ({source}); "
            "point it at the directory scripts/build-payload.sh wrote"
        )

    payload = read_payload(source)
    if is_current(repo_root, payload):
        return f"{PAYLOAD_DIRNAME}/ is already current with {source}"
    stage(repo_root, payload)
    return f"staged {source} into {PAYLOAD_DIRNAME}/ and {len(payload.wheels)} wheels"


def main(argv: list[str]) -> int:
    """Entry point: stage into the working directory; 0 on success, 1 on refusal."""
    if argv:
        _fail(f"takes no arguments (got {' '.join(argv)})")
        return 2
    try:
        message = run(Path.cwd())
    except StaleLibrary as exc:
        sys.stderr.write(
            f"canopy library: {LIBRARY_SUBPATH} does not match its write-binding report\n"
        )
        for line in exc.lines:
            sys.stderr.write(f"canopy library: {line}\n")
        return 1
    except StageError as exc:
        _fail(str(exc))
        return 1
    except OSError as exc:
        _fail(f"{type(exc).__name__}: {exc}")
        return 1
    print(f"canopy payload: {message}")
    return 0


def _fail(message: str) -> None:
    sys.stderr.write("canopy payload: " + " ".join(message.split()) + "\n")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
