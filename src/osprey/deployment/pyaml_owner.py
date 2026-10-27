"""Assert that the ``pyaml`` import package has exactly one owning distribution.

pyAML ships under more than one distribution name — ``accelerator-middle-layer``
upstream and ``accelerator-middle-layer-apg`` as a staging release — and every
one of them writes the same ``pyaml/`` package directory. Two of them in
one environment do not fail to install: the second silently overwrites files of
the first, and uninstalling either later deletes files the other still claims.
Neither resolver nor ``pip check`` sees it, because the two names do not
conflict as requirements.

The image recipes that overlay locally-built wheels on a primed framework run
this module after the overlay::

    python -m osprey.deployment.pyaml_owner

It exits 0 when exactly one installed distribution owns ``pyaml`` and 1
otherwise, naming the owners it found. It only reads installed metadata: it
never uninstalls or reinstalls anything, so a failing build is fixed where the
second owner came from, not papered over here.

Standard library only, so it runs in any image whatever else is installed.
It cannot reach the CLI's renderer for the same reason, so it writes its one
verdict line straight to the stream.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable
from importlib import metadata

IMPORT_PACKAGE = "pyaml"


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _owns(dist: metadata.Distribution, import_package: str) -> bool:
    """True when ``dist`` installs the top-level ``import_package``.

    The install record (``RECORD``) is authoritative: a distribution owns the
    package when it wrote a file under ``<import_package>/`` or the module
    ``<import_package>.py``. ``top_level.txt`` stands in only for a distribution
    that ships no record.
    """
    files = dist.files
    if files is not None:
        for entry in files:
            parts = entry.parts
            if not parts:
                continue
            if parts[0] == import_package and len(parts) > 1:
                return True
            if len(parts) == 1 and parts[0] == f"{import_package}.py":
                return True
        return False
    top_level = dist.read_text("top_level.txt") or ""
    return import_package in (line.strip() for line in top_level.splitlines())


def owners(import_package: str = IMPORT_PACKAGE, path: Iterable[str] | None = None) -> list[str]:
    """Return the sorted, normalized names of the distributions owning ``import_package``.

    Args:
        import_package: The top-level import name to look for.
        path: Directories to search for installed distributions; ``None``
            searches ``sys.path``, the environment the interpreter imports from.
    """
    dists = (
        metadata.distributions(path=list(path)) if path is not None else metadata.distributions()
    )
    found: set[str] = set()
    for dist in dists:
        name = dist.metadata["Name"]
        if name and _owns(dist, import_package):
            found.add(_normalize(name))
    return sorted(found)


def main(path: Iterable[str] | None = None) -> int:
    """Write the verdict and return the process exit status (0 = one owner)."""
    found = owners(IMPORT_PACKAGE, path)
    if len(found) == 1:
        sys.stdout.write(f"{IMPORT_PACKAGE}: one owning distribution ({found[0]})\n")
        return 0
    if not found:
        sys.stderr.write(
            f"ERROR: no installed distribution owns the `{IMPORT_PACKAGE}` package; "
            "the framework requires exactly one\n"
        )
    else:
        sys.stderr.write(
            f"ERROR: the `{IMPORT_PACKAGE}` package has {len(found)} owning distributions "
            f"({', '.join(found)}); they write the same files, so exactly one may be "
            "installed. Remove the requirement that brought in the extra one.\n"
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
