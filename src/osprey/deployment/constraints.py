"""Print a frozen constraints file minus the lines naming the given wheels.

The project image freezes what the framework resolved to
``/opt/osprey/constraints.txt``. Its wheel layer installs locally-built wheels
under that file, and the file pins each of those distributions at the version
being replaced, so the layer first drops the lines naming them::

    python -m osprey.deployment.constraints /opt/osprey/constraints.txt /tmp/ctx/*.whl

The filtered file goes to stdout. A wheel may then change its own version and
nothing else: any other pin it would move still fails the install.

A wheel's distribution is the part of its file name before the first ``-``; a
constraint line's is its leading requirement name. Both are compared in the
normalized form (runs of ``-``, ``_`` and ``.`` read as one ``-``, lower case),
so ``osprey_framework-1.1-py3-none-any.whl`` drops ``osprey-framework==1.0``.
Every other line, comments and blank lines included, is kept as written.

Standard library only, so it runs in any image whatever else is installed.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import PurePosixPath

#: Characters that end the requirement name at the start of a constraint line.
_NAME_END = re.compile(r"[^A-Za-z0-9._-]")


def normalize(name: str) -> str:
    """``name`` in its normalized distribution form (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def wheel_distribution(path: str) -> str:
    """The normalized distribution name of the wheel file at ``path``."""
    return normalize(PurePosixPath(path).name.split("-", 1)[0])


def line_distribution(line: str) -> str:
    """The normalized requirement name a constraint line starts with, or ``""``."""
    return normalize(_NAME_END.split(line.strip(), maxsplit=1)[0])


def minus_wheels(lines: Iterable[str], wheels: Iterable[str]) -> list[str]:
    """The constraint ``lines`` that name none of the distributions of ``wheels``."""
    dropped = {wheel_distribution(path) for path in wheels}
    return [line for line in lines if line_distribution(line) not in dropped]


def main(argv: Sequence[str] | None = None) -> int:
    """``<constraints> <wheel>...``: write the filtered constraints to stdout."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        sys.stderr.write(
            "usage: python -m osprey.deployment.constraints <constraints> <wheel>...\n"
        )
        return 2
    constraints, wheels = args[0], args[1:]
    with open(constraints, encoding="utf-8") as handle:
        sys.stdout.writelines(minus_wheels(handle, wheels))
    return 0


if __name__ == "__main__":
    sys.exit(main())
