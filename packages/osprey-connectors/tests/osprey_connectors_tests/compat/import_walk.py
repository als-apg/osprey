"""Every module of an installed framework imports beside these connectors.

The import scanner reads the sites a framework names; this runs them. Each
module ``pkgutil.walk_packages`` finds under the package is imported in turn,
so an import-time failure that no site spells -- a connectors module that
loads but raises in the framework's hands, a re-export that moved -- is
reported with the module that hit it.

A module is skipped only when it is listed in :data:`SKIPPED` with the reason.
Those are modules that cannot import in a plain ``pip install
osprey-framework`` by design, whatever connectors sit beside them: they need a
runtime that only a rendered deployment provides. A prefix skips the
subpackage under it.

Usage::

    python -I import_walk.py [--package osprey]

Prints each failure as ``<module>: <error>`` and a summary on stderr. Exits 1
on any failure. Standard library only.
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
import traceback
from collections.abc import Iterable

#: Module (or subpackage) -> why it is not imported here.
SKIPPED: dict[str, str] = {
    # Jinja-rendered and copied into a deployment; the hooks import their
    # siblings (``osprey_hook_log``) as top-level modules, which exist only
    # in the rendered ``.claude/hooks/`` directory.
    "osprey.templates": "deployment templates, imported only once rendered",
    # The soft-IOC server process. Its server library (pcaspy, later lume-pva)
    # comes with the `va` extra, which a plain install does not carry.
    "osprey.services.virtual_accelerator.serving.runner": "needs the `va` extra",
}


def _skipped(name: str) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in SKIPPED)


def walk(package: str) -> tuple[list[str], list[tuple[str, str]]]:
    """Import every module of ``package`` that :data:`SKIPPED` does not name.

    Returns:
        The modules imported, and ``(module, error)`` for each that failed.
    """
    failures: list[tuple[str, str]] = []
    root = importlib.import_module(package)

    def _on_error(name: str) -> None:
        # The walk retries a package the loop below already failed to import.
        if not _skipped(name) and all(failed != name for failed, _ in failures):
            failures.append((name, _last_error()))

    imported = [package]
    for info in pkgutil.walk_packages(root.__path__, f"{package}.", onerror=_on_error):
        if _skipped(info.name):
            continue
        if info.name in sys.modules:
            imported.append(info.name)
            continue
        try:
            importlib.import_module(info.name)
        except (Exception, SystemExit) as error:  # SystemExit at import is a failure too
            failures.append((info.name, f"{error.__class__.__name__}: {error}"))
        else:
            imported.append(info.name)
    return imported, failures


def _last_error() -> str:
    kind, value, _ = sys.exc_info()
    if kind is None:
        return "failed to walk"
    return "".join(traceback.format_exception_only(kind, value)).strip()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--package", default="osprey")
    args = parser.parse_args(list(argv) if argv is not None else None)
    imported, failures = walk(args.package)
    for name, error in failures:
        print(f"{name}: {error}")
    print(
        f"imported {len(imported)} modules of {args.package}, {len(failures)} failed, "
        f"{len(SKIPPED)} subtree(s) skipped: {', '.join(sorted(SKIPPED))}",
        file=sys.stderr,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
