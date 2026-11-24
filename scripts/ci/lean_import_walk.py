#!/usr/bin/env python3
"""Import every module of an installed OSPREY component wheel and refuse the rest.

Run with the interpreter of a venv that holds the stream's wheel (and its lean
extras) and nothing else of OSPREY, in isolated mode so neither the checkout
nor ``PYTHONPATH`` can lend it a module the wheel does not ship::

    /tmp/lean-check/bin/python -I scripts/ci/lean_import_walk.py connectors

The stream key names a row of ``release_streams.py``; the walk reads its
``import_name``, ``lean_forbidden``, ``lean_entry_modules`` and ``typed``.
Callers: the ``package`` check (connectors and facility), ci.yml's
``osprey-panels-standalone`` job (panels) and release-package.yml's
verify-install job (connectors, installed from PyPI). A missed ``osprey.*``
rewrite in a leaf nothing imports transitively would otherwise ship, and so
would a module that pulls in the agent stack.

Exit 0 and ``LEAN WHEEL OK (<n> modules)`` on success; an exception otherwise.
"""

from __future__ import annotations

import importlib
import importlib.resources
import importlib.util
import pkgutil
import sys
from pathlib import Path

#: The stream table beside this script. ``python -I`` drops nothing we need
#: here, but it is loaded by path so the walk never depends on ``sys.path``.
TABLE = Path(__file__).resolve().with_name("release_streams.py")
#: The ``sys.modules`` name the table is registered under; ``@dataclass``
#: resolves the frozen records' annotations through it.
TABLE_MODULE = "_osprey_release_streams"


def _load_table():
    spec = importlib.util.spec_from_file_location(TABLE_MODULE, TABLE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the stream table at {TABLE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[TABLE_MODULE] = module
    spec.loader.exec_module(module)
    return module


def _raise(name: str) -> None:
    # walk_packages swallows a package's import error unless onerror raises.
    raise ImportError(f"failed to walk {name}")


def _loaded(forbidden: frozenset[str]) -> list[str]:
    return sorted({name.split(".")[0] for name in sys.modules} & forbidden)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    table = _load_table()
    keys = [stream.key for stream in table.STREAMS]
    if len(args) != 1 or args[0] not in keys:
        print(f"usage: lean_import_walk.py <{'|'.join(keys)}>", file=sys.stderr)
        return 2
    stream = next(s for s in table.STREAMS if s.key == args[0])
    forbidden = stream.lean_forbidden

    # The entry points first, so a failure names the module a user reaches.
    for name in stream.lean_entry_modules:
        importlib.import_module(name)
    bad = _loaded(forbidden)
    assert not bad, f"lean wheel loaded: {bad}"

    package = importlib.import_module(stream.import_name)
    modules = [
        module.name
        for module in pkgutil.walk_packages(
            package.__path__, f"{stream.import_name}.", onerror=_raise
        )
    ]
    for name in modules:
        importlib.import_module(name)
    bad = _loaded(forbidden)
    assert not bad, f"a lean-wheel module imported {bad}"
    if stream.typed:
        # The PEP 561 marker: without it a type checker treats the installed
        # package as untyped and every name imported from it as Any.
        marker = importlib.resources.files(stream.import_name) / "py.typed"
        assert marker.is_file(), "py.typed missing from the wheel"
    print(f"LEAN WHEEL OK ({len(modules)} modules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
