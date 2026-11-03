#!/usr/bin/env python3
"""Import every module of an installed osprey-connectors and refuse the framework.

Run with the interpreter of a venv that holds the connectors wheel and nothing
of the framework, in isolated mode so neither the checkout nor ``PYTHONPATH``
can lend it a module the wheel does not ship::

    /tmp/lean-check/bin/python -I scripts/ci/connectors_lean_import_walk.py

Two callers, one check: ci.yml's ``package`` job walks the wheel it built with
its dependencies resolved fresh, and release-connectors.yml walks the wheel a
connectors tag is about to publish, installed the same way from PyPI. A missed
``osprey.*`` rewrite in a leaf nothing imports transitively would otherwise
ship, and so would a module that pulls in the agent stack.

Exit 0 and ``LEAN WHEEL OK (<n> modules)`` on success; an exception otherwise.
"""

from __future__ import annotations

import importlib
import importlib.resources
import pkgutil
import sys

#: Top-level packages the lean wheel must never load: the framework itself and
#: the agent/web stack it brings.
FORBIDDEN = frozenset({"litellm", "openai", "anthropic", "fastapi", "playwright", "osprey"})


def _raise(name: str) -> None:
    # walk_packages swallows a package's import error unless onerror raises.
    raise ImportError(f"failed to walk {name}")


def main() -> int:
    # The entry points first, so a failure names the module a user reaches.
    import osprey_connectors
    import osprey_connectors.config
    import osprey_connectors.control_system
    import osprey_connectors.control_system._epics_connector
    import osprey_connectors.errors
    import osprey_connectors.logger
    import osprey_connectors.simulation

    bad = sorted({name.split(".")[0] for name in sys.modules} & FORBIDDEN)
    assert not bad, f"lean wheel loaded: {bad}"

    modules = [
        module.name
        for module in pkgutil.walk_packages(
            osprey_connectors.__path__, "osprey_connectors.", onerror=_raise
        )
    ]
    for name in modules:
        importlib.import_module(name)
    assert "osprey" not in sys.modules, "a lean-wheel module imported the osprey framework"
    # The PEP 561 marker: without it a type checker treats the installed
    # package as untyped and every name imported from it as Any.
    marker = importlib.resources.files("osprey_connectors") / "py.typed"
    assert marker.is_file(), "py.typed missing from the wheel"
    print(f"LEAN WHEEL OK ({len(modules)} modules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
