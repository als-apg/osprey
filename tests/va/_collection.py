"""The vacuous-green guard a Virtual Accelerator test module ends with.

A module that imports cleanly but collects nothing (or half of what it should)
passes silently; each module states a floor for its own count and checks it
with :func:`assert_module_collected`.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def assert_module_collected(request: pytest.FixtureRequest, module: str, minimum: int) -> None:
    """Fail unless the session collected at least *minimum* tests from *module*.

    Args:
        request: The calling test's request.
        module: The test module's ``__file__``.
        minimum: The module's floor for its own test count.
    """
    name = Path(module).name
    collected = [
        item for item in request.session.items if item.nodeid.split("::")[0].endswith(name)
    ]

    assert len(collected) >= minimum, f"{name} collected {len(collected)} tests, floor is {minimum}"
