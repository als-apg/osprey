"""The engine suite needs the optimizer's ``service`` extra (xopt, torch, polars).

The main unit lanes install only the base package (through ``dev``); there the
engine tests are left out of collection, and the model and environment tests
beside them (pydantic and YAML only) still run. The ``optimizer`` CI job
installs the extra and runs everything.
"""

import importlib.util
from collections.abc import Iterator

import pytest

# The directory itself, not a glob under it: its conftest imports xopt too. The
# owner-verdict rows import ``osprey_optimizer.api``, whose package init needs the
# extra as well.
collect_ignore = (
    []
    if all(importlib.util.find_spec(m) for m in ("xopt", "torch", "polars"))
    else ["engine", "test_owner_write_verdict.py"]
)


@pytest.fixture(autouse=True)
def _no_write_authority_left_installed() -> Iterator[None]:
    """A test that runs a worker in process leaves its process authority behind; drop it."""
    yield
    from osprey_optimizer.environments.write_authority import install_process_authority

    install_process_authority(None)
