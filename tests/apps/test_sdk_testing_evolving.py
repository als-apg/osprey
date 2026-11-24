"""An application's tests ask for a simulator whose time-driven models run.

A synthetic deployment states ``control_system.connector.virtual_accelerator.evolve``
in its profile's ``config:`` like any deployment would. The connector the
factory builds inside the served environment then evolves: the injection
fixture's cycle signal, which only time moves, is read both low and high.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from click.testing import CliRunner

from osprey.apps.sdk.testing import synthetic_deployment
from tests.facility._injection_fixture import BUSY, COUNTERS, FIXTURE

APP = "probe-app"

#: The longest the test watches the cycle signal, several of its 1 s periods.
WATCH_S = 20.0


def _new_app(parent: Path, name: str) -> Path:
    """A fresh application, as ``osprey app new`` writes it into a repository of its own."""
    from osprey.cli.app_cmd import app

    source = parent / "source"
    source.mkdir()
    (source / "profile.yml").write_text("name: source\n", encoding="utf-8")
    made = CliRunner().invoke(app, ["new", name, "--repo", str(source)])
    assert made.exit_code == 0, made.output
    return source / "apps" / name


def _fixture_tree() -> dict[str, str]:
    """The injection fixture as the tree a synthetic deployment takes."""
    return {
        str(path.relative_to(FIXTURE)): path.read_text(encoding="utf-8")
        for path in sorted(FIXTURE.rglob("*"))
        if path.is_file()
    }


async def _cycle_levels() -> set[float]:
    """The levels the cycle signal reads on the configured connector, until it has shown both."""
    from osprey_connectors.factory import ConnectorFactory

    connector = await ConnectorFactory.create_control_system_connector()
    seen: set[float] = set()
    try:
        deadline = time.monotonic() + WATCH_S
        while len(seen) < 2 and time.monotonic() < deadline:
            seen.add(float((await connector.read_channel(BUSY)).value))
            await asyncio.sleep(0.05)
    finally:
        await connector.disconnect()
    return seen


def test_a_deployment_that_states_evolve_runs_its_time_driven_models(tmp_path: Path) -> None:
    deployment = synthetic_deployment(
        tmp_path / "site",
        apps={APP: _new_app(tmp_path, APP)},
        facility=_fixture_tree(),
        profile={
            "apps": {APP: {"config": {"channel": COUNTERS[0]}}},
            "config": {"control_system.connector.virtual_accelerator.evolve": True},
        },
    )
    served = deployment.serve("operator")

    with served.environment():
        levels = asyncio.run(_cycle_levels())

    assert levels == {0.0, 1.0}
