"""SC7 acceptance: default-config-check for a scaffolded Control Assistant.

A freshly scaffolded Control Assistant project must:
  1. Start on the live stand-in (the preset declares
     ``virtual_accelerator.live_standin``, so a second copy of the soft IOC is
     deployed as the deployment's own third control target and is the baseline
     a session sits on — a machine that behaves like hardware and moves
     nothing). The virtual accelerator ships beside it, one ``osprey set``
     away.
  2. Engage the in-process simulator when the simulator's ``serving`` is
     switched to ``in_process`` (real ConnectorFactory resolution of the
     scaffolded control_system section, not just a string check) — the
     documented fallback for environments with no containers to depend on.
  3. Leave the epics block untouched by that switch — and untouched by the
     build in the first place.

Complements tests/templates/test_preset_va_block.py (which renders the raw
.j2 template in isolation) by exercising the real lifecycle end to end:
`osprey init` materializes the source zone, `osprey build` renders it, and
`osprey set config.control_system.connector.virtual_accelerator.serving=in_process`
performs the flip — which, because `osprey set`
writes the profile and never the render, means the flip is only visible in
`build/config.yml` after a second build. That sequencing IS the contract, so
the tests below run it rather than short-circuiting it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.init_cmd import init
from osprey.cli.set_cmd import set as set_cmd
from osprey.connectors.control_system.va_in_process_connector import VAInProcessConnector
from osprey.connectors.factory import (
    ConnectorFactory,
    isolated_connector_registries,
    register_builtin_connectors,
)
from tests.facility.served_tree import in_process_config

# The epics block a scaffolded Control Assistant renders: the app template's
# own values, verbatim, and the same block
# tests/templates/test_preset_va_block.py pins on the raw render as
# SHIPPED_EPICS_BLOCK. The stand-in has its own connector block —
# `control_system.connector.live_standin`, seven leaves the build derives from
# `virtual_accelerator.live_standin` — so `epics:` stays the machine the
# facility authors. `live` means that machine on a deployment running a
# stand-in exactly as on one that is not, which is why pointing this repo at a
# real facility is one edit here and nothing else. The gateways, the
# `probe_channel` and the operator acknowledgment are all commented out
# (facility-specific, nothing shipped set), so a scaffolded block carries the
# timeout and nothing else — a stock deployment's live target reads "not
# configured" until the go-live edit authors it.
SCAFFOLDED_EPICS_BLOCK = {"timeout_s": 5.0}


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _build(runner: CliRunner, repo: Path) -> None:
    """Render *repo*'s build/ zone, failing the test if it does not."""
    result = runner.invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])
    assert result.exit_code == 0, result.output


@pytest.fixture
def scaffolded_repo(runner: CliRunner, tmp_path: Path) -> Path:
    """A fresh Control Assistant deployment repo, built once."""
    repo = tmp_path / "smoke"
    result = runner.invoke(init, [str(repo), "--preset", "control-assistant", "--no-git"])
    assert result.exit_code == 0, result.output
    _build(runner, repo)
    assert (repo / "build" / "config.yml").exists()
    return repo


#: The documented flip to the simulator in process.
SERVE_IN_PROCESS = "config.control_system.connector.virtual_accelerator.serving=in_process"


def _serve_in_process(runner: CliRunner, repo: Path) -> None:
    """The documented flip: edit the profile, then re-render it.

    Both halves, deliberately. `osprey set` writes profile.yml and leaves the
    render alone, so a test that stopped after the set would assert against the
    build the flip has not reached yet.
    """
    result = runner.invoke(set_cmd, ["--repo", str(repo), SERVE_IN_PROCESS])
    assert result.exit_code == 0, result.output
    _build(runner, repo)


def _load_config(repo: Path) -> dict:
    return yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def clean_connector_factory():
    """Isolate ConnectorFactory global state across tests.

    The registries start empty so ``register_builtin_connectors()`` is
    observed doing the registration; snapshot/restore brackets the clear so
    registrations made elsewhere in the process survive teardown.
    """
    with isolated_connector_registries(clear=True):
        yield


class TestFreshProjectDefaultsToTheSimulator:
    """State 1: a freshly scaffolded project starts on the sandbox simulator."""

    def test_default_control_system_type_is_the_simulator(self, scaffolded_repo: Path):
        config = _load_config(scaffolded_repo)
        assert config["control_system"]["type"] == "virtual_accelerator"

    def test_virtual_accelerator_and_epics_blocks_present(self, scaffolded_repo: Path):
        """The authored connector blocks are fully materialized even though
        neither is the active type — each is ready to flip to. The simulator
        is served from its container until the profile says otherwise."""
        connector = _load_config(scaffolded_repo)["control_system"]["connector"]
        assert "serving" not in connector["virtual_accelerator"]
        assert "virtual_accelerator" in connector
        assert "epics" in connector


class TestServingInProcessEngagesTheConnector:
    """State 2: serving the simulator in process engages VAInProcessConnector —
    the documented fallback flip for environments with no containers to depend
    on."""

    def test_cli_switch_updates_config_serving(self, runner: CliRunner, scaffolded_repo: Path):
        _serve_in_process(runner, scaffolded_repo)

        config = _load_config(scaffolded_repo)
        assert config["control_system"]["type"] == "virtual_accelerator"
        assert config["control_system"]["connector"]["virtual_accelerator"]["serving"] == (
            "in_process"
        )

    def test_the_set_alone_does_not_move_the_render(self, runner: CliRunner, scaffolded_repo: Path):
        """`osprey set` edits the source and nothing else: until a build runs,
        the deployment still answers as the simulator it was scaffolded on.
        This is the property the flip test above depends on, so it is asserted
        rather than assumed."""
        result = runner.invoke(set_cmd, ["--repo", str(scaffolded_repo), SERVE_IN_PROCESS])
        assert result.exit_code == 0, result.output

        assert "control_system.connector.virtual_accelerator.serving: in_process" in (
            scaffolded_repo / "profile.yml"
        ).read_text(encoding="utf-8")
        rendered = _load_config(scaffolded_repo)["control_system"]["connector"]
        assert "serving" not in rendered["virtual_accelerator"]

    @pytest.mark.asyncio
    async def test_scaffolded_in_process_section_resolves_to_the_in_process_connector(
        self, runner: CliRunner, scaffolded_repo: Path, monkeypatch
    ):
        """The scaffolded project's control_system section, fed through the
        real ConnectorFactory, produces a VAInProcessConnector instance — not just a
        config string. Unlike the VA/epics connectors, VAInProcessConnector has no
        real network I/O in connect(), so no stubbing is needed.

        The in-process simulator serves the render's own simulator view, which
        the app's config loader would find beside build/config.yml. This test
        reads build/config.yml directly, so it hands the connector that view
        itself, from the repo root.
        """
        _serve_in_process(runner, scaffolded_repo)

        register_builtin_connectors()
        cs_config = _load_config(scaffolded_repo)["control_system"]
        assert cs_config["type"] == "virtual_accelerator"

        monkeypatch.chdir(scaffolded_repo)
        render_view = scaffolded_repo / "build" / "data" / "simulator"
        cs_config["connector"]["virtual_accelerator"].update(in_process_config(render_view))
        connector = await ConnectorFactory.create_control_system_connector(cs_config)
        try:
            assert isinstance(connector, VAInProcessConnector)
            assert connector._connected is True
        finally:
            await connector.disconnect()


class TestEpicsBlockRemainsUntouched:
    """State 3: the epics block still reads exactly as it was authored.

    The class finally tests what its name says. A deployment that stands up a
    stand-in derives seven leaves under
    ``control_system.connector.live_standin`` and writes nothing at all under
    ``epics:`` — so the gateways a fresh render carries are the app template's
    own, and the one edit that points this repo at a real facility is the one
    the template invites."""

    def test_epics_block_unchanged_before_switch(self, scaffolded_repo: Path):
        """A build of the shipped preset leaves the authored gateways alone."""
        epics = _load_config(scaffolded_repo)["control_system"]["connector"]["epics"]
        assert epics == SCAFFOLDED_EPICS_BLOCK

    def test_epics_block_unchanged_after_serving_in_process(
        self, runner: CliRunner, scaffolded_repo: Path
    ):
        """Switching the connector must not perturb the epics block — the write
        reaches exactly one key, and the re-render carries the rest through
        unchanged."""
        _serve_in_process(runner, scaffolded_repo)

        epics = _load_config(scaffolded_repo)["control_system"]["connector"]["epics"]
        assert epics == SCAFFOLDED_EPICS_BLOCK
