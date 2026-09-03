"""The build advisory for PVA channels that name no address source.

The connector-host child scrubs every inherited ``EPICS_PVA_*`` variable
before the connector runs, so a ``pva_gateway`` block is the only route a PVA
address list has into p4p. A deployment that lists ``pva_channels`` and sets
``EPICS_PVA_ADDR_LIST`` in its environment instead gets subnet auto-discovery,
which from an off-subnet host is a bare timeout on every read. The build is
where that is worth saying.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.build_profile_reach import pva_address_source_advisories


def _config(**epics: Any) -> dict[str, Any]:
    """A rendered config whose ``epics`` connector block carries *epics*."""
    return {
        "control_system": {
            "type": "epics",
            "connector": {"mock": {"response_delay_ms": 0}, "epics": {"timeout": 5.0, **epics}},
        }
    }


def test_pva_channels_without_a_gateway_are_advised() -> None:
    advisories = pva_address_source_advisories(_config(pva_channels=["*:image"]))

    assert len(advisories) == 1
    assert "control_system.connector.epics.pva_gateway" in advisories[0]
    assert "EPICS_PVA_" in advisories[0]


def test_a_string_glob_counts_as_configured() -> None:
    """``connect()`` accepts a single glob as a string, so the advisory must too."""
    assert len(pva_address_source_advisories(_config(pva_channels="*:image"))) == 1


def test_an_empty_gateway_block_is_no_address_source() -> None:
    """``connect()`` treats an empty ``pva_gateway`` exactly like an absent one."""
    advisories = pva_address_source_advisories(_config(pva_channels=["*:image"], pva_gateway={}))

    assert len(advisories) == 1


def test_a_gateway_block_is_silent() -> None:
    config = _config(pva_channels=["*:image"], pva_gateway={"address": "10.0.0.1 10.0.0.2"})

    assert pva_address_source_advisories(config) == []


def test_no_pva_channels_is_silent() -> None:
    assert pva_address_source_advisories(_config()) == []
    assert pva_address_source_advisories(_config(pva_channels=[])) == []
    assert pva_address_source_advisories(_config(pva_channels=["  "])) == []


def test_a_config_without_connector_blocks_is_silent() -> None:
    assert pva_address_source_advisories({}) == []
    assert pva_address_source_advisories({"control_system": {"connector": None}}) == []


def test_every_connector_block_is_checked() -> None:
    """The advisory names the block it found, whichever connector type owns it."""
    config = {
        "control_system": {
            "connector": {
                "epics": {"pva_channels": ["*:image"], "pva_gateway": {"address": "gw"}},
                "virtual_accelerator": {"pva_channels": ["*:image"]},
            }
        }
    }

    advisories = pva_address_source_advisories(config)

    assert len(advisories) == 1
    assert "control_system.connector.virtual_accelerator.pva_gateway" in advisories[0]


# ---------------------------------------------------------------------------
# Build wiring — a real `osprey build` names the gap exactly once
# ---------------------------------------------------------------------------

_PROFILE = """\
extends: hello-world
name: PVA Advisory Fixture
config:
  control_system.connector.epics.pva_channels: ["*:image"]
{gateway}"""

_GATEWAY = """\
  control_system.connector.epics.pva_gateway:
    address: cam1.example.org cam2.example.org
"""


def _build(tmp_path: Path, profile: str) -> str:
    """Run a real ``osprey build`` on *profile*; return its captured output.

    ``--skip-deps --skip-lifecycle``: the virtualenv install and the profile's
    shell phases render nothing this module reads and cost minutes.
    """
    repo = tmp_path / "pva-advisory-fixture"
    repo.mkdir()
    (repo / "profile.yml").write_text(profile, encoding="utf-8")

    result = CliRunner().invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])

    assert result.exit_code == 0, result.output
    # Rich wraps to the console width; single-space the output before matching.
    return re.sub(r"\s+", " ", result.output)


def test_build_names_the_missing_gateway_once(tmp_path: Path) -> None:
    """One line, not one per render pass. Advisory: the build still succeeds."""
    output = _build(tmp_path, _PROFILE.format(gateway=""))

    assert output.count("control_system.connector.epics.pva_gateway") == 1
    assert "EPICS_PVA_" in output


def test_build_with_a_gateway_is_silent(tmp_path: Path) -> None:
    output = _build(tmp_path, _PROFILE.format(gateway=_GATEWAY))

    assert "pva_gateway" not in output
