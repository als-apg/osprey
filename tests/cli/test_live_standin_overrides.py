"""``virtual_accelerator.live_standin:`` and the rendered config it derives.

The stand-in is a control target of its own — ``standin``, served by the EPICS
connector out of ``control_system.connector.live_standin`` — so what these tests
pin is not that some keys were written, it is that the build says exactly two
things and no more:

* the stand-in's own block dials it, on loopback, at the port the profile named,
  with the port written out (``EPICSConnector`` has no ``fill_gateway_ports``,
  so an omitted port is the EPICS default rather than the stand-in's) — and the
  *sandbox* VA's gateway rows still carry no port, because those really are
  default-filled and a written one would state the same fact twice;
* that block carries the same probe channel the VA block proves, since a target
  without one is never switched to and a rehearsal you cannot switch into
  rehearses nothing.

Everything else about the deployment is the profile's. The facility's authored
``epics`` block is ``live`` and the build never writes a key there; limits
checking, write posture and the operator acknowledgment describe how the
deployment is *run* rather than where one of its targets lives, so they are not
derived and not refused. A profile that spells one of the stand-in's own leaves
is refused, because that block is the one thing the build does own.

The off-state matters as much: with no ``live_standin`` key the same build must
render no stand-in block at all.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner
from ruamel.yaml import YAML

from osprey.cli.build_cmd import build
from osprey.cli.build_profile_schema import VAConfig
from osprey.cli.build_profile_standin import (
    LIVE_STANDIN_DERIVED_KEYS,
    PROBE_CHANNEL_KEY,
    VA_PROBE_CHANNEL_KEY,
    live_standin_config_overrides,
    live_standin_duplicate_key_errors,
)

#: The port the stand-in is reserved on throughout the codebase — both config
#: templates' gateway examples read 5094 rather than this, on purpose, so an
#: override written there cannot collide with a stand-in.
STANDIN_PORT = 5074

#: What the Control Assistant template's VA block proves, and therefore what
#: the stand-in must prove too.
VA_PROBE_CHANNEL = "SR:VAC:GAUGE:SR01:PRESSURE:RB"

#: The rendered subtree the ``standin`` target is configured from.
STANDIN_PREFIX = "control_system.connector.live_standin"

#: The ``epics`` block the template ships. This is the ``live`` target's, and
#: the whole point of the third target is that it reads the same whether or
#: not a stand-in was asked for — pinned the same way
#: tests/cli/test_rendered_va_block.py pins it. The gateways ship commented
#: out (authoring them is the go-live edit), so the shipped block is the
#: timeout and nothing else.
SHIPPED_EPICS_BLOCK = {"timeout": 5.0}


def _va(live_standin: int | None) -> VAConfig:
    return VAConfig(port=5064, live_standin=live_standin)


def _rendered_with_probe(channel: str | None) -> dict[str, Any]:
    """A rendered config shaped like the template's, VA probe channel set or not."""
    va_block: dict[str, Any] = {"timeout": 5.0}
    if channel is not None:
        va_block["probe_channel"] = channel
    return {"control_system": {"connector": {"virtual_accelerator": va_block}}}


# ── The overrides the block derives ──────────────────────────────────────────


def test_live_standin_overrides_are_empty_without_the_key() -> None:
    """No stand-in asked for, nothing derived — there is no third target."""
    assert (
        live_standin_config_overrides(_va(None), {}, _rendered_with_probe(VA_PROBE_CHANNEL)) == {}
    )
    assert live_standin_config_overrides(None, {}, _rendered_with_probe(VA_PROBE_CHANNEL)) == {}


def test_live_standin_overrides_point_the_standin_target_at_the_stand_in() -> None:
    """Both gateway lanes on loopback at the named port, probe carried over."""
    overrides = live_standin_config_overrides(
        _va(STANDIN_PORT), {}, _rendered_with_probe(VA_PROBE_CHANNEL)
    )
    assert overrides == {
        "control_system.connector.live_standin.gateways.read_only.address": "localhost",
        "control_system.connector.live_standin.gateways.read_only.port": STANDIN_PORT,
        "control_system.connector.live_standin.gateways.read_only.use_name_server": True,
        "control_system.connector.live_standin.gateways.write_access.address": "localhost",
        "control_system.connector.live_standin.gateways.write_access.port": STANDIN_PORT,
        "control_system.connector.live_standin.gateways.write_access.use_name_server": True,
        "control_system.connector.live_standin.probe_channel": VA_PROBE_CHANNEL,
    }
    assert set(overrides) == set(LIVE_STANDIN_DERIVED_KEYS)


def test_live_standin_overrides_write_nothing_outside_the_stand_ins_own_block() -> None:
    """Seven leaves under one prefix — no ``epics``, no limits, no acknowledgment.

    Stated as a claim about the whole returned mapping rather than as three
    absences, so a key derived for some future reason has to be added here on
    purpose. The stand-in is a target, and a target's config is where it is and
    what proves it reachable; how the deployment is run is the profile's.
    """
    overrides = live_standin_config_overrides(
        _va(STANDIN_PORT), {}, _rendered_with_probe(VA_PROBE_CHANNEL)
    )

    assert len(LIVE_STANDIN_DERIVED_KEYS) == 7
    assert all(key.startswith(f"{STANDIN_PREFIX}.") for key in overrides), overrides
    assert not [key for key in overrides if "connector.epics" in key]
    assert not [key for key in overrides if "limits_checking" in key]
    assert not [key for key in overrides if "target_switch" in key]


def test_live_standin_overrides_take_the_probe_channel_the_profile_spells() -> None:
    """A profile that names its own VA probe channel is the one the render will show."""
    for spelling in (
        {VA_PROBE_CHANNEL_KEY: "MY:OWN:CHANNEL"},
        {"control_system.connector.virtual_accelerator": {"probe_channel": "MY:OWN:CHANNEL"}},
        {
            "control_system": {
                "connector": {"virtual_accelerator": {"probe_channel": "MY:OWN:CHANNEL"}}
            }
        },
    ):
        overrides = live_standin_config_overrides(
            _va(STANDIN_PORT), spelling, _rendered_with_probe(VA_PROBE_CHANNEL)
        )
        assert overrides[PROBE_CHANNEL_KEY] == "MY:OWN:CHANNEL", spelling


def test_live_standin_overrides_fall_back_to_the_rendered_probe_channel() -> None:
    """Profile silent: what the template rendered is what the VA block will say."""
    overrides = live_standin_config_overrides(
        _va(STANDIN_PORT), {}, _rendered_with_probe(VA_PROBE_CHANNEL)
    )
    assert overrides[PROBE_CHANNEL_KEY] == VA_PROBE_CHANNEL


def test_live_standin_overrides_omit_the_probe_channel_when_there_is_none() -> None:
    """No channel proves the VA, so none proves the stand-in — and that is honest.

    A target with no probe channel is never switched to, which is the correct
    state for a deployment that has not named one: better an unswitchable
    rehearsal than a switch proved by an invented channel.
    """
    overrides = live_standin_config_overrides(_va(STANDIN_PORT), {}, _rendered_with_probe(None))
    assert PROBE_CHANNEL_KEY not in overrides
    assert len(overrides) == len(LIVE_STANDIN_DERIVED_KEYS) - 1


# ── One fact, two homes ──────────────────────────────────────────────────────


def test_live_standin_overrides_refuse_a_dotted_duplicate() -> None:
    """A dotted stand-in gateway key beside the stand-in is refused."""
    errors = live_standin_duplicate_key_errors(
        _va(STANDIN_PORT),
        {f"{STANDIN_PREFIX}.gateways.read_only.port": 5064},
    )
    assert len(errors) == 1
    assert f"{STANDIN_PREFIX}.gateways.read_only.port" in errors[0]
    assert "The stand-in owns that key" in errors[0]


def test_live_standin_overrides_refuse_a_nested_duplicate() -> None:
    """Spelling-independent: a nested subtree reaches the same leaf and is refused too."""
    errors = live_standin_duplicate_key_errors(
        _va(STANDIN_PORT),
        {
            "control_system": {
                "connector": {
                    "live_standin": {"gateways": {"read_only": {"address": "gw.example.org"}}}
                }
            }
        },
    )
    assert len(errors) == 1
    assert f"{STANDIN_PREFIX}.gateways.read_only.address" in errors[0]


def test_live_standin_overrides_accumulate_every_duplicate_at_once() -> None:
    """Every offending key in one report, so a profile is fixed in one pass."""
    errors = live_standin_duplicate_key_errors(
        _va(STANDIN_PORT),
        {
            f"{STANDIN_PREFIX}.gateways.read_only.port": 5064,
            f"{STANDIN_PREFIX}.gateways.write_access.address": "gw.example.org",
            f"{STANDIN_PREFIX}.probe_channel": "SOME:CHANNEL",
        },
    )
    assert len(errors) == 3


def test_live_standin_overrides_point_an_author_at_the_facilitys_own_block() -> None:
    """The way out is the ``epics`` block, not deleting the stand-in.

    The refusal that shipped with the second-target design told an author to
    stop asking for a stand-in, because the stand-in *was* ``live``. It is a
    third target now: an author who wants to address a machine addresses the one
    their facility runs, and keeps the rehearsal.
    """
    errors = live_standin_duplicate_key_errors(
        _va(STANDIN_PORT), {f"{STANDIN_PREFIX}.gateways.read_only.address": "gw.example.org"}
    )

    assert "control_system.connector.epics" in errors[0]
    assert "never touches it" in errors[0]
    assert "Going live" not in errors[0]
    assert "delete `virtual_accelerator.live_standin`" not in errors[0]


def test_live_standin_overrides_do_not_refuse_the_facilitys_epics_block() -> None:
    """``live`` is the profile's to spell, stand-in or no stand-in.

    The sharpest statement of the third-target model: a facility already pointed
    at its own machine can stand a rehearsal up beside it without moving a line
    of its own configuration.
    """
    authored = {
        "control_system.connector.epics.gateways.read_only.address": "cagw.example.org",
        "control_system.connector.epics.gateways.read_only.port": 5064,
        "control_system.connector.epics.gateways.write_access.address": "cagw.example.org",
        "control_system.connector.epics.gateways.write_access.port": 5084,
        "control_system.connector.epics.probe_channel": "SR:VAC:GAUGE:SR01:PRESSURE:RB",
    }
    assert live_standin_duplicate_key_errors(_va(STANDIN_PORT), authored) == []
    assert (
        live_standin_duplicate_key_errors(
            _va(STANDIN_PORT),
            {
                "control_system": {
                    "connector": {"epics": {"gateways": {"read_only": {"port": 5064}}}}
                }
            },
        )
        == []
    )


def test_live_standin_overrides_do_not_refuse_the_deployments_own_posture() -> None:
    """Limits checking and the acknowledgment are the profile's, not the build's.

    They were derived — and therefore refused — while the stand-in was a rewrite
    of ``live``. Nothing derives them now, so refusing them would take a key
    away from a profile and give it to nobody.
    """
    assert (
        live_standin_duplicate_key_errors(
            _va(STANDIN_PORT),
            {
                "control_system.limits_checking.enabled": True,
                "control_system.limits_checking.allow_unlisted_channels": False,
                "control_system.target_switch.live_gateway_acknowledged": "cagw.example.org:5064",
            },
        )
        == []
    )


def test_live_standin_overrides_do_not_refuse_the_persona_write_posture() -> None:
    """``live_standin.writes_enabled`` is the read-only persona's own key.

    The refusal is a LEAF allowlist, never a prefix over the stand-in's block:
    where the stand-in is and what proves it reachable are the build's; whether
    a given login may write to it is the persona's.
    """
    assert (
        live_standin_duplicate_key_errors(
            _va(STANDIN_PORT), {f"{STANDIN_PREFIX}.writes_enabled": False}
        )
        == []
    )
    assert (
        live_standin_duplicate_key_errors(
            _va(STANDIN_PORT),
            {"control_system": {"connector": {"live_standin": {"writes_enabled": False}}}},
        )
        == []
    )


def test_live_standin_overrides_do_not_refuse_the_source_probe_channel() -> None:
    """The VA block's own probe channel is where the derived one comes from."""
    assert (
        live_standin_duplicate_key_errors(_va(STANDIN_PORT), {VA_PROBE_CHANNEL_KEY: "X:Y:Z"}) == []
    )


def test_live_standin_overrides_refuse_nothing_without_the_key() -> None:
    """With no stand-in there is no block to own, so nothing is reserved."""
    spelled = dict.fromkeys(LIVE_STANDIN_DERIVED_KEYS, "anything")
    assert live_standin_duplicate_key_errors(_va(None), spelled) == []
    assert live_standin_duplicate_key_errors(None, spelled) == []


# ── The whole build ──────────────────────────────────────────────────────────

#: Neither a venv nor lifecycle hooks is what the stand-in's keys are about,
#: and a real `uv` install per build would dominate this module's runtime.
CI_FLAGS = ["--skip-deps", "--skip-lifecycle"]

_ruamel = YAML(typ="rt")


def _set_live_standin(repo: Path, port: int | None) -> None:
    """Set or clear ``virtual_accelerator.live_standin`` in the repo's profile.

    Written through ruamel rather than by string surgery so the fixture keeps
    working whichever way the shipped preset spells the block once it carries a
    stand-in by default.
    """
    profile_path = repo / "profile.yml"
    with profile_path.open("r", encoding="utf-8") as fh:
        profile = _ruamel.load(fh)
    block = profile["virtual_accelerator"]
    if port is None:
        block.pop("live_standin", None)
    else:
        block["live_standin"] = port
    with profile_path.open("w", encoding="utf-8") as fh:
        _ruamel.dump(profile, fh)


def _add_config_entry(repo: Path, key: str, value: Any) -> None:
    """Add one dotted entry to the repo profile's own ``config:`` block."""
    profile_path = repo / "profile.yml"
    with profile_path.open("r", encoding="utf-8") as fh:
        profile = _ruamel.load(fh)
    profile["config"][key] = value
    with profile_path.open("w", encoding="utf-8") as fh:
        _ruamel.dump(profile, fh)


def _remove_config_entries(repo: Path, *keys: str) -> None:
    """Drop dotted entries from the repo profile's own ``config:`` block."""
    profile_path = repo / "profile.yml"
    with profile_path.open("r", encoding="utf-8") as fh:
        profile = _ruamel.load(fh)
    for key in keys:
        profile["config"].pop(key, None)
    with profile_path.open("w", encoding="utf-8") as fh:
        _ruamel.dump(profile, fh)


def _build(runner: CliRunner, repo: Path):
    previous = Path.cwd()
    os.chdir(repo)
    try:
        return runner.invoke(build, CI_FLAGS)
    finally:
        os.chdir(previous)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.mark.slow
class TestTheRenderedDeploymentDialsTheStandIn:
    """What ``osprey build`` writes into ``build/config.yml``, both ways."""

    def test_live_standin_overrides_reach_the_rendered_config(self, runner, lifecycle_repo) -> None:
        """The stand-in's own block, as the connector will read it."""
        _set_live_standin(lifecycle_repo, STANDIN_PORT)

        result = _build(runner, lifecycle_repo)
        assert result.exit_code == 0, result.output

        config = yaml.safe_load((lifecycle_repo / "build" / "config.yml").read_text())
        standin = config["control_system"]["connector"]["live_standin"]
        assert standin["gateways"] == {
            "read_only": {
                "address": "localhost",
                "port": STANDIN_PORT,
                "use_name_server": True,
            },
            "write_access": {
                "address": "localhost",
                "port": STANDIN_PORT,
                "use_name_server": True,
            },
        }
        assert standin["probe_channel"] == VA_PROBE_CHANNEL

    def test_live_standin_overrides_leave_the_live_target_alone(
        self, runner, lifecycle_repo
    ) -> None:
        """The facility's authored ``epics`` block reads exactly as it was shipped.

        ``live`` means the machine the facility named, on a deployment running a
        stand-in exactly as on one that is not. Asserted on a build that DID ask
        for a stand-in, because that is the build that used to overwrite it.
        """
        _set_live_standin(lifecycle_repo, STANDIN_PORT)

        result = _build(runner, lifecycle_repo)
        assert result.exit_code == 0, result.output

        control_system = yaml.safe_load((lifecycle_repo / "build" / "config.yml").read_text())[
            "control_system"
        ]
        assert control_system["connector"]["epics"] == SHIPPED_EPICS_BLOCK
        assert "probe_channel" not in control_system["connector"]["epics"]

    def test_live_standin_overrides_take_the_limits_posture_from_the_profile(
        self, runner, lifecycle_repo
    ) -> None:
        """The strict pair is rendered because the profile authored it."""
        _set_live_standin(lifecycle_repo, STANDIN_PORT)

        result = _build(runner, lifecycle_repo)
        assert result.exit_code == 0, result.output

        rendered = (lifecycle_repo / "build" / "config.yml").read_text(encoding="utf-8")
        limits = yaml.safe_load(rendered)["control_system"]["limits_checking"]
        assert limits["enabled"] is True
        assert limits["allow_unlisted_channels"] is False

    def test_live_standin_overrides_derive_no_limits_posture(self, runner, lifecycle_repo) -> None:
        """Take the pair out of the profile and nothing puts it back.

        The other half of the claim above, and the one that actually separates
        "authored" from "derived": with the two keys gone from ``config:`` the
        render falls back to what the template ships — permissive — even though
        a stand-in is still asked for. While the stand-in was ``live`` the build
        flipped this key itself and then had to rewrite the comment beside it to
        stop the rendered line contradicting its own value. Neither happens now:
        how strictly a deployment runs is a fact about the deployment, and the
        profile is where it is stated.
        """
        _set_live_standin(lifecycle_repo, STANDIN_PORT)
        _remove_config_entries(
            lifecycle_repo,
            "control_system.limits_checking.enabled",
            "control_system.limits_checking.allow_unlisted_channels",
        )

        result = _build(runner, lifecycle_repo)
        assert result.exit_code == 0, result.output

        rendered = (lifecycle_repo / "build" / "config.yml").read_text(encoding="utf-8")
        limits = yaml.safe_load(rendered)["control_system"]["limits_checking"]
        assert limits["allow_unlisted_channels"] is True
        line = next(row for row in rendered.splitlines() if "allow_unlisted_channels" in row)
        assert "false refuses any channel the database does not list" in line

    def test_live_standin_overrides_leave_the_sandbox_gateways_portless(
        self, runner, lifecycle_repo
    ) -> None:
        """The VA's own rows are default-filled from its service port, so no port."""
        _set_live_standin(lifecycle_repo, STANDIN_PORT)

        result = _build(runner, lifecycle_repo)
        assert result.exit_code == 0, result.output

        config = yaml.safe_load((lifecycle_repo / "build" / "config.yml").read_text())
        va_gateways = config["control_system"]["connector"]["virtual_accelerator"]["gateways"]
        for row in va_gateways.values():
            assert "port" not in row, va_gateways

    def test_live_standin_overrides_refuse_a_baseline_with_no_stand_in(
        self, runner, lifecycle_repo, caplog
    ) -> None:
        """Dropping the key alone leaves the baseline naming a machine nobody serves.

        The exemplar starts every session on the stand-in, so ``live_standin``
        is both a connector block and a baseline — and the two are one decision.
        Removing the port without moving the baseline is the incoherent middle
        state, and it is refused naming both keys rather than built into a
        deployment whose every session dials a dead port.
        """
        _set_live_standin(lifecycle_repo, None)

        with caplog.at_level(logging.ERROR):
            result = _build(runner, lifecycle_repo)

        assert result.exit_code != 0
        assert "control_system.type: live_standin with no" in caplog.text
        assert "virtual_accelerator.live_standin" in caplog.text
        assert not (lifecycle_repo / "build" / "config.yml").exists()

    def test_live_standin_overrides_change_nothing_when_the_key_is_absent(
        self, runner, lifecycle_repo
    ) -> None:
        """No stand-in: no third target, and the shipped production block stands.

        The baseline moves back to the sandbox VA with the port, because the two
        are one decision (see the refusal above) — this is the deployment an
        operator who never asked for a stand-in actually has.
        """
        _set_live_standin(lifecycle_repo, None)
        _add_config_entry(lifecycle_repo, "control_system.type", "virtual_accelerator")

        result = _build(runner, lifecycle_repo)
        assert result.exit_code == 0, result.output

        rendered = (lifecycle_repo / "build" / "config.yml").read_text(encoding="utf-8")
        config = yaml.safe_load(rendered)
        control_system = config["control_system"]
        assert "live_standin" not in control_system["connector"]
        assert control_system["connector"]["epics"] == SHIPPED_EPICS_BLOCK
        assert "probe_channel" not in control_system["connector"]["epics"]
        # Still the profile's own pair, unchanged by the stand-in going away:
        # nothing about the limits posture was ever the stand-in's to decide.
        assert control_system["limits_checking"]["allow_unlisted_channels"] is False

    def test_live_standin_overrides_refuse_a_duplicate_at_build_time(
        self, runner, lifecycle_repo, caplog
    ) -> None:
        """The refusal is reached by a real build, not only by its unit.

        Before anything is written: the render aborts on the profile, so a repo
        with a duplicated stand-in gateway key never gets a ``build/`` at all.
        """
        _set_live_standin(lifecycle_repo, STANDIN_PORT)
        _add_config_entry(
            lifecycle_repo,
            f"{STANDIN_PREFIX}.gateways.read_only.address",
            "gw.example.org",
        )

        with caplog.at_level(logging.ERROR):
            result = _build(runner, lifecycle_repo)

        assert result.exit_code != 0
        assert "The stand-in owns that key" in caplog.text
        assert f"{STANDIN_PREFIX}.gateways.read_only.address" in caplog.text
        assert not (lifecycle_repo / "build" / "config.yml").exists()

    def test_live_standin_overrides_renders_an_unperturbed_latticeless_standin(
        self, runner, lifecycle_repo
    ) -> None:
        """``VA_LATTICE=none`` builds, and its stand-in serves the manifest clean.

        This shape used to be refused, on the reasoning that the stand-in ships
        a readout perturbation and a latticeless IOC exits rather than applying
        one. The perturbation is the half that gives way: the shipped offsets
        displace the builtin PyAT model, so a chain that resolves ``VA_LATTICE``
        elsewhere renders the EMPTY default and gets a stand-in serving the
        facility's own manifest unperturbed — a facility can rehearse against
        its real channel set instead of being turned away.

        Only a deployment that ASKED for faults it cannot apply is still
        refused, and that refusal lives at validation
        (``build_profile_va_faults.live_standin_lattice_errors``), where a
        non-empty ``VA_STANDIN_BPM_ERRORS`` beside this pin is read.
        """
        _set_live_standin(lifecycle_repo, STANDIN_PORT)
        (lifecycle_repo / ".env").write_text("VA_LATTICE=none\n", encoding="utf-8")

        result = _build(runner, lifecycle_repo)
        assert result.exit_code == 0, result.output

        rendered = (
            lifecycle_repo / "build" / "services" / "virtual_accelerator" / "docker-compose.yml"
        ).read_text(encoding="utf-8")
        # The empty default reaches the container as an empty fault set: `-`
        # substitutes only for an unset variable, so nothing rounds it back up.
        assert 'VA_BPM_ERRORS: "${VA_STANDIN_BPM_ERRORS-}"' in rendered
        assert "stand-in serves the facility manifest unperturbed" in result.output
