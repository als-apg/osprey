"""``virtual_accelerator.live_standin:`` and the rendered config it derives.

The stand-in's whole promise is that "live" behaves like the live machine, so
what these tests pin is not that some keys were written — it is that the four
statements the rendered config makes about the live target are the true ones:

* the ``epics`` gateways dial the stand-in, on loopback, at the port the
  profile named, with the port written out (``EPICSConnector`` has no
  ``fill_gateway_ports``, so an omitted port is the EPICS default rather than
  the stand-in's) — and the *sandbox* VA's gateway rows still carry no port,
  because those really are default-filled and a written one would state the
  same fact twice;
* the ``epics`` target carries the same probe channel the VA block proves,
  since a target without one is never switched to and a rehearsal you cannot
  switch into rehearses nothing;
* limits checking runs strict, and the rendered line *says* strict — the
  template ships that key permissive with an inline comment calling it a
  tutorial convenience, and a comment left standing beside its own
  contradiction is the sentence an operator reads when judging whether the
  deployment is safe;
* a profile that also spells any of those keys is refused, naming the three
  go-live steps, rather than having its own gateway address silently outranked.

The off-state matters as much: with no ``live_standin`` key the same build must
render the shipped ALS production ``epics`` block untouched, permissive comment
and all.
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
    STRICT_LIMITS_COMMENT,
    VA_PROBE_CHANNEL_KEY,
    live_standin_config_overrides,
    live_standin_duplicate_key_errors,
    rewrite_strict_limits_comment,
)

#: The port the stand-in is reserved on throughout the codebase — both config
#: templates' gateway examples read 5094 rather than this, on purpose, so an
#: override written there cannot collide with a stand-in.
STANDIN_PORT = 5074

#: What the Control Assistant template's VA block proves, and therefore what
#: the stand-in must prove too.
VA_PROBE_CHANNEL = "SR:VAC:GAUGE:SR01:PRESSURE:RB"

#: The shipped ALS production ``epics`` gateways — the "before" a build with no
#: stand-in must still render, pinned the same way tests/cli/test_rendered_va_block.py
#: pins it.
SHIPPED_EPICS_GATEWAYS = {
    "read_only": {
        "address": "cagw-alsdmz.als.lbl.gov",
        "port": 5064,
        "use_name_server": False,
    },
    "write_access": {
        "address": "cagw-alsdmz.als.lbl.gov",
        "port": 5084,
        "use_name_server": False,
    },
}


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
    """No stand-in asked for, nothing derived — the epics block stays the facility's."""
    assert (
        live_standin_config_overrides(_va(None), {}, _rendered_with_probe(VA_PROBE_CHANNEL)) == {}
    )
    assert live_standin_config_overrides(None, {}, _rendered_with_probe(VA_PROBE_CHANNEL)) == {}


def test_live_standin_overrides_point_the_live_target_at_the_stand_in() -> None:
    """Both gateways on loopback at the named port, strict limits, probe carried over."""
    overrides = live_standin_config_overrides(
        _va(STANDIN_PORT), {}, _rendered_with_probe(VA_PROBE_CHANNEL)
    )
    assert overrides == {
        "control_system.connector.epics.gateways.read_only.address": "localhost",
        "control_system.connector.epics.gateways.read_only.port": STANDIN_PORT,
        "control_system.connector.epics.gateways.read_only.use_name_server": True,
        "control_system.connector.epics.gateways.write_access.address": "localhost",
        "control_system.connector.epics.gateways.write_access.port": STANDIN_PORT,
        "control_system.connector.epics.gateways.write_access.use_name_server": True,
        "control_system.connector.epics.probe_channel": VA_PROBE_CHANNEL,
        "control_system.limits_checking.enabled": True,
        "control_system.limits_checking.allow_unlisted_channels": False,
    }
    assert set(overrides) == set(LIVE_STANDIN_DERIVED_KEYS)


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
    """A dotted gateway key beside the stand-in is refused, naming the way out."""
    errors = live_standin_duplicate_key_errors(
        _va(STANDIN_PORT),
        {"control_system.connector.epics.gateways.read_only.port": 5064},
    )
    assert len(errors) == 1
    assert "control_system.connector.epics.gateways.read_only.port" in errors[0]
    assert "Going live is three steps: delete `virtual_accelerator.live_standin`" in errors[0]


def test_live_standin_overrides_refuse_a_nested_duplicate() -> None:
    """Spelling-independent: a nested subtree reaches the same leaf and is refused too."""
    errors = live_standin_duplicate_key_errors(
        _va(STANDIN_PORT),
        {
            "control_system": {
                "connector": {"epics": {"gateways": {"read_only": {"address": "gw.example.org"}}}}
            }
        },
    )
    assert len(errors) == 1
    assert "control_system.connector.epics.gateways.read_only.address" in errors[0]
    assert "Going live is three steps: delete `virtual_accelerator.live_standin`" in errors[0]


def test_live_standin_overrides_accumulate_every_duplicate_at_once() -> None:
    """Every offending key in one report, so a profile is fixed in one pass."""
    errors = live_standin_duplicate_key_errors(
        _va(STANDIN_PORT),
        {
            "control_system.connector.epics.gateways.read_only.port": 5064,
            "control_system.limits_checking.allow_unlisted_channels": True,
            "control_system.connector.epics.probe_channel": "SOME:CHANNEL",
        },
    )
    assert len(errors) == 3


def test_live_standin_overrides_do_not_refuse_the_persona_write_posture() -> None:
    """``epics.writes_enabled`` is the read-only persona's own key and stays legal.

    The refusal is a LEAF allowlist, never a prefix over the epics block: where
    the live target is and how strictly it runs are the stand-in's; whether a
    given login may write to it is the persona's.
    """
    assert (
        live_standin_duplicate_key_errors(
            _va(STANDIN_PORT), {"control_system.connector.epics.writes_enabled": False}
        )
        == []
    )
    assert (
        live_standin_duplicate_key_errors(
            _va(STANDIN_PORT),
            {"control_system": {"connector": {"epics": {"writes_enabled": False}}}},
        )
        == []
    )


def test_live_standin_overrides_do_not_refuse_the_source_probe_channel() -> None:
    """The VA block's own probe channel is where the derived one comes from."""
    assert (
        live_standin_duplicate_key_errors(_va(STANDIN_PORT), {VA_PROBE_CHANNEL_KEY: "X:Y:Z"}) == []
    )


def test_live_standin_overrides_refuse_nothing_without_the_key() -> None:
    """With no stand-in the epics block is the facility's to spell however it likes."""
    spelled = dict.fromkeys(LIVE_STANDIN_DERIVED_KEYS, "anything")
    assert live_standin_duplicate_key_errors(_va(None), spelled) == []
    assert live_standin_duplicate_key_errors(None, spelled) == []


# ── The rendered line says what the value means ──────────────────────────────

STALE_COMMENT_FIXTURE = """control_system:
  writes_enabled: false

  # Runtime Channel Limits Checking
  limits_checking:
    enabled: true
    database_path: "data/channel_limits.json"
    allow_unlisted_channels: false  # Permissive mode for tutorial

  # Write Verification Configuration
  # How the connector confirms a write took effect.
  write_verification:
    default_level: callback
"""


def test_live_standin_overrides_retruth_the_strict_limits_line(tmp_path: Path) -> None:
    """The tutorial-convenience comment is replaced once the value is strict."""
    config_path = tmp_path / "config.yml"
    config_path.write_text(STALE_COMMENT_FIXTURE, encoding="utf-8")

    assert rewrite_strict_limits_comment(config_path) is True

    rendered = config_path.read_text(encoding="utf-8")
    line = next(row for row in rendered.splitlines() if "allow_unlisted_channels" in row)
    assert "Permissive" not in line
    assert STRICT_LIMITS_COMMENT in line
    assert "allow_unlisted_channels: false" in line
    # The banner ruamel parks in that same comment token belongs to the file's
    # layout, not to the key, and must survive the rewrite.
    assert "# Write Verification Configuration" in rendered
    assert "# How the connector confirms a write took effect." in rendered
    assert yaml.safe_load(rendered)["control_system"]["write_verification"] == {
        "default_level": "callback"
    }


def test_live_standin_overrides_retruth_a_line_with_no_comment(tmp_path: Path) -> None:
    """The generic project template ships the key bare; it still gets the reason."""
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "control_system:\n"
        "  limits_checking:\n"
        "    enabled: true\n"
        "    allow_unlisted_channels: false\n"
        "\n"
        "  # Write Verification Configuration\n"
        "  write_verification:\n"
        "    default_level: callback\n",
        encoding="utf-8",
    )

    assert rewrite_strict_limits_comment(config_path) is True

    rendered = config_path.read_text(encoding="utf-8")
    assert f"allow_unlisted_channels: false {STRICT_LIMITS_COMMENT}" in rendered
    assert "# Write Verification Configuration" in rendered


def test_live_standin_overrides_leave_a_config_without_the_key_alone(tmp_path: Path) -> None:
    """Nothing to retruth, nothing rewritten — the file is not even re-emitted."""
    config_path = tmp_path / "config.yml"
    original = "control_system:\n  type: mock\n"
    config_path.write_text(original, encoding="utf-8")

    assert rewrite_strict_limits_comment(config_path) is False
    assert config_path.read_text(encoding="utf-8") == original


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
        """Gateways, probe channel and posture, as the connector will read them."""
        _set_live_standin(lifecycle_repo, STANDIN_PORT)

        result = _build(runner, lifecycle_repo)
        assert result.exit_code == 0, result.output

        config = yaml.safe_load((lifecycle_repo / "build" / "config.yml").read_text())
        control_system = config["control_system"]
        epics = control_system["connector"]["epics"]
        assert epics["gateways"] == {
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
        assert epics["probe_channel"] == VA_PROBE_CHANNEL
        assert control_system["limits_checking"]["enabled"] is True
        assert control_system["limits_checking"]["allow_unlisted_channels"] is False

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

    def test_live_standin_overrides_render_a_truthful_limits_comment(
        self, runner, lifecycle_repo
    ) -> None:
        """The rendered line explains the strictness instead of contradicting it."""
        _set_live_standin(lifecycle_repo, STANDIN_PORT)

        result = _build(runner, lifecycle_repo)
        assert result.exit_code == 0, result.output

        rendered = (lifecycle_repo / "build" / "config.yml").read_text(encoding="utf-8")
        line = next(row for row in rendered.splitlines() if "allow_unlisted_channels" in row)
        assert "Permissive" not in line
        assert STRICT_LIMITS_COMMENT in line

    def test_live_standin_overrides_change_nothing_when_the_key_is_absent(
        self, runner, lifecycle_repo
    ) -> None:
        """No stand-in: the shipped production epics block and its comment stand."""
        _set_live_standin(lifecycle_repo, None)

        result = _build(runner, lifecycle_repo)
        assert result.exit_code == 0, result.output

        rendered = (lifecycle_repo / "build" / "config.yml").read_text(encoding="utf-8")
        config = yaml.safe_load(rendered)
        control_system = config["control_system"]
        assert control_system["connector"]["epics"]["gateways"] == SHIPPED_EPICS_GATEWAYS
        assert "probe_channel" not in control_system["connector"]["epics"]
        assert control_system["limits_checking"]["allow_unlisted_channels"] is True
        line = next(row for row in rendered.splitlines() if "allow_unlisted_channels" in row)
        assert "Permissive mode for tutorial" in line
        assert STRICT_LIMITS_COMMENT not in line

    def test_live_standin_overrides_refuse_a_duplicate_at_build_time(
        self, runner, lifecycle_repo, caplog
    ) -> None:
        """The refusal is reached by a real build, not only by its unit.

        Before anything is written: the render aborts on the profile, so a repo
        with a duplicated gateway key never gets a ``build/`` at all.
        """
        _set_live_standin(lifecycle_repo, STANDIN_PORT)
        _add_config_entry(
            lifecycle_repo,
            "control_system.connector.epics.gateways.read_only.address",
            "gw.example.org",
        )

        with caplog.at_level(logging.ERROR):
            result = _build(runner, lifecycle_repo)

        assert result.exit_code != 0
        assert "Going live is three steps: delete `virtual_accelerator.live_standin`" in caplog.text
        assert "control_system.connector.epics.gateways.read_only.address" in caplog.text
        assert not (lifecycle_repo / "build" / "config.yml").exists()

    def test_live_standin_overrides_refuse_a_latticeless_build(
        self, runner, lifecycle_repo, caplog, monkeypatch
    ) -> None:
        """The build's half of the lattice gate: no manifest, and a pinned channel file.

        Validation cannot answer this one — whether a render produced a channel
        manifest is only knowable once it has run — so the refusal lives at the
        end of the build, beside the ``.env`` write that gates on the very same
        file. Without it the stand-in would be deployed and exit at container
        start, which is a failure an operator meets at ``osprey up`` instead of
        here.
        """
        _set_live_standin(lifecycle_repo, STANDIN_PORT)
        (lifecycle_repo / ".env").write_text(
            "VA_CHANNELS_FILE=facility_channels.json\n", encoding="utf-8"
        )
        # The one seam that decides "this build generated no channel manifest":
        # a data tree with no paradigm databases behind it returns None here,
        # and the build then writes nothing into build/data/simulation/.
        monkeypatch.setattr(
            "osprey.services.virtual_accelerator.manifest.build.prepare_project_manifest",
            lambda data_root, tier: None,
        )

        with caplog.at_level(logging.ERROR):
            result = _build(runner, lifecycle_repo)

        assert result.exit_code != 0
        assert "needs a lattice-backed virtual accelerator" in caplog.text
        assert "VA_CHANNELS_FILE='facility_channels.json'" in caplog.text
        assert "delete `virtual_accelerator.live_standin`" in caplog.text
