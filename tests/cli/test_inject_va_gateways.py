"""Deploying the VA service must leave a target able to reach it.

A session is switched to a target the rendered ``config.yml`` already describes —
the switch never edits config — so a project that deploys the virtual
accelerator and carries no ``control_system.connector.virtual_accelerator``
block has a soft-IOC running and nothing able to point at it. Projects built
from the current generic template render that block themselves; these tests
cover the other configs the injector meets: the ones written before the template
had it, and the hand-maintained ones.

The rule the injector cannot break is the other half: a gateway table that is
already there was written by somebody, and an injector that "corrects" it is an
edit nobody asked for. So the tests pin both directions — written when absent,
byte-identical when present, including a table the author left half-filled.

The last test closes CF-3 against the real predicate rather than against a
restatement of it: with the gateways injected and the probe channel named, the
VA target is ELIGIBLE with no hand editing, and on a bare injected config the
ONE thing still missing is the probe channel.
"""

from __future__ import annotations

import logging
from typing import Any

import yaml as pyyaml

from osprey.cli.build_injectors import _inject_va
from osprey.cli.build_profile_schema import VAConfig
from osprey.mcp_server.control_system.target_eligibility import (
    REASON_PROBE_CHANNEL_MISSING,
    evaluate_eligibility,
    target_availability,
)

#: A config with no ``virtual_accelerator`` connector block at all — the shape
#: of every project rendered before the generic template grew one. Carries a
#: real archiver so the honesty rule (VA + mock archiver = invented history) is
#: not what answers the eligibility question here.
CONFIG_WITHOUT_VA_BLOCK = """\
services:
  postgresql:
    path: ./services/postgresql

# Services to deploy with `osprey up`
deployed_services:
  - postgresql

# ============================================================
# CONTROL SYSTEM
# ============================================================

control_system:
  type: "mock"
  writes_enabled: false
  connector:
    mock:
      response_delay_ms: 0
    epics:
      timeout: 5.0
      gateways:
        read_only:
          address: your-gateway.example.com
          port: 5064
          use_name_server: false

archiver:
  type: "epics_archiver"

# ============================================================
# SAFETY CONTROLS
# ============================================================

# Approval workflow for sensitive operations
approval:
  enabled: true
"""

#: A connector block an operator wrote by hand: one role, an explicit
#: non-default port, name-server off. Nothing here is what the injector would
#: write, which is the point — it must survive untouched.
CONFIG_WITH_AUTHORED_GATEWAYS = """\
services:
  postgresql:
    path: ./services/postgresql

deployed_services:
  - postgresql

control_system:
  type: "mock"
  connector:
    virtual_accelerator:
      timeout: 5.0
      probe_channel: SR:BPM:1:X
      # The VA lives on another host, so the port IS written out here.
      gateways:
        write_access:
          address: va-host.example.org
          port: 5199          # published port, not the served one
          use_name_server: false
"""

#: The connector block exists (a project that configured a timeout and a probe
#: channel) but names no gateways at all.
CONFIG_WITH_BLOCK_NO_GATEWAYS = """\
services:
  postgresql:
    path: ./services/postgresql

deployed_services:
  - postgresql

control_system:
  type: "mock"
  connector:
    virtual_accelerator:
      timeout: 5.0
      probe_channel: SR:BPM:1:X

# Trailing section banner
approval:
  enabled: true
"""

#: What the injector installs: both roles on localhost in CA name-server mode,
#: and no ``port`` — that is derived from services.virtual_accelerator.port.
EXPECTED_GATEWAYS: dict[str, Any] = {
    "read_only": {"address": "localhost", "use_name_server": True},
    "write_access": {"address": "localhost", "use_name_server": True},
}


def _line_no(text: str, needle: str) -> int:
    for i, line in enumerate(text.splitlines()):
        if needle in line:
            return i
    raise AssertionError(f"{needle!r} not found in:\n{text}")


def _inject(tmp_path, template: str) -> str:
    (tmp_path / "config.yml").write_text(template, encoding="utf-8")
    _inject_va(VAConfig(port=5064), tmp_path)
    return (tmp_path / "config.yml").read_text(encoding="utf-8")


def _va_block(text: str) -> Any:
    return pyyaml.safe_load(text)["control_system"]["connector"]["virtual_accelerator"]


def test_absent_block_gets_the_canonical_gateway_table(tmp_path):
    text = _inject(tmp_path, CONFIG_WITHOUT_VA_BLOCK)

    assert _va_block(text) == {"gateways": EXPECTED_GATEWAYS}


def test_injected_gateways_carry_no_port(tmp_path):
    """Ledger 56: an unset port follows services.virtual_accelerator.port.

    Writing one here would state the deployed soft-IOC's port a second time, and
    two spellings of one fact are free to disagree the moment the service moves.
    """
    text = _inject(tmp_path, CONFIG_WITHOUT_VA_BLOCK)

    for role, gateway in _va_block(text)["gateways"].items():
        assert "port" not in gateway, f"{role} gateway must not name a port"


def test_injected_block_names_no_probe_channel(tmp_path):
    """The channel comes from the project's own machine model, so none is guessed."""
    text = _inject(tmp_path, CONFIG_WITHOUT_VA_BLOCK)

    assert "probe_channel" not in _va_block(text)


def test_absent_block_write_keeps_section_comments_anchored(tmp_path):
    text = _inject(tmp_path, CONFIG_WITHOUT_VA_BLOCK)

    # The new sub-block lands inside the existing connector table, above the
    # sections that follow it — not after the SAFETY CONTROLS banner.
    assert _line_no(text, "    virtual_accelerator:") < _line_no(text, "archiver:")
    assert _line_no(text, "# SAFETY CONTROLS") < _line_no(text, "approval:")
    assert _line_no(text, "# Approval workflow") < _line_no(text, "approval:")
    # Comments that were already in the control-system section stay above it.
    assert _line_no(text, "# CONTROL SYSTEM") < _line_no(text, "control_system:")

    # The blocks it sits beside are untouched.
    connector = pyyaml.safe_load(text)["control_system"]["connector"]
    assert connector["mock"] == {"response_delay_ms": 0}
    assert connector["epics"]["gateways"]["read_only"]["port"] == 5064


def test_authored_gateways_are_left_byte_identical(tmp_path):
    """A table the author wrote is theirs — weird port, one role, quotes and all."""
    text = _inject(tmp_path, CONFIG_WITH_AUTHORED_GATEWAYS)

    block = _va_block(text)
    assert block["gateways"] == {
        "write_access": {
            "address": "va-host.example.org",
            "port": 5199,
            "use_name_server": False,
        }
    }
    # The missing read_only role is NOT filled in: the presence of the key means
    # the author owns the table, and completing it is an edit nobody asked for.
    assert "read_only" not in block["gateways"]
    assert block["probe_channel"] == "SR:BPM:1:X"
    # Byte-level: the whole authored block, comments and inline spacing
    # included, is reproduced verbatim.
    authored = CONFIG_WITH_AUTHORED_GATEWAYS.split("  connector:\n", 1)[1]
    assert authored in text


def test_existing_block_without_gateways_gains_only_gateways(tmp_path):
    text = _inject(tmp_path, CONFIG_WITH_BLOCK_NO_GATEWAYS)

    assert _va_block(text) == {
        "timeout": 5.0,
        "probe_channel": "SR:BPM:1:X",
        "gateways": EXPECTED_GATEWAYS,
    }
    # The banner that trailed the connector block stays above what it introduces.
    assert _line_no(text, "# Trailing section banner") < _line_no(text, "approval:")
    assert _line_no(text, "      gateways:") < _line_no(text, "# Trailing section banner")


def test_injection_is_idempotent(tmp_path):
    once = _inject(tmp_path, CONFIG_WITHOUT_VA_BLOCK)
    _inject_va(VAConfig(port=5064), tmp_path)
    twice = (tmp_path / "config.yml").read_text(encoding="utf-8")

    assert twice == once


def test_services_write_is_unchanged(tmp_path):
    """The behavior this step was bolted onto still behaves."""
    text = _inject(tmp_path, CONFIG_WITHOUT_VA_BLOCK)

    config = pyyaml.safe_load(text)
    assert config["services"]["virtual_accelerator"] == {
        "path": "./services/virtual_accelerator",
        "port": 5064,
    }
    assert config["deployed_services"] == ["postgresql", "virtual_accelerator"]


def test_a_non_mapping_connector_entry_is_left_alone(tmp_path):
    """Whatever ``virtual_accelerator: <scalar>`` meant, it is not ours to replace."""
    text = _inject(
        tmp_path,
        "services:\n  postgresql:\n    path: ./services/postgresql\n"
        "deployed_services:\n  - postgresql\n"
        "control_system:\n  connector:\n    virtual_accelerator: disabled\n",
    )

    assert pyyaml.safe_load(text)["control_system"]["connector"]["virtual_accelerator"] == (
        "disabled"
    )


# ---------------------------------------------------------------------------
# CF-3: the built project reports va ELIGIBLE with no hand editing
# ---------------------------------------------------------------------------


def test_injected_config_is_eligible_once_the_probe_channel_is_named(tmp_path):
    """The criterion, asserted through Task 2.4's real predicate.

    The template renders the probe channel; the injector renders the gateways.
    Together — and with no hand editing of either — the VA target passes
    eligibility. The probe channel is added here the way the template carries
    it, since this config deliberately starts from one that predates the block.
    """
    text = _inject(tmp_path, CONFIG_WITHOUT_VA_BLOCK)
    config = pyyaml.safe_load(text)
    config["control_system"]["connector"]["virtual_accelerator"]["probe_channel"] = "SR:BPM:1:X"

    verdict = evaluate_eligibility(config, "va")

    assert verdict.eligible, verdict.detail
    assert verdict.reason is None

    # And the roster reports it available from a session sitting on the live
    # baseline — the "va ELIGIBLE in the roster" half of the criterion.
    availability = target_availability(config, "va", session_target="live", baseline_target="live")
    assert availability.eligible
    assert availability.available_now


def test_the_only_thing_missing_after_injection_is_the_probe_channel(tmp_path):
    """Everything the injector CAN derive, it derived.

    A bare injected config is ineligible for exactly one reason, and the reason
    names the piece an operator still has to supply — not gateways, not the
    connector block, not the archiver pairing.
    """
    text = _inject(tmp_path, CONFIG_WITHOUT_VA_BLOCK)

    verdict = evaluate_eligibility(pyyaml.safe_load(text), "va")

    assert not verdict.eligible
    assert verdict.reason == REASON_PROBE_CHANNEL_MISSING
    assert "probe_channel" in verdict.detail


# ---------------------------------------------------------------------------
# The live stand-in: a second instance of the same service, acknowledged
# ---------------------------------------------------------------------------
#
# ``virtual_accelerator.live_standin: <port>`` deploys a SECOND soft-IOC
# container and wires it in as the deployment's ``live`` target, so an operator
# can rehearse the whole go-live ritual against something safe. Three writes
# make that work with no hand editing, and each of them is pinned below:
#
# * ``services.live_standin`` — same ``path`` as ``virtual_accelerator``,
#   because it is a second instance of one template, not a second service;
# * ``deployed_services`` — the compose template reads its instance list from
#   there, so a block that is not deployed conjures no container;
# * ``control_system.target_switch.live_gateway_acknowledged`` — the gate that
#   otherwise stops a session switching TO ``live``. Nothing is left for an
#   operator to confirm about a loopback container this build addressed itself.
#
# The text assertions run against the REAL Control Assistant template rather
# than a literal fixture, because what they pin is how the write lands among
# comments that ship — the wrapped inline comments beside it, the prose above
# ``target_switch:``, and the commented-out example the write has to remove.

#: Channel Access port of the stand-in in these tests. Not 5064: the two
#: instances serve different machines and may never share a port.
STANDIN_PORT = 5074

#: The note the build hangs above the acknowledgment it wrote. Restated here
#: rather than imported, because this is the prose an operator reads in their
#: own config.yml — a reword should have to be made twice, on purpose.
EXPECTED_ACK_NOTE = (
    "    # Written by `osprey build` for the live stand-in: the `epics` gateways\n"
    "    # above dial the second virtual-accelerator container on this loopback\n"
    "    # port, so the acknowledgment names it. When you go live, delete\n"
    "    # `virtual_accelerator.live_standin` from the build profile, set your\n"
    "    # gateways, and replace this value by hand with your own live gateway's\n"
    "    # hostname.\n"
)


def _render_control_assistant_template() -> str:
    """The shipped app template, rendered the way the build renders it.

    Same helper as tests/cli/test_rendered_va_block.py's: the text pins below
    are only worth having if they are against what ships.
    """
    from osprey.cli.templates.manager import TemplateManager

    return TemplateManager().jinja_env.get_template("apps/control_assistant/config.yml.j2").render()


def _inject_standin(tmp_path, template: str | None = None, *, port: int | None = STANDIN_PORT):
    """Run the injector over *template* with the stand-in port set (or not)."""
    if template is None:
        template = _render_control_assistant_template()
    (tmp_path / "config.yml").write_text(template, encoding="utf-8")
    _inject_va(VAConfig(port=5064, live_standin=port), tmp_path)
    return (tmp_path / "config.yml").read_text(encoding="utf-8")


def _target_switch(text: str) -> Any:
    return pyyaml.safe_load(text)["control_system"]["target_switch"]


class TestNoStandinChangesNothing:
    """``live_standin`` unset is the shipped default, and it has to be inert."""

    def test_the_rendered_config_is_what_it_was_before_the_stand_in_existed(self, tmp_path):
        template = _render_control_assistant_template()
        without = _inject_standin(tmp_path, template, port=None)

        assert _inject(tmp_path, template) == without

    def test_no_stand_in_service_and_no_stand_in_entry(self, tmp_path):
        config = pyyaml.safe_load(_inject_standin(tmp_path, port=None))

        assert "live_standin" not in config["services"]
        assert "live_standin" not in config["deployed_services"]
        assert "virtual_accelerator" in config["deployed_services"]

    def test_the_target_switch_block_is_left_exactly_as_the_template_ships_it(self, tmp_path):
        """Including the commented-out example: nothing wrote the key, so it stays."""
        template = _render_control_assistant_template()
        text = _inject_standin(tmp_path, template, port=None)

        block = (
            "  target_switch:\n" + template.split("  target_switch:\n", 1)[1].split("\n\n", 1)[0]
        )
        assert block in text
        assert "# live_gateway_acknowledged: cagw-alsdmz.als.lbl.gov" in text
        assert "live_gateway_acknowledged" not in _target_switch(text)


class TestStandinServiceRegistration:
    def test_the_stand_in_shares_the_virtual_accelerators_template_directory(self, tmp_path):
        """One template, one path, two containers — not a second service tree."""
        services = pyyaml.safe_load(_inject_standin(tmp_path))["services"]

        assert services["live_standin"] == {
            "path": "./services/virtual_accelerator",
            "port": STANDIN_PORT,
        }
        assert services["virtual_accelerator"]["path"] == services["live_standin"]["path"]

    def test_both_instances_are_deployed_exactly_once(self, tmp_path):
        """The compose template reads its instance list off ``deployed_services``."""
        deployed = pyyaml.safe_load(_inject_standin(tmp_path))["deployed_services"]

        assert deployed.count("virtual_accelerator") == 1
        assert deployed.count("live_standin") == 1

    def test_an_authored_env_passthrough_survives_the_block_replacement(self, tmp_path):
        """``_carry_authored_keys`` covers the second instance too.

        ``env:`` is the one key on a service block that belongs to the author
        rather than to the injector, and it lands in config.yml *before* the
        injectors run — so a stand-in that dropped it would accept the
        declaration and then silently deliver no passthrough.
        """
        template = _render_control_assistant_template().replace(
            "services:\n",
            "services:\n  live_standin:\n    env:\n      - MY_HOST_VAR\n",
            1,
        )

        services = pyyaml.safe_load(_inject_standin(tmp_path, template))["services"]

        assert services["live_standin"]["env"] == ["MY_HOST_VAR"]
        assert services["live_standin"]["port"] == STANDIN_PORT

    def test_the_gateway_rows_still_carry_no_port(self, tmp_path):
        """Ledger 56 holds with a stand-in deployed: the port is derived, not written."""
        text = _inject_standin(tmp_path, CONFIG_WITHOUT_VA_BLOCK)

        for role, gateway in _va_block(text)["gateways"].items():
            assert "port" not in gateway, f"{role} gateway must not name a port"


class TestAcknowledgment:
    def test_the_stand_ins_loopback_endpoint_is_acknowledged(self, tmp_path):
        assert _target_switch(_inject_standin(tmp_path))["live_gateway_acknowledged"] == (
            f"localhost:{STANDIN_PORT}"
        )

    def test_the_note_sits_directly_above_the_key(self, tmp_path):
        text = _inject_standin(tmp_path)

        assert (
            EXPECTED_ACK_NOTE + f"    live_gateway_acknowledged: localhost:{STANDIN_PORT}\n"
        ) in text

    def test_the_key_is_written_once_and_the_commented_example_is_gone(self, tmp_path):
        """Two spellings of one setting side by side is what this write must not leave."""
        text = _inject_standin(tmp_path)

        assert text.count("    live_gateway_acknowledged:") == 1
        assert "# live_gateway_acknowledged:" not in text

    def test_the_template_prose_above_target_switch_survives_once(self, tmp_path):
        """The write lands inside the block; the header explaining it does not move."""
        text = _inject_standin(tmp_path)

        assert text.count("# The `live_gateway_acknowledged` key below is the operator") == 1
        assert _line_no(text, "# The `live_gateway_acknowledged` key below") < _line_no(
            text, "  target_switch:"
        )

    def test_the_wrapped_inline_comments_beside_it_are_not_torn_in_half(self, tmp_path):
        """The failure mode this write is most likely to cause, pinned directly.

        ``probe_interval_s``'s inline comment wraps onto a continuation line,
        and both lines live on ONE ruamel comment token — together with the
        commented-out example the write removes.
        """
        text = _inject_standin(tmp_path)

        assert (
            "    probe_interval_s: 30    # Seconds between background reachability probes of\n"
            "                            # every target's gateways\n"
        ) in text
        assert (
            "    drain_timeout_s: 5      # Seconds in-flight operations get to finish on the\n"
            "                            # old target before it is torn down regardless\n"
        ) in text

    def test_a_rebuild_reproduces_the_file_byte_for_byte(self, tmp_path):
        once = _inject_standin(tmp_path)
        _inject_va(VAConfig(port=5064, live_standin=STANDIN_PORT), tmp_path)
        twice = (tmp_path / "config.yml").read_text(encoding="utf-8")

        assert twice == once

    def test_a_config_with_no_target_switch_block_gains_the_acknowledgment(self, tmp_path):
        """The configs that predate the block, and the hand-maintained ones."""
        text = _inject_standin(tmp_path, CONFIG_WITHOUT_VA_BLOCK)

        assert _target_switch(text)["live_gateway_acknowledged"] == f"localhost:{STANDIN_PORT}"

    def test_an_operator_authored_acknowledgment_is_never_downgraded(self, tmp_path, caplog):
        """It names an operator's own machine, and a loopback container is not it."""
        template = _render_control_assistant_template().replace(
            "    # live_gateway_acknowledged: cagw-alsdmz.als.lbl.gov\n",
            "    live_gateway_acknowledged: cagw.example.com   # ours, checked\n",
            1,
        )

        with caplog.at_level(logging.WARNING):
            text = _inject_standin(tmp_path, template)

        assert _target_switch(text)["live_gateway_acknowledged"] == "cagw.example.com"
        # The value AND the comment the operator wrote beside it.
        assert "    live_gateway_acknowledged: cagw.example.com   # ours, checked\n" in text
        assert "# Written by `osprey build`" not in text
        assert "live_gateway_acknowledged" in caplog.text
