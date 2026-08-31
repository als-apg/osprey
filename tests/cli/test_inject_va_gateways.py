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
from osprey.port_layout import DEFAULT_PORT_BASE, layout_ports

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
# The live stand-in: a second instance of the same service, a THIRD target
# ---------------------------------------------------------------------------
#
# ``virtual_accelerator.live_standin: <port>`` deploys a SECOND soft-IOC
# container and gives the deployment a THIRD control target, ``standin``, so an
# operator can rehearse against something that cannot move a magnet. Two writes
# make that work with no hand editing, and each of them is pinned below:
#
# * ``services.live_standin`` — same ``path`` as ``virtual_accelerator``,
#   because it is a second instance of one template, not a second service;
# * ``deployed_services`` — the compose template reads its instance list from
#   there, so a block that is not deployed conjures no container.
#
# The third write this injector used to make is now the thing it must NOT make:
# ``control_system.target_switch.live_gateway_acknowledged``. ``live`` means the
# machine the facility authored under ``epics:`` — on a stand-in deployment
# exactly as on one without — so reaching it is ``control_target_set live``,
# which asks the profile for its own acknowledgment and strict limits. The
# stand-in is reached as ``control_target_set standin`` and needs neither. A
# build that wrote the acknowledgment would be answering, on the operator's
# behalf, a question about a machine it never addressed.
#
# The text assertions run against the REAL Control Assistant template rather
# than a literal fixture, because what they pin is that the shipped
# ``target_switch`` block — its prose, its wrapped inline comments, and the
# commented-out example an operator fills in by hand — comes through untouched.

#: Channel Access port of the stand-in in these tests. Not 5064: the two
#: instances serve different machines and may never share a port. An arbitrary
#: operator-chosen number, deliberately NOT the layout's ``va_standin`` slot —
#: every use passes it explicitly as ``VAConfig(live_standin=<port>)``, which is
#: the "a facility placed the second soft-IOC somewhere specific" branch, so
#: nothing here reads it as a default.
STANDIN_PORT = 5074


def _render_control_assistant_template() -> str:
    """The shipped app template, rendered the way the build renders it.

    Same helper as tests/cli/test_rendered_va_block.py's: the text pins below
    are only worth having if they are against what ships.
    """
    from osprey.cli.templates.manager import TemplateManager

    return (
        TemplateManager()
        .jinja_env.get_template("apps/control_assistant/config.yml.j2")
        .render(
            port_base=DEFAULT_PORT_BASE,
            osprey_ports=layout_ports(DEFAULT_PORT_BASE),
        )
    )


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
        assert "# live_gateway_acknowledged: your-ca-gateway.example.com" in text
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


class TestTheAcknowledgmentIsNeverWritten:
    """The stand-in is a target of its own, so no gate about ``live`` moves.

    ``control_target_set live`` reaches the facility's ``epics`` gateways on a
    stand-in deployment exactly as on one without, and it asks that deployment's
    own profile for the operator acknowledgment. Everything below pins the same
    rule from a different angle: whatever the config said about
    ``target_switch`` before the injector ran, it says afterwards.
    """

    def test_the_injector_writes_no_operator_acknowledgment(self, tmp_path):
        """The key stays absent — the profile's to state, not the build's."""
        assert "live_gateway_acknowledged" not in _target_switch(_inject_standin(tmp_path))

    def test_the_commented_example_is_left_for_the_operator_to_fill_in(self, tmp_path):
        """It is the template author's example; nothing wrote a value beside it."""
        text = _inject_standin(tmp_path)

        assert "    # live_gateway_acknowledged: your-ca-gateway.example.com\n" in text
        assert text.count("    live_gateway_acknowledged:") == 0

    def test_the_build_hangs_no_note_of_its_own_in_the_config(self, tmp_path):
        """No `Written by \\`osprey build\\`` prose survives anywhere in the render."""
        assert "# Written by `osprey build`" not in _inject_standin(tmp_path)

    def test_the_target_switch_block_is_identical_with_and_without_a_stand_in(self, tmp_path):
        """The one assertion the whole section reduces to.

        A stand-in changes ``services`` and ``deployed_services``; the
        target-switch block — prose, wrapped inline comments, commented example
        and all — is byte-identical to the render that deploys no stand-in.
        """
        template = _render_control_assistant_template()

        def _block(text: str) -> str:
            head = "  target_switch:\n"
            return head + text.split(head, 1)[1].split("\n\n", 1)[0]

        with_standin = _block(_inject_standin(tmp_path, template))
        without = _block(_inject_standin(tmp_path, template, port=None))

        assert with_standin == without
        assert _block(template) == with_standin

    def test_the_wrapped_inline_comments_are_not_torn_in_half(self, tmp_path):
        """``probe_interval_s``'s comment wraps; both lines are ONE ruamel token."""
        text = _inject_standin(tmp_path)

        assert (
            "    probe_interval_s: 30    # Seconds between background reachability probes of\n"
            "                            # every target's gateways\n"
        ) in text
        assert (
            "    drain_timeout_s: 5      # Seconds in-flight operations get to finish on the\n"
            "                            # old target before it is torn down regardless\n"
        ) in text

    def test_a_config_with_no_target_switch_block_gains_none(self, tmp_path):
        """The block is not conjured to hold a key the build no longer writes."""
        config = pyyaml.safe_load(_inject_standin(tmp_path, CONFIG_WITHOUT_VA_BLOCK))

        assert "target_switch" not in config["control_system"]

    def test_an_operator_authored_acknowledgment_is_left_exactly_as_written(self, tmp_path):
        """It names their own machine, and this build has no opinion about it."""
        template = _render_control_assistant_template().replace(
            "    # live_gateway_acknowledged: your-ca-gateway.example.com\n",
            "    live_gateway_acknowledged: cagw.example.com   # ours, checked\n",
            1,
        )

        text = _inject_standin(tmp_path, template)

        assert _target_switch(text)["live_gateway_acknowledged"] == "cagw.example.com"
        # The value AND the comment the operator wrote beside it.
        assert "    live_gateway_acknowledged: cagw.example.com   # ours, checked\n" in text

    def test_a_rebuild_reproduces_the_file_byte_for_byte(self, tmp_path):
        once = _inject_standin(tmp_path)
        _inject_va(VAConfig(port=5064, live_standin=STANDIN_PORT), tmp_path)
        twice = (tmp_path / "config.yml").read_text(encoding="utf-8")

        assert twice == once


class TestTheStandinPostBuildHint:
    """What the build TELLS the operator, now that it edits nothing for them.

    The hint is the only place the third target is explained at build time, so
    it has to name both switches by the command that performs them — and it may
    not still describe going live as a profile edit and a rebuild.
    """

    def _hint(self, tmp_path, caplog) -> str:
        with caplog.at_level(logging.DEBUG, logger="build"):
            _inject_standin(tmp_path)
        return caplog.text

    def test_the_hint_names_the_command_that_goes_live(self, tmp_path, caplog):
        assert "control_target_set live" in self._hint(tmp_path, caplog)

    def test_the_hint_names_the_stand_in_as_its_own_target(self, tmp_path, caplog):
        text = self._hint(tmp_path, caplog)

        assert "control_target_set standin" in text
        assert "`standin` target" in text

    def test_the_hint_names_how_to_start_a_deployment_on_the_stand_in(self, tmp_path, caplog):
        assert "osprey set connector=live_standin" in self._hint(tmp_path, caplog)

    def test_the_hint_no_longer_asks_for_a_profile_edit_and_a_rebuild(self, tmp_path, caplog):
        """The old ritual — delete ``live_standin``, repoint gateways, rewrite the key."""
        text = self._hint(tmp_path, caplog)

        assert "Going live" not in text
        assert "live_gateway_acknowledged" not in text
        assert "delete `virtual_accelerator.live_standin`" not in text

    def test_no_stand_in_prints_no_stand_in_hint(self, tmp_path, caplog):
        with caplog.at_level(logging.DEBUG, logger="build"):
            _inject_standin(tmp_path, port=None)

        assert "control_target_set" not in caplog.text
