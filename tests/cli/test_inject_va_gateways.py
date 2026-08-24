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
