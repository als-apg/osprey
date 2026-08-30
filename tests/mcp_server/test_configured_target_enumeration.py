"""Enumerators walk the targets a deployment configures, not the vocabulary.

``osprey_connectors.types.CONTROL_TARGETS`` is the list of machines that *can*
exist — it grew a third name when the live stand-in became a target of its own.
Anything that enumerates targets *for a deployment* has to walk
:func:`~osprey_connectors.types.configured_targets` instead, or widening the
vocabulary silently grows a ``standin`` row, probe and posture on every
deployment that stands no soft IOC up: a machine that is not there, described as
if it were.

Pinned here for the two enumerators an operator sees directly — the
``control_target`` roster and the background endpoint prober — because the
regression they guard against is invisible in a three-target deployment, where
both spellings agree.
"""

from __future__ import annotations

from osprey.mcp_server.control_system.endpoint_prober import EndpointProber
from osprey.mcp_server.control_system.tools.control_target import target_rows

_EPICS_BLOCK = {"gateways": {"read_only": {"address": "gw.example.org", "port": 5064}}}
_VA_BLOCK = {"gateways": {"read_only": {"address": "localhost", "port": 5074}}}
_STANDIN_BLOCK = {"gateways": {"read_only": {"address": "localhost", "port": 5075}}}


def _config(*, standin: bool) -> dict:
    """A VA-baseline deployment with the facility's own machine beside it.

    Two targets without *standin*, three with — the same config in both cases
    except for the one block that decides whether the stand-in exists.
    """
    connector = {"epics": dict(_EPICS_BLOCK), "virtual_accelerator": dict(_VA_BLOCK)}
    if standin:
        connector["live_standin"] = dict(_STANDIN_BLOCK)
    return {"control_system": {"type": "virtual_accelerator", "connector": connector}}


def test_two_target_roster_has_no_standin_row():
    """A deployment with no ``live_standin`` block gets no ``standin`` row.

    Absent, not present-and-unavailable: a row saying a stand-in cannot be
    switched to is a claim that there is one, and an operator reading the
    roster would go looking for the machine it names.
    """
    rows = target_rows(_config(standin=False), session_target="va", baseline="va")

    assert list(rows) == ["live", "va"]


def test_three_target_roster_carries_the_standin_row():
    """The same config plus the block: the row appears, in vocabulary order."""
    rows = target_rows(_config(standin=True), session_target="va", baseline="va")

    assert list(rows) == ["live", "va", "standin"]
    assert rows["standin"]["is_baseline"] is False


def test_prober_defaults_skip_an_unconfigured_standin():
    """The background sweep probes configured targets only.

    A prober that defaulted to the vocabulary would open a socket for a
    stand-in gateway this deployment never renders, and publish it as down for
    the life of the process.
    """
    assert EndpointProber(_config(standin=False)).targets == ("live", "va")
    assert EndpointProber(_config(standin=True)).targets == ("live", "va", "standin")


def test_prober_defaults_survive_an_unreadable_config():
    """No section, no crash: the baseline alone is what such a deployment is on."""
    assert EndpointProber({}).targets == ("live",)
    assert EndpointProber(None).targets == ("live",)
