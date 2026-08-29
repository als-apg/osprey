"""Unit tests for the frozen host-port layout.

:mod:`osprey.port_layout` is the one place a framework host port is spelled, so
these tests pin the numbers themselves — every slot's port at the default base,
the block a deployment reserves, and the band each per-user family owns. They
also hold the layout and the companion-server registry against each other: the
registry names port families and the layout gives them offsets, and neither half
can be edited without the other noticing.

Two properties are structural rather than numeric and are worth naming:

* Per-user families sit at least ``INDEX_MAX + 1`` apart, which is what lets the
  user index be read straight off the port.
* The module is a stdlib-only leaf. Everything from the registry to the docs
  extension imports it, so a single ``osprey`` import here would put a cycle in
  reach of half the codebase. That is checked on a fresh interpreter, not on the
  already-populated ``sys.modules`` of the test session.
"""

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from osprey.port_layout import (
    BLOCK_SIZE,
    DEFAULT_PORT_BASE,
    INDEX_MAX,
    LAYOUT,
    PORT_BASE_CONFIG_KEY,
    SLOTS_BY_NAME,
    VA_STANDIN_MAX,
    WORKER_MAX,
    PortSlot,
    block_range,
    default_port,
    layout_ports,
    resolve_port_base,
)
from osprey.registry.web import FRAMEWORK_WEB_SERVERS, framework_web_port_default

#: Every slot's port at :data:`DEFAULT_PORT_BASE`, spelled out rather than
#: derived: a test that recomputes the offsets it is checking would pass on a
#: typo. Indexed slots appear at the first index of their band, which is what
#: :func:`layout_ports` emits — worker 1 for the dispatch band, index 0 for the
#: rest.
EXPECTED_PORTS_AT_DEFAULT_BASE = {
    "nginx": 10000,
    "auth": 10001,
    "dispatcher": 10010,
    "worker": 10011,
    "openobserve": 10050,
    "qmd": 10060,
    "tiled": 10070,
    "bluesky_web": 10071,
    "bluesky": 10080,
    "bluesky_second_lane": 10081,
    "va_standin": 10090,
    "web": 10100,
    "artifact": 10200,
    "ariel": 10300,
    "lattice": 10400,
    "channel_finder": 10500,
    "okf": 10600,
    "system_health": 10700,
    "postgres": 10800,
    "mongo": 10801,
    "graphdb_bolt": 10802,
    "graphdb_http": 10803,
    "facility": 10900,
}

#: The per-user families and the first port of each band at the default base.
#: Pinned by name so that renaming a family in the registry without moving its
#: band — or moving a band without telling the registry — fails here.
EXPECTED_FAMILY_PORTS = {
    "web": 10100,
    "artifact": 10200,
    "ariel": 10300,
    "lattice": 10400,
    "channel_finder": 10500,
    "okf": 10600,
    "system_health": 10700,
}

#: ``src`` on the test tree, for the child interpreter that checks import purity.
_SRC = str(Path(__file__).resolve().parents[1] / "src")


def _family_slots() -> tuple[PortSlot, ...]:
    """Return the per-user family slots, in layout order.

    Returns:
        Every :class:`~osprey.port_layout.PortSlot` whose ``per_index`` flag is
        set — that is, the registry web-server families, and not the other
        indexed slots (workers, extra VA instances, the facility band).
    """
    return tuple(entry for entry in LAYOUT if entry.per_index)


def _registry_port_families() -> set[str]:
    """Return the port-family names the layout has to have a slot for.

    Returns:
        Every companion web server's family — ``port_family`` when it is set,
        otherwise the registry key it is filed under — plus ``web``, the web
        terminal's own family, which is not a companion server and so has no
        registry entry.
    """
    return {"web"} | {
        definition.port_family or key for key, definition in FRAMEWORK_WEB_SERVERS.items()
    }


def _fresh_import_modules() -> list[str]:
    """Return the non-stdlib modules a fresh import of ``port_layout`` pulls in.

    Runs in a child interpreter, because this session imported the module long
    ago and its own ``sys.modules`` no longer shows what the import costs. The
    parent package ``__init__`` runs for any submodule import and pulls in
    ``osprey`` and ``osprey.version``; those two and the module itself are the
    only ``osprey`` names an import is allowed to add.

    Returns:
        The offending module names, sorted — empty when the module is the
        stdlib-only leaf it claims to be.

    Raises:
        AssertionError: If the child interpreter fails to import the module.
    """
    code = (
        "import json, sys;"
        "before = set(sys.modules);"
        "import osprey.port_layout;"
        "allowed = {'osprey', 'osprey.version', 'osprey.port_layout'};"
        "delta = set(sys.modules) - before - allowed;"
        "print(json.dumps(sorted("
        "m for m in delta if m.split('.')[0] not in sys.stdlib_module_names)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=_SRC),
        check=False,
    )
    assert result.returncode == 0, f"fresh import of osprey.port_layout failed:\n{result.stderr}"
    return json.loads(result.stdout)


class TestLayoutTable:
    """The numbers themselves, and the shape of the table that holds them."""

    def test_every_slot_sits_where_the_layout_says_at_the_default_base(self):
        """``layout_ports`` at the default base matches the pinned table exactly."""
        assert layout_ports(DEFAULT_PORT_BASE) == EXPECTED_PORTS_AT_DEFAULT_BASE

    def test_the_pinned_table_covers_every_slot(self):
        """No slot may be added to the layout without being pinned here."""
        assert set(EXPECTED_PORTS_AT_DEFAULT_BASE) == set(SLOTS_BY_NAME)

    def test_default_port_agrees_with_layout_ports(self):
        """The single-slot lookup and the whole-table render cannot disagree."""
        for name, port in EXPECTED_PORTS_AT_DEFAULT_BASE.items():
            first_index = 1 if name == "worker" else 0
            assert default_port(name, first_index) == port

    def test_the_whole_block_moves_with_the_base(self):
        """One knob moves every port: the table at another base is the table shifted."""
        moved = layout_ports(20000)
        assert moved == {
            name: port + 10000 for name, port in EXPECTED_PORTS_AT_DEFAULT_BASE.items()
        }

    def test_layout_is_in_ascending_offset_order(self):
        """Surfaces group by tier and order by offset, so the table stays sorted."""
        offsets = [entry.offset for entry in LAYOUT]
        assert offsets == sorted(offsets)

    def test_every_slot_offset_is_inside_the_block(self):
        """A slot past the block would publish a port another deployment owns."""
        for entry in LAYOUT:
            assert 0 <= entry.offset < BLOCK_SIZE, (
                f"slot {entry.name!r} is at offset {entry.offset}, outside the "
                f"{BLOCK_SIZE}-port block a deployment reserves"
            )


class TestFamilyBands:
    """The per-user families: one hundred ports each, index readable off the port."""

    def test_families_are_at_least_a_full_band_apart(self):
        """Consecutive families are ``INDEX_MAX + 1`` apart or more."""
        families = _family_slots()
        for lower, upper in zip(families[:-1], families[1:], strict=True):
            gap = upper.offset - lower.offset
            assert gap >= INDEX_MAX + 1, (
                f"family {lower.name!r} (+{lower.offset}) and {upper.name!r} "
                f"(+{upper.offset}) are only {gap} ports apart, but a family holds "
                f"indices 0..{INDEX_MAX}, so user {gap} of {lower.name!r} would land "
                f"on user 0 of {upper.name!r}"
            )

    def test_no_other_slot_sits_inside_a_family_band(self):
        """A singleton slot inside a band would collide with some user's panel."""
        for family in _family_slots():
            top = family.offset + INDEX_MAX
            for other in LAYOUT:
                if other.name == family.name:
                    continue
                assert not family.offset <= other.offset <= top, (
                    f"slot {other.name!r} is at +{other.offset}, inside the band "
                    f"{family.name!r} owns (+{family.offset}..+{top}); it would collide "
                    f"with user {other.offset - family.offset} of that family"
                )

    def test_family_first_ports_are_pinned_by_name(self):
        """The family-to-port binding is what the docs and deployments quote."""
        actual = {entry.name: default_port(entry.name, 0) for entry in _family_slots()}
        assert actual == EXPECTED_FAMILY_PORTS

    def test_a_user_index_reads_off_the_port(self):
        """User *i* of a family is at the family's first port plus *i*."""
        for name, first in EXPECTED_FAMILY_PORTS.items():
            assert default_port(name, INDEX_MAX) == first + INDEX_MAX
            with pytest.raises(ValueError, match=f"{name!r}"):
                default_port(name, INDEX_MAX + 1)


class TestRegistryReconciliation:
    """The registry names the families; the layout gives them offsets."""

    def test_every_family_has_a_slot_and_every_slot_has_a_family(self):
        """The two sets are equal, and the failure says which side is short."""
        families = _registry_port_families()
        slotted = {entry.name for entry in _family_slots()}
        unslotted = sorted(families - slotted)
        unregistered = sorted(slotted - families)
        assert families == slotted, (
            "osprey.registry.web and osprey.port_layout disagree about port "
            f"families. Registry families with no per-index LAYOUT slot: "
            f"{unslotted or 'none'}. Per-index LAYOUT slots no registry entry "
            f"claims: {unregistered or 'none'}. Either add the slot to LAYOUT or "
            "point the registry entry at an existing family with port_family."
        )

    def test_registry_defaults_are_the_layout_defaults(self):
        """The registry cannot drift from the layout: it reads its ports from it."""
        for key, definition in FRAMEWORK_WEB_SERVERS.items():
            expected = default_port(definition.port_family or key, 0, DEFAULT_PORT_BASE)
            assert framework_web_port_default(key) == expected, (
                f"companion server {key!r} defaults to "
                f"{framework_web_port_default(key)}, but its family "
                f"{definition.port_family or key!r} is at {expected}"
            )

    def test_registry_defaults_are_pinned_by_registry_key(self):
        """The port each companion server actually lands on, spelled out."""
        assert {key: framework_web_port_default(key) for key in FRAMEWORK_WEB_SERVERS} == {
            "artifact": 10200,
            "ariel": 10300,
            "channel_finder": 10500,
            "lattice_dashboard": 10400,
            "okf": 10600,
            "system_health": 10700,
        }

    def test_registry_defaults_follow_a_moved_base(self):
        """A deployment on another base is never described in default-base terms."""
        for key in FRAMEWORK_WEB_SERVERS:
            assert framework_web_port_default(key, base=20000) == (
                framework_web_port_default(key) + 10000
            )


class TestBlockRange:
    """The span of ports a deployment reserves."""

    def test_block_is_a_thousand_ports_from_the_base(self):
        """The pair is inclusive, so the last port is ``base + BLOCK_SIZE - 1``."""
        assert block_range(DEFAULT_PORT_BASE) == (10000, 10999)
        assert block_range(20000) == (20000, 20999)

    def test_block_holds_every_slot(self):
        """Every port the layout publishes is inside the block it advertises."""
        first, last = block_range(DEFAULT_PORT_BASE)
        for name, port in EXPECTED_PORTS_AT_DEFAULT_BASE.items():
            assert first <= port <= last, f"{name} at {port} is outside {first}-{last}"

    def test_indexed_bands_stay_inside_the_block(self):
        """The top of each open-ended band is in-block too, not only its first port."""
        _, last = block_range(DEFAULT_PORT_BASE)
        tops = {
            "worker": default_port("worker", WORKER_MAX),
            "va_standin": default_port("va_standin", VA_STANDIN_MAX),
            "system_health": default_port("system_health", INDEX_MAX),
            "facility": default_port("facility", INDEX_MAX),
        }
        for name, port in tops.items():
            assert port <= last, f"the top of the {name} band ({port}) runs past {last}"

    def test_block_range_refuses_a_base_it_cannot_span(self):
        """A base validated nowhere else is still validated here."""
        with pytest.raises(ValueError, match=PORT_BASE_CONFIG_KEY):
            block_range(65000)


class TestResolvePortBase:
    """Reading ``deployment.port_base`` out of a rendered config."""

    @pytest.mark.parametrize(
        "config",
        [
            None,
            {},
            {"deployment": {}},
            {"deployment": {"port_base": "x"}},
            {"deployment": None},
            {"deployment": {"port_base": None}},
        ],
        ids=["none", "empty", "no-key", "not-an-int", "null-block", "null-value"],
    )
    def test_absent_or_malformed_is_the_default(self, config):
        """A config that never mentions the key is the common case, not an error."""
        assert resolve_port_base(config) == DEFAULT_PORT_BASE

    def test_a_configured_base_is_honoured(self):
        """An in-range base comes back unchanged and without complaint."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert resolve_port_base({"deployment": {"port_base": 20000}}) == 20000

    def test_a_privileged_base_is_refused(self):
        """1000 would need root to bind, so it is an intent that cannot be honoured."""
        with pytest.raises(ValueError) as excinfo:
            resolve_port_base({"deployment": {"port_base": 1000}})
        message = str(excinfo.value)
        assert PORT_BASE_CONFIG_KEY in message
        assert "1000" in message
        assert "1024" in message

    def test_a_base_whose_block_overruns_the_port_range_is_refused(self):
        """65000 + 999 is past 65535, so the top of the block could not exist."""
        with pytest.raises(ValueError) as excinfo:
            resolve_port_base({"deployment": {"port_base": 65000}})
        message = str(excinfo.value)
        assert PORT_BASE_CONFIG_KEY in message
        assert "65999" in message

    def test_a_base_in_the_ephemeral_range_warns_but_resolves(self):
        """Advisory, not a refusal: the operator may know their local port range."""
        with pytest.warns(UserWarning, match=r"block \(40000-40999\) overlaps"):
            assert resolve_port_base({"deployment": {"port_base": 40000}}) == 40000


class TestImportPurity:
    """The module is a leaf, and everything downstream depends on it staying one."""

    def test_port_layout_imports_only_stdlib(self):
        """A fresh import adds no third-party module and no further ``osprey`` one."""
        offenders = _fresh_import_modules()
        assert offenders == [], (
            "osprey.port_layout is imported by the registry, the template manager, "
            "the build, the preflight and the docs extension, so it must import "
            f"nothing but the stdlib. A fresh import pulled in: {offenders}"
        )
