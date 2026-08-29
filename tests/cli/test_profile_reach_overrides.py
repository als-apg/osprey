"""What an attached render is told about the services its host deploys.

A web-terminal persona is an attached project: it scaffolds no services, so
the injectors that write ``services.<name>`` blocks never run for it — yet its
in-container clients resolve their endpoints from exactly those blocks, and
bottom out in a compiled-in default (or refuse at first use) when a block is
absent. The build therefore PROJECTS the client-facing facts the Reach
Contract registry declares from the hosting deployment's render into the
attached render. These tests pin the projection's rules — gated on the
attached render's own switches, panel keys on the profile's selection, nothing
without a host — the refusal of a profile that spells a projected key by hand,
and the client resolvers' half of the contract.
"""

from __future__ import annotations

import copy

import pytest

from osprey.bluesky_bridge_connection import (
    DEFAULT_BRIDGE_PORT,
    DEFAULT_BRIDGE_URL,
    bridge_url_from_config,
    resolve_bridge_url,
)
from osprey.cli.build_profile_reach import (
    attached_render_overrides,
    orphan_panel_fragments,
    reach_override_errors,
    selected_panel_errors,
)
from osprey.cli.build_profile_schema import BlueskyConfig
from osprey.deployment.reach import REACH_CONTRACTS, project_attached_overrides

HOST = {
    "services": {
        "qmd": {"port": 18180, "interval": 30},
        "graphdb": {"port_host": 17687, "heap_max_size": "1G"},
        "postgresql": {"port_host": 15432, "username": "ariel", "database_name": "ariel"},
        "openobserve": {"port": 15080, "retention_days": 14},
        "bluesky": {"port": 18090, "tiled_port": 18091},
        "virtual_accelerator": {"port": 15064},
    },
    "web": {
        "panels": {
            "events": {
                "url": "${EVENT_DISPATCHER_URL:-http://localhost:18020}",
                "path": "/dashboard",
                "label": "EVENTS",
                "health_endpoint": "/health",
            },
            "bluesky": {
                "url": "${BLUESKY_WEB_URL:-http://localhost:18095}",
                "path": "/bluesky/",
                "label": "BLUESKY",
            },
        }
    },
}

#: An attached render that switches every consumer on.
#:
#: The virtual accelerator's consumer is keyed on the ``va`` target RESOLVING
#: and its connector block being configured, not on ``control_system.type`` —
#: a render is a client of the simulator because it carries the block a VA
#: connector is built from, whichever target the deployment boots on. So the
#: block is spelled here the way the build writes it (``address``, the
#: connector's own leaf), and a fixture that named only the type would switch
#: the consumer off and quietly drop the port from the projection.
ALL_ON = {
    "ariel": {"database": {}, "search_modules": {"hybrid": {"enabled": True}}},
    "claude_code": {
        "servers": {"bluesky": {"enabled": True}},
        "telemetry": {"enabled": True, "backend": "openobserve"},
    },
    "control_system": {
        "type": "virtual_accelerator",
        "connector": {"virtual_accelerator": {"gateways": {"read_only": {"address": "localhost"}}}},
    },
}


# ---------------------------------------------------------------------------
# The projection
# ---------------------------------------------------------------------------


def test_every_live_consumer_is_told_its_hosts_fact():
    projected = project_attached_overrides(HOST, ALL_ON, selected_panels=("events", "bluesky"))
    assert projected == {
        "services.qmd.port": 18180,
        "services.graphdb.port_host": 17687,
        "services.postgresql.port_host": 15432,
        "services.postgresql.username": "ariel",
        "services.postgresql.database_name": "ariel",
        "services.openobserve.port": 15080,
        "services.bluesky.port": 18090,
        "web.panels.bluesky.url": "${BLUESKY_WEB_URL:-http://localhost:18095}",
        "web.panels.bluesky.path": "/bluesky/",
        "web.panels.bluesky.label": "BLUESKY",
        "web.panels.events.url": "${EVENT_DISPATCHER_URL:-http://localhost:18020}",
        "web.panels.events.path": "/dashboard",
        "web.panels.events.label": "EVENTS",
        "web.panels.events.health_endpoint": "/health",
        "services.virtual_accelerator.port": 15064,
    }


def test_a_two_lane_host_projects_both_lanes_port_and_target():
    """``bluesky.second_lane`` renders a second bridge and stamps ``target`` on
    both lane blocks; a persona told only lane 1's port would be a single-lane
    render whose session, switched to the other machine, is refused rather
    than routed. Port AND target: the lane resolver picks the active lane by
    matching each lane's declared target against the session's."""
    host = copy.deepcopy(HOST)
    host["services"]["bluesky"]["target"] = "va"
    host["services"]["bluesky_live"] = {
        "path": "./services/bluesky",
        "port": 18190,
        "target": "live",
        "ca_name_servers": "${EPICS_CA_NAME_SERVERS:?set it}",
    }
    projected = project_attached_overrides(host, ALL_ON)
    lane_keys = {k: v for k, v in projected.items() if ".bluesky" in k}
    assert lane_keys == {
        "services.bluesky.port": 18090,
        "services.bluesky.target": "va",
        "services.bluesky_live.port": 18190,
        "services.bluesky_live.target": "live",
    }


def test_a_single_lane_host_projects_no_lane_target():
    """``target`` is written only on a two-lane deploy; a single-lane persona
    stays byte-for-byte the pre-lane render, which is that resolver's
    single-lane answer."""
    projected = project_attached_overrides(HOST, ALL_ON)
    assert not any(key.endswith(".target") for key in projected)
    assert not any("bluesky_va" in key or "bluesky_live" in key for key in projected)


def test_a_persona_with_the_bluesky_server_off_is_told_no_lane():
    host = copy.deepcopy(HOST)
    host["services"]["bluesky_va"] = {"port": 18190, "target": "va"}
    off = copy.deepcopy(ALL_ON)
    off["claude_code"]["servers"]["bluesky"] = {"enabled": False}
    projected = project_attached_overrides(host, off)
    assert not any("bluesky" in key for key in projected)


def test_only_the_client_facing_keys_are_projected():
    """The host's ``interval``, ``heap_max_size``, ``retention_days``,
    ``tiled_port`` describe the service it RUNS; a client needs none of them."""
    projected = project_attached_overrides(HOST, ALL_ON, selected_panels=("events", "bluesky"))
    assert not any(
        key.endswith((".interval", ".heap_max_size", ".retention_days", ".tiled_port"))
        for key in projected
    )


def test_a_consumer_switched_off_is_told_nothing():
    """A fact projected into a render with no consumer for it would grow a
    surface — a ``services.graphdb`` block makes the graph server render."""
    off = {
        "ariel": {"database": {}, "search_modules": {"hybrid": {"enabled": False}}},
        "claude_code": {
            "servers": {"graph": {"enabled": False}, "bluesky": {"enabled": False}},
            "telemetry": {"enabled": False},
        },
        "control_system": {"type": "mock"},
    }
    projected = project_attached_overrides(HOST, off)
    assert set(projected) == {
        # The ARIEL database: the ``ariel:`` block is present, so its DSN is
        # still derived — and must be derived from the host's Postgres.
        "services.postgresql.port_host",
        "services.postgresql.username",
        "services.postgresql.database_name",
    }


def test_panel_keys_follow_the_profiles_panel_selection():
    """A custom panel exists in a render only through its ``web.panels`` keys, so
    the rendered config cannot say whether the profile selected it — the
    ``web_panels`` list can, and does."""
    assert not any(
        key.startswith("web.panels.") for key in project_attached_overrides(HOST, ALL_ON)
    )
    events_only = project_attached_overrides(HOST, ALL_ON, selected_panels=("events",))
    assert {k for k in events_only if k.startswith("web.panels.")} == {
        "web.panels.events.url",
        "web.panels.events.path",
        "web.panels.events.label",
        "web.panels.events.health_endpoint",
    }


def test_a_host_that_dropped_a_service_projects_nothing_for_it():
    host = {"services": {"postgresql": {"port_host": 5432}}}
    projected = project_attached_overrides(host, ALL_ON, selected_panels=("events",))
    assert projected == {"services.postgresql.port_host": 5432}


def test_no_host_projects_nothing():
    assert project_attached_overrides(None, ALL_ON, selected_panels=("events",)) == {}
    assert project_attached_overrides({}, ALL_ON) == {}


def test_the_build_entry_point_is_the_registrys_projection():
    assert attached_render_overrides(HOST, ALL_ON, selected_panels=("events",)) == (
        project_attached_overrides(HOST, ALL_ON, selected_panels=("events",))
    )


def test_the_projected_keys_are_the_ones_the_injectors_and_remedies_name():
    """The attached render and the deploying render must name each fact under
    one spelling, or a client following the deploying spelling reads nothing
    in a persona. Pinned against the deploy preflight's own remedy table."""
    from osprey.deployment.host_ports import _SERVICE_REMEDY_KEYS

    projected = {p.key for c in REACH_CONTRACTS.values() for p in c.projected}
    for compose_name, key in _SERVICE_REMEDY_KEYS.items():
        if compose_name in (
            "dispatch-worker",
            "tiled",
            "bluesky-web",
            "event-dispatcher",
            "mongodb",
        ):
            continue  # not a client-facing key: the worker base, a port only the bridge dials, panel URLs, the archive's own path
        assert key in projected, (
            f"{compose_name}: {key} is a published port no attached render is told"
        )


# ---------------------------------------------------------------------------
# A hand-spelled duplicate is refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [
        {"services.bluesky.port": 1},
        {"services.bluesky": {"port": 1}},
    ],
    ids=["dotted", "nested-under-prefix"],
)
def test_a_spelling_that_contradicts_the_projection_is_refused(config):
    """One fact with two homes is free to disagree, and the disagreeing case
    is the silent one — so it is the one refused, naming both values."""
    errors = reach_override_errors(config, {"services.bluesky.port": 18090})
    assert len(errors) == 1
    assert "services.bluesky.port" in errors[0]
    assert "18090" in errors[0] and "1" in errors[0]


@pytest.mark.parametrize(
    "config",
    [
        {"services.bluesky.port": 18090},
        {"services.bluesky": {"port": 18090}},
    ],
    ids=["dotted", "nested-under-prefix"],
)
def test_a_spelling_that_agrees_is_the_same_fact(config):
    """A persona profile inherits the hosting profile's ``config:`` dotted
    keys, so a host that moved its bridge port there spells it in every
    persona's merged config — as the host's own value. Refusing that would
    refuse every moved port; the spelling and the projection agree."""
    assert reach_override_errors(config, {"services.bluesky.port": 18090}) == []


def test_nothing_projected_refuses_nothing():
    assert reach_override_errors({"services.bluesky.port": 1}, {}) == []
    assert reach_override_errors({}, {"services.bluesky.port": 18090}) == []


# ---------------------------------------------------------------------------
# The client's half: the bridge URL follows the published port
# ---------------------------------------------------------------------------


def test_bridge_url_follows_the_published_port():
    assert bridge_url_from_config({"services": {"bluesky": {"port": 18090}}}) == (
        "http://127.0.0.1:18090"
    )


def test_bridge_url_config_url_beats_the_published_port():
    config = {"bluesky": {"bridge_url": "http://bridge:9/"}, "services": {"bluesky": {"port": 1}}}
    assert bridge_url_from_config(config) == "http://bridge:9"


@pytest.mark.parametrize("config", [None, {}, {"bluesky": {}}, {"services": {"bluesky": {}}}])
def test_bridge_url_falls_back_to_the_default(config):
    assert bridge_url_from_config(config) == DEFAULT_BRIDGE_URL
    assert DEFAULT_BRIDGE_URL.endswith(f":{DEFAULT_BRIDGE_PORT}")


def test_default_port_is_the_profiles_default():
    """The compiled-in fallback and the schema default are one number, so a
    profile that declares nothing and a client that reads nothing agree."""
    assert DEFAULT_BRIDGE_PORT == BlueskyConfig().port


def test_resolve_bridge_url_reads_the_config_when_the_env_is_empty(monkeypatch, tmp_path):
    """The MCP server definition exports ``${BLUESKY_BRIDGE_URL:-}`` — an unset
    variable reaches the server EMPTY and must count as no override."""
    import yaml

    from osprey.utils.workspace import reset_config_cache

    (tmp_path / "config.yml").write_text(
        yaml.safe_dump({"services": {"bluesky": {"port": 18090}}}), encoding="utf-8"
    )
    monkeypatch.setenv("BLUESKY_BRIDGE_URL", "")
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))
    reset_config_cache()
    try:
        assert resolve_bridge_url() == "http://127.0.0.1:18090"
    finally:
        reset_config_cache()


def test_a_non_mapping_config_block_spells_nothing():
    """A profile with no ``config:`` at all (None) contradicts no projection."""
    assert reach_override_errors(None, {"services.qmd.port": 8180}) == []


@pytest.mark.parametrize(
    "spelling",
    [
        {"services.bluesky.port": 9999},
        {"services.bluesky": {"port": 9999}},
        {"services": {"bluesky.port": 9999}},
        {"services": {"bluesky": {"port": 9999}}},
    ],
)
def test_every_spelling_of_a_contradicting_pin_is_refused(spelling):
    """Dotted, prefix-over-mapping, fully nested and mixed: each is legal YAML
    for the same rendered leaf, so each is found — a spelling missed here is
    a pin the projection would silently overwrite."""
    (error,) = reach_override_errors(spelling, {"services.bluesky.port": 18090})
    assert "9999" in error and "18090" in error


def test_every_spelling_of_an_agreeing_pin_is_allowed():
    for spelling in (
        {"services.bluesky.port": 18090},
        {"services": {"bluesky": {"port": 18090}}},
    ):
        assert reach_override_errors(spelling, {"services.bluesky.port": 18090}) == []


def test_a_selected_tab_told_no_address_is_refused():
    """``web_panels: [events]`` is a consumer the profile switches on; when
    the host it was told about runs no dispatcher, the render would simply
    lack the tab. Named instead, with the service and the key."""
    rendered = {"web": {"panels": {"bluesky": {"url": "http://localhost:10071"}}}}
    (error,) = selected_panel_errors(["events", "bluesky", "okf"], rendered, told_by="the host")
    assert "web.panels.events.url" in error
    assert "event_dispatcher" in error
    assert "the host" in error
    assert selected_panel_errors(["bluesky"], rendered, told_by="the host") == []
    assert selected_panel_errors(["okf"], {}, told_by="the host") == []


def test_an_inherited_fragment_of_an_unselected_tab_is_dropped():
    """A host that pins ``web.panels.events.path`` hands every persona the
    fragment; a persona that never selected the tab drops it, one that did
    keeps it (the projection supplies the url), and a fragment that names a
    url is a tab in its own right and stays."""
    rendered = {
        "web": {
            "panels": {
                "events": {"path": "/custom-route"},
                "bluesky": {"url": "http://localhost:10071", "path": "/bluesky/"},
                "okf": {"enabled": True},
            }
        }
    }
    assert orphan_panel_fragments([], rendered) == ["web.panels.events"]
    assert orphan_panel_fragments(["events"], rendered) == []
    assert orphan_panel_fragments(["okf"], {"web": {"panels": {"okf": {"enabled": True}}}}) == []
