"""The panel layout: every row lands on one tile in one band, said once."""

from __future__ import annotations

from typing import Any

from osprey.health.layout import COMPONENTS, LAYERS, RULES, build_layers
from osprey.health.models import NOT_USED, CheckResult, Status


def _row(name: str, category: str, status: str = "ok", message: str = "m", **kw: Any) -> dict:
    return CheckResult(name, category, Status(status), message, **kw).to_dict()


def _components(layers: list[dict]) -> dict[str, dict]:
    return {c["id"]: c for layer in layers for c in layer["components"]}


def test_every_rule_names_a_known_component_and_layer():
    layer_ids = {layer.id for layer in LAYERS}
    assert {c.layer for c in COMPONENTS.values()} <= layer_ids
    assert {r.component for r in RULES if r.component is not None} <= set(COMPONENTS)


def test_rows_are_annotated_and_grouped_into_bands_in_order():
    rows = [
        _row("disk_space", "file_system", message="Disk 30% full (1308.4 GB free)"),
        _row("graphdb_connection", "graphdb", message="reachable (bolt://h:10802)", latency_ms=11),
        _row("hooks.osprey_limits", "safety_chain", value="registered · sha = render"),
    ]
    layers = build_layers(rows, {})

    assert [layer["id"] for layer in layers] == ["chain", "data", "host"]
    graph = _components(layers)["graph-store"]
    assert graph["fact"] == "bolt :10802"
    assert graph["rows"] == [1]
    assert rows[1]["label"] == "bolt connection"
    assert rows[1]["method"] == "bolt connect + RETURN 1"
    assert rows[1]["display"] == "open · 11 ms"
    assert _components(layers)["disk"]["fact"] == "30 % used"


def test_a_faulty_tile_carries_its_issue_text():
    rows = [
        _row("agent_record_backend", "agent_record", value="clickhouse · 76 entries"),
        _row("agent_record_transcripts.alice", "agent_record", value="readable"),
        _row(
            "agent_record_transcripts.logbook",
            "agent_record",
            "warning",
            value="root unreadable (ENOENT)",
            fix="docker compose exec agent-record ls /data/transcripts",
        ),
    ]
    tile = _components(build_layers(rows, {}))["telemetry-store"]

    assert tile["status"] == "warning"
    assert tile["issue"] == "transcripts logbook: root unreadable (ENOENT)"
    assert tile["header"] == "agent record · clickhouse · 76 entries"
    transcripts = next(g for g in tile["groups"] if g["id"] == "transcripts")
    assert transcripts["rows"] == [1, 2]
    assert transcripts["status"] == "warning"
    assert transcripts["fix"] == "docker compose exec agent-record ls /data/transcripts"


def test_many_faults_in_one_group_are_counted_once():
    rows = [
        _row(f"agent_record_transcripts.{n}", "agent_record", "warning", value="unreadable")
        for n in ("a", "b", "c")
    ] + [_row("agent_record_transcripts.d", "agent_record", value="readable")]
    tile = _components(build_layers(rows, {}))["telemetry-store"]

    assert tile["issue"] == "transcripts: 3 / 4 unreadable"


def test_providers_split_by_role_and_unused_ones_are_set_apart():
    config = {"claude_code": {"provider": "agent"}, "ariel": {"m": {"provider": "emb"}}}
    rows = [
        _row("agent", "providers", message="API accessible and authenticated"),
        _row("emb", "providers", message="Accessible"),
        _row("spare", "providers", "skip", value=NOT_USED),
    ]
    tiles = _components(build_layers(rows, config))

    assert tiles["model-provider"]["fact"] == "agent"
    assert tiles["model-provider"]["unused"] == [2]
    assert tiles["model-provider"]["status"] == "ok"
    assert tiles["embed-vision"]["rows"] == [1]


def test_machine_rows_are_annotated_but_not_drawn():
    rows = [_row("simulator_va_SR", "simulator", value="ok")]
    assert build_layers(rows, {}) == []
    assert rows[0]["layer"] == "machine"


def test_an_undeclared_category_gets_its_own_facility_tile():
    rows = [_row("beamline_check", "beamline_x")]
    layers = build_layers(rows, {})

    assert layers[-1]["id"] == "facility"
    tile = layers[-1]["components"][0]
    assert tile["id"] == "facility.beamline_x"
    assert tile["name"] == "beamline x"


def test_mcp_tile_is_drawn_from_the_configured_servers():
    tile = _components(build_layers([], {}, mcp_configured=["controls", "workspace"]))[
        "mcp-servers"
    ]
    assert tile["fact"] == "2 servers"
    assert tile["status"] == "ok"


def test_used_by_of_folded_rows_reaches_the_tile():
    rows = [_row("graphdb_connection", "graphdb", used_by=("channel finder", "graph MCP server"))]
    assert _components(build_layers(rows, {}))["graph-store"]["used_by"] == [
        "channel finder",
        "graph MCP server",
    ]


def test_panels_group_by_panel_and_count_up_panels():
    rows = [
        _row("web_panels.ariel", "web_panels", value="up", message="LOGBOOK: reachable"),
        _row("web_panels.canopy", "web_panels", "skip", message="CANOPY: started — not probed"),
        _row("web_panels.okf", "web_panels", "warning", message="KNOWLEDGE: unreachable"),
    ]
    tile = _components(build_layers(rows, {}))["panels"]

    assert [g["name"] for g in tile["groups"]] == ["Logbook", "Canopy", "Knowledge"]
    assert tile["issue"] == "Knowledge HTTP /health: unreachable"
    assert rows[1]["display"] == "started — not probed"
