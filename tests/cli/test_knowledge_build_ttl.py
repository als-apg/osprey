"""Tests for ``osprey knowledge build-ttl``.

Covers:
- ``build-ttl --help`` reads as operator documentation and names its inputs,
  its defaults and the verb to run next.
- A run against an explicit channel database writes a Turtle file that parses
  and carries one ``narad_p:hasBinding`` per address.
- The two description inputs: ``--descriptions`` defaults to the ``in_context``
  database sitting beside the hierarchical one, both texts reach the corpus,
  and an absent neighbour asks for both flags by name.
- A graph-mode project is told it ships no channel database by design, rather
  than being sent down the hierarchical-paradigm dead end.
- The direction source is reported, and actually decides: the same database
  emits no ``writesSignal`` under a limits file that grants nothing, and two
  under the PV-grammar fallback.
- Every failure is one legible line and a non-zero exit: no channel database
  where the config points, no config key at all, a ``--limits`` path that
  names nothing, and an address the six-token grammar cannot read.
- A LinkML schema handed to ``--ontology`` is pointed at ``compile-ontology``
  rather than reported as malformed JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from osprey.cli.knowledge_cmd import knowledge

NARAD_P = "https://narad.example.org/property/"
NARAD_SEM = "https://narad.example.org/schema/shared_semantics/"

# The eight addresses the fixture database expands to: two families, two
# devices each, two signals each.
FIXTURE_ADDRESSES = (
    "SR:DIAG:BPM:01:POSITION:X",
    "SR:DIAG:BPM:01:POSITION:Y",
    "SR:DIAG:BPM:02:POSITION:X",
    "SR:DIAG:BPM:02:POSITION:Y",
    "SR:MAG:DIPOLE:01:CURRENT:RB",
    "SR:MAG:DIPOLE:01:CURRENT:SP",
    "SR:MAG:DIPOLE:02:CURRENT:RB",
    "SR:MAG:DIPOLE:02:CURRENT:SP",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _hierarchical_payload() -> dict[str, Any]:
    """A miniature hierarchical database in the shipped file's own schema.

    Two families across two systems so the corpus has more than one device
    class in it, and one ``:SP`` subfield so read and write are both exercised.
    """
    return {
        "_comment": "Miniature hierarchical channel database for tests.",
        "hierarchy": {
            "levels": [
                {"name": "ring", "type": "tree"},
                {"name": "system", "type": "tree"},
                {"name": "family", "type": "tree"},
                {"name": "device", "type": "instances"},
                {"name": "field", "type": "tree"},
                {"name": "subfield", "type": "tree"},
            ],
            "naming_pattern": "{ring}:{system}:{family}:{device}:{field}:{subfield}",
        },
        "tree": {
            "SR": {
                "_description": "Storage Ring (SR).",
                "MAG": {
                    "_description": "Magnet System (MAG).",
                    "DIPOLE": {
                        "_description": "Dipole Bending Magnets.",
                        "DEVICE": {
                            "_expansion": {
                                "_type": "range",
                                "_pattern": "{:02d}",
                                "_range": [1, 2],
                            },
                            "CURRENT": {
                                "_description": "Magnet Excitation Current (Amperes).",
                                "SP": {"_description": "Current Setpoint (read-write)."},
                                "RB": {"_description": "Current Readback (read-only)."},
                            },
                        },
                    },
                },
                "DIAG": {
                    "_description": "Beam Diagnostics (DIAG).",
                    "BPM": {
                        "_description": "Beam Position Monitors.",
                        "DEVICE": {
                            "_expansion": {
                                "_type": "range",
                                "_pattern": "{:02d}",
                                "_range": [1, 2],
                            },
                            "POSITION": {
                                "_description": "Beam position (millimeters).",
                                "X": {"_description": "Horizontal Position (read-only)."},
                                "Y": {"_description": "Vertical Position (read-only)."},
                            },
                        },
                    },
                },
            }
        },
    }


def _ontology_payload() -> dict[str, Any]:
    """A minimal FAMILY-to-class table covering the fixture's two families.

    DIPOLE is deliberately mapped to ``BendingMagnet`` rather than the shipped
    table's ``Dipole``, so a corpus carrying that class proves the given table
    was read instead of the default one.
    """
    return {
        "root": "AcceleratorDevice",
        "family_to_class": {"BPM": "BeamPositionMonitor", "DIPOLE": "BendingMagnet"},
        "classes": {
            "AcceleratorDevice": {"parent": None, "altLabels": []},
            "Magnet": {"parent": "AcceleratorDevice", "altLabels": ["magnet"]},
            "Instrumentation": {"parent": "AcceleratorDevice", "altLabels": ["diagnostic"]},
            "BendingMagnet": {"parent": "Magnet", "altLabels": ["bend", "dipole"]},
            "BeamPositionMonitor": {"parent": "Instrumentation", "altLabels": ["bpm"]},
        },
    }


@pytest.fixture()
def ontology_table(tmp_path: Path) -> Path:
    """Write the minimal ontology table and return its path."""
    path = tmp_path / "ontology.json"
    path.write_text(json.dumps(_ontology_payload()), encoding="utf-8")
    return path


@pytest.fixture()
def channel_db(tmp_path: Path) -> Path:
    """Write the miniature hierarchical database and return its path."""
    path = tmp_path / "hierarchical.json"
    path.write_text(json.dumps(_hierarchical_payload()), encoding="utf-8")
    return path


def _in_context_payload() -> dict[str, Any]:
    """The in-context database of the same miniature machine.

    A flat list of the same eight addresses, each with the one sentence written
    for that channel.  This is what ``--descriptions`` reads.
    """
    return {
        "_metadata": {"version": "1.0", "tier": "test", "total_channels": len(FIXTURE_ADDRESSES)},
        "channels": [
            {
                "channel": address.replace(":", "_"),
                "address": address,
                "description": f"Per-channel sentence for {address}.",
            }
            for address in FIXTURE_ADDRESSES
        ],
    }


@pytest.fixture()
def descriptions_db(tmp_path: Path) -> Path:
    """Write the in-context database somewhere the hierarchical one is not."""
    path = tmp_path / "elsewhere" / "in_context.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_in_context_payload()), encoding="utf-8")
    return path


@pytest.fixture()
def source_tree(tmp_path: Path) -> Path:
    """The OSPREY source tree's own layout: both databases side by side.

    ``tiers/tier3/`` holds ``hierarchical.json`` and ``in_context.json`` as
    neighbours, which is the layout the ``--descriptions`` default is a
    convenience for.  Returns the hierarchical file.
    """
    tier = tmp_path / "tiers" / "tier3"
    tier.mkdir(parents=True)
    (tier / "hierarchical.json").write_text(json.dumps(_hierarchical_payload()), encoding="utf-8")
    (tier / "in_context.json").write_text(json.dumps(_in_context_payload()), encoding="utf-8")
    return tier / "hierarchical.json"


@pytest.fixture()
def readonly_limits(tmp_path: Path) -> Path:
    """A channel limits file that grants no address write access.

    Deliberately at odds with the PV grammar: every address, ``:SP`` included,
    is explicitly non-writable.  A corpus built against this file and one built
    against the grammar therefore differ, which is what makes the reported
    direction source checkable rather than decorative.
    """
    path = tmp_path / "channel_limits.json"
    payload: dict[str, Any] = {
        "_comment": "Test limits: nothing is writable.",
        "defaults": {"writable": True},
    }
    for address in FIXTURE_ADDRESSES:
        payload[address] = {"writable": False}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flat(result: Any) -> str:
    """Rich wraps at the console width, so compare on whitespace-collapsed text."""
    return " ".join(result.output.split())


def _parse(ttl_path: Path) -> Any:
    """Parse *ttl_path* as Turtle.

    rdflib is imported outright rather than skipped past: it is a CI
    dependency (the ``dev`` extra), so an absent one is a broken environment
    and four of these tests would otherwise pass by not running.
    """
    import rdflib

    graph = rdflib.Graph()
    graph.parse(str(ttl_path), format="turtle")
    return graph


def _predicate_count(ttl_path: Path, predicate: str) -> int:
    """Parse *ttl_path* and count triples carrying ``narad_p:<predicate>``."""
    import rdflib

    graph = _parse(ttl_path)
    return len(list(graph.triples((None, rdflib.URIRef(NARAD_P + predicate), None))))


def _no_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the direction pass see a project with no limits key configured.

    ``direction`` binds ``get_config_value`` at import time, so the patch has
    to land in that module's namespace rather than on the config module.
    """

    def _lookup(path: str, default: object = None, config_path: object = None) -> object:
        return default

    monkeypatch.setattr(
        "osprey.services.facility_knowledge.ttl_generator.direction.get_config_value",
        _lookup,
    )


def _config_channel_db(monkeypatch: pytest.MonkeyPatch, value: object) -> None:
    """Make ``channel_finder.pipelines.hierarchical.database.path`` read as *value*."""

    def _lookup(path: str, default: object = None, config_path: object = None) -> object:
        if path == "channel_finder.pipelines.hierarchical.database.path":
            return value
        return default

    monkeypatch.setattr("osprey.utils.config.get_config_value", _lookup)


def _config_graph_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the project read as a graph-paradigm one.

    A graph project sets ``pipeline_mode`` and ships no channel database at
    all, so the hierarchical database key is unset here too — which is exactly
    the state the graph arm has to pre-empt.
    """

    def _lookup(path: str, default: object = None, config_path: object = None) -> object:
        if path == "channel_finder.pipeline_mode":
            return "graph"
        return default

    monkeypatch.setattr("osprey.utils.config.get_config_value", _lookup)


# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------


def test_build_ttl_help_documents_inputs_and_next_step() -> None:
    """The listing shows the verb, and its own help says what it reads and writes."""
    runner = CliRunner()

    listing = runner.invoke(knowledge, ["--help"])
    assert listing.exit_code == 0, listing.output
    assert "build-ttl" in listing.output

    detail = runner.invoke(knowledge, ["build-ttl", "--help"])
    assert detail.exit_code == 0, detail.output
    flat = _flat(detail)
    assert "OUTPUT is the Turtle file to write" in flat
    assert "channel_finder.pipelines.hierarchical.database.path" in flat
    assert "control_system.limits_checking.database_path" in flat
    assert "--descriptions" in flat
    assert "in_context.json" in flat
    assert "osprey knowledge seed-graph" in flat
    # Written for operators, not for the framework's authors.
    assert "Claude Code" not in detail.output


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_build_ttl_writes_a_parsable_corpus(
    channel_db: Path, descriptions_db: Path, readonly_limits: Path, tmp_path: Path
) -> None:
    """An explicit database and limits file produce one binding per address."""
    output = tmp_path / "out" / "demo.ttl"

    result = CliRunner().invoke(
        knowledge,
        [
            "build-ttl",
            str(output),
            "--channel-db",
            str(channel_db),
            "--descriptions",
            str(descriptions_db),
            "--limits",
            str(readonly_limits),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output.is_file()
    assert _predicate_count(output, "hasBinding") == len(FIXTURE_ADDRESSES)
    assert _predicate_count(output, "fullPv") == len(FIXTURE_ADDRESSES)

    flat = _flat(result)
    assert "4 devices, 8 channel bindings, 4 signals" in flat
    # The parent directory did not exist before the run.
    assert output.parent.is_dir()


def test_build_ttl_reports_the_limits_direction_source(
    channel_db: Path, descriptions_db: Path, readonly_limits: Path, tmp_path: Path
) -> None:
    """A limits file is named in the report, and it is what decided the directions."""
    output = tmp_path / "limits.ttl"

    result = CliRunner().invoke(
        knowledge,
        [
            "build-ttl",
            str(output),
            "--channel-db",
            str(channel_db),
            "--descriptions",
            str(descriptions_db),
            "--limits",
            str(readonly_limits),
        ],
    )

    assert result.exit_code == 0, result.output
    flat = _flat(result)
    assert "direction from channel limits" in flat
    assert "channel_limits.json" in flat
    # Every address was declared non-writable, so nothing writes.
    assert _predicate_count(output, "writesSignal") == 0
    assert _predicate_count(output, "readsSignal") == len(FIXTURE_ADDRESSES)


def test_build_ttl_reports_the_grammar_fallback(
    channel_db: Path, descriptions_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no limits file anywhere, the PV grammar decides and says so."""
    _no_config(monkeypatch)
    output = tmp_path / "grammar.ttl"

    result = CliRunner().invoke(
        knowledge,
        [
            "build-ttl",
            str(output),
            "--channel-db",
            str(channel_db),
            "--descriptions",
            str(descriptions_db),
        ],
    )

    assert result.exit_code == 0, result.output
    flat = _flat(result)
    assert "direction from PV grammar" in flat
    assert "control_system.limits_checking.database_path" in flat
    # The two ':SP' addresses write; the other six read.
    assert _predicate_count(output, "writesSignal") == 2
    assert _predicate_count(output, "readsSignal") == 6


def test_build_ttl_reads_the_configured_database(
    channel_db: Path, descriptions_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no --channel-db, the config key names the database."""
    _no_config(monkeypatch)
    _config_channel_db(monkeypatch, str(channel_db))
    output = tmp_path / "configured.ttl"

    result = CliRunner().invoke(
        knowledge, ["build-ttl", str(output), "--descriptions", str(descriptions_db)]
    )

    assert result.exit_code == 0, result.output
    assert _predicate_count(output, "hasBinding") == len(FIXTURE_ADDRESSES)


def test_build_ttl_finds_the_descriptions_beside_the_channel_database(
    source_tree: Path, readonly_limits: Path, tmp_path: Path
) -> None:
    """With the two databases as neighbours, --descriptions need not be typed.

    This is the OSPREY source tree's own layout, and the only place the default
    applies: both files sit under ``tiers/tier3/``.
    """
    output = tmp_path / "sibling.ttl"

    result = CliRunner().invoke(
        knowledge,
        [
            "build-ttl",
            str(output),
            "--channel-db",
            str(source_tree),
            "--limits",
            str(readonly_limits),
        ],
    )

    assert result.exit_code == 0, result.output
    # Per-channel prose from the in-context neighbour, one per address.
    assert _predicate_count(output, "description") == len(FIXTURE_ADDRESSES)
    # Level prose from the hierarchical tree, on the nodes whose keys resolve it.
    assert _predicate_count(output, "fieldDescription") == len(FIXTURE_ADDRESSES)
    assert _predicate_count(output, "subfieldDescription") == len(FIXTURE_ADDRESSES)
    assert _predicate_count(output, "familyDescription") == 4
    assert _predicate_count(output, "systemDescription") == 4
    assert _predicate_count(output, "ringDescription") == 4


def test_build_ttl_takes_both_databases_by_name(
    channel_db: Path, descriptions_db: Path, readonly_limits: Path, tmp_path: Path
) -> None:
    """Named explicitly, the two inputs need not sit anywhere near each other."""
    output = tmp_path / "explicit.ttl"

    result = CliRunner().invoke(
        knowledge,
        [
            "build-ttl",
            str(output),
            "--channel-db",
            str(channel_db),
            "--descriptions",
            str(descriptions_db),
            "--limits",
            str(readonly_limits),
        ],
    )

    assert result.exit_code == 0, result.output
    assert _predicate_count(output, "description") == len(FIXTURE_ADDRESSES)
    assert "Per-channel sentence for SR:MAG:DIPOLE:01:CURRENT:SP." in output.read_text(
        encoding="utf-8"
    )


def test_build_ttl_without_a_descriptions_neighbour_names_both_flags(
    channel_db: Path, readonly_limits: Path, tmp_path: Path
) -> None:
    """No neighbour to default to: the command asks for both inputs by name."""
    output = tmp_path / "out.ttl"

    result = CliRunner().invoke(
        knowledge,
        [
            "build-ttl",
            str(output),
            "--channel-db",
            str(channel_db),
            "--limits",
            str(readonly_limits),
        ],
    )

    assert result.exit_code != 0
    flat = _flat(result)
    assert "Traceback" not in result.output
    assert "--descriptions" in flat
    assert "--channel-db" in flat
    assert "in_context.json" in flat
    assert not output.exists()


def test_build_ttl_uses_a_given_ontology_table(
    channel_db: Path,
    descriptions_db: Path,
    ontology_table: Path,
    readonly_limits: Path,
    tmp_path: Path,
) -> None:
    """--ontology decides the device classes the corpus declares."""
    import rdflib

    output = tmp_path / "custom.ttl"

    result = CliRunner().invoke(
        knowledge,
        [
            "build-ttl",
            str(output),
            "--channel-db",
            str(channel_db),
            "--descriptions",
            str(descriptions_db),
            "--limits",
            str(readonly_limits),
            "--ontology",
            str(ontology_table),
        ],
    )

    assert result.exit_code == 0, result.output
    types = {str(obj) for _, _, obj in _parse(output).triples((None, rdflib.RDF.type, None))}
    assert NARAD_SEM + "BendingMagnet" in types
    assert NARAD_SEM + "BeamPositionMonitor" in types
    # The shipped table's name for the same family is nowhere in the corpus.
    assert NARAD_SEM + "Dipole" not in types


def test_build_ttl_ontology_missing_a_family_is_one_line(
    channel_db: Path, descriptions_db: Path, readonly_limits: Path, tmp_path: Path
) -> None:
    """A table that covers only some families names the gap and the option."""
    payload = _ontology_payload()
    del payload["family_to_class"]["DIPOLE"]
    table = tmp_path / "partial.json"
    table.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "out.ttl"

    result = CliRunner().invoke(
        knowledge,
        [
            "build-ttl",
            str(output),
            "--channel-db",
            str(channel_db),
            "--descriptions",
            str(descriptions_db),
            "--limits",
            str(readonly_limits),
            "--ontology",
            str(table),
        ],
    )

    assert result.exit_code != 0
    flat = _flat(result)
    assert "Traceback" not in result.output
    assert "DIPOLE" in flat
    assert "--ontology" in flat


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


def test_build_ttl_in_a_graph_project_names_the_two_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A graph project ships no channel database at all, and is told so.

    Without this arm the run dies on the unset hierarchical database key, which
    reads as a broken project rather than as the deliberate posture it is.
    """
    _config_graph_mode(monkeypatch)
    output = tmp_path / "out.ttl"

    result = CliRunner().invoke(knowledge, ["build-ttl", str(output)])

    assert result.exit_code != 0
    flat = _flat(result)
    assert "Traceback" not in result.output
    assert "graph" in flat
    assert "--channel-db" in flat
    assert "--descriptions" in flat
    # Neither the missing-key dead end nor the wrong-paradigm one.
    assert "is not set in config" not in flat
    assert "hierarchical-paradigm" not in flat
    assert not output.exists()


def test_build_ttl_names_the_paradigm_that_is_actually_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A project built for another paradigm gets a diagnosis, not a bare miss."""
    databases = tmp_path / "data" / "channel_databases"
    databases.mkdir(parents=True)
    (databases / "in_context.json").write_text("{}", encoding="utf-8")
    _config_channel_db(monkeypatch, str(databases / "hierarchical.json"))

    result = CliRunner().invoke(knowledge, ["build-ttl", str(tmp_path / "out.ttl")])

    assert result.exit_code != 0
    flat = _flat(result)
    assert "Traceback" not in result.output
    assert "in_context.json" in flat
    assert "hierarchical-paradigm" in flat
    assert "--channel-db" in flat
    assert not (tmp_path / "out.ttl").exists()


def test_build_ttl_reports_a_project_with_no_channel_databases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty database directory is said to be empty."""
    databases = tmp_path / "data" / "channel_databases"
    databases.mkdir(parents=True)
    _config_channel_db(monkeypatch, str(databases / "hierarchical.json"))

    result = CliRunner().invoke(knowledge, ["build-ttl", str(tmp_path / "out.ttl")])

    assert result.exit_code != 0
    flat = _flat(result)
    assert "no channel databases in" in flat.lower()
    assert "--channel-db" in flat


def test_build_ttl_without_a_database_or_a_config_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing given and nothing configured names both the key and the option."""
    _config_channel_db(monkeypatch, None)

    result = CliRunner().invoke(knowledge, ["build-ttl", str(tmp_path / "out.ttl")])

    assert result.exit_code != 0
    flat = _flat(result)
    assert "Traceback" not in result.output
    assert "channel_finder.pipelines.hierarchical.database.path" in flat
    assert "--channel-db" in flat


def test_build_ttl_missing_limits_file_is_one_line(
    channel_db: Path, descriptions_db: Path, tmp_path: Path
) -> None:
    """A --limits path that names nothing is refused, with the fallback spelled out."""
    output = tmp_path / "out.ttl"

    result = CliRunner().invoke(
        knowledge,
        [
            "build-ttl",
            str(output),
            "--channel-db",
            str(channel_db),
            "--descriptions",
            str(descriptions_db),
            "--limits",
            str(tmp_path / "absent.json"),
        ],
    )

    assert result.exit_code != 0
    flat = _flat(result)
    assert "Traceback" not in result.output
    assert "--limits" in flat
    assert "control_system.limits_checking.database_path" in flat
    assert not output.exists()


def test_build_ttl_rejects_a_malformed_address(descriptions_db: Path, tmp_path: Path) -> None:
    """An address outside the six-token grammar is named, not traced."""
    payload = _hierarchical_payload()
    # A database that loads cleanly but joins its last two levels with '_',
    # so every expanded address is five colon tokens instead of six.
    payload["hierarchy"]["naming_pattern"] = "{ring}:{system}:{family}:{device}:{field}_{subfield}"
    db_path = tmp_path / "short.json"
    db_path.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "out.ttl"

    result = CliRunner().invoke(
        knowledge,
        [
            "build-ttl",
            str(output),
            "--channel-db",
            str(db_path),
            "--descriptions",
            str(descriptions_db),
        ],
    )

    assert result.exit_code != 0
    flat = _flat(result)
    assert "Traceback" not in result.output
    assert "RING:SYSTEM:FAMILY:DEVICE:FIELD:SUBFIELD" in flat
    assert not output.exists()


def test_build_ttl_points_a_yaml_ontology_at_the_compiler(
    channel_db: Path, descriptions_db: Path, tmp_path: Path
) -> None:
    """A LinkML schema handed to --ontology is sent to compile-ontology first.

    ``--ontology`` reads a compiled JSON table, so a ``.yaml`` schema fails to
    parse.  The failure has to name the verb that turns one into the other,
    otherwise the operator is left with 'not valid JSON' about a file that was
    never meant to be JSON.
    """
    schema = tmp_path / "facility.yaml"
    schema.write_text(
        "id: https://example.org/facility\n"
        "name: facility\n"
        "classes:\n"
        "  AcceleratorDevice:\n"
        "    class_uri: narad_sem:AcceleratorDevice\n",
        encoding="utf-8",
    )
    output = tmp_path / "out.ttl"

    result = CliRunner().invoke(
        knowledge,
        [
            "build-ttl",
            str(output),
            "--channel-db",
            str(channel_db),
            "--descriptions",
            str(descriptions_db),
            "--ontology",
            str(schema),
        ],
    )

    assert result.exit_code != 0
    flat = _flat(result)
    assert "Traceback" not in result.output
    assert "compile-ontology" in flat
    assert not output.exists()
