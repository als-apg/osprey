"""The whole ``osprey mml`` chain, run on every committed fixture.

``import`` → ``map --init`` → the fixture's reviewed ``mapping.yaml`` →
``map --check`` → ``emit --duckdb``, twice, under ``CliRunner``. The single
chain each fixture gets is shared by every assertion in this module, so what is
pinned here is one facility installed the way a facility is installed:

* the chain stays green and a second pass changes no byte of any emitted file
  except the DuckDB one, whose ``channels`` row count is the export's distinct
  PVs and whose ``systems`` row count is ``section_order``, after each pass;
* nothing is lost between the export and the artifacts — the channel strings
  the reviewer's answers keep, the corpus's ``fullPv`` literals and the channel
  database's channels are one set, and the binding count agrees between the
  corpus, the judged census and ``knowledge build-index``;
* every mapped family arrives in all three places a deployment reads it (the
  channel database, the corpus as a class-typed device population, and an OKF
  family page), carrying the prose and the types the agent answers from;
* the forms only some facilities have — Tango names, both channel keys on one
  field, two families differing only by case — survive as themselves;
* what the reviewer answered about a row beyond a family's devices, a device
  bound by no channel and a PV shared by several devices is what the corpus,
  the channel database, the DuckDB copy and the family page hold.

An export that carries a virtual accelerator chains one verb further, into
``verify``, and is discovered from the fixtures rather than listed: a directory
holding a ``*.va.json`` sibling runs the whole lane, writes the five files a
served machine boots from beside everything the shorter chain already wrote,
drives that machine through addresses the same run kept, and holds it against
the exported response matrix. The 1.0 fixtures pin the other side of that:
their emit says the lane was skipped and writes not one of its files.

Two facility-scale numbers ride on the ALS export, which never enters the repo:
with ``OSPREY_ALS_MML_EXPORT`` and ``OSPREY_ALS_MML_MAPPING`` set, the same
chain runs on it and the binding and channel counts of success criterion 5 are
pinned. Unset, that lane skips with the reason.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner
from rdflib import RDF, Graph, URIRef

from osprey.cli.main import cli
from osprey.services.channel_finder.databases.middle_layer import (
    CHANNEL_KEYS,
    MiddleLayerDatabase,
)
from osprey.services.facility_knowledge.okf.bundle import OKFBundle
from osprey.services.facility_knowledge.ttl_generator.model import (
    BINDING_IRI_PREFIX,
    DEVICE_IRI_PREFIX,
)
from osprey.services.mml.emit.channel_db import FIELD_METADATA_KEYS, PROVENANCE_KEY
from osprey.services.mml.mapping.schema import Mapping, parse_mapping
from osprey.services.virtual_accelerator.bindings import load_bindings, setpoints

# Every fixture chains through ``emit``, which needs the knowledge extra, and
# through ``--duckdb``.
pytest.importorskip("linkml_runtime")
pytest.importorskip("duckdb")

from tests.cli.test_mml_emit import VA_ARTIFACTS, VA_SKIPPED  # noqa: E402
from tests.cli.test_mml_import_chain import FIXTURE_IMPORTS  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "mml"

#: The fixture trees a 2.0 export commits, discovered rather than listed: the
#: virtual accelerator of an export lives in its ``*.va.json`` sibling, so a
#: directory that carries one chains all the way through ``verify``. A tree
#: committed later joins every case in :class:`TestTheVirtualAcceleratorChain`
#: without a name being typed here.
TWO_ZERO_TREES = tuple(
    sorted(
        directory.name
        for directory in FIXTURES.iterdir()
        if directory.is_dir() and any(directory.glob("*.va.json"))
    )
)

#: The namespaces every emitted corpus writes its facts and its types in.
NARAD_P = "https://narad.example.org/property/"
NARAD_SEM = "https://narad.example.org/schema/shared_semantics/"

#: The field-level keys ``emit`` is allowed to carry into the channel database.
ALLOWED_FIELD_KEYS = frozenset({*CHANNEL_KEYS, "Description", *FIELD_METADATA_KEYS})

ALS_EXPORT_ENV = "OSPREY_ALS_MML_EXPORT"
ALS_MAPPING_ENV = "OSPREY_ALS_MML_MAPPING"

#: Success criterion 5's pinned ALS totals, after broadcast expansion.
ALS_BINDINGS = 13674
ALS_DISTINCT_PVS = 11209


# ===================================================================
# Running the chain
# ===================================================================


@dataclass(frozen=True)
class Pass:
    """What one pass of the chain left behind.

    Attributes:
        artifacts: Every emitted non-DuckDB file under ``data/``, keyed by its
            path relative to the repo root.
        duck: The ``channels`` and ``systems`` row counts of the DuckDB import.
        emit: Everything ``mml emit`` reported on that pass, for the lines it
            says rather than writes.
        verify: Everything ``mml verify`` reported, or ``None`` on a chain that
            stops at ``emit`` because the export carries no virtual accelerator.
    """

    artifacts: dict[str, bytes]
    duck: dict[str, int]
    emit: str
    verify: str | None


@dataclass(frozen=True)
class Chain:
    """One fixture installed end to end, and everything read back off it.

    Attributes:
        name: The fixture directory's name.
        root: The deployment repo the chain ran in.
        ao: The canonical ``ao.json`` the import wrote.
        document: The reviewed ``mapping.yaml`` as a plain document.
        mapping: That document parsed.
        passes: One :class:`Pass` per run of the chain, in order.
        graph: The emitted Turtle corpus, parsed.
    """

    name: str
    root: Path
    ao: dict[str, Any]
    document: dict[str, Any]
    mapping: Mapping
    passes: tuple[Pass, ...]
    graph: Graph

    @property
    def token(self) -> str:
        """The facility token every IRI and output filename carries."""
        assert self.mapping.facility.token is not None
        return self.mapping.facility.token

    @property
    def ttl(self) -> Path:
        return self.root / "data" / f"{self.token}.ttl"

    @property
    def bundle(self) -> Path:
        return self.root / "data" / "facility_knowledge"

    @property
    def database(self) -> MiddleLayerDatabase:
        return MiddleLayerDatabase(str(self.root / "data" / "channel_databases/middle_layer.json"))

    @property
    def profile(self) -> str:
        return (self.root / "data" / "mml" / "PROFILE.md").read_text(encoding="utf-8")


def _run(*args: str) -> Any:
    """Invoke the CLI, refusing anything but a clean exit."""
    result = CliRunner().invoke(cli, list(args), catch_exceptions=False)
    assert "Traceback" not in result.output
    assert result.exit_code == 0, f"osprey {' '.join(args)}:\n{result.output}"
    return result


def _artifacts(root: Path) -> dict[str, bytes]:
    """Every emitted non-DuckDB file under ``data/``, keyed by relative path.

    ``data/mml/`` holds the chain's inputs rather than its output, and the
    DuckDB file timestamps every row it imports, so neither is comparable
    between passes.
    """
    data = root / "data"
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(data.rglob("*"))
        if path.is_file()
        and path.suffix != ".duckdb"
        and not path.relative_to(data).as_posix().startswith("mml/")
    }


def _duck_counts(path: Path) -> dict[str, int]:
    """The ``channels`` and ``systems`` row counts of a DuckDB import."""
    import duckdb

    connection = duckdb.connect(str(path), read_only=True)
    try:
        return {
            table: connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in ("channels", "systems")
        }
    finally:
        connection.close()


def run_chain(
    root: Path,
    inputs: tuple[str, ...],
    flags: tuple[str, ...],
    mapping_source: Path,
    *,
    name: str,
    passes: int = 2,
    verify: bool = False,
) -> Chain:
    """Install one facility from its export, ``passes`` times over.

    The reviewed mapping is copied over the skeleton ``map --init`` writes, as
    a facility's own review would leave it; every later pass re-inits with
    ``--force`` so the whole chain runs, not only its tail.

    Args:
        root: The deployment repo to build in. Created if absent.
        inputs: The export paths to import.
        flags: The ``--system`` arguments the export's form needs.
        mapping_source: The reviewed ``mapping.yaml`` to install.
        name: The fixture's name, for reporting.
        passes: How many times to run the whole chain.
        verify: Whether each pass ends in ``mml verify``. Only an export that
            carries a virtual accelerator has one to verify; the report lands
            under ``data/mml/``, which is not part of the emitted tree.

    Returns:
        The finished chain, with one :class:`Pass` per run.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "profile.yml").write_text("name: scratch\n", encoding="utf-8")
    where = ("--repo", str(root))
    mapping_path = root / "data" / "mml" / "mapping.yaml"

    records: list[Pass] = []
    for index in range(passes):
        _run("mml", "import", *inputs, *flags, *where)
        _run("mml", "map", "--init", *(("--force",) if index else ()), *where)
        shutil.copy(mapping_source, mapping_path)
        _run("mml", "map", "--check", *where)
        emit = _run("mml", "emit", "--duckdb", *where)
        checked = _run("mml", "verify", *where).output if verify else None
        records.append(
            Pass(
                artifacts=_artifacts(root),
                duck=_duck_counts(root / "data/channel_databases/middle_layer.duckdb"),
                emit=emit.output,
                verify=checked,
            )
        )

    document = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    mapping = parse_mapping(document)
    ao = json.loads((root / "data" / "mml" / "ao.json").read_text(encoding="utf-8"))
    graph = Graph()
    graph.parse(root / "data" / f"{mapping.facility.token}.ttl", format="turtle")
    return Chain(
        name=name,
        root=root,
        ao=ao,
        document=document,
        mapping=mapping,
        passes=tuple(records),
        graph=graph,
    )


@pytest.fixture(scope="module")
def chains(tmp_path_factory: pytest.TempPathFactory) -> Callable[[str], Chain]:
    """A per-fixture chain, built once and shared by every assertion."""
    built: dict[str, Chain] = {}

    def build(name: str) -> Chain:
        if name not in built:
            sources, flags, _ = FIXTURE_IMPORTS[name]
            built[name] = run_chain(
                tmp_path_factory.mktemp(name),
                tuple(str(FIXTURES / source) for source in sources),
                flags,
                FIXTURES / name / "mapping.yaml",
                name=name,
            )
        return built[name]

    return build


@pytest.fixture(scope="module", params=sorted(FIXTURE_IMPORTS))
def chain(request: pytest.FixtureRequest, chains: Callable[[str], Chain]) -> Chain:
    """Every committed fixture in turn, installed end to end."""
    return chains(request.param)


@pytest.fixture(scope="module")
def two_zero_chains(tmp_path_factory: pytest.TempPathFactory) -> Callable[[str], Chain]:
    """A per-fixture 2.0 chain, built once and shared by every assertion.

    The whole verb sequence a facility runs: ``import`` pulls the export's
    ``va`` and ``response`` siblings and its deck in beside the Accelerator
    Objects, ``map --init`` writes a skeleton carrying a virtual-accelerator
    block, the reviewed mapping answers it, and ``emit`` and ``verify`` have a
    machine to build and to hold against the exported response matrix.
    """
    built: dict[str, Chain] = {}

    def build(name: str) -> Chain:
        if name not in built:
            exports = sorted((FIXTURES / name).glob("*.ao.json"))
            assert exports, f"{name} commits a virtual accelerator but no export"
            built[name] = run_chain(
                tmp_path_factory.mktemp(f"{name}-2-0"),
                tuple(str(path) for path in exports),
                (),
                FIXTURES / name / "mapping.yaml",
                name=name,
                verify=True,
            )
        return built[name]

    return build


@pytest.fixture(scope="module", params=TWO_ZERO_TREES)
def two_zero_chain(
    request: pytest.FixtureRequest, two_zero_chains: Callable[[str], Chain]
) -> Chain:
    """Every committed 2.0 fixture in turn, installed end to end."""
    return two_zero_chains(request.param)


# ===================================================================
# Reading the export without the chain's own code
# ===================================================================


def _families(ao: dict[str, Any]) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Yield ``(system, family, body)`` for every family body the export holds."""
    for system, families in ao.items():
        if system.startswith("_") or not isinstance(families, dict):
            continue
        for family, body in families.items():
            if family.startswith("_") or not isinstance(body, dict):
                continue
            yield system, family, body


def _channel_fields(body: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(field, body)`` for every channel-bearing field of a family body."""
    for field, field_body in body.items():
        if not isinstance(field_body, dict):
            continue
        if any(key in field_body for key in CHANNEL_KEYS):
            yield field, field_body


def _signal_groups(ao: dict[str, Any]) -> Iterator[tuple[str, str, str, dict[str, Any]]]:
    """Yield ``(system, family, field, body)`` for every channel-bearing group."""
    for system, family, body in _families(ao):
        for field, field_body in _channel_fields(body):
            yield system, family, field, field_body


def _device_count(body: dict[str, Any]) -> int:
    """How many devices one family body states, counted as the import walk counts.

    ``DeviceList`` states one row per device, and a bare pair of numbers is one
    row; the array may sit on the family or in its setup block. A body stating
    no device list falls back to its longest channel list, which leaves no slot
    beyond the devices at all.
    """
    rows = body.get("DeviceList")
    if rows is None:
        for key in ("setup", "_setup"):
            setup = body.get(key)
            if isinstance(setup, dict) and "DeviceList" in setup:
                rows = setup["DeviceList"]
                break
    if isinstance(rows, list) and rows:
        return len(rows) if all(isinstance(row, list) for row in rows) else 1
    lengths = [
        len(value) if isinstance(value, list) else 1
        for _, field_body in _channel_fields(body)
        for key in CHANNEL_KEYS
        if isinstance(value := field_body.get(key), str | list)
    ]
    return max(lengths, default=0)


def _slots(body: dict[str, Any]) -> Iterator[tuple[str, int, str]]:
    """Yield ``(key, index, address)`` for every non-blank slot of every channel key."""
    for key in CHANNEL_KEYS:
        value = body.get(key)
        items = [value] if isinstance(value, str) else value
        if not isinstance(items, list):
            continue
        for index, item in enumerate(items):
            if isinstance(item, str) and item.strip():
                yield key, index, item.strip()


def _dropped_rows(document: dict[str, Any]) -> set[tuple[str, str, str]]:
    """Every ``(raw family, field, signal)`` the reviewer answered ``drop``."""
    dropped: set[tuple[str, str, str]] = set()
    for family, entry in (document.get("judgments") or {}).items():
        for field, signals in (entry.get("rows_beyond_devices") or {}).items():
            for signal, answer in signals.items():
                if answer == "drop":
                    dropped.add((family, field, signal))
    return dropped


def _kept_addresses(ao: dict[str, Any], document: dict[str, Any]) -> set[str]:
    """Every non-blank channel string the export carries and judgment keeps.

    A ``drop`` answer cuts one row beyond a family's devices, never a channel
    string: it reaches a slot only where that slot sits past the family's
    device count, and only in the system whose device count it sits past. The
    same PV bound to a device of the family, or beyond the devices in one
    system but on a device in another, stays. So an address leaves the facility
    only when every slot carrying it is a dropped row.

    The whole reading is re-derived from ``ao.json`` and the mapping document,
    so this never agrees with the emitter by sharing its reading of them.
    """
    dropped = _dropped_rows(document)
    kept: dict[str, bool] = {}
    for _, family, body in _families(ao):
        beyond = _device_count(body)
        for field, field_body in _channel_fields(body):
            for _, index, address in _slots(field_body):
                cut = index >= beyond and (family, field, address) in dropped
                kept[address] = kept.get(address, False) or not cut
    return {address for address, survives in kept.items() if survives}


def _owners(ao: dict[str, Any]) -> dict[str, set[tuple[str, str, str, int]]]:
    """Every channel string's ``(system, family, field, ordinal)`` owners.

    Ordinals are 1-based, as the document and ``PROFILE.md`` write them.
    """
    owners: dict[str, set[tuple[str, str, str, int]]] = {}
    for system, family, field, body in _signal_groups(ao):
        for _, index, address in _slots(body):
            owners.setdefault(address, set()).add((system, family, field, index + 1))
    return owners


def _populations(ao: dict[str, Any]) -> set[tuple[str, str]]:
    """Every ``(system, family)`` the export gives at least one channel."""
    found: set[tuple[str, str]] = set()
    for system, family, _, body in _signal_groups(ao):
        if any(True for _ in _slots(body)):
            found.add((system, family))
    return found


def _duck_units(root: Path) -> dict[str, str]:
    """Every DuckDB ``channels`` row as ``channel_name -> units``."""
    import duckdb

    connection = duckdb.connect(str(root / "data/channel_databases/middle_layer.duckdb"), True)
    try:
        return dict(connection.execute("SELECT channel_name, units FROM channels").fetchall())
    finally:
        connection.close()


def _duck_channel(root: Path, name: str) -> dict[str, Any]:
    """One ``channels`` row of the DuckDB copy, keyed by column name."""
    import duckdb

    connection = duckdb.connect(str(root / "data/channel_databases/middle_layer.duckdb"), True)
    try:
        cursor = connection.execute("SELECT * FROM channels WHERE channel_name = ?", [name])
        row = cursor.fetchone()
        assert row is not None, f"{name} holds no row of the DuckDB copy"
        return dict(zip([column[0] for column in cursor.description], row, strict=True))
    finally:
        connection.close()


def _exported_units(ao: dict[str, Any]) -> dict[str, set[str]]:
    """Every channel string's possible hardware units, read off the export alone.

    A field states one unit as a string or as one entry per device; a per-device
    list states one unit only when its non-blank entries agree. A PV bound by
    several fields may be reachable from any of their units.
    """
    found: dict[str, set[str]] = {}
    for _, _, _, body in _signal_groups(ao):
        value = body.get("HWUnits")
        if isinstance(value, list):
            distinct = {u.strip() for u in value if isinstance(u, str) and u.strip()}
            unit = distinct.pop() if len(distinct) == 1 else ""
        else:
            unit = value.strip() if isinstance(value, str) else ""
        for _, _, address in _slots(body):
            found.setdefault(address, set()).add(unit)
    return found


def _integral_slots(ao: dict[str, Any]) -> Iterator[tuple[str, str, str, Any]]:
    """Yield every ``(system, family, key, entry)`` of an index or status array."""
    for system, families in ao.items():
        if system.startswith("_") or not isinstance(families, dict):
            continue
        for family, body in families.items():
            if family.startswith("_") or not isinstance(body, dict):
                continue
            blocks = [body, *(v for v in body.values() if isinstance(v, dict))]
            for block in blocks:
                for key in ("DeviceList", "ElementList", "Status"):
                    value = block.get(key)
                    if not isinstance(value, list):
                        continue
                    for entry in value:
                        for item in entry if isinstance(entry, list) else [entry]:
                            yield system, family, key, item


def _census_bindings(root: Path, mapping: Mapping) -> int:
    """The import walk's own binding count, judged and broadcast-expanded.

    The census is the one production reader this module leans on, because the
    binding count is what the import walk itself claims. Handing it the mapping
    keeps that claim on the judged grain the emitters write.
    """
    from osprey.services.mml.canonical import read_canonical
    from osprey.services.mml.census import take_census

    ao, ad = read_canonical(root / "data" / "mml")
    return take_census(ao, ad, mapping).totals.bindings


def _raw_census_bindings(root: Path) -> int:
    """The same count on the export's own grain, before any answer is applied.

    What one answer costs the facility is the distance between this count and
    the judged one.
    """
    from osprey.services.mml.canonical import read_canonical
    from osprey.services.mml.census import take_census

    ao, ad = read_canonical(root / "data" / "mml")
    return take_census(ao, ad).totals.bindings


# ===================================================================
# Reading the emitted corpus
# ===================================================================


def _devices(graph: Graph) -> set[URIRef]:
    return set(graph.subjects(URIRef(f"{NARAD_P}deviceId"), None))


def _bindings(graph: Graph) -> set[URIRef]:
    return set(graph.subjects(URIRef(f"{NARAD_P}bindingId"), None))


def _objects(graph: Graph, subject: URIRef, name: str) -> list[Any]:
    return list(graph.objects(subject, URIRef(f"{NARAD_P}{name}")))


def _full_pvs(graph: Graph) -> list[str]:
    return [str(value) for value in graph.objects(None, URIRef(f"{NARAD_P}fullPv"))]


def _build_index_counts(root: Path, ttl: Path, output: Path) -> dict[str, int]:
    """Run ``knowledge build-index`` on a corpus and read back what it reports."""
    result = _run("knowledge", "build-index", "--ttl", str(ttl), "--output", str(output))
    # Rich wraps the report line at the console width; the counts survive the
    # wrap, the line breaks do not.
    flat = " ".join(result.output.split())
    counts = {}
    for name in ("bindings", "channels"):
        match = re.search(rf"([\d,]+) {name}", flat)
        assert match, f"build-index reported no {name} count:\n{result.output}"
        counts[name] = int(match.group(1).replace(",", ""))
    return counts


def _binding_field(graph: Graph, binding: URIRef) -> str:
    """The field name a binding's ``bindingId`` ends in.

    The last segment is ``<field>_<subfield>``, and the subfield names the
    channel key the slot came from — ``val`` for a CA name, ``tango`` for a
    Tango one — so the field is everything before the final underscore.
    """
    return str(_objects(graph, binding, "bindingId")[0]).rsplit(":", 1)[-1].rsplit("_", 1)[0]


def _device_bindings(chain: Chain, prefix: str) -> dict[str, list[str]]:
    """Every device IRI under ``prefix``, mapped to its bindings' field names."""
    return {
        str(device): sorted(
            _binding_field(chain.graph, URIRef(str(binding)))
            for binding in _objects(chain.graph, device, "hasBinding")
        )
        for device in _devices(chain.graph)
        if str(device).startswith(prefix)
    }


def _device_prefix(chain: Chain, section: str, token: str) -> str:
    """The IRI every device of one family in one sub-machine starts with."""
    return f"{DEVICE_IRI_PREFIX}{chain.token}_{section}_{token}_"


def _profile_blocks(profile: str, heading: str) -> str:
    """Every ``PROFILE.md`` block under one heading, at any level, run together."""
    kept: list[str] = []
    inside = False
    for line in profile.splitlines():
        if line.startswith("#"):
            inside = line.strip().lstrip("# ") == heading
            continue
        if inside:
            kept.append(line)
    return "\n".join(kept)


def _shared_pv_section(profile: str) -> str:
    """The facility-wide ``Shared PVs`` block of ``PROFILE.md``."""
    lines = profile.splitlines()
    heads = [number for number, line in enumerate(lines) if line.strip() == "### Shared PVs"]
    if not heads:
        return ""
    start = heads[0]
    # The facility-wide block ends where the first system's own sections begin.
    tails = [number for number in range(start + 1, len(lines)) if lines[number].startswith("## ")]
    return "\n".join(lines[start : tails[0] if tails else len(lines)])


# ===================================================================
# Criterion 1 — the chain runs, twice, and lands where it says
# ===================================================================


class TestTheChainRuns:
    def test_every_artifact_is_written(self, chain: Chain) -> None:
        data = chain.root / "data"
        expected = (
            data / "channel_databases" / "middle_layer.json",
            data / "channel_databases" / "middle_layer.duckdb",
            data / "ontology" / f"{chain.token}.yaml",
            data / "facility_ontology.json",
            data / "facility_knowledge" / "facility.md",
            data / "facility_knowledge" / "index.md",
            chain.ttl,
        )
        for path in expected:
            assert path.is_file(), f"{path} was not written"
        assert list((data / "facility_knowledge" / "families").glob("*.md"))

    def test_a_one_zero_export_says_the_va_lane_is_skipped_and_writes_none_of_it(
        self, chain: Chain
    ) -> None:
        # Every committed fixture is a 1.0 export: it carries no virtual
        # accelerator, so each pass adds the one line that says so and not one
        # of the five files a served machine boots from.
        assert not (chain.root / "data" / "mml" / "va.json").is_file()

        for index, record in enumerate(chain.passes):
            assert VA_SKIPPED in record.emit, f"pass {index + 1}"
            written = sorted(set(record.artifacts) & set(VA_ARTIFACTS))
            assert not written, f"pass {index + 1} wrote {written}"

    def test_a_second_pass_changes_no_emitted_byte(self, chain: Chain) -> None:
        first, second = chain.passes[0], chain.passes[1]

        assert second.artifacts == first.artifacts

    def test_duckdb_counts_hold_after_each_pass(self, chain: Chain) -> None:
        distinct = len(_kept_addresses(chain.ao, chain.document))
        systems = len(chain.mapping.section_order)

        for index, record in enumerate(chain.passes):
            assert record.duck["channels"] == distinct, f"pass {index + 1}"
            assert record.duck["systems"] == systems, f"pass {index + 1}"

    def test_every_device_iri_carries_the_facility_token(self, chain: Chain) -> None:
        prefix = f"{DEVICE_IRI_PREFIX}{chain.token}_"
        devices = _devices(chain.graph)

        assert devices
        for device in devices:
            assert str(device).startswith(prefix), device

    def test_every_binding_iri_carries_the_facility_token(self, chain: Chain) -> None:
        prefix = f"{BINDING_IRI_PREFIX}narad_endpoint_{chain.token}_"
        bindings = _bindings(chain.graph)

        assert bindings
        for binding in bindings:
            assert str(binding).startswith(prefix), binding

    def test_the_database_names_exactly_section_order(self, chain: Chain) -> None:
        database = chain.database

        assert [entry["name"] for entry in database.list_systems()] == list(
            chain.mapping.section_order
        )
        assert database.get_statistics()["systems"] == len(chain.mapping.section_order)

    def test_the_provenance_slot_stays_a_string(self, chain: Chain) -> None:
        # A mapping here would be counted as a system by the loader's census.
        database = json.loads(
            (chain.root / "data/channel_databases/middle_layer.json").read_text(encoding="utf-8")
        )

        assert isinstance(database[PROVENANCE_KEY], str)


# ===================================================================
# Criterion 1, the other half — an export that carries a machine
# ===================================================================


class TestTheVirtualAcceleratorChain:
    """A 2.0 export chains one verb further, and the machine it serves is the
    machine the same run described.

    The fixtures above stop at ``emit`` because a 1.0 export has no virtual
    accelerator to serve. An export that carries one runs ``import`` → ``map
    --init`` → the reviewed mapping → ``map --check`` → ``emit --duckdb`` →
    ``verify``, and the five files a served machine boots from join everything
    the shorter chain already wrote.
    """

    def test_a_two_zero_export_is_committed(self) -> None:
        # Every other case in this class is parametrised over the discovery,
        # so a fixture directory that stops carrying a ``*.va.json`` would
        # empty them all silently rather than fail.
        assert TWO_ZERO_TREES, "no fixture export carries a *.va.json sibling"

    def test_the_lane_runs_and_writes_all_five_files(self, two_zero_chain: Chain) -> None:
        assert (two_zero_chain.root / "data" / "mml" / "va.json").is_file()

        for index, record in enumerate(two_zero_chain.passes):
            assert VA_SKIPPED not in record.emit, f"pass {index + 1}"
            missing = sorted(set(VA_ARTIFACTS) - set(record.artifacts))
            assert not missing, f"pass {index + 1} is missing {missing}"

    def test_a_second_pass_changes_no_emitted_byte(self, two_zero_chain: Chain) -> None:
        first, second = two_zero_chain.passes[0], two_zero_chain.passes[1]

        assert set(first.artifacts) >= set(VA_ARTIFACTS)
        assert second.artifacts == first.artifacts

    def test_every_bound_address_is_one_the_chain_kept(self, two_zero_chain: Chain) -> None:
        # The machine is driven through the same addresses the deployment
        # serves: a binding onto an address the review dropped, or onto one the
        # channel database never heard of, would be a knob with nothing behind it.
        bound = setpoints(load_bindings(two_zero_chain.root / "data/simulation/va_bindings.json"))
        kept = _kept_addresses(two_zero_chain.ao, two_zero_chain.document)
        database = {entry["channel"] for entry in two_zero_chain.database.get_all_channels()}

        assert bound
        assert set(bound) <= kept
        assert set(bound) <= database

    def test_verify_compares_the_matrix_the_export_carries(self, two_zero_chain: Chain) -> None:
        report = two_zero_chain.root / "data" / "mml" / "VA-REPORT.md"

        assert report.is_file()
        for index, record in enumerate(two_zero_chain.passes):
            assert record.verify is not None, f"pass {index + 1}"
            assert "Not one entry" not in record.verify, f"pass {index + 1}: {record.verify}"
            assert "response entries" in record.verify, f"pass {index + 1}: {record.verify}"

    def test_verify_answers_the_same_on_every_pass(self, two_zero_chain: Chain) -> None:
        # The passes install the same export over the same mapping, so a
        # comparison that moved between them would be the chain, not the machine.
        verdicts = {record.verify for record in two_zero_chain.passes}

        assert len(verdicts) == 1, verdicts


# ===================================================================
# Criterion 2 — zero loss between the export and the artifacts
# ===================================================================


class TestZeroLoss:
    def test_the_export_the_corpus_and_the_database_hold_one_set(self, chain: Chain) -> None:
        exported = _kept_addresses(chain.ao, chain.document)
        corpus = set(_full_pvs(chain.graph))
        database = {entry["channel"] for entry in chain.database.get_all_channels()}

        assert exported
        assert corpus == exported, f"corpus differs by {corpus ^ exported}"
        assert database == exported, f"channel database differs by {database ^ exported}"

    def test_the_corpus_binds_every_non_blank_slot(self, chain: Chain) -> None:
        expected = _census_bindings(chain.root, chain.mapping)

        assert len(_bindings(chain.graph)) == expected
        assert len(_full_pvs(chain.graph)) == expected

    def test_build_index_counts_the_same_bindings_and_channels(
        self, chain: Chain, tmp_path: Path
    ) -> None:
        counts = _build_index_counts(chain.root, chain.ttl, tmp_path / "index.json")

        assert counts["bindings"] == _census_bindings(chain.root, chain.mapping)
        assert counts["channels"] == len(_kept_addresses(chain.ao, chain.document))

    def test_the_profile_lists_every_shared_pv_with_all_its_owners(self, chain: Chain) -> None:
        shared = {
            address: owners for address, owners in _owners(chain.ao).items() if len(owners) > 1
        }
        assert shared, f"{chain.name} carries no shared PV to report"
        section = _shared_pv_section(chain.profile)

        for address, owners in shared.items():
            assert f"`{address}`" in section, address
            for system, family, field, index in owners:
                assert f"({system}, {family}, {field}, {index})" in section


# ===================================================================
# Criterion 3 — every mapped family arrives everywhere it is read
# ===================================================================


class TestEveryFamilyArrives:
    def test_each_population_reaches_database_corpus_and_bundle(self, chain: Chain) -> None:
        devices = {str(device) for device in _devices(chain.graph)}
        database = chain.database
        populations = sorted(_populations(chain.ao))

        assert populations, f"{chain.name} exports no channel-bearing family"
        for system, family in populations:
            section = chain.mapping.systems[system].name
            token = chain.mapping.mapped(family)
            klass = chain.mapping.families[family].class_

            families = {entry["name"] for entry in database.list_families(section)}
            assert token in families, f"{section}:{token} is not a family of the database"
            page = chain.bundle / "families" / f"{section}-{token}.md"
            assert page.is_file(), f"{page} was not written"

            prefix = f"{DEVICE_IRI_PREFIX}{chain.token}_{section}_{token}_"
            population = [iri for iri in devices if iri.startswith(prefix)]
            assert population, f"the corpus holds no device of {section}:{token}"
            for iri in population:
                types = [str(value) for value in chain.graph.objects(URIRef(iri), RDF.type)]
                assert types == [f"{NARAD_SEM}{klass}"], f"{iri} is typed {types}"

    def test_every_binding_carries_a_description(self, chain: Chain) -> None:
        for binding in _bindings(chain.graph):
            described = _objects(chain.graph, binding, "description")
            assert described and str(described[0]).strip(), binding

    def test_system_prose_reaches_the_database_and_every_device(self, chain: Chain) -> None:
        for entry in chain.database.list_systems():
            assert entry["description"].strip(), entry["name"]
        for device in _devices(chain.graph):
            prose = _objects(chain.graph, device, "systemDescription")
            assert prose and str(prose[0]).strip(), device

    def test_inspect_fields_types_and_describes_every_field(self, chain: Chain) -> None:
        database = chain.database

        for system, family in sorted(_populations(chain.ao)):
            section = chain.mapping.systems[system].name
            token = chain.mapping.mapped(family)
            fields = database.inspect_fields(section, token)
            emitted = {name: body for name, body in fields.items() if name != "setup"}

            assert emitted, f"{section}:{token} inspects as no field at all"
            for name, body in emitted.items():
                assert body["type"] in CHANNEL_KEYS, f"{section}:{token}:{name} {body['type']}"
                assert body["description"].strip(), f"{section}:{token}:{name}"

    def test_a_field_holds_no_subfield_and_no_key_outside_the_allowlist(self, chain: Chain) -> None:
        database = chain.database

        for system, family in sorted(_populations(chain.ao)):
            section = chain.mapping.systems[system].name
            token = chain.mapping.mapped(family)
            for name in database.inspect_fields(section, token):
                if name == "setup":
                    continue
                for key, body in database.inspect_fields(section, token, name).items():
                    where = f"{section}:{token}:{name}:{key}"
                    assert body["type"] != "dict (subfield)", where
                    assert key in ALLOWED_FIELD_KEYS, where


class TestTheOntologyTable:
    def _table(self, chain: Chain) -> dict[str, Any]:
        return json.loads(
            (chain.root / "data" / "facility_ontology.json").read_text(encoding="utf-8")
        )

    def test_every_mapped_family_is_typed_by_its_class(self, chain: Chain) -> None:
        table = self._table(chain)

        for _, family in sorted(_populations(chain.ao)):
            token = chain.mapping.mapped(family)
            assert table["family_to_class"][token] == chain.mapping.families[family].class_

    def test_alt_labels_are_the_sorted_union_of_the_sharing_aliases(self, chain: Chain) -> None:
        table = self._table(chain)
        sharing: dict[str, set[str]] = {}
        # A packaged class carries no branch, and keeps its packaged labels
        # beside the aliases of the families that chose it.
        packaged: set[str] = set()
        for _, family in sorted(_populations(chain.ao)):
            entry = chain.mapping.families[family]
            assert entry.class_ is not None
            sharing.setdefault(entry.class_, set()).update(entry.aliases)
            if entry.branch is None:
                packaged.add(entry.class_)

        assert sharing
        for klass, aliases in sharing.items():
            labels = table["classes"][klass]["altLabels"]
            assert labels == sorted(labels), klass
            if klass in packaged:
                assert aliases <= set(labels), f"{klass} lost {sorted(aliases - set(labels))}"
            else:
                assert labels == sorted(aliases), klass

    def test_every_class_walks_its_parents_up_to_the_root(self, chain: Chain) -> None:
        table = self._table(chain)
        classes = table["classes"]

        for klass in {family.class_ for family in chain.mapping.families.values() if family.class_}:
            seen = []
            current: str | None = klass
            while current is not None:
                assert current in classes, f"{current} is missing from the compiled table"
                seen.append(current)
                current = classes[current]["parent"]
            assert seen[-1] == table["root"], seen

    def test_a_packaged_class_keeps_its_packaged_ancestors(self, chains: Callable) -> None:
        # The paired fixture types HCM as the packaged HCorrector.
        paired = chains("paired")
        table = self._table(paired)

        assert table["family_to_class"]["HCM"] == "HCorrector"
        ancestors, current = [], table["classes"]["HCorrector"]["parent"]
        while current is not None:
            ancestors.append(current)
            current = table["classes"][current]["parent"]
        assert ancestors == ["Corrector", "Magnet", "AcceleratorDevice"]


class TestTheKnowledgeBundle:
    def test_the_bundle_validates(self, chain: Chain) -> None:
        result = _run("knowledge", "validate", str(chain.bundle))

        assert "valid" in result.output

    def test_the_root_index_names_the_facility_and_the_families(self, chain: Chain) -> None:
        index = (chain.bundle / "index.md").read_text(encoding="utf-8")
        headings = [line for line in index.splitlines() if line.startswith("# ")]

        assert headings == ["# Facility", "# Subdirectories"], headings
        facility, subdirectories = index.split("# Subdirectories")
        assert "facility.md" in facility.split("# Facility")[1]
        assert "families" in subdirectories

    def test_the_family_index_lists_every_page_under_one_heading(self, chain: Chain) -> None:
        index = (chain.bundle / "families" / "index.md").read_text(encoding="utf-8")
        headings = [line for line in index.splitlines() if line.startswith("# ")]

        assert headings == ["# DeviceFamily"], headings
        for page in sorted((chain.bundle / "families").glob("*.md")):
            if page.name != "index.md":
                assert page.name in index, page.name

    def test_every_advertised_concept_resolves_to_a_file(self, chain: Chain) -> None:
        # `knowledge validate` reads index front matter, never index links.
        concepts = OKFBundle(chain.bundle).list_concepts()

        assert concepts
        for entry in concepts:
            assert (chain.bundle / f"{entry.concept_id}.md").is_file(), entry.concept_id

    def test_a_second_regen_index_changes_no_byte(self, chain: Chain) -> None:
        indexes = (chain.bundle / "index.md", chain.bundle / "families" / "index.md")
        _run("knowledge", "regen-index", str(chain.bundle))
        before = [path.read_bytes() for path in indexes]

        _run("knowledge", "regen-index", str(chain.bundle))

        assert [path.read_bytes() for path in indexes] == before


# ===================================================================
# Criterion 8 — the forms only some facilities have
# ===================================================================


class TestFacilitySpecificForms:
    def test_tango_names_bind_as_tango(self, chains: Callable[[str], Chain]) -> None:
        chain = chains("tango")
        protocols = {
            str(value)
            for binding in _bindings(chain.graph)
            for value in _objects(chain.graph, binding, "protocol")
        }
        addresses = {entry["channel"] for entry in chain.database.get_all_channels()}

        assert protocols == {"tango"}
        assert addresses == _kept_addresses(chain.ao, chain.document)

    def test_a_dual_key_field_binds_both_keys_per_slot(
        self, chains: Callable[[str], Chain]
    ) -> None:
        chain = chains("dualkey")
        body = next(
            body
            for _, family, field, body in _signal_groups(chain.ao)
            if family == "SF" and field == "Monitor"
        )
        by_protocol: dict[str, set[str]] = {}
        for binding in _bindings(chain.graph):
            protocol = str(_objects(chain.graph, binding, "protocol")[0])
            address = str(_objects(chain.graph, binding, "fullPv")[0])
            by_protocol.setdefault(protocol, set()).add(address)

        assert set(body) >= set(CHANNEL_KEYS), "the fixture lost its dual-key field"
        assert by_protocol["ca"] and by_protocol["tango"]
        assert not by_protocol["ca"] & by_protocol["tango"]
        assert len(_bindings(chain.graph)) == _census_bindings(chain.root, chain.mapping)

    def test_list_channel_names_answers_per_protocol(self, chains: Callable[[str], Chain]) -> None:
        database = chains("dualkey").database

        default = database.list_channel_names("STOR", "SF", "Monitor")
        tango = database.list_channel_names("STOR", "SF", "Monitor", protocol="tango")

        assert default == ["ST-SF-01:I", "ST-SF-02:I"]
        assert tango == ["st/ps/sf-01/current", "st/ps/sf-02/current"]

    def test_case_duplicate_families_keep_both_populations(
        self, chains: Callable[[str], Chain]
    ) -> None:
        chain = chains("casedup")
        mapped = {chain.mapping.mapped(raw) for raw in ("BPMx", "bpmx")}

        assert set(chain.document["directions"]) >= {"BPMx.Monitor", "bpmx.Monitor"}
        assert len(mapped) == 2, mapped
        families = {entry["name"] for entry in chain.database.list_families("MAIN")}
        assert mapped <= families
        for token in mapped:
            page = chain.bundle / "families" / f"MAIN-{token}.md"
            assert page.is_file(), page

    def test_the_corpus_opens_with_its_mapping_provenance(
        self, chains: Callable[[str], Chain]
    ) -> None:
        chain = chains("paired")

        assert chain.ttl.read_text(encoding="utf-8").splitlines()[0] == (
            "# osprey:direction-source mapping"
        )

    def test_the_prompt_snapshot_names_the_mapping_as_the_direction_source(self) -> None:
        from osprey.services.facility_knowledge.seeder import prompt_snapshot

        line = prompt_snapshot.DIRECTION_PROVENANCE_LINES["mapping"]

        assert "MML export" in line
        assert "mapping file" in line


class TestTheReviewersJudgment:
    """What the reviewer answered is what the artifacts hold.

    The three questions a skeleton asks — a row beyond a family's devices, a
    device bound by no channel, a PV shared by several devices — are answered
    in the committed fixtures, and each answer is readable in the corpus, the
    channel database, the DuckDB copy and the family page.
    """

    def test_a_row_carved_into_a_field_stands_beside_the_field_it_left(
        self, chains: Callable[[str], Chain]
    ) -> None:
        """NSLS-II's DCCT reads three quantities off one device, one field each."""
        chain = chains("nsls2")
        prefix = _device_prefix(chain, "StorageRing", "DCCT")
        exported = set(chain.ao["StorageRing"]["DCCT"]["Monitor"]["ChannelNames"])

        bindings = _device_bindings(chain, prefix)
        family = chain.database.data["StorageRing"]["DCCT"]

        assert list(bindings.values()) == [["Lifetime", "Monitor", "Total"]]
        landed = [family[field]["ChannelNames"] for field in ("Lifetime", "Monitor", "Total")]
        assert all(len(slots) == 1 for slots in landed), landed
        assert {slots[0] for slots in landed} == exported

    def test_a_carved_out_field_keeps_the_units_of_the_field_it_left(
        self, chains: Callable[[str], Chain]
    ) -> None:
        """A carved row is served in the hardware unit its own field states."""
        chain = chains("nsls2")
        units = _duck_units(chain.root)

        family = chain.database.data["StorageRing"]["DCCT"]
        for field in ("Lifetime", "Monitor", "Total"):
            channel = family[field]["ChannelNames"][0]
            assert units[channel] == "mA", field

    def test_a_dropped_device_leaves_the_family_at_its_bound_devices(
        self, chains: Callable[[str], Chain]
    ) -> None:
        """NSLS-II drops its third tune device, and nothing counts it again."""
        chain = chains("nsls2")
        prefix = _device_prefix(chain, "StorageRing", "TUNE")
        exported = chain.ao["StorageRing"]["TUNE"]["DeviceList"]

        bindings = _device_bindings(chain, prefix)
        page = (chain.bundle / "families" / "StorageRing-TUNE.md").read_text(encoding="utf-8")

        assert len(exported) == 3, "the fixture lost the device the reviewer drops"
        assert len(chain.database.data["StorageRing"]["TUNE"]["setup"]["DeviceList"]) == 2
        assert sorted(bindings.values()) == [["Monitor"], ["Monitor"]]
        assert "- Devices: 2" in page

    def test_the_profile_files_both_shapes_as_questions_for_the_reviewer(
        self, chains: Callable[[str], Chain]
    ) -> None:
        """A shape a reviewer must rule on is asked about, not filed as a hazard.

        ``PROFILE.md`` is rendered by ``import``, off the raw export and before
        any answer exists. A one-device family whose channel list runs to three
        rows, and a three-device family whose longest list reaches two, are both
        questions rather than partial lists, so they stand under ``Judgment
        required`` with the words the ``judgments:`` block takes.
        """
        chain = chains("nsls2")

        asked = _profile_blocks(chain.profile, "Judgment required")
        partial = _profile_blocks(chain.profile, "Partial channel lists")

        assert "`StorageRing.DCCT`, 1 device" in asked
        for signal in chain.ao["StorageRing"]["DCCT"]["Monitor"]["ChannelNames"][1:]:
            assert f"row beyond devices `Monitor` `ChannelNames` `{signal}`" in asked
        assert "`StorageRing.TUNE`, 3 devices" in asked
        assert "unbound device ordinal 3" in asked
        assert "DCCT" not in partial
        assert "TUNE" not in partial

    def test_a_carved_field_inherits_the_metadata_of_the_field_it_left(
        self, chains: Callable[[str], Chain]
    ) -> None:
        """A new field is served like its source, and says which family it serves."""
        chain = chains("casedup")
        source = chain.ao["MAIN"]["CH"]["Monitor"]

        carved = chain.database.data["MAIN"]["CH"]["SumCurrent"]
        row = _duck_channel(chain.root, "MN-CH:Sum-Mon")

        assert carved["MemberOf"] == source["MemberOf"]
        assert carved["HWUnits"] == source["HWUnits"]
        assert row["units"] == source["HWUnits"]
        assert row["member_of"] == ", ".join(source["MemberOf"])

    def test_a_row_answered_device_mints_a_device_the_broadcast_reaches(
        self, chains: Callable[[str], Chain]
    ) -> None:
        """A fourth skew quad is a device, and the shared setpoint drives it too."""
        chain = chains("paired")
        prefix = _device_prefix(chain, "RING", "SQ")

        bindings = _device_bindings(chain, prefix)

        assert bindings[f"{prefix}4"] == ["Monitor", "Setpoint"]
        assert sum(len(fields) for fields in bindings.values()) == 8
        assert chain.database.data["RING"]["SQ"]["Setpoint"]["ChannelNames"] == ["QK:SQ:SP"] * 4

    def test_a_kept_device_is_a_device_with_nothing_bound_to_it(
        self, chains: Callable[[str], Chain]
    ) -> None:
        """A device the reviewer keeps is addressable, and answers with nothing."""
        chain = chains("dialect")
        prefix = _device_prefix(chain, "BOOST", "TUNE")

        bindings = _device_bindings(chain, prefix)
        channels = chain.database.data["BOOST"]["TUNE"]["Monitor"]["ChannelNames"]

        assert bindings[f"{prefix}3"] == []
        assert channels == ["BS:TUNE1:Freq", "BS:TUNE2:Freq", ""]
        assert chain.database.list_channel_names("BOOST", "TUNE", "Monitor", devices=[3]) == []

    def test_a_shared_pv_answered_by_owner_reaches_that_device_alone(
        self, chains: Callable[[str], Chain]
    ) -> None:
        """Two Quokka quadrupoles read one record, and the owner answers for it."""
        chain = chains("wrapped")
        prefix = _device_prefix(chain, "INJ", "QM")
        database = chain.database

        bindings = _device_bindings(chain, prefix)
        channels = database.data["INJ"]["QM"]["Monitor"]["ChannelNames"]

        assert channels == ["IJ:QM1:RB", "IJ:QM2:RB", ""]
        assert database.list_channel_names("INJ", "QM", "Monitor", devices=[2]) == ["IJ:QM2:RB"]
        assert database.list_channel_names("INJ", "QM", "Monitor", devices=[3]) == []
        assert bindings[f"{prefix}3"] == ["On", "OnControl", "Setpoint"]

    def test_an_owner_answer_costs_a_binding_and_no_channel(
        self, chains: Callable[[str], Chain]
    ) -> None:
        """A device the answer passes over gives up its slot, never the PV.

        The facility keeps every channel string it exported, because the owning
        device still binds the shared one; only the binding the other device
        made to it leaves the count.
        """
        chain = chains("wrapped")
        kept = _kept_addresses(chain.ao, chain.document)

        assert _census_bindings(chain.root, chain.mapping) == _raw_census_bindings(chain.root) - 1
        assert "IJ:QM2:RB" in kept
        assert "IJ:QM2:RB" in _duck_units(chain.root)

    def test_the_profile_asks_about_a_supply_group_and_lists_both_its_owners(
        self, chains: Callable[[str], Chain]
    ) -> None:
        """A shared PV is a question, and every device that carries it is named.

        ``PROFILE.md`` is rendered by ``import``, off the raw export and before
        any answer exists, so both devices stand under ``Shared PVs`` however
        the reviewer later answers, and the group they form stands under
        ``Judgment required`` with the words the ``judgments:`` block takes.
        """
        chain = chains("wrapped")

        asked = _profile_blocks(chain.profile, "Judgment required")
        shared = _shared_pv_section(chain.profile)

        assert "supply group 2: `keep_all | {2: <owning ordinal>}`" in asked
        assert "PVs `IJ:QM2:RB`" in asked
        assert "(INJ, QM, Monitor, 2)" in shared
        assert "(INJ, QM, Monitor, 3)" in shared


class TestTheDerivedViews:
    """The DuckDB copy and the facility page say what the source says.

    Both are derived views: the SQL surface holds the unit a field is served
    in, never the MML ``Units`` mode word, and the facility page names every
    sub-machine whether or not accelerator data reached it.
    """

    def test_duckdb_units_hold_the_engineering_unit(self, chain: Chain) -> None:
        """Every ``channels.units`` is a unit the export states, or nothing."""
        exported = _exported_units(chain.ao)

        for name, unit in _duck_units(chain.root).items():
            assert unit in exported.get(name, {""}), (chain.name, name, unit)

    def test_no_duckdb_row_holds_a_units_mode_word(self, chain: Chain) -> None:
        """``Hardware``/``Physics`` name which unit serves a field, not the unit."""
        held = set(_duck_units(chain.root).values())

        assert not held & {"Hardware", "Physics", "[]"}, chain.name

    def test_the_facility_page_names_every_sub_machine_with_its_prose(self, chain: Chain) -> None:
        """One section per system in ``section_order``, led by the mapping's prose."""
        body = (chain.bundle / "facility.md").read_text(encoding="utf-8")

        assert f"- Sub-machines: {', '.join(chain.mapping.section_order)}" in body
        for name in chain.mapping.section_order:
            assert f"## {name}" in body, chain.name
        for entry in chain.mapping.systems.values():
            assert entry.description is None or entry.description in body

    def test_the_two_system_fixture_names_both_systems(
        self, chains: Callable[[str], Chain]
    ) -> None:
        """A facility with no accelerator data still shows its structure."""
        body = (chains("dialect").bundle / "facility.md").read_text(encoding="utf-8")

        assert "- Sub-machines: RING, BOOST" in body
        assert body.index("## RING") < body.index("## BOOST")

    def test_the_facility_page_carries_injection_energy_and_lattice_model(
        self, chains: Callable[[str], Chain]
    ) -> None:
        """``InjectionEnergy`` and ``ATModel`` are facts, and energies are labelled."""
        body = (chains("mat").bundle / "facility.md").read_text(encoding="utf-8")

        assert "- Energy (GeV): 2.4" in body
        assert "- Injection energy (GeV): 0.1" in body
        assert "- Lattice file: quokka_booster_lattice" in body

    def test_emit_names_the_bindings_the_duckdb_copy_holds_no_row_of(self, tmp_path: Path) -> None:
        """A shared or broadcast PV is one SQL row, and emit says so as it writes."""
        sources, flags, _ = FIXTURE_IMPORTS["tango"]
        root = tmp_path / "tango"
        root.mkdir()
        (root / "profile.yml").write_text("name: scratch\n", encoding="utf-8")
        where = ("--repo", str(root))
        _run("mml", "import", *(str(FIXTURES / source) for source in sources), *flags, *where)
        _run("mml", "map", "--init", *where)
        shutil.copy(FIXTURES / "tango" / "mapping.yaml", root / "data" / "mml" / "mapping.yaml")
        _run("mml", "map", "--check", *where)

        output = _run("mml", "emit", "--duckdb", *where).output

        assert "4 of 17 bindings share a PV with another" in output
        assert "the DuckDB channels table holds 13 rows" in output
        assert "ring/mag/cor-c3/current is bound by" in output
        assert "RING.SF.Setpoint broadcasts one TangoNames entry to every device" in output


class TestTheSourceSpelling:
    """What the export says survives the lane it arrived on, and a rename."""

    def test_index_and_status_arrays_are_integers_in_both_lanes(self, chain: Chain) -> None:
        """A MATLAB double index reaches ``ao.json`` spelled as the JSON lane spells it."""
        for system, family, key, item in _integral_slots(chain.ao):
            assert not isinstance(item, float), (chain.name, system, family, key, item)

    def test_raw_type_is_the_export_word_not_the_rename(
        self, chains: Callable[[str], Chain]
    ) -> None:
        """``rawType`` falls back to the family token the export itself carries."""
        chain = chains("casedup")
        predicate = URIRef(f"{NARAD_P}rawType")

        held = {str(o) for o in chain.graph.objects(None, predicate)}

        assert "bpmx" in held
        assert "bpmx_slow" not in held

    def test_an_unaligned_family_array_reaches_neither_paradigm(
        self, chains: Callable[[str], Chain]
    ) -> None:
        """A one-slot ``CommonNames`` on a two-device family names no device."""
        chain = chains("dialect")
        setup = chain.database.data["RING"]["QF"]["setup"]

        assert "CommonNames" not in setup
        assert chain.database.get_common_names("RING", "QF") is None

    def test_a_per_device_unit_list_reaches_the_field_prose(
        self, chains: Callable[[str], Chain]
    ) -> None:
        """One unit repeated per device is still that field's hardware unit."""
        page = (chains("wrapped").bundle / "families" / "INJ-QM.md").read_text(encoding="utf-8")

        assert "hardware units Amps" in page


# ===================================================================
# Criterion 5 — the ALS lane, which never enters the repo
# ===================================================================


@pytest.mark.skipif(
    not (os.environ.get(ALS_EXPORT_ENV) and os.environ.get(ALS_MAPPING_ENV)),
    reason=(
        f"{ALS_EXPORT_ENV} and {ALS_MAPPING_ENV} are not both set; "
        "the ALS full-chain lane needs the real export and its reviewed mapping"
    ),
)
def test_the_als_export_chains_to_its_pinned_counts(tmp_path: Path) -> None:
    export = Path(os.environ[ALS_EXPORT_ENV]).expanduser().resolve()
    mapping = Path(os.environ[ALS_MAPPING_ENV]).expanduser().resolve()

    chain = run_chain(tmp_path / "als", (str(export),), (), mapping, name="als")

    assert _census_bindings(chain.root, chain.mapping) == ALS_BINDINGS
    assert len(_bindings(chain.graph)) == ALS_BINDINGS
    assert len(_kept_addresses(chain.ao, chain.document)) == ALS_DISTINCT_PVS
    counts = _build_index_counts(chain.root, chain.ttl, tmp_path / "index.json")
    assert counts == {"bindings": ALS_BINDINGS, "channels": ALS_DISTINCT_PVS}
    for record in chain.passes:
        assert record.duck["channels"] == ALS_DISTINCT_PVS
        assert record.duck["systems"] == len(chain.mapping.section_order)
    assert chain.passes[1].artifacts == chain.passes[0].artifacts
