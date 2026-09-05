"""Scale guard for the flat search index: does it still answer at machine size?

The rest of the lane proves the index is *correct* on corpora small enough to
count by hand. This module proves it is *usable* on one that is not. A real
facility corpus is two orders of magnitude larger than the shipped demo, so the
question this file answers is whether the flat table and its seven-statement
search stay inside an operator's patience when the ``bindings`` table holds a
hundred thousand rows rather than three thousand.

Four things are measured, and every one of them prints what it measured before
it asserts anything. What the printed lines are for is the *trend* -- a build
that goes from forty seconds to ninety has regressed even though it never
failed, and the run history is where that is visible.

The latency budgets are held to a workstation's numbers rather than widened
until a loaded runner fits inside them, which is why this module does not run
in the parallel lane. Measured: a search over the hundred-thousand-row index
takes 42 ms here and 298 ms in the shared lane, and the whole parity matrix
over the demo corpus 5.5 ms here and 34 ms there -- the same ~7x on a corpus of
a hundred thousand rows and on one of three thousand. A factor that does not
move with the data is the box, not the code: four xdist workers on a four-vCPU
runner, with DuckDB asking for cores none of them can spare. Coverage
instrumentation was measured too and is not the cause (+6%). Budgets widened to
survive that contention would sit twelve times above the real number and could
no longer see a threefold regression, so the tests are taken out of the lane
instead and the budgets left where the measurement puts them.

The whole module is therefore marked ``channel_finder_benchmark`` as well as
``slow``: it is deselected from CI's unit lane and runs on demand, in the
benchmark job (``gh workflow run ci.yml -f run_benchmarks=true``) or locally
with ``pytest tests/services/channel_finder/graph_index/test_scale.py``. The
cost of that is real and worth naming: nothing runs these automatically, so a
latency regression lands unnoticed until someone asks for a benchmark run.

:data:`PARITY_MATRIX` is exported for the parity lane, which replays the same
filter shapes against the store and against the index and compares the answers.
It lives here because this module is what proves the shapes are cheap enough to
replay in bulk.
"""

from __future__ import annotations

import logging
import time
from importlib.resources import as_file, files
from pathlib import Path
from statistics import median
from typing import Any

import pytest

from osprey.services.channel_finder.graph_index.builder import (
    CALLER_META_KEYS,
    EDGE_READS,
    EDGE_WRITES,
    BindingRow,
    ClassRow,
    build_from_rows,
    build_graph_index,
    channels_from_rows,
)
from osprey.services.channel_finder.graph_index.reader import (
    GraphIndex,
    GraphIndexAbsence,
    open_graph_index,
)
from osprey.services.channel_finder.graph_index.taxonomy import class_name

pytestmark = [pytest.mark.slow, pytest.mark.channel_finder_benchmark]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------

#: Writing a hundred thousand derived rows into a fresh DuckDB file.
#:
#: A workstation takes about 1.0 s in all -- 0.01 ms per binding row -- of
#: which roughly two thirds is the two bulk loads: the twelve-column
#: ``bindings`` table and the three-column ``channels`` table pivoted into
#: registered frames and read columnwise by ``INSERT ... SELECT``. The history
#: is worth keeping: the writer first inserted row by row through
#: ``executemany``, which cost ~0.26 ms per binding row and ~0.13 ms per
#: channel row whatever the batch size (26 s and 13 s, 36.7 s in all), because
#: that walks DuckDB's prepared-statement path once per row. The ``LIST``
#: columns were never the cost; the per-row path was.
#:
#: So the budget is the criterion the proposal states, with room for a CI box a
#: good deal slower than a laptop. It is not a target: it is the point past
#: which the writer has stopped being a bulk load and started being something
#: else. A build that lands near it should be read as the columnar path having
#: quietly become a per-row one again, not as a flake.
BUILD_BUDGET_SECONDS = 20.0

#: The whole build -- read, parse, derive, write -- over the shipped demo corpus,
#: the one every deployment's first ``osprey build`` runs. A workstation takes
#: about a second. The same build took 7.8 s on a saturated four-worker macOS
#: runner in the shared lane, with a 156-second sibling on the same worker,
#: which is the measurement that moved this guard here from ``test_build.py``.
DEMO_BUILD_SECONDS = 5.0

#: The median of twenty varied searches over the hundred-thousand-row index.
#: The finder redraws its page on every filter click, so this is the number the
#: interaction is made of.
SCALE_SEARCH_P50_SECONDS = 0.150

#: The first search after opening the *demo* index -- cold caches, nothing
#: mapped yet.
DEMO_FIRST_SEARCH_SECONDS = 0.100

#: The median of the whole parity matrix over the demo index, once warm. The
#: measured number is around five milliseconds, so this is a generous bound
#: whose job is to catch a change of order, not a slow afternoon.
DEMO_WARM_P50_SECONDS = 0.030


# ---------------------------------------------------------------------------
# The synthetic machine
# ---------------------------------------------------------------------------

#: 500 sections x 20 device families x 10 ordinals = 100 000 bindings.
SECTION_COUNT = 500
ORDINAL_COUNT = 10

_NARAD = "https://narad.example.org/schema/shared_semantics/"
_DEVICE = "https://narad.example.org/data/device/"
_BINDING = "https://narad.example.org/data/binding/"

#: The synthetic taxonomy: every family names a leaf class, and most leaves
#: roll up through a mid class to ``AcceleratorDevice``. Keeping it shallow and
#: explicit is deliberate -- the ancestor walk itself is pinned in
#: ``test_parse_corpus.py``; what is wanted here is a realistic *width* of
#: ``class_uris`` per row, not another test of the walk.
_ROOT = f"{_NARAD}AcceleratorDevice"

#: The ancestor chains, nearest first. ``_ROOT_CHAIN`` is the two-class case
#: (a device type hanging straight off the root); a family with an empty chain
#: is the one-class case.
_MAGNET_CHAIN = ("Magnet", "AcceleratorDevice")
_INSTRUMENT_CHAIN = ("Instrumentation", "AcceleratorDevice")
_VACUUM_CHAIN = ("Vacuum", "AcceleratorDevice")
_RF_CHAIN = ("RadioFrequency", "AcceleratorDevice")
_ROOT_CHAIN = ("AcceleratorDevice",)

#: ``(family, system, leaf class, ancestor classes, signal suffixes, edge kind)``.
#: ``ancestors`` is the chain *above* the leaf, nearest first, so a row carries
#: one, two or three class URIs depending on how deeply its family is
#: classified -- the same unevenness a real ontology has, where a handful of
#: device types hang off the root and one or two are not classified at all. The
#: edge kinds cycle through all four directions the rail filters on, so the
#: ``dir`` facet over the synthetic index is populated in every bucket.
_FAMILIES: tuple[tuple[str, str, str, tuple[str, ...], tuple[str, ...], str], ...] = (
    ("BPM", "DIAG", "BeamPositionMonitor", _INSTRUMENT_CHAIN, ("X", "Y", "STATUS"), "R"),
    ("BLM", "DIAG", "BeamLossMonitor", _INSTRUMENT_CHAIN, ("LOSS", "STATUS"), "R"),
    ("BCM", "DIAG", "BeamCurrentMonitor", _ROOT_CHAIN, ("CURRENT", "STATUS"), "R"),
    ("SCR", "DIAG", "Screen", _INSTRUMENT_CHAIN, ("IMAGE", "IN", "OUT"), "RW"),
    ("QF", "MAG", "Quadrupole", _MAGNET_CHAIN, ("SP", "RB", "ON", "FAULT"), "RW"),
    ("QD", "MAG", "Quadrupole", _MAGNET_CHAIN, ("SP", "RB", "ON"), "RW"),
    ("SF", "MAG", "Sextupole", _MAGNET_CHAIN, ("SP", "RB"), "W"),
    ("SD", "MAG", "Sextupole", _MAGNET_CHAIN, ("SP", "RB"), "W"),
    ("HCM", "MAG", "HCorrector", _MAGNET_CHAIN, ("SP", "RB", "ON"), "RW"),
    ("VCM", "MAG", "VCorrector", _MAGNET_CHAIN, ("SP", "RB", "ON"), "RW"),
    ("BEND", "MAG", "Dipole", _MAGNET_CHAIN, ("SP", "RB", "ON", "FAULT"), "RW"),
    ("SKQ", "MAG", "SkewQuadrupole", _MAGNET_CHAIN, ("SP", "RB"), "W"),
    ("IVU", "MAG", "InsertionDevice", _ROOT_CHAIN, ("GAP", "GAP_RB"), "RW"),
    ("IPMP", "VAC", "Pump", _VACUUM_CHAIN, ("CURRENT", "ON"), "R"),
    ("GAUGE", "VAC", "Gauge", _VACUUM_CHAIN, ("PRESSURE", "STATUS"), "R"),
    ("VALVE", "VAC", "Valve", _VACUUM_CHAIN, ("OPEN", "CLOSED", "CMD"), "RW"),
    ("CAV", "RF", "AcceleratingCavity", _RF_CHAIN, ("AMP", "PHASE", "TUNE"), "RW"),
    ("MOD", "RF", "Modulator", _ROOT_CHAIN, ("VOLTAGE", "ON"), "W"),
    ("LLRF", "RF", "LowLevelRF", _RF_CHAIN, ("SETPOINT", "READBACK"), "RW"),
    ("PLATE", "VAC", "Aperture", (), ("POSITION",), "none"),
)

#: The device families, in the order above -- handy for shaping filters.
FAMILY_NAMES = tuple(family for family, *_ in _FAMILIES)

#: Every system code the synthetic machine uses.
SYSTEM_CODES = ("DIAG", "MAG", "VAC", "RF")


def _section_code(index: int) -> str:
    """``SR001`` .. ``SR500`` -- a section code an operator would recognise."""
    return f"SR{index + 1:03d}"


def _edges_for(kind: str) -> list[str]:
    """The ``edges`` list the reader turns into ``R``/``W``/``RW``/``none``."""
    if kind == "R":
        return [EDGE_READS]
    if kind == "W":
        return [EDGE_WRITES]
    if kind == "RW":
        return sorted((EDGE_READS, EDGE_WRITES))
    return []


def _signal_names(family: str, suffixes: tuple[str, ...]) -> list[str]:
    """Two to four signal names, spelled the way the demo corpus spells them."""
    return sorted(f"{family.lower()}_{suffix.lower()}" for suffix in suffixes)


def _haystack(parts: list[str]) -> str:
    """The lowercased join the builder writes into ``bindings.haystack``.

    ``parse_corpus`` builds this inline rather than through a named helper, so
    the join rule is replicated here rather than imported: ``fullPv``, then the
    description, the device name, the signal names and the class names, joined
    with single spaces and lowercased. If the builder ever exposes the helper,
    this function should be deleted in favour of it -- a synthetic haystack
    that drifts from the real one would make every token filter below a test of
    nothing.
    """
    return " ".join(parts).lower()


def synthetic_binding_rows() -> list[BindingRow]:
    """A hundred thousand ``BindingRow``s over a plausible synthetic machine.

    One row per ``(section, family, ordinal)``: 500 x 20 x 10. Each row carries
    a realistic ``fullPv``, a prose description, a device name, two to four
    signal names and one to three class URIs, and the rows are sorted the way
    ``parse_corpus`` sorts them so the written table is ordered as a real build
    would order it.
    """
    rows: list[BindingRow] = []
    for section_index in range(SECTION_COUNT):
        section = _section_code(section_index)
        for family, system, leaf, ancestors, suffixes, edge_kind in _FAMILIES:
            # One to three classes: the leaf alone for a family the ontology
            # never classified, the leaf and the root for one that hangs
            # straight off it, the whole chain for the rest.
            class_uris = sorted(f"{_NARAD}{name}" for name in (leaf, *ancestors))
            names_of_classes = [class_name(uri) for uri in class_uris]
            signal_names = _signal_names(family, suffixes)
            edges = _edges_for(edge_kind)
            for ordinal in range(1, ORDINAL_COUNT + 1):
                device_name = f"{section}C:{family}{ordinal}"
                device_uri = f"{_DEVICE}{section}_{family}_{ordinal}"
                full_pv = f"{device_name}:{suffixes[0]}"
                description = f"{leaf} {ordinal} in section {section} of the {system} system"
                rows.append(
                    BindingRow(
                        binding_uri=f"{_BINDING}{section}_{family}_{ordinal}",
                        full_pv=full_pv,
                        description=description,
                        device_uri=device_uri,
                        device_name=device_name,
                        section=section,
                        system=system,
                        edges=list(edges),
                        signal_uris=[f"{_NARAD}signal/{name}" for name in signal_names],
                        signal_names=list(signal_names),
                        class_uris=list(class_uris),
                        haystack=_haystack(
                            [full_pv, description, device_name, *signal_names, *names_of_classes]
                        ),
                    )
                )
    rows.sort(key=lambda row: (row.full_pv, row.device_uri, row.binding_uri))
    return rows


def synthetic_class_rows(rows: list[BindingRow]) -> list[ClassRow]:
    """The ``classes`` table for *rows*, counted the way the ontology counts it.

    ``direct_devices`` is the devices that name the class outright;
    ``rollup_devices`` adds the devices of every class beneath it. The
    synthetic taxonomy is two levels under one root, so the rollup is worked
    out from the parent links rather than walked.
    """
    direct: dict[str, set[str]] = {}
    parents: dict[str, set[str]] = {}
    for _family, _system, leaf, ancestors, _suffixes, _edge in _FAMILIES:
        chain = [f"{_NARAD}{name}" for name in (leaf, *ancestors)]
        for uri in chain:
            parents.setdefault(uri, set())
        # Each link points at the next class up; the topmost has no parent.
        for child, parent in zip(chain, chain[1:], strict=False):
            parents[child].add(parent)

    for row in rows:
        for uri in row.class_uris:
            direct.setdefault(uri, set()).add(row.device_uri)

    # Rollup: a class holds its own devices plus every descendant's.
    children: dict[str, set[str]] = {uri: set() for uri in parents}
    for child, ancestors in parents.items():
        for ancestor in ancestors:
            children.setdefault(ancestor, set()).add(child)

    def _rollup(uri: str, seen: set[str]) -> set[str]:
        if uri in seen:
            return set()
        seen.add(uri)
        devices = set(direct.get(uri, ()))
        for child in children.get(uri, ()):
            devices |= _rollup(child, seen)
        return devices

    class_rows = [
        ClassRow(
            uri=uri,
            name=class_name(uri),
            alt_labels=[],
            parents=sorted(parents.get(uri, ())),
            direct_devices=len(direct.get(uri, ())),
            rollup_devices=len(_rollup(uri, set())),
        )
        for uri in sorted(direct)
    ]
    class_rows.sort(key=lambda row: row.name)
    return class_rows


# ---------------------------------------------------------------------------
# The parity matrix
# ---------------------------------------------------------------------------

_DEMO_CLASS_MAGNET = f"{_NARAD}Magnet"
_DEMO_CLASS_BPM = f"{_NARAD}BeamPositionMonitor"
_DEMO_CLASS_ROOT = _ROOT

#: Every filter shape the parity lane replays against the store and against
#: the index, and the shapes this module times over the demo corpus. The values
#: are the demo machine's own: sections ``SR``/``BR``/``BTS``, systems
#: ``MAG``/``DIAG``/``VAC``/``RF``, classes from the ``narad`` shared
#: semantics, and signal names the demo bindings actually carry.
#:
#: Imported by the parity lane as::
#:
#:     from tests.services.channel_finder.graph_index.test_scale import PARITY_MATRIX
#:
#: Keep it a list of plain ``dict``s of keyword arguments to
#: :meth:`GraphIndex.search` -- the parity lane passes each one straight
#: through, and translates the same keys into the store's Cypher parameters.
PARITY_MATRIX: list[dict[str, Any]] = [
    # -- no filter at all, and paging through it ---------------------------
    {},
    {"skip": 50},
    {"skip": 2500},
    {"page_size": 10},
    {"page_size": 200},
    {"facet_cap": 5},
    # -- tokens only -------------------------------------------------------
    {"tokens": ["sr"]},
    {"tokens": ["bpm"]},
    {"tokens": ["quad"]},
    {"tokens": ["current"]},
    {"tokens": ["magnet", "current"]},
    {"tokens": ["setpoint"]},
    {"tokens": ["readback"]},
    {"tokens": ["nothingmatchesthis"]},
    # -- one section (which also excludes every device placed nowhere) -----
    {"sections": ["SR"]},
    {"sections": ["BR"]},
    {"sections": ["BTS"]},
    {"sections": ["SR", "BR"]},
    {"sections": ["SR"], "skip": 50},
    # -- one system --------------------------------------------------------
    {"systems": ["MAG"]},
    {"systems": ["DIAG"]},
    {"systems": ["VAC"]},
    {"systems": ["RF"]},
    {"systems": ["MAG", "DIAG"]},
    # -- section and system together ---------------------------------------
    {"sections": ["SR"], "systems": ["MAG"]},
    {"sections": ["SR"], "systems": ["DIAG"]},
    {"sections": ["BR"], "systems": ["MAG"]},
    {"sections": ["BTS"], "systems": ["DIAG"]},
    # -- class filters, which roll their subclasses up ---------------------
    {"cls": _DEMO_CLASS_ROOT},
    {"cls": _DEMO_CLASS_MAGNET},
    {"cls": _DEMO_CLASS_BPM},
    {"cls": _DEMO_CLASS_MAGNET, "sections": ["SR"]},
    {"cls": _DEMO_CLASS_BPM, "systems": ["DIAG"]},
    # -- signal filters ----------------------------------------------------
    {"signals": ["bpm_position_x"]},
    {"signals": ["bpm_position_x", "bpm_position_y"]},
    {"signals": ["hcm_current_sp"]},
    {"signals": ["hcm_current_sp"], "sections": ["SR"]},
    # -- direction filters -------------------------------------------------
    {"dirs": ["R"]},
    {"dirs": ["W"]},
    {"dirs": ["RW"]},
    {"dirs": ["none"]},
    {"dirs": ["R", "W"]},
    {"dirs": ["W"], "systems": ["MAG"]},
    # -- everything at once, and page two of it ----------------------------
    {
        "tokens": ["sr"],
        "sections": ["SR"],
        "systems": ["MAG"],
        "cls": _DEMO_CLASS_MAGNET,
        "dirs": ["R"],
    },
    {
        "tokens": ["sr"],
        "sections": ["SR"],
        "systems": ["MAG"],
        "cls": _DEMO_CLASS_MAGNET,
        "dirs": ["R"],
        "skip": 50,
    },
    {"tokens": ["bpm"], "signals": ["bpm_position_x"], "dirs": ["R"], "page_size": 100},
]

#: Twenty shapes over the synthetic machine, whose facet values are its own.
SCALE_SHAPES: list[dict[str, Any]] = [
    {},
    {"tokens": ["sr001"]},
    {"tokens": ["quadrupole"]},
    {"tokens": ["magnet", "sp"]},
    {"tokens": ["bpm"]},
    {"sections": ["SR001"]},
    {"sections": ["SR001", "SR002", "SR003"]},
    {"systems": ["MAG"]},
    {"systems": ["DIAG"]},
    {"sections": ["SR001"], "systems": ["MAG"]},
    {"cls": _ROOT},
    {"cls": f"{_NARAD}Magnet"},
    {"cls": f"{_NARAD}Quadrupole", "sections": ["SR001"]},
    {"signals": ["bpm_x"]},
    {"signals": ["qf_sp", "qd_sp"]},
    {"dirs": ["R"]},
    {"dirs": ["RW"]},
    {"dirs": ["none"]},
    {"dirs": ["R", "W"], "systems": ["MAG"]},
    {"tokens": ["sr250"], "skip": 50, "page_size": 100, "facet_cap": 50},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


#: The shapes the *demo* corpus answers with nothing, and legitimately so: a
#: token no binding carries, and the two directions the demo machine happens
#: not to use (every demo binding either reads or writes, never both, and none
#: is unconnected). They stay in the matrix because a parity lane has to prove
#: the two backends agree about emptiness too. Every *other* shape must match
#: something -- a class URI or a signal name typed wrong would otherwise sit in
#: the matrix looking like coverage while testing nothing.
DEMO_EMPTY_SHAPES: list[dict[str, Any]] = [
    {"tokens": ["nothingmatchesthis"]},
    {"dirs": ["RW"]},
    {"dirs": ["none"]},
]


def _open(index_path: Path) -> GraphIndex:
    """Open *index_path*, failing the test rather than returning an absence."""
    index = open_graph_index(index_path)
    if isinstance(index, GraphIndexAbsence):
        pytest.fail(f"the index could not be opened: {index.detail}")
    return index


def _time_shapes(index: GraphIndex, shapes: list[dict[str, Any]]) -> list[float]:
    """Run every shape once, returning the seconds each one took."""
    timings = []
    for shape in shapes:
        started = time.perf_counter()
        index.search(**shape)
        timings.append(time.perf_counter() - started)
    return timings


def _report(label: str, timings: list[float]) -> tuple[float, float, float]:
    """Print and log ``p50``/``p95``/``max`` for *timings*, and return them."""
    ordered = sorted(timings)
    p50 = median(ordered)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    worst = ordered[-1]
    line = (
        f"{label}: {len(timings)} shapes, "
        f"p50 {p50 * 1000:.1f} ms, p95 {p95 * 1000:.1f} ms, max {worst * 1000:.1f} ms"
    )
    print(line)
    logger.info(line)
    return p50, p95, worst


# ---------------------------------------------------------------------------
# The synthetic machine, built and searched
# ---------------------------------------------------------------------------


class TestHundredThousandBindings:
    """A machine two orders of magnitude past the demo, built and searched."""

    @pytest.fixture(scope="class")
    def built(self, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, int, float]:
        rows = synthetic_binding_rows()
        classes = synthetic_class_rows(rows)
        channels = channels_from_rows(rows)
        index_path = tmp_path_factory.mktemp("scale") / "graph.duckdb"
        meta = {
            "corpus_sha256": "c" * 64,
            "corpus_filename": "synthetic_machine.ttl",
            "binding_count": len(rows),
            "device_count": len({row.device_uri for row in rows}),
            "class_count": len(classes),
            "signal_count": len({uri for row in rows for uri in row.signal_uris}),
            "section_count": len({row.section for row in rows if row.section}),
        }
        assert set(meta) == set(CALLER_META_KEYS), sorted(meta)

        started = time.perf_counter()
        report = build_from_rows(rows, classes, channels, index_path, meta)
        elapsed = time.perf_counter() - started

        assert report.binding_count == len(rows)
        return index_path, len(rows), elapsed

    def test_the_synthetic_machine_is_the_size_it_claims(self, built):
        _, row_count, _ = built

        assert row_count == SECTION_COUNT * len(_FAMILIES) * ORDINAL_COUNT
        assert row_count == 100_000

    def test_writing_a_hundred_thousand_rows_stays_inside_its_budget(self, built):
        index_path, row_count, elapsed = built

        size_mib = index_path.stat().st_size / (1024 * 1024)
        line = (
            f"build_from_rows over {row_count} bindings took {elapsed:.2f} s "
            f"({elapsed / row_count * 1000:.3f} ms per binding, "
            f"{size_mib:.1f} MiB on disk)"
        )
        print(line)
        logger.info(line)
        assert elapsed < BUILD_BUDGET_SECONDS, (
            f"writing {row_count} rows took {elapsed:.2f} s, budget {BUILD_BUDGET_SECONDS} s"
        )

    def test_the_index_answers_the_whole_machine(self, built):
        index_path, row_count, _ = built

        with _open(index_path) as index:
            payload = index.search(page_size=10)

        assert payload["total"] == row_count
        assert payload["devices"] == row_count
        assert len(payload["rows"]) == 10
        assert {entry["value"] for entry in payload["facets"]["system"]} == set(SYSTEM_CODES)
        assert {entry["value"] for entry in payload["facets"]["dir"]} == {"R", "W", "RW", "none"}

    def test_a_search_over_a_hundred_thousand_rows_stays_interactive(self, built):
        index_path, _, _ = built

        with _open(index_path) as index:
            index.search()  # Warm the page cache; the cold call is not the guard.
            timings = _time_shapes(index, SCALE_SHAPES)

        p50, _, _ = _report("search over the 100k synthetic index", timings)
        assert p50 < SCALE_SEARCH_P50_SECONDS, (
            f"the median of {len(timings)} searches was {p50 * 1000:.1f} ms, "
            f"budget {SCALE_SEARCH_P50_SECONDS * 1000:.0f} ms"
        )


# ---------------------------------------------------------------------------
# The shipped demo corpus, over the parity matrix
# ---------------------------------------------------------------------------


class TestDemoCorpusOverTheParityMatrix:
    """The corpus a deployment actually ships, over every shape parity replays."""

    @pytest.fixture(scope="class")
    def demo_build(self, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, float]:
        """The demo index, built once for the class, and how long the build took."""
        resource = (
            files("osprey.templates")
            .joinpath("apps")
            .joinpath("control_assistant")
            .joinpath("data")
            .joinpath("demo_machine.ttl")
        )
        index_path = tmp_path_factory.mktemp("demo") / "graph.duckdb"
        with as_file(resource) as path:
            started = time.perf_counter()
            build_graph_index(path, index_path)
            elapsed = time.perf_counter() - started
        return index_path, elapsed

    @pytest.fixture(scope="class")
    def demo_index_path(self, demo_build: tuple[Path, float]) -> Path:
        return demo_build[0]

    def test_building_the_demo_corpus_stays_inside_its_budget(self, demo_build: tuple[Path, float]):
        _, elapsed = demo_build

        line = f"build_graph_index over the demo corpus: {elapsed:.2f} s"
        print(line)
        logger.info(line)
        assert elapsed < DEMO_BUILD_SECONDS, (
            f"building the demo index took {elapsed:.2f} s, budget {DEMO_BUILD_SECONDS} s"
        )

    def test_the_parity_matrix_covers_every_filter_the_rail_offers(self):
        assert len(PARITY_MATRIX) >= 40, len(PARITY_MATRIX)
        assert {} in PARITY_MATRIX, "the unfiltered shape has to be in the matrix"
        assert any("sections" in shape for shape in PARITY_MATRIX)
        assert any("systems" in shape for shape in PARITY_MATRIX)
        assert any("cls" in shape for shape in PARITY_MATRIX)
        assert any("signals" in shape for shape in PARITY_MATRIX)
        assert any("dirs" in shape for shape in PARITY_MATRIX)
        assert any("tokens" in shape for shape in PARITY_MATRIX)
        assert any(shape.get("skip") == 50 for shape in PARITY_MATRIX), "page two is missing"

    def test_every_shape_in_the_matrix_matches_something_real(self, demo_index_path: Path):
        """A shape that matches nothing by accident is coverage that is not there.

        Every filter value in the matrix is spelled the way the demo corpus
        spells it, so every shape but the three deliberately empty ones has to
        come back with rows. This is what catches a class URI or a signal name
        that drifts when the demo corpus is regenerated.
        """
        with _open(demo_index_path) as index:
            empty = [shape for shape in PARITY_MATRIX if index.search(**shape)["total"] == 0]

        assert empty == DEMO_EMPTY_SHAPES, (
            f"the shapes matching nothing on the demo corpus are not the expected ones: {empty}"
        )

    def test_the_first_search_after_opening_answers_at_once(self, demo_index_path: Path):
        index = _open(demo_index_path)
        try:
            started = time.perf_counter()
            payload = index.search()
            elapsed = time.perf_counter() - started
        finally:
            index.close()

        line = f"first search on the demo index: {elapsed * 1000:.1f} ms"
        print(line)
        logger.info(line)
        assert payload["total"] > 0
        assert elapsed < DEMO_FIRST_SEARCH_SECONDS, (
            f"the first search took {elapsed * 1000:.1f} ms, "
            f"budget {DEMO_FIRST_SEARCH_SECONDS * 1000:.0f} ms"
        )

    def test_the_whole_parity_matrix_stays_inside_the_warm_budget(self, demo_index_path: Path):
        with _open(demo_index_path) as index:
            index.search()  # Warm the page cache; the cold call is timed above.
            timings = _time_shapes(index, PARITY_MATRIX)

        p50, _, _ = _report("the parity matrix over the demo index", timings)
        assert p50 < DEMO_WARM_P50_SECONDS, (
            f"the warm median over {len(timings)} shapes was {p50 * 1000:.1f} ms, "
            f"budget {DEMO_WARM_P50_SECONDS * 1000:.0f} ms"
        )
