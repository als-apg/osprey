"""The graph view's bytes: unchanged where no relation is stated, and reproducible.

- A facility with no relation claims and no aliases renders byte-identically to
  the graph writer before relations existed (the writer at ``BASE``, loaded from
  git as a module), under every graph setting: a small tree and the demo build.
  The demo's built graph keeps its ``narad_p:`` predicate set, so the demo pins
  that count its triples (``EXPECTED_TRIPLES_LOADED`` and the census) stand as
  they were at ``BASE``.
- Two builds of the design's Appendix A tree, their claims split over two claim
  files in different orders and the files written in a different order, give
  the same Turtle, header included, for both variants and every setting.
- Rendering one facility file twice in one process with different settings
  gives each setting its own text: the per-build memo never hands one
  setting's Turtle to another.
"""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import random
import subprocess
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from osprey.facility.records import FacilityRecords
from osprey.facility.views.graph import (
    CONTESTED_SUFFIX,
    RELATION_PREDICATES,
    GraphSettings,
    _turtle,
    graph_text,
)
from osprey.facility.views.graph_iri import namespace
from tests.facility._appendix_a import VARIANTS, appendix_a_tree, build_appendix_a
from tests.facility._graph_relations import (
    LOCATED_IN,
    NARAD_P,
    PART_OF,
    expected_relation_edges,
)
from tests.facility.test_graph_view import (
    GRAPH_VIEW,
    PREDICATES,
    _doc,
    _graph,
    _narad_predicates,
)

if TYPE_CHECKING:
    from tests.facility.conftest import BuiltProject

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The commit whose graph writer predates relation claims: the byte reference.
BASE = "d10a3447b7"

ALL_SETTINGS = [
    GraphSettings(claims, basis)
    for claims, basis in itertools.product(("all", "contested", "none"), ("all", "stated"))
]


def _at_base(path: str) -> str:
    return subprocess.run(
        ["git", "show", f"{BASE}:{path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


@pytest.fixture(scope="module")
def base_writer() -> types.ModuleType:
    """The graph view module as it stood at ``BASE``."""
    module = types.ModuleType("base_graph")
    source = _at_base("src/osprey/facility/views/graph.py")
    exec(compile(source, "base_graph.py", "exec"), module.__dict__)
    return module


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TestNoRelationsIsTheBaseWriter:
    def test_the_small_tree_renders_as_the_base_writer_did(
        self, base_writer: types.ModuleType
    ) -> None:
        document = _doc()
        assert "relations" not in document
        reference = _sha(base_writer.graph_text(FacilityRecords(_doc())))
        for settings in ALL_SETTINGS:
            assert _sha(graph_text(FacilityRecords(_doc()), settings)) == reference, settings

    @pytest.mark.slow
    def test_the_demo_renders_as_the_base_writer_did(
        self, base_writer: types.ModuleType, built_demo: BuiltProject
    ) -> None:
        facility = built_demo.facility
        assert not facility.get("relations")
        built = (built_demo.build_dir / GRAPH_VIEW).read_bytes()
        reference = hashlib.sha256(
            base_writer.graph_text(FacilityRecords(facility)).encode("utf-8")
        ).hexdigest()
        assert hashlib.sha256(built).hexdigest() == reference
        for settings in ALL_SETTINGS:
            assert _sha(graph_text(FacilityRecords(facility), settings)) == reference, settings

    @pytest.mark.slow
    def test_the_demo_graph_keeps_its_predicate_set(self, built_demo: BuiltProject) -> None:
        graph = _graph((built_demo.build_dir / GRAPH_VIEW).read_text(encoding="utf-8"))
        assert _narad_predicates(graph) == PREDICATES

    def test_the_demo_triple_pins_stand_as_they_were(self) -> None:
        store_test = "tests/integration/test_graphdb_store.py"
        census = "tests/facility/golden/demo_census.json"

        def pinned(source: str) -> int:
            for node in ast.parse(source).body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "EXPECTED_TRIPLES_LOADED"
                    for target in node.targets
                ):
                    return ast.literal_eval(node.value)
            raise AssertionError("EXPECTED_TRIPLES_LOADED is not assigned")

        current = (REPO_ROOT / store_test).read_text(encoding="utf-8")
        assert pinned(current) == pinned(_at_base(store_test))
        assert json.loads((REPO_ROOT / census).read_text(encoding="utf-8")) == json.loads(
            _at_base(census)
        )


def _split_claims(variant: str, seed: int) -> dict[str, Any]:
    """The Appendix A tree with its claims over two files, shuffled by ``seed``.

    Which file holds a claim is fixed (its provenance names the file); the order
    of the claims inside each file and the order the files are written in are
    not.
    """
    tree = appendix_a_tree(variant)  # type: ignore[arg-type]
    claims = sorted(tree.pop("records/relations.yaml"), key=lambda claim: claim["id"])
    rng = random.Random(seed)
    authored, imported = claims[0::2], claims[1::2]
    rng.shuffle(authored)
    rng.shuffle(imported)
    tree["records/relations.yaml"] = authored
    tree["imported/controls/relations.yaml"] = imported
    paths = list(tree)
    rng.shuffle(paths)
    return {path: tree[path] for path in paths}


class TestShuffledClaimsBuildTheSameGraph:
    @pytest.fixture(scope="class")
    def builds(
        self, tmp_path_factory: pytest.TempPathFactory
    ) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
        built = {}
        for variant in VARIANTS:
            first, second = _split_claims(variant, 1), _split_claims(variant, 2)
            assert first["records/relations.yaml"] != second["records/relations.yaml"]
            assert list(first) != list(second)
            built[variant] = (
                build_appendix_a(tmp_path_factory.mktemp(f"{variant}-1"), tree=first),
                build_appendix_a(tmp_path_factory.mktemp(f"{variant}-2"), tree=second),
            )
        return built

    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize("settings", ALL_SETTINGS, ids=str)
    def test_the_two_builds_hash_equal(
        self,
        builds: dict[str, tuple[dict[str, Any], dict[str, Any]]],
        variant: str,
        settings: GraphSettings,
    ) -> None:
        first, second = builds[variant]
        assert len(first["relations"]) == len(appendix_a_tree()["records/relations.yaml"])
        assert _sha(graph_text(FacilityRecords(first), settings)) == _sha(
            graph_text(FacilityRecords(second), settings)
        )


class TestTheMemoKeepsEachSetting:
    @pytest.fixture(scope="class")
    def records(self, tmp_path_factory: pytest.TempPathFactory) -> FacilityRecords:
        return FacilityRecords(build_appendix_a(tmp_path_factory.mktemp("memo")))

    def test_each_setting_gets_its_own_text(self, records: FacilityRecords) -> None:
        texts = {settings: graph_text(records, settings) for settings in ALL_SETTINGS}
        for settings, text in texts.items():
            assert text == _turtle(records, settings), settings
        assert len(set(texts.values())) == len(ALL_SETTINGS)

    def test_rendering_back_and_forth_changes_the_text_as_specified(
        self, records: FacilityRecords
    ) -> None:
        claim_nodes = namespace("relation")
        for first, second in itertools.permutations(ALL_SETTINGS, 2):
            one, other = graph_text(records, first), graph_text(records, second)
            assert graph_text(records, first) == one
            for settings, text in ((first, one), (second, other)):
                # Claim nodes follow ``claims``; the edges follow ``basis`` only.
                assert (claim_nodes in text) == (settings.claims != "none"), settings
                written, expected = (
                    _relation_edges(text),
                    expected_relation_edges(records, settings),
                )
                # Place parents and device places also write partOf and locatedIn.
                assert expected <= written, settings
                assert {edge for edge in written if edge[1] not in _PLACE_EDGES} == {
                    edge for edge in expected if edge[1] not in _PLACE_EDGES
                }, settings
            # The header hashes the facility file, which no setting changes.
            assert one.split("\n", 1)[0] == other.split("\n", 1)[0]

    def test_the_settings_change_what_appendix_a_writes(self, records: FacilityRecords) -> None:
        everything = graph_text(records, GraphSettings("all", "all"))
        assert CONTESTED_SUFFIX in everything
        assert expected_relation_edges(records, GraphSettings("all", "stated")) < (
            expected_relation_edges(records, GraphSettings("all", "all"))
        )


#: The relation predicates the view also writes from place and device records.
_PLACE_EDGES = frozenset({PART_OF, LOCATED_IN})


def _relation_edges(text: str) -> frozenset[tuple[str, str, str]]:
    """The triples of ``text`` whose predicate is a relation predicate or its twin."""
    names = {
        f"{NARAD_P}{name}{suffix}"
        for name in RELATION_PREDICATES.values()
        for suffix in ("", CONTESTED_SUFFIX)
    }
    graph = _graph(text)
    return frozenset(
        (str(subject), str(predicate), str(obj))
        for subject, predicate, obj in graph
        if str(predicate) in names and not str(subject).startswith(NARAD_P)
    )
