"""Channel-finder view parity against the frozen pre-LINE goldens.

The goldens under ``tests/facility/golden/`` were captured from the demo's
committed sources. Each states in its ``_reproduce`` line the command that
rewrites it, or that it is frozen and none does. This module's loader
reads them, and the tests below hold the frozen invariants every later
comparison relies on: the fingerprint's size, role split and pinned sha256,
the in_context size, the standalone address set, the limits projection and
the byte identity of the pre-LINE channel-finder index copies.

The parity tests hold the hierarchical and middle-layer views, called as pure
functions on the shared control-assistant build's facility file, to those
copies: the same address set less the declared fingerprint additions, the
machine, system and family descriptions reachable, every benchmark target
indexed, each copy's Family with an equal ``DeviceList``, and no fewer
benchmark queries answerable by whole index cells than the copies answer. The in_context view,
called the same way, holds the in_context golden's address set, and every
in_context benchmark target is one of its rows. Each query file is checked
against the indexes its pipeline scores.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from tests.facility.conftest import BuiltProject

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
CF_INDEX_DIR = GOLDEN_DIR / "cf_index_pre_line"

QUERY_DIR = (
    REPO_ROOT / "src/osprey/templates/apps/control_assistant/data/benchmarks/cross_paradigm/queries"
)

#: The benchmark queries the pre-LINE tree indexes are scored against; every
#: query names the addresses it targets in ``targeted_pv``.
PRE_LINE_QUERIES = QUERY_DIR / "tree_queries.json"

#: The benchmark queries the in_context index is scored against.
IN_CONTEXT_QUERIES = QUERY_DIR / "in_context_queries.json"

#: The channel-finder pipelines with an index file, keyed by their index name.
PRE_LINE_INDEXES = ("hierarchical", "in_context", "middle_layer")

FINGERPRINT_SHA256 = "644f41850784e0251ac4be0081c6891b7f933b119efc037a7455ebe89c54e9c0"
FINGERPRINT_ROW_KEYS = ["address", "role", "value_type", "names", "description"]


@cache
def load_golden(name: str) -> Any:
    """Parse ``tests/facility/golden/<name>``."""
    return json.loads((GOLDEN_DIR / name).read_text(encoding="utf-8"))


def fingerprint_rows() -> list[dict[str, Any]]:
    """The frozen demo fingerprint rows, sorted by address."""
    return load_golden("demo_fingerprint.json")["rows"]


def fingerprint_addresses() -> set[str]:
    return {row["address"] for row in fingerprint_rows()}


def in_context_golden_addresses() -> set[str]:
    """The addresses of the frozen in_context golden."""
    return {row["address"] for row in load_golden("in_context_size.json")["rows"]}


@cache
def pre_line_index(index: str) -> Any:
    """Parse the frozen pre-LINE channel-finder index ``index``."""
    return json.loads((CF_INDEX_DIR / f"{index}.json").read_text(encoding="utf-8"))


@cache
def pre_line_queries() -> list[dict[str, Any]]:
    """The benchmark queries the pre-LINE indexes answer."""
    return json.loads(PRE_LINE_QUERIES.read_text(encoding="utf-8"))


@cache
def in_context_queries() -> list[dict[str, Any]]:
    """The benchmark queries the in_context index answers."""
    return json.loads(IN_CONTEXT_QUERIES.read_text(encoding="utf-8"))


def addition_addresses() -> set[str]:
    """The addresses the demo declares beyond the frozen fingerprint."""
    return {row["address"] for row in load_golden("demo_fingerprint_additions.json")["rows"]}


def _children(node: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    for key, child in node.items():
        if not key.startswith("_") and isinstance(child, dict):
            yield key, child


def _instances(expansion: dict[str, Any]) -> list[str]:
    if expansion["_type"] == "list":
        return list(expansion["_instances"])
    low, high = expansion["_range"]
    return [expansion["_pattern"].format(number) for number in range(low, high + 1)]


def hierarchical_leaves(index: dict[str, Any]) -> list[tuple[tuple[str, ...], str]]:
    """Every leaf of a hierarchical index as (its node path, its address).

    A level of type ``instances`` expands its ``_expansion`` node into one path
    step per instance. A leaf's address is its ``_channel_part`` when that is
    non-empty, else the naming pattern filled with its path.
    """
    hierarchy = index["hierarchy"]
    levels = hierarchy["levels"]
    leaves: list[tuple[tuple[str, ...], str]] = []

    def walk(node: dict[str, Any], path: tuple[str, ...]) -> None:
        if len(path) == len(levels):
            names = {level["name"]: step for level, step in zip(levels, path, strict=True)}
            address = node.get("_channel_part") or hierarchy["naming_pattern"].format(**names)
            leaves.append((path, address))
            return
        if levels[len(path)]["type"] == "instances":
            for _key, container in _children(node):
                for instance in _instances(container["_expansion"]):
                    walk(container, (*path, instance))
        else:
            for key, child in _children(node):
                walk(child, (*path, key))

    walk(index["tree"], ())
    return leaves


def hierarchical_channel_parts(index: dict[str, Any]) -> set[str]:
    """The non-empty ``_channel_part`` of every leaf of a hierarchical index."""
    depth = len(index["hierarchy"]["levels"])
    parts: set[str] = set()

    def walk(node: dict[str, Any], level: int) -> None:
        if level == depth:
            if node.get("_channel_part"):
                parts.add(node["_channel_part"])
            return
        for _key, child in _children(node):
            walk(child, level + 1)

    walk(index["tree"], 0)
    return parts


def hierarchical_subtrees(index: dict[str, Any]) -> list[list[frozenset[str]]]:
    """For each level of a hierarchical index, the address set of each node at it."""
    leaves = hierarchical_leaves(index)
    subtrees: list[list[frozenset[str]]] = []
    for depth in range(1, len(index["hierarchy"]["levels"]) + 1):
        nodes: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for path, address in leaves:
            nodes[path[:depth]].add(address)
        subtrees.append([frozenset(addresses) for addresses in nodes.values()])
    return subtrees


def hierarchical_descriptions(index: dict[str, Any]) -> set[str]:
    """The ``_description`` of every node of a hierarchical index above its leaves."""
    depth = len(index["hierarchy"]["levels"])
    found: set[str] = set()

    def walk(node: dict[str, Any], level: int) -> None:
        if level == depth:
            return
        for _key, child in _children(node):
            if child.get("_description"):
                found.add(child["_description"])
            walk(child, level + 1)

    walk(index["tree"], 1)
    return found


def middle_layer_cells(index: dict[str, Any]) -> list[frozenset[str]]:
    """The address set of every ``ChannelNames`` list of a middle-layer index."""
    cells: list[frozenset[str]] = []

    def walk(node: dict[str, Any]) -> None:
        for _key, child in _children(node):
            if "ChannelNames" in child:
                cells.append(frozenset(child["ChannelNames"]))
            else:
                walk(child)

    walk(index)
    return cells


def middle_layer_families(index: dict[str, Any]) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Every (System, Family, Family node) of a middle-layer index."""
    for system, system_node in _children(index):
        for family, family_node in _children(system_node):
            yield system, family, family_node


def is_union_of(target: frozenset[str], cells: Iterable[frozenset[str]]) -> bool:
    """Whether ``target`` is exactly a union of whole ``cells``."""
    covered: set[str] = set()
    for cell in cells:
        if cell <= target:
            covered |= cell
    return covered == target


def answerable_count(hierarchical: dict[str, Any], middle_layer: dict[str, Any]) -> int:
    """How many tree benchmark queries both indexes answer with whole cells.

    A query counts when its ``targeted_pv`` is a union of whole middle-layer
    ``ChannelNames`` cells and a union of whole hierarchical subtrees at some
    level.
    """
    cells = middle_layer_cells(middle_layer)
    subtrees = hierarchical_subtrees(hierarchical)
    count = 0
    for query in pre_line_queries():
        target = frozenset(query["targeted_pv"])
        if is_union_of(target, cells) and any(is_union_of(target, nodes) for nodes in subtrees):
            count += 1
    return count


def golden_family_descriptions() -> dict[tuple[str, str], str]:
    """Each pre-LINE family's description by (machine, family).

    The hierarchical copy keeps them at the family level, under a system level
    the middle-layer copy does not have.
    """
    tree = pre_line_index("hierarchical")["tree"]
    return {
        (machine, family): node["_description"]
        for machine, machine_node in _children(tree)
        for _system, system_node in _children(machine_node)
        for family, node in _children(system_node)
    }


def golden_system_descriptions() -> dict[tuple[str, str], str]:
    """Each pre-LINE system's description by (machine, system).

    The hierarchical copy keeps them at the system level; the middle-layer
    index files each system's group as a Family of its machine.
    """
    tree = pre_line_index("hierarchical")["tree"]
    return {
        (machine, system): node["_description"]
        for machine, machine_node in _children(tree)
        for system, node in _children(machine_node)
    }


def golden_machine_descriptions() -> dict[str, str]:
    """Each pre-LINE machine's description by machine."""
    tree = pre_line_index("hierarchical")["tree"]
    return {machine: node["_description"] for machine, node in _children(tree)}


def _rows_sha256(rows: list[dict[str, Any]]) -> str:
    compact = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(compact.encode("utf-8")).hexdigest()


def _golden_files() -> list[str]:
    return sorted(
        path.relative_to(GOLDEN_DIR).as_posix()
        for path in GOLDEN_DIR.rglob("*.json")
        if path.parent != CF_INDEX_DIR or path.name == "MANIFEST.json"
    )


@pytest.mark.parametrize("name", _golden_files())
def test_every_golden_names_its_reproduce_command(name: str) -> None:
    golden = load_golden(name)
    assert isinstance(golden, dict)
    assert golden.get("_reproduce"), f"{name} has no _reproduce line"


def test_fingerprint_is_frozen() -> None:
    golden = load_golden("demo_fingerprint.json")
    rows = fingerprint_rows()
    assert len(rows) == golden["count"] == 2908
    assert [row["address"] for row in rows] == sorted(fingerprint_addresses())
    assert all(list(row) == FINGERPRINT_ROW_KEYS for row in rows)
    roles = {"setpoint": 0, "readback": 0}
    for row in rows:
        roles[row["role"]] += 1
    assert roles == golden["roles"] == {"setpoint": 396, "readback": 2512}
    assert {row["value_type"] for row in rows} == {"bool", "float"}
    assert sum(row["value_type"] == "bool" for row in rows) == 1246
    assert _rows_sha256(rows) == golden["sha256"] == FINGERPRINT_SHA256


def test_fingerprint_additions_are_new_addresses_in_the_row_shape() -> None:
    rows = load_golden("demo_fingerprint_additions.json")["rows"]
    addresses = [row["address"] for row in rows]
    assert addresses == sorted(set(addresses))
    assert all(list(row) == FINGERPRINT_ROW_KEYS for row in rows)
    assert not set(addresses) & fingerprint_addresses()


def test_in_context_size_is_frozen() -> None:
    golden = load_golden("in_context_size.json")
    assert golden["size"] == len(golden["rows"]) == 569
    assert {row["address"] for row in golden["rows"]} <= fingerprint_addresses()


def test_standalone_address_set_is_frozen() -> None:
    golden = load_golden("cf_standalone_addresses.json")
    addresses = golden["addresses"]
    assert golden["count"] == len(addresses) == 1228
    assert addresses == sorted(set(addresses))


def test_limits_name_only_setpoints_of_the_fingerprint() -> None:
    golden = load_golden("limits.json")
    channels = golden["channels"]
    assert "defaults" not in golden
    assert golden["count"] == len(channels) == 3
    setpoints = {row["address"] for row in fingerprint_rows() if row["role"] == "setpoint"}
    assert set(channels) <= setpoints
    assert all(
        list(record) == ["min_value", "max_value", "max_step", "writable", "confirm"]
        for record in channels.values()
    )


def test_pre_line_index_copies_match_their_manifest() -> None:
    manifest = load_golden("cf_index_pre_line/MANIFEST.json")["files"]
    assert sorted(manifest) == [f"{index}.json" for index in PRE_LINE_INDEXES]
    for name, entry in manifest.items():
        content = (CF_INDEX_DIR / name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == entry["sha256"], name


def test_pre_line_in_context_index_holds_the_fingerprint_addresses() -> None:
    rows = pre_line_index("in_context")["channels"]
    assert {row["address"] for row in rows} == fingerprint_addresses()


def test_pre_line_queries_target_fingerprint_addresses() -> None:
    queries = pre_line_queries()
    assert len(queries) == 60
    addresses = fingerprint_addresses()
    for query in queries:
        assert query["targeted_pv"], query["user_query"]
        assert set(query["targeted_pv"]) <= addresses, query["user_query"]


def test_in_context_queries_target_in_context_golden_addresses() -> None:
    queries = in_context_queries()
    assert len(queries) == 20
    addresses = in_context_golden_addresses()
    for query in queries:
        assert query["targeted_pv"], query["user_query"]
        assert set(query["targeted_pv"]) <= addresses, query["user_query"]


@pytest.fixture(scope="module")
def hierarchical_view(built_control_assistant: BuiltProject) -> dict[str, Any]:
    from osprey.facility.views.channel_finder import hierarchical_document

    return hierarchical_document(built_control_assistant.facility)


@pytest.fixture(scope="module")
def middle_layer_view(built_control_assistant: BuiltProject) -> dict[str, Any]:
    from osprey.facility.views.channel_finder import middle_layer_document

    document, _left_out, _keyed = middle_layer_document(built_control_assistant.facility)
    return document


@pytest.fixture(scope="module")
def in_context_view(built_control_assistant: BuiltProject) -> dict[str, Any]:
    from osprey.facility.views.channel_finder import in_context_document

    return in_context_document(built_control_assistant.facility)


def test_the_pre_line_copies_hold_the_fingerprint_addresses() -> None:
    hierarchical = {
        address for _path, address in hierarchical_leaves(pre_line_index("hierarchical"))
    }
    middle_layer = set().union(*middle_layer_cells(pre_line_index("middle_layer")))
    assert hierarchical == middle_layer == fingerprint_addresses()


def test_the_pre_line_copies_describe_three_machines_and_28_families() -> None:
    families = golden_family_descriptions()
    assert len(golden_machine_descriptions()) == 3
    assert len(families) == 28
    middle_layer = pre_line_index("middle_layer")
    assert {
        (system, family) for system, family, _node in middle_layer_families(middle_layer)
    } == set(families)


@pytest.mark.slow
@pytest.mark.xdist_group("built_control_assistant")
def test_the_hierarchical_view_holds_the_pre_line_addresses(
    hierarchical_view: dict[str, Any],
) -> None:
    golden = {address for _path, address in hierarchical_leaves(pre_line_index("hierarchical"))}
    view = {address for _path, address in hierarchical_leaves(hierarchical_view)}
    assert view - addition_addresses() == golden


@pytest.mark.slow
@pytest.mark.xdist_group("built_control_assistant")
def test_the_middle_layer_view_holds_the_pre_line_addresses(
    middle_layer_view: dict[str, Any],
) -> None:
    golden = set().union(*middle_layer_cells(pre_line_index("middle_layer")))
    view = set().union(*middle_layer_cells(middle_layer_view))
    assert view - addition_addresses() == golden


@pytest.mark.slow
@pytest.mark.xdist_group("built_control_assistant")
def test_the_in_context_view_holds_the_golden_addresses(in_context_view: dict[str, Any]) -> None:
    view = {row["address"] for row in in_context_view["channels"]}
    assert view == in_context_golden_addresses()


@pytest.mark.slow
@pytest.mark.xdist_group("built_control_assistant")
def test_the_hierarchical_view_keeps_the_machine_and_family_descriptions(
    hierarchical_view: dict[str, Any],
) -> None:
    machines = {key: node.get("_description") for key, node in _children(hierarchical_view["tree"])}
    for machine, description in golden_machine_descriptions().items():
        assert machines[machine] == description, machine
    reachable = hierarchical_descriptions(hierarchical_view)
    for family, description in golden_family_descriptions().items():
        assert description in reachable, family


@pytest.mark.slow
@pytest.mark.xdist_group("built_control_assistant")
def test_the_middle_layer_view_keeps_the_machine_and_family_descriptions(
    middle_layer_view: dict[str, Any],
) -> None:
    for machine, description in golden_machine_descriptions().items():
        assert middle_layer_view[machine]["_description"] == description, machine
    for (system, family), description in golden_family_descriptions().items():
        assert middle_layer_view[system][family]["_description"] == description, (system, family)


@pytest.mark.slow
@pytest.mark.xdist_group("built_control_assistant")
def test_the_middle_layer_view_keeps_the_system_descriptions_as_families(
    middle_layer_view: dict[str, Any],
) -> None:
    systems = golden_system_descriptions()
    assert len(systems) == 8
    for (machine, system), description in systems.items():
        assert middle_layer_view[machine][system]["_description"] == description, (machine, system)


@pytest.mark.slow
@pytest.mark.xdist_group("built_control_assistant")
def test_every_benchmark_target_is_a_tree_leaf_and_a_middle_layer_channel(
    hierarchical_view: dict[str, Any], middle_layer_view: dict[str, Any]
) -> None:
    leaves = hierarchical_channel_parts(hierarchical_view)
    channels = set().union(*middle_layer_cells(middle_layer_view))
    for query in [*pre_line_queries(), *in_context_queries()]:
        targets = set(query["targeted_pv"])
        assert targets <= leaves, query["user_query"]
        assert targets <= channels, query["user_query"]


@pytest.mark.slow
@pytest.mark.xdist_group("built_control_assistant")
def test_every_in_context_benchmark_target_is_an_in_context_row(
    in_context_view: dict[str, Any],
) -> None:
    rows = {row["address"] for row in in_context_view["channels"]}
    for query in in_context_queries():
        assert set(query["targeted_pv"]) <= rows, query["user_query"]


@pytest.mark.slow
@pytest.mark.xdist_group("built_control_assistant")
def test_every_pre_line_family_keeps_its_device_list(middle_layer_view: dict[str, Any]) -> None:
    for system, family, golden in middle_layer_families(pre_line_index("middle_layer")):
        view = middle_layer_view[system][family]["_setup"]["DeviceList"]
        assert view == golden["_setup"]["DeviceList"], (system, family)


@pytest.mark.slow
@pytest.mark.xdist_group("built_control_assistant")
def test_the_views_answer_no_fewer_tree_queries_than_the_pre_line_copies(
    hierarchical_view: dict[str, Any], middle_layer_view: dict[str, Any]
) -> None:
    golden = answerable_count(pre_line_index("hierarchical"), pre_line_index("middle_layer"))
    assert golden > 0
    assert answerable_count(hierarchical_view, middle_layer_view) >= golden
