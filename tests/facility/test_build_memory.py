"""A large facility builds in memory proportional to its facility file, and gives the same bytes.

The build reads a facility's layer files, validates the facility file it builds,
writes the facility file and its views into every render, reads the file back
for the render's templates, and derives the channel search index from the graph
view. Each step used to hold copies of the whole facility at once: the node
tree of every layer file, a model object per record, the serialised file, a
second parse of it per reader, every view's bytes, and three indexes of every
triple of the Turtle. A facility of 175,000 channels peaked near 6 GB.

``TestTheMemoryBound`` builds a synthetic facility and holds the traced peak of
that whole sequence, and what it leaves held, to a multiple of the facility
file's size. The other classes hold each change to the output it replaced:
the same bytes, the same values, the same errors.
"""

from __future__ import annotations

import json
import tracemalloc
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pydantic
import pytest
import yaml

from osprey.facility.build import build_facility
from osprey.facility.render import FACILITY_FILE, facility_bytes, json_bytes
from osprey.facility.schema import Facility
from osprey.facility.sources import _Loader, _loader_on, read_yaml
from osprey.facility.validate import _model_failures
from osprey.facility.views.facts import parsed_facility_file
from osprey.facility.views.outputs import WrittenViews, render_facility_outputs
from osprey.services.channel_finder.graph_index import builder
from osprey.services.channel_finder.graph_index.builder import build_graph_index, parse_corpus
from osprey.services.graph_store.seeder import ttl_file_sha256, ttl_sha256
from osprey_connectors import yaml_loader
from tests._builds import FRAMEWORK_CHECKS
from tests.facility._synthetic_trees import BPM, QUAD, write_tree
from tests.facility.served_tree import IN_PROCESS_RENDER_CONFIG
from tests.services.channel_finder.graph_index import corpora

#: Devices in the measured facility, four channels each.
DEVICES = 2500
#: The traced peak of the whole sequence, as a multiple of the facility file's
#: size. Before these changes the sequence peaked at about 24 times the file;
#: after, at about 9.
PEAK_BOUND = 12
#: What the sequence leaves held once done, the facility's own document
#: included, as a multiple of the file. About 20 before, 3 after.
HELD_BOUND = 6
#: Renders written from one facility, as a build writes one per login and image.
RENDERS = 3


def _tree(devices: int) -> dict[str, Any]:
    """A facility of ``devices`` devices in 40 sectors of one ring, four channels each."""
    places = [{"id": "RING"}] + [{"id": f"RING/S{s:02d}"} for s in range(1, 41)]
    records: list[dict[str, Any]] = []
    channels: list[dict[str, Any]] = []
    for n in range(devices):
        sector = f"RING/S{n % 40 + 1:02d}"
        device = f"{sector}/D{n:05d}"
        records.append({"id": device, "class": QUAD if n % 2 else BPM, "place": sector})
        stem = f"X{n:05d}"
        on = {"device": device}
        channels += [
            {
                "id": f"{stem}:SP",
                "role": "setpoint",
                "pair": f"{stem}:RB",
                "on": on,
                "tags": ["magnet", "setting"],
                "description": "Set the current",
            },
            {
                "id": f"{stem}:RB",
                "on": on,
                "tags": ["magnet", "reading"],
                "description": "Read the current",
            },
            {"id": f"{stem}:T1", "on": on, "tags": ["temperature"]},
            {"id": f"{stem}:T2", "on": on, "tags": ["temperature"]},
        ]
    return {
        "records/places.yaml": places,
        "records/devices.yaml": records,
        "records/channels.yaml": channels,
    }


def _build_and_render(root: Path, devices: int) -> tuple[Any, Path]:
    """Build, render ``RENDERS`` times, read each render's file back, index the graph."""
    from osprey.cli.templates.claude_code import _read_facility
    from osprey.integrations.canopy.screen_guide import _guide_facility

    facility_dir = write_tree(root / "facility", _tree(devices))
    document = build_facility(facility_dir, project_name="synthetic", extra_checks=FRAMEWORK_CHECKS)
    written: WrittenViews = {}
    for index in range(RENDERS):
        render = root / f"render{index}"
        render.mkdir()
        render_facility_outputs(
            render,
            document,
            IN_PROCESS_RENDER_CONFIG,
            facility_dir,
            omitted_reported=set(),
            views_written=written,
            built_databases={},
        )
        _read_facility(render)
        _guide_facility(render / FACILITY_FILE)
    first = root / "render0"
    build_graph_index(first / "data" / "graph" / "facility.ttl", root / "index.duckdb")
    return document, first


@pytest.fixture
def traced() -> Iterator[None]:
    """Trace allocations for the test, and leave tracing as it was."""
    was = tracemalloc.is_tracing()
    if not was:
        tracemalloc.start()
    try:
        yield
    finally:
        if not was:
            tracemalloc.stop()


class TestTheMemoryBound:
    @pytest.mark.usefixtures("traced")
    def test_the_build_holds_a_large_facility_to_a_multiple_of_its_file(
        self, tmp_path: Path
    ) -> None:
        # Once small, so the libraries the sequence imports are not counted.
        _build_and_render(tmp_path / "warm", 10)
        start = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()

        document, render = _build_and_render(tmp_path / "large", DEVICES)

        held, peak = tracemalloc.get_traced_memory()
        size = (render / FACILITY_FILE).stat().st_size
        assert (peak - start) < PEAK_BOUND * size, (peak - start) / size
        assert (held - start) < HELD_BOUND * size, (held - start) / size
        assert len(document["channels"]) == 4 * DEVICES


class TestTheRenders:
    @pytest.fixture(scope="class")
    def built(self, tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, Path]:
        root = tmp_path_factory.mktemp("renders")
        document, _ = _build_and_render(root, 40)
        return document, root

    def test_every_render_holds_the_bytes_the_writer_gives(self, built: tuple[Any, Path]) -> None:
        document, root = built
        expected = (
            json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode()
        first = root / "render0"
        for index in range(RENDERS):
            render = root / f"render{index}"
            assert (render / FACILITY_FILE).read_bytes() == expected
            for path in sorted((first / "data").rglob("*")):
                if path.is_file():
                    relative = path.relative_to(first)
                    assert (render / relative).read_bytes() == path.read_bytes(), relative

    def test_a_reader_of_the_build_file_gets_the_build_document(
        self, built: tuple[Any, Path]
    ) -> None:
        document, root = built
        assert parsed_facility_file(root / "render1" / FACILITY_FILE) is document

    def test_a_file_that_differs_is_parsed(self, built: tuple[Any, Path], tmp_path: Path) -> None:
        document, _ = built
        other = {**document, "places": document["places"][:1]}
        path = tmp_path / FACILITY_FILE
        path.write_bytes(json_bytes(other))
        parsed = parsed_facility_file(path)
        assert parsed is not document
        assert parsed == other

    def test_a_view_changed_since_it_was_written_is_written_again(self, tmp_path: Path) -> None:
        facility_dir = write_tree(tmp_path / "facility", _tree(8))
        document = build_facility(
            facility_dir, project_name="synthetic", extra_checks=FRAMEWORK_CHECKS
        )
        written: WrittenViews = {}
        renders = []
        for index in range(2):
            render = tmp_path / f"render{index}"
            render.mkdir()
            if index == 1:
                (renders[0] / "data" / "graph" / "facility.ttl").write_text("changed")
                (renders[0] / FACILITY_FILE).write_text("changed")
            render_facility_outputs(
                render,
                document,
                IN_PROCESS_RENDER_CONFIG,
                facility_dir,
                omitted_reported=set(),
                views_written=written,
            )
            renders.append(render)
        fresh = tmp_path / "fresh"
        fresh.mkdir()
        render_facility_outputs(
            fresh, document, IN_PROCESS_RENDER_CONFIG, facility_dir, omitted_reported=set()
        )
        for relative in (Path("data/graph/facility.ttl"), Path(FACILITY_FILE)):
            assert (renders[1] / relative).read_bytes() == (fresh / relative).read_bytes()


class TestTheSerialiser:
    @pytest.mark.parametrize("sort_keys", [True, False])
    def test_json_bytes_are_the_indented_dump(self, sort_keys: bool) -> None:
        document = {
            "b": [1, 2.5, None, True, {"z": "ü", "a": []}],
            "a": {"nested": {"x": "line\nbreak", "y": "☃"}},
            "c": "",
        }
        dumped = json.dumps(document, indent=2, sort_keys=sort_keys, ensure_ascii=False)
        assert json_bytes(document, sort_keys=sort_keys) == (dumped + "\n").encode("utf-8")

    def test_json_bytes_refuse_nan(self) -> None:
        with pytest.raises(ValueError, match="JSON compliant"):
            json_bytes({"x": float("nan")})

    def test_facility_bytes_are_the_indented_sorted_dump(self) -> None:
        document = {"schema": "osprey.facility.facility/1", "channels": [{"id": "Ä", "b": 1}]}
        dumped = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False)
        assert facility_bytes(document) == (dumped + "\n").encode("utf-8")


def _whole_load(text: str) -> Any:
    """``read_yaml`` as it read every file before: the whole document at once."""
    loader = _loader_on(yaml_loader.safe_loader())
    try:
        return yaml.load(text, Loader=loader)
    except yaml.YAMLError:
        if loader is _Loader:
            raise
    return yaml.load(text, Loader=_Loader)


class TestTheLayerFileReader:
    @pytest.mark.parametrize(
        "text",
        [
            "- id: A\n  'on': {device: D}\n  tags: [x, y]\n- id: B\n  value: 1.5\n",
            "- 1\n- yes\n- on\n- true\n- ~\n- 2026-01-01\n- '007'\n",
            "[]\n",
            "- [a, [b, c]]\n- {k: v}\n",
            "--- \n- a\n...\n",
            "# only a comment\n- a\n",
        ],
    )
    def test_a_list_reads_as_the_whole_load_reads_it(self, text: str) -> None:
        assert read_yaml(text) == _whole_load(text)

    @pytest.mark.parametrize(
        "text",
        [
            "- &one {id: A}\n- *one\n",
            "!!seq\n- a\n",
            "&root\n- a\n",
            "id: A\n",
            "just a scalar\n",
            "",
        ],
    )
    def test_what_is_not_a_plain_list_reads_as_the_whole_load_reads_it(self, text: str) -> None:
        assert read_yaml(text) == _whole_load(text)

    def test_an_alias_across_items_shares_its_value_as_the_whole_load_does(self) -> None:
        items = read_yaml("- &one {id: A}\n- *one\n")
        assert items[0] is items[1]

    @pytest.mark.parametrize(
        "text",
        ["- a\n- [unclosed\n", "- a\n---\n- b\n", "- *nowhere\n", "- {a: 1\n"],
    )
    def test_an_error_is_the_whole_load_error(self, text: str) -> None:
        with pytest.raises(yaml.YAMLError) as streamed:
            read_yaml(text)
        with pytest.raises(yaml.YAMLError) as whole:
            _whole_load(text)
        assert str(streamed.value) == str(whole.value)


def _whole_failures(document: dict[str, Any]) -> list[tuple[Any, ...]]:
    try:
        Facility.model_validate(document)
    except pydantic.ValidationError as exc:
        return [(tuple(item["loc"]), item["type"], item["msg"]) for item in exc.errors()]
    return []


class TestTheSchemaCheck:
    @pytest.fixture(scope="class")
    def document(self, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
        root = tmp_path_factory.mktemp("schema")
        return build_facility(
            write_tree(root / "facility", _tree(6)),
            project_name="synthetic",
            extra_checks=FRAMEWORK_CHECKS,
        )

    @pytest.mark.parametrize(
        "break_it",
        [
            lambda d: None,
            lambda d: d["channels"].__setitem__(2, "not a record"),
            lambda d: d["channels"][0].__setitem__("unknown_field", 1),
            lambda d: d["channels"][3].pop("id"),
            lambda d: d["devices"][1].__setitem__("class", ["a list"]),
            lambda d: d.__setitem__("channels", {"not": "a list"}),
            lambda d: d.__setitem__("schema", "not a header"),
            lambda d: d.__setitem__("stray", 1),
            lambda d: (d["places"].append(7), d["channels"][1].__setitem__("on", 3)),
        ],
    )
    def test_each_failure_is_the_whole_model_failure(
        self, document: dict[str, Any], break_it: Any
    ) -> None:
        broken = json.loads(json.dumps(document))
        break_it(broken)
        per_record = [(loc, item["type"], item["msg"]) for loc, item in _model_failures(broken)]
        assert sorted(per_record, key=repr) == sorted(_whole_failures(broken), key=repr)


class TestTheCorpusParse:
    @pytest.mark.parametrize(
        "corpus",
        [
            corpora.SUBCLASS_CHAIN,
            corpora.BOTH_EDGES,
            corpora.SHARED_FULL_PV,
            corpora.BINDING_UNDER_TWO_DEVICES,
            corpora.DEVICE_WITHOUT_SECTION_OR_SYSTEM,
            corpora.CLASS_CYCLE,
            corpora.DEEP_CHAIN,
            corpora.UNTYPED_TARGETS,
            corpora.PLACED_DEVICES,
            corpora.NO_BINDINGS,
        ],
    )
    def test_the_rows_are_those_of_a_parse_that_keeps_every_triple(
        self, corpus: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        kept = parse_corpus(corpus)

        monkeypatch.setattr(builder, "_read_index_store", _every_triple)
        assert parse_corpus(corpus) == kept
        assert parse_corpus(corpus.encode("utf-8")) == kept

    def test_the_graph_view_of_a_facility_parses_as_with_every_triple(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, render = _build_and_render(tmp_path, 12)
        text = (render / "data" / "graph" / "facility.ttl").read_text(encoding="utf-8")
        kept = parse_corpus(text)
        monkeypatch.setattr(builder, "_read_index_store", _every_triple)
        assert parse_corpus(text) == kept
        assert kept.binding_rows


def _every_triple(_kept: Any) -> Any:
    """rdflib's own memory store, which keeps every triple, in place of the read index."""
    from rdflib.plugins.stores.memory import Memory

    store = Memory()
    store.release = lambda: None  # type: ignore[attr-defined]
    return store


class TestTheCorpusDigest:
    @pytest.mark.parametrize(
        "content",
        [b"@prefix a: <x> .\n", b"line\r\nwindows\r\n", b"old\rmac\r", "café\n".encode(), b""],
    )
    def test_the_file_digest_is_the_text_digest(self, tmp_path: Path, content: bytes) -> None:
        path = tmp_path / "corpus.ttl"
        path.write_bytes(content)
        assert ttl_file_sha256(path) == ttl_sha256(path.read_text(encoding="utf-8"))

    def test_a_file_that_is_not_utf8_raises_as_read_text_does(self, tmp_path: Path) -> None:
        path = tmp_path / "corpus.ttl"
        path.write_bytes(b"ok\n\xff\xfe broken")
        with pytest.raises(UnicodeDecodeError) as streamed:
            ttl_file_sha256(path)
        with pytest.raises(UnicodeDecodeError) as whole:
            path.read_text(encoding="utf-8")
        assert str(streamed.value) == str(whole.value)

    def test_a_large_corpus_is_hashed_across_blocks(self, tmp_path: Path) -> None:
        path = tmp_path / "corpus.ttl"
        # A two-byte character on the boundary of the first block.
        path.write_bytes(b"a" * ((1 << 20) - 1) + "é".encode() + b"\n" * 10)
        assert ttl_file_sha256(path) == ttl_sha256(path.read_text(encoding="utf-8"))
