"""The build step that derives the graph paradigm's channel search index.

A graph-mode project's channels live in its Turtle corpus. Every consumer of
them -- the roster, the explorer, the agent's keyword tool -- reads a DuckDB
index derived from that corpus rather than parsing it again, and ``osprey
build`` is where the derivation happens: once per build, on the host, shared by
every render pass.

So there are five outcomes and all five are pinned here: a render ships an index
built from ITS corpus; a build with personas parses the corpus once and every
render carries the same file; a corpus that is configured but not staged is a
fact and no file; and a corpus that is not valid Turtle -- or not text at all --
is a warning and no file, with the build still succeeding. Each of those five
carries ``real_graph_index``, because the derivation is what they are about.

The rest of the suite does not want that cost. Rendering a graph-mode project
is ordinary background work in dozens of build and deployment tests, and every
one of them would parse the same corpus again, so ``tests/conftest.py`` stubs
the build's index step out: it builds each corpus once per session and copies
that file into every render. The last three tests here are about the stub
itself -- that a render still carries a real index the roster can read, that a
second corpus becomes a second cache entry built once, and that the marker
above really does put the real hook back.

The roster facade is the oracle in one place only, the stub's own test: reading
a render's index through ``registered_channels`` is what shows the file is a
real index and not merely a file. The five marked tests read the index
directly, which is also what pins that it sits exactly where
``resolve_graph_index_path`` sends every reader.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from osprey.cli.phase_reporter import PhaseReporter, install_reporter
from osprey.services.virtual_accelerator.manifest.build import LIMITS_FILENAME
from tests.services.channel_finder.graph_index import corpora

#: Where a render's index goes, as ``services.graphdb.index_path`` defaults.
_INDEX_RELATIVE = Path("data") / "channel_databases" / "graph.duckdb"

#: What the profile points ``services.graphdb.ttl_path`` at, inside its own tree.
_CORPUS_RELATIVE = "./data/facility.ttl"


@pytest.fixture(autouse=True)
def _plain_reporter():
    """Print facts without color, so an assertion reads the words alone."""
    previous = install_reporter(PhaseReporter(color=False))
    yield
    install_reporter(previous)


@pytest.fixture(autouse=True)
def _cold_roster():
    """Every test resolves its own source cold; none inherits another's read."""
    import osprey.channel_roster as channel_roster

    channel_roster._roster_cache.clear()
    yield
    channel_roster._roster_cache.clear()


def _graph_repo(root: Path, *, corpus: str | None, personas: tuple[str, ...] = ()) -> Path:
    """A graph-mode deployment repo with its OWN corpus and no ``tiers/``.

    The data tree carries the per-tree sources a build reads and no paradigm
    channel database at all, which is what a graph-mode facility looks like:
    its channels are in the corpus. ``corpus=None`` writes no corpus file,
    leaving ``ttl_path`` naming a file that is not staged.
    """
    from tests.fixtures.lifecycle_repo import FACILITY_ONTOLOGY_JSON

    root.mkdir(parents=True, exist_ok=True)
    data = root / "data"
    (data / "simulation").mkdir(parents=True)
    (data / "simulation" / "machine.json").write_text(json.dumps({"channels": {}}))
    (data / "machine_state_channels.json").write_text(json.dumps({"_comment": "empty"}))
    (data / LIMITS_FILENAME).write_text("{}\n")
    (data / "facility_knowledge").mkdir()
    (data / "facility_ontology.json").write_text(FACILITY_ONTOLOGY_JSON)
    if corpus is not None:
        (data / "facility.ttl").write_text(corpus, encoding="utf-8")
    (root / "profile.yml").write_text(
        "name: Graph Index\n"
        "app_template: control_assistant\n"
        "provider: anthropic\n"
        "channel_finder_mode: graph\n"
        "data: data\n"
        "config:\n"
        f"  services.graphdb.ttl_path: {_CORPUS_RELATIVE}\n"
    )
    for persona in personas:
        (root / "personas").mkdir(exist_ok=True)
        (root / "personas" / f"{persona}.yml").write_text(f"name: {persona}\n")
    return root


def _build(repo: Path):
    """Run ``osprey build`` on *repo* the way CI does: no venv, no hooks."""
    from click.testing import CliRunner

    from osprey.cli.build_cmd import build as build_command

    return CliRunner().invoke(
        build_command, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"]
    )


def _indexes(build_dir: Path) -> list[Path]:
    """Every ``graph.duckdb`` this build wrote: the deployment's, then each persona's."""
    own = build_dir / _INDEX_RELATIVE
    return ([own] if own.is_file() else []) + sorted(build_dir.glob(f"*/{_INDEX_RELATIVE}"))


@pytest.mark.real_graph_index
def test_a_profile_build_derives_the_index_from_its_own_corpus(tmp_path: Path) -> None:
    """The render ships an index built from the corpus the PROFILE staged.

    The failure this rules out is the quiet one: an index derived from the
    framework's bundled demo corpus sitting in a project whose operators read
    their own facility's channels off it. So the digest is compared against the
    profile's own corpus, and the channels table against the three bindings that
    corpus declares.
    """
    from osprey.cli.build_cmd import _rendered_config
    from osprey.deployment.graphdb_service import resolve_graph_index_path
    from osprey.services.channel_finder.graph_index import open_graph_index
    from osprey.services.facility_knowledge.seeder.graph_seeder import ttl_sha256

    repo = _graph_repo(tmp_path / "graph-own-corpus", corpus=corpora.SUBCLASS_CHAIN)

    result = _build(repo)

    assert result.exit_code == 0, result.output
    render = repo / "build"
    index_path = render / _INDEX_RELATIVE
    assert index_path.is_file(), (
        f"the build wrote no channel search index; it holds {_indexes(render)}"
    )
    # The one path every reader resolves. An index written anywhere else is an
    # index the roster, the explorer and the agent's tool all report as missing.
    rendered = _rendered_config(render)
    assert resolve_graph_index_path(rendered, config_dir=render) == index_path

    # Not the source corpus: the render's own copy of it, which is what the
    # containers built from this render will carry.
    staged = render / "data" / "facility.ttl"
    assert staged.is_file()

    index = open_graph_index(index_path)
    try:
        assert getattr(index, "meta", None) is not None, f"index absent: {index}"
        assert index.meta.corpus_sha256 == ttl_sha256(staged.read_text(encoding="utf-8"))
        assert index.meta.corpus_filename == "facility.ttl"
        assert index.meta.binding_count == 3
        assert index.meta.device_count == 1
        rows = index.cursor().execute("SELECT address, direction FROM channels ORDER BY address")
        assert rows.fetchall() == [
            ("SR:MAG:QF1:CURRENT:RB", "read"),
            ("SR:MAG:QF1:CURRENT:SP", "write"),
            ("SR:MAG:QF1:NOTE", None),
        ]
    finally:
        index.close()

    # The manifest checksums every text file in the render; a derived binary
    # carrying its own digest would turn every rebuild into apparent drift.
    manifest = json.loads((render / ".osprey-manifest.json").read_text(encoding="utf-8"))
    assert not [name for name in manifest["file_checksums"] if name.endswith(".duckdb")]


@pytest.mark.real_graph_index
def test_a_persona_build_parses_the_corpus_once_and_ships_one_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two personas, three renders, ONE parse.

    Deriving an index is an rdflib parse of the whole corpus, seconds on a real
    facility, and every render pass of one build stages the same corpus. The
    memo is what makes a persona build cost the same as a plain one, so both
    halves are pinned: the builder ran once, and every render still carries the
    file rather than the first one keeping it to itself.
    """
    import osprey.services.channel_finder.graph_index as graph_index

    real = graph_index.build_graph_index
    calls: list[Path] = []

    def counted(ttl_path: Path, index_path: Path):
        calls.append(Path(ttl_path))
        return real(ttl_path, index_path)

    monkeypatch.setattr(graph_index, "build_graph_index", counted)

    repo = _graph_repo(
        tmp_path / "graph-personas",
        corpus=corpora.SUBCLASS_CHAIN,
        personas=("reader", "writer"),
    )

    result = _build(repo)

    assert result.exit_code == 0, result.output
    written = _indexes(repo / "build")
    # A deployment and two personas: exactly three renders, exactly three files.
    assert len(written) == 3, f"a render carries no index: {written}"
    assert len(calls) == 1, f"the corpus was parsed {len(calls)} times: {calls}"
    first = written[0].read_bytes()
    for path in written[1:]:
        assert path.read_bytes() == first, f"{path} differs from {written[0]}"


@pytest.mark.real_graph_index
def test_a_corpus_that_is_not_staged_is_a_fact_and_no_file(tmp_path: Path) -> None:
    """``ttl_path`` naming a file this build does not stage builds nothing.

    A legal state, not a failure: the project keeps the device card it reads
    from the store it dials. What it must not do is finish silently, leaving an
    operator to discover the absence from an empty explorer.
    """
    repo = _graph_repo(tmp_path / "graph-no-corpus", corpus=None)

    result = _build(repo)

    assert result.exit_code == 0, result.output
    assert not (repo / "build" / _INDEX_RELATIVE).exists()
    printed = " ".join(result.output.split())
    assert "No channel search index" in printed
    assert "services.graphdb.ttl_path" in printed
    assert "facility.ttl" in printed


@pytest.mark.real_graph_index
def test_a_corpus_that_is_not_turtle_warns_and_the_build_still_succeeds(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A corpus that will not parse costs the index, not the build.

    The store is seeded from the same file by a later verb, and every reader of
    the index already says what its absence means and how to fix it. Refusing
    here would stop a build whose deployment is otherwise sound.
    """
    repo = _graph_repo(tmp_path / "graph-bad-corpus", corpus=corpora.INVALID_TURTLE)

    with caplog.at_level(logging.WARNING):
        result = _build(repo)

    assert result.exit_code == 0, result.output
    assert not (repo / "build" / _INDEX_RELATIVE).exists()
    assert "facility.ttl could not be turned into a channel search index" in caplog.text


@pytest.mark.real_graph_index
def test_a_corpus_that_is_not_text_warns_and_the_build_still_succeeds(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Bytes that are not UTF-8 cost the index, not the build.

    The corpus is read twice -- once here for the memo digest, once by the
    builder -- and the first read is the one that meets a file that is not text
    at all. Both reads answer the same way a corpus that will not parse does,
    so a staged file the build cannot decode cannot take the whole render down.
    """
    repo = _graph_repo(tmp_path / "graph-binary-corpus", corpus="")
    (repo / "data" / "facility.ttl").write_bytes(b"\xff\xfe\x00binary, not turtle\x00")

    with caplog.at_level(logging.WARNING):
        result = _build(repo)

    assert result.exit_code == 0, result.output
    assert not (repo / "build" / _INDEX_RELATIVE).exists()
    assert "facility.ttl could not be turned into a channel search index" in caplog.text


def _preset_repo(tmp_path: Path, name: str) -> Path:
    """A control-assistant project as ``osprey init`` writes one, in graph mode.

    The bundled preset, not a hand-built profile: its corpus is the one the
    suite renders over and over, so it is the corpus the session cache exists
    for.
    """
    from click.testing import CliRunner

    from osprey.cli.init_cmd import init

    repo = tmp_path / name
    override = tmp_path / f"{name}-override.yml"
    override.write_text("channel_finder_mode: graph\n", encoding="utf-8")
    created = CliRunner().invoke(
        init, [str(repo), "--preset", "control-assistant", "--no-git", "-O", str(override)]
    )
    assert created.exit_code == 0, created.output
    return repo


def test_the_stub_ships_a_real_index_the_roster_can_read(tmp_path: Path) -> None:
    """Under the suite-wide stub, a preset render still carries a true index.

    The stub is only allowed to skip the *parse*, never to change what a render
    holds, because most of the suite renders graph-mode projects for reasons
    that have nothing to do with this step and would silently start asserting
    against a fiction. So the render is read back the way its containers will
    read it: the index sits where every reader resolves it, its meta names the
    digest of the corpus THIS render staged, and the roster facade enumerates
    channels from it.
    """
    from osprey.channel_roster import registered_channels
    from osprey.cli import build_cmd
    from osprey.services.channel_finder.graph_index import open_graph_index
    from osprey.services.facility_knowledge.seeder.graph_seeder import ttl_sha256

    assert getattr(build_cmd._build_graph_index, "stubbed_by_conftest", False), (
        "this test is about the stub, so it must not carry the real_graph_index marker"
    )

    repo = _preset_repo(tmp_path, "preset-graph")

    result = _build(repo)

    assert result.exit_code == 0, result.output
    render = repo / "build"
    rendered = build_cmd._rendered_config(render)
    rendered["config_dir"] = str(render)

    # Resolved rather than spelled: the preset names its own corpus, and the
    # point is that the stub honoured whatever the render actually staged.
    target = build_cmd._graph_index_target(render, rendered)
    assert target is not None, "the preset render staged no corpus"
    assert target.index_path.is_file(), (
        f"the build wrote no channel search index; it holds {_indexes(render)}"
    )

    index = open_graph_index(target.index_path)
    try:
        assert getattr(index, "meta", None) is not None, f"index absent: {index}"
        assert index.meta.corpus_sha256 == ttl_sha256(
            target.corpus_path.read_text(encoding="utf-8")
        )
        assert index.meta.binding_count > 0
    finally:
        index.close()

    roster = registered_channels(rendered)
    assert roster.records, f"the roster read no channels off the index: {roster.absence}"


def test_a_second_corpus_becomes_a_second_cache_entry_built_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, graph_index_cache
) -> None:
    """One cache entry per corpus, and each corpus parsed once for the session.

    Three builds, two corpora: the parse is what the cache is saving, so the
    builder has to run twice and not three times, and the second corpus must
    not be served the first one's file. Both digests are dropped from the cache
    first, so the count is the same whatever else in the session had already
    rendered these corpora.
    """
    import osprey.services.channel_finder.graph_index as graph_index
    from osprey.services.facility_knowledge.seeder.graph_seeder import ttl_sha256
    from tests.fixtures.lifecycle_repo import DEMO_MACHINE_TTL

    repeated, single = DEMO_MACHINE_TTL, corpora.BOTH_EDGES
    digests = {corpus: ttl_sha256(corpus) for corpus in (repeated, single)}
    assert len(set(digests.values())) == 2, "the two corpora are the same text"
    for digest in digests.values():
        graph_index_cache.entries.pop(digest, None)

    real = graph_index.build_graph_index
    calls: list[Path] = []

    def counted(ttl_path: Path, index_path: Path):
        calls.append(Path(ttl_path))
        return real(ttl_path, index_path)

    monkeypatch.setattr(graph_index, "build_graph_index", counted)

    renders = []
    for name, corpus in (("first", repeated), ("second", repeated), ("other", single)):
        result = _build(_graph_repo(tmp_path / name, corpus=corpus))
        assert result.exit_code == 0, result.output
        renders.append(tmp_path / name / "build" / _INDEX_RELATIVE)

    assert len(calls) == 2, f"the two corpora were parsed {len(calls)} times: {calls}"

    cached = {corpus: graph_index_cache.entries[digest] for corpus, digest in digests.items()}
    assert len(set(cached.values())) == 2, f"one file is serving both corpora: {cached}"
    for path in cached.values():
        assert path.parent == graph_index_cache.directory
        assert path.is_file()

    # Every render carries its own copy, and the two renders of the same corpus
    # carry the same bytes -- a cache that handed one render the other corpus's
    # index would be worse than no cache at all.
    assert all(path.is_file() for path in renders), renders
    assert renders[0].read_bytes() == renders[1].read_bytes()
    assert renders[2].read_bytes() != renders[0].read_bytes()


@pytest.mark.real_graph_index
def test_the_marker_puts_the_real_builder_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, graph_index_cache
) -> None:
    """A marked test parses its own corpus, whatever the cache holds for it.

    The opt-out has to be complete, because the five tests above assert on what
    the derivation does and would pass vacuously against a copied file. So the
    cache is deliberately poisoned with an index built from a DIFFERENT corpus:
    a marked test that quietly went through the stub would ship that file, and
    the digest in its meta says which corpus the render really parsed.
    """
    import osprey.services.channel_finder.graph_index as graph_index
    from osprey.cli import build_cmd
    from osprey.services.channel_finder.graph_index import open_graph_index
    from osprey.services.facility_knowledge.seeder.graph_seeder import ttl_sha256

    assert not getattr(build_cmd._build_graph_index, "stubbed_by_conftest", False), (
        "the real_graph_index marker did not restore the real build step"
    )

    corpus = corpora.SHARED_FULL_PV
    digest = ttl_sha256(corpus)
    other_corpus = tmp_path / "wrong-corpus.ttl"
    other_corpus.write_text(corpora.SUBCLASS_CHAIN, encoding="utf-8")
    wrong = graph_index_cache.directory / "wrong-corpus.duckdb"
    graph_index.build_graph_index(other_corpus, wrong)
    monkeypatch.setitem(graph_index_cache.entries, digest, wrong)

    real = graph_index.build_graph_index
    calls: list[Path] = []

    def counted(ttl_path: Path, index_path: Path):
        calls.append(Path(ttl_path))
        return real(ttl_path, index_path)

    monkeypatch.setattr(graph_index, "build_graph_index", counted)

    repo = _graph_repo(tmp_path / "marked", corpus=corpus)

    result = _build(repo)

    assert result.exit_code == 0, result.output
    assert len(calls) == 1, f"the real builder ran {len(calls)} times: {calls}"

    index_path = repo / "build" / _INDEX_RELATIVE
    assert index_path.is_file()
    index = open_graph_index(index_path)
    try:
        assert getattr(index, "meta", None) is not None, f"index absent: {index}"
        assert index.meta.corpus_sha256 == digest, "the render carries the poisoned cache entry"
    finally:
        index.close()
