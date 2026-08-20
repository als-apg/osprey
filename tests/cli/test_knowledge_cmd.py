"""Tests for the ``osprey knowledge`` CLI command group.

Covers:
- ``knowledge --help`` lists regen-index, validate, and seed-from-ttl.
- ``knowledge regen-index <bundle>`` writes index files and is idempotent.
- ``knowledge validate <bundle>`` exits 0 on a clean bundle; exits 1 and
  collects ALL failures (broken concept doc + broken index.md in one run).
- ``knowledge seed-from-ttl <ttl> <bundle>`` writes stubs on first run,
  is idempotent on re-run, skips diffs without --force, overwrites with
  --force, and exits cleanly when rdflib is absent.
- ``knowledge seed-graph [ttl]`` reaches each of the five seeding states
  against a stubbed graph store, resolves a configured relative ``ttl_path``
  against the config file's directory rather than the process CWD, and turns
  a missing driver, an unreachable store and a rejected credential into clean
  CLI errors.
"""

from __future__ import annotations

import hashlib
import textwrap
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.knowledge_cmd import knowledge

# Result types only: importing the seeder module pulls in no neo4j driver, which
# is the lazy-import contract tests/services/facility_knowledge/test_import_isolation.py
# pins.
from osprey.services.facility_knowledge.seeder.graph_seeder import (
    BootstrapResult,
    BootstrapStatus,
    ImportResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def okf_bundle(tmp_path: Path) -> Path:
    """Create a minimal OKF bundle with one concept in a subdirectory.

    Structure::

        bundle/
          concepts/
            beam.md       (type: Concept, title: Beam)

    Returns:
        Path to the bundle root.
    """
    bundle = tmp_path / "bundle"
    concepts = bundle / "concepts"
    concepts.mkdir(parents=True)

    beam = concepts / "beam.md"
    beam.write_text(
        textwrap.dedent("""\
            ---
            type: Concept
            title: Beam
            description: The primary electron or photon beam.
            ---

            Body text about the beam.
        """),
        encoding="utf-8",
    )
    return bundle


# ---------------------------------------------------------------------------
# --help smoke test
# ---------------------------------------------------------------------------


def test_knowledge_help_lists_regen_index() -> None:
    """``knowledge --help`` output includes ``regen-index``."""
    runner = CliRunner()
    result = runner.invoke(knowledge, ["--help"])
    assert result.exit_code == 0, result.output
    assert "regen-index" in result.output


# ---------------------------------------------------------------------------
# regen-index
# ---------------------------------------------------------------------------


def test_regen_index_writes_indexes(okf_bundle: Path) -> None:
    """``regen-index <bundle>`` writes index.md files and exits 0."""
    runner = CliRunner()
    result = runner.invoke(knowledge, ["regen-index", str(okf_bundle)])
    assert result.exit_code == 0, result.output

    # Bundle root index and concepts sub-directory index should both exist.
    assert (okf_bundle / "index.md").is_file()
    assert (okf_bundle / "concepts" / "index.md").is_file()

    # Summary line is present.
    assert "Wrote" in result.output
    assert "index file" in result.output


def test_regen_index_idempotent(okf_bundle: Path) -> None:
    """Running ``regen-index`` twice produces bit-identical index files."""
    runner = CliRunner()

    runner.invoke(knowledge, ["regen-index", str(okf_bundle)])
    root_index = okf_bundle / "index.md"
    sub_index = okf_bundle / "concepts" / "index.md"

    first_root = root_index.read_text(encoding="utf-8")
    first_sub = sub_index.read_text(encoding="utf-8")

    runner.invoke(knowledge, ["regen-index", str(okf_bundle)])

    assert root_index.read_text(encoding="utf-8") == first_root
    assert sub_index.read_text(encoding="utf-8") == first_sub


def test_regen_index_no_bundle_and_no_config_errors() -> None:
    """``regen-index`` without an arg and no config setting exits nonzero."""
    runner = CliRunner()
    # Patch get_config_value to return None so the fallback path is exercised.
    import unittest.mock as mock

    with mock.patch(
        "osprey.cli.knowledge_cmd.knowledge.commands",
        wraps=knowledge.commands,
    ):
        with mock.patch("osprey.utils.config.get_config_value", return_value=None):
            result = runner.invoke(knowledge, ["regen-index"])

    # Should exit non-zero or print an error message.
    assert result.exit_code != 0 or "Error" in (result.output or "")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def test_validate_help_lists_validate() -> None:
    """``knowledge --help`` output includes ``validate``."""
    runner = CliRunner()
    result = runner.invoke(knowledge, ["--help"])
    assert result.exit_code == 0, result.output
    assert "validate" in result.output


def test_validate_clean_bundle_exits_zero(okf_bundle: Path) -> None:
    """``validate <bundle>`` exits 0 when all documents are valid."""
    runner = CliRunner()
    result = runner.invoke(knowledge, ["validate", str(okf_bundle)])
    assert result.exit_code == 0, result.output + (result.stderr or "")
    assert "valid" in result.output.lower()


def test_validate_broken_concept_doc_exits_nonzero(okf_bundle: Path) -> None:
    """``validate`` flags a concept doc missing required frontmatter keys."""
    broken = okf_bundle / "concepts" / "broken.md"
    broken.write_text(
        textwrap.dedent("""\
            ---
            type: Concept
            ---

            Missing title and description.
        """),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(knowledge, ["validate", str(okf_bundle)])
    assert result.exit_code != 0
    assert "broken.md" in result.output


def test_validate_broken_index_md_exits_nonzero(okf_bundle: Path) -> None:
    """``validate`` flags an index.md with disallowed frontmatter."""
    # A sub-directory index.md must NOT have frontmatter (OKF §6).
    bad_index = okf_bundle / "concepts" / "index.md"
    bad_index.parent.mkdir(parents=True, exist_ok=True)
    bad_index.write_text(
        textwrap.dedent("""\
            ---
            title: Should not be here
            ---

            # Concepts
        """),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(knowledge, ["validate", str(okf_bundle)])
    assert result.exit_code != 0
    assert "index.md" in result.output


def test_validate_collects_all_failures(okf_bundle: Path) -> None:
    """``validate`` reports EVERY failing file, not just the first one."""
    # Write two broken concept docs simultaneously.
    broken1 = okf_bundle / "concepts" / "no_title.md"
    broken1.write_text(
        textwrap.dedent("""\
            ---
            type: Concept
            description: Missing title.
            ---

            Body.
        """),
        encoding="utf-8",
    )
    broken2 = okf_bundle / "concepts" / "no_description.md"
    broken2.write_text(
        textwrap.dedent("""\
            ---
            type: Concept
            title: Missing Description
            ---

            Body.
        """),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(knowledge, ["validate", str(okf_bundle)])
    assert result.exit_code != 0
    # Both failures must appear in the output.
    assert "no_title.md" in result.output
    assert "no_description.md" in result.output


def test_validate_malformed_yaml_index_and_broken_concept_collect_all(
    tmp_path: Path,
) -> None:
    """Malformed-YAML root index.md + broken concept → both reported, no traceback.

    Regression test for the collect-all bug: validate_index() internally calls
    OKFDocument.parse(), which raises OKFDocumentError (not OKFIndexError) on
    invalid YAML frontmatter.  Without the fix the exception escapes the
    per-file catch, aborts the loop, and produces a raw traceback.
    """
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    # Root index.md with intentionally malformed YAML.
    (bundle / "index.md").write_text(
        "---\n: bad: yaml: {\n---\n\n# Index\n",
        encoding="utf-8",
    )

    # A concept doc missing required frontmatter keys.
    (bundle / "device.md").write_text(
        textwrap.dedent("""\
            ---
            type: Concept
            ---

            Missing title and description.
        """),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(knowledge, ["validate", str(bundle)])

    assert result.exit_code != 0
    # Both files must be reported.
    assert "index.md" in result.output
    assert "device.md" in result.output
    # No raw Python traceback should appear.
    assert "Traceback" not in result.output


# ---------------------------------------------------------------------------
# seed-from-ttl
# ---------------------------------------------------------------------------

# Minimal synthetic TTL — mirrors the fixture in test_ttl_seeder.py but kept
# local so this test file is self-contained.
_MINI_TTL = textwrap.dedent("""\
    @prefix narad_p: <https://narad.example.org/property/> .
    @prefix narad_sem: <https://narad.example.org/schema/shared_semantics/> .

    <https://narad.example.org/device/tst_SEC_QF1> a narad_sem:Quadrupole ;
        narad_p:deviceId "narad:device:tst:SEC:QF1" ;
        narad_p:sectionCode "SEC" ;
        narad_p:sourceName "QF1" .

    <https://narad.example.org/device/tst_SEC_QD1> a narad_sem:Quadrupole ;
        narad_p:deviceId "narad:device:tst:SEC:QD1" ;
        narad_p:sectionCode "SEC" ;
        narad_p:sourceName "QD1" .
""")


@pytest.fixture()
def mini_ttl(tmp_path: Path) -> Path:
    """Write the minimal TTL fixture to a temp file."""
    p = tmp_path / "mini.ttl"
    p.write_text(_MINI_TTL, encoding="utf-8")
    return p


@pytest.fixture()
def seed_bundle(tmp_path: Path) -> Path:
    """An empty OKF bundle directory for seed-from-ttl tests."""
    bundle = tmp_path / "seed_bundle"
    bundle.mkdir()
    return bundle


def test_seed_from_ttl_help_listed() -> None:
    """``knowledge --help`` output includes ``seed-from-ttl``."""
    runner = CliRunner()
    result = runner.invoke(knowledge, ["--help"])
    assert result.exit_code == 0, result.output
    assert "seed-from-ttl" in result.output


def test_seed_from_ttl_no_rdflib_clean_error(
    mini_ttl: Path, seed_bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing rdflib produces a clean ClickException, not a traceback."""
    import sys
    import unittest.mock as mock

    # Make rdflib un-importable for this test only.
    with mock.patch.dict(sys.modules, {"rdflib": None}):
        runner = CliRunner()
        result = runner.invoke(knowledge, ["seed-from-ttl", str(mini_ttl), str(seed_bundle)])

    # Exit code must be non-zero; output must mention the knowledge extra.
    assert result.exit_code != 0
    assert "knowledge" in result.output.lower() or "rdflib" in result.output.lower()
    # No Python traceback should appear.
    assert "Traceback" not in result.output


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("rdflib") is None,
    reason="rdflib not installed (knowledge extra required)",
)
def test_seed_from_ttl_writes_stubs(mini_ttl: Path, seed_bundle: Path) -> None:
    """Fresh seed writes one stub per device node and exits 0."""
    runner = CliRunner()
    result = runner.invoke(knowledge, ["seed-from-ttl", str(mini_ttl), str(seed_bundle)])
    assert result.exit_code == 0, result.output
    stubs = list(seed_bundle.glob("*.md"))
    assert len(stubs) == 2  # two device nodes in _MINI_TTL
    assert "written" in result.output


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("rdflib") is None,
    reason="rdflib not installed (knowledge extra required)",
)
def test_seed_from_ttl_idempotent(mini_ttl: Path, seed_bundle: Path) -> None:
    """Re-running seed-from-ttl on an unchanged TTL produces no new writes."""
    runner = CliRunner()
    runner.invoke(knowledge, ["seed-from-ttl", str(mini_ttl), str(seed_bundle)])

    # Capture file contents after first run.
    before = {p.name: p.read_text(encoding="utf-8") for p in seed_bundle.glob("*.md")}

    result = runner.invoke(knowledge, ["seed-from-ttl", str(mini_ttl), str(seed_bundle)])
    assert result.exit_code == 0, result.output

    after = {p.name: p.read_text(encoding="utf-8") for p in seed_bundle.glob("*.md")}
    assert before == after  # bit-identical; nothing changed
    assert "unchanged" in result.output


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("rdflib") is None,
    reason="rdflib not installed (knowledge extra required)",
)
def test_seed_from_ttl_skips_diff_without_force(mini_ttl: Path, seed_bundle: Path) -> None:
    """Existing stub with different body is NOT overwritten without --force."""
    runner = CliRunner()
    runner.invoke(knowledge, ["seed-from-ttl", str(mini_ttl), str(seed_bundle)])

    # Hand-edit one stub to simulate a human-modified file.
    stub_files = sorted(seed_bundle.glob("*.md"))
    edited = stub_files[0]
    original = edited.read_text(encoding="utf-8")
    edited.write_text(original + "\n# Hand-edited section\n", encoding="utf-8")
    modified_content = edited.read_text(encoding="utf-8")

    result = runner.invoke(knowledge, ["seed-from-ttl", str(mini_ttl), str(seed_bundle)])
    assert result.exit_code == 0, result.output  # no failure — just a skip

    # File must NOT have been overwritten.
    assert edited.read_text(encoding="utf-8") == modified_content
    assert "force" in result.output.lower()


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("rdflib") is None,
    reason="rdflib not installed (knowledge extra required)",
)
def test_seed_from_ttl_force_overwrites(mini_ttl: Path, seed_bundle: Path) -> None:
    """--force causes an existing stub with a different body to be overwritten."""
    runner = CliRunner()
    runner.invoke(knowledge, ["seed-from-ttl", str(mini_ttl), str(seed_bundle)])

    stub_files = sorted(seed_bundle.glob("*.md"))
    edited = stub_files[0]
    original = edited.read_text(encoding="utf-8")
    edited.write_text(original + "\n# Hand-edited section\n", encoding="utf-8")

    result = runner.invoke(knowledge, ["seed-from-ttl", "--force", str(mini_ttl), str(seed_bundle)])
    assert result.exit_code == 0, result.output

    # File must have been restored to the generated content.
    assert edited.read_text(encoding="utf-8") == original
    assert "overwritten" in result.output


# ---------------------------------------------------------------------------
# seed-graph
# ---------------------------------------------------------------------------

# The graph store is never dialed here: every ``graph_seeder`` primitive is
# replaced by a recorder, so these tests pin the VERB's policy (which state it
# reports, what it refuses, in which order it calls the primitives) and leave
# the Cypher to tests/services/facility_knowledge/test_graph_seeder.py.

_TTL_TEXT = "@prefix narad: <https://narad.example.org/> .\n"

#: sha256 of :data:`_TTL_TEXT`, which is the digest the verb must put on the
#: marker node. Recomputed here rather than imported so a change to the verb's
#: hashing rule fails instead of following along.
_TTL_SHA256 = hashlib.sha256(_TTL_TEXT.encode("utf-8")).hexdigest()


@dataclass
class _GraphStub:
    """Recorder standing in for a bootstrapped graph store.

    The attributes before ``calls`` are the store's state, set by a test before
    it invokes the verb; everything from ``calls`` down is what the verb did.
    """

    marker: str | None = None
    resources: int = 0
    bootstrap_result: BootstrapResult = field(
        default_factory=lambda: BootstrapResult(status=BootstrapStatus.INITIALIZED)
    )
    import_result: ImportResult = field(
        default_factory=lambda: ImportResult(
            termination_status="OK", triples_loaded=8114, triples_parsed=8114
        )
    )
    open_error: Exception | None = None

    calls: list[str] = field(default_factory=list)
    connection: tuple[str, str, str] | None = None
    imported_text: str | None = None
    written_marker: str | None = None


@pytest.fixture()
def graph(monkeypatch: pytest.MonkeyPatch) -> _GraphStub:
    """Replace every ``graph_seeder`` primitive with a recorder."""
    from osprey.services.facility_knowledge.seeder import graph_seeder

    stub = _GraphStub()
    session = object()

    @contextmanager
    def _open_session(uri: str, username: str, password: str, **_: object):
        stub.connection = (uri, username, password)
        if stub.open_error is not None:
            raise stub.open_error
        stub.calls.append("open_session")
        yield session

    def _bootstrap(_session: object) -> BootstrapResult:
        stub.calls.append("bootstrap")
        return stub.bootstrap_result

    def _resource_count(_session: object) -> int:
        stub.calls.append("resource_count")
        return stub.resources

    def _read_marker(_session: object) -> str | None:
        stub.calls.append("read_marker")
        return stub.marker

    def _write_marker(_session: object, sha256: str) -> None:
        stub.calls.append("write_marker")
        stub.written_marker = sha256

    def _import_ttl(_session: object, text: str) -> ImportResult:
        stub.calls.append("import_ttl")
        stub.imported_text = text
        return stub.import_result

    def _wipe(_session: object) -> None:
        stub.calls.append("wipe")

    monkeypatch.setattr(graph_seeder, "open_session", _open_session)
    monkeypatch.setattr(graph_seeder, "bootstrap", _bootstrap)
    monkeypatch.setattr(graph_seeder, "resource_count", _resource_count)
    monkeypatch.setattr(graph_seeder, "read_marker", _read_marker)
    monkeypatch.setattr(graph_seeder, "write_marker", _write_marker)
    monkeypatch.setattr(graph_seeder, "import_ttl", _import_ttl)
    monkeypatch.setattr(graph_seeder, "wipe", _wipe)
    return stub


@pytest.fixture()
def graph_ttl(tmp_path: Path) -> Path:
    """A TTL file whose text hashes to :data:`_TTL_SHA256`."""
    path = tmp_path / "corpus.ttl"
    path.write_text(_TTL_TEXT, encoding="utf-8")
    return path


def _patch_config(monkeypatch: pytest.MonkeyPatch, block: object) -> None:
    """Make ``services.graphdb`` read as *block* wherever the verb looks it up."""

    def _get_config_value(path: str, default: object = None, config_path: object = None) -> object:
        if path == "services.graphdb":
            return block
        if path == "services.graphdb.ttl_path":
            return block.get("ttl_path", default) if isinstance(block, dict) else default
        return default

    monkeypatch.setattr("osprey.utils.config.get_config_value", _get_config_value)


def _flat(result) -> str:
    """Rich wraps at the console width, so compare on whitespace-collapsed text."""
    return " ".join(result.output.split())


def test_seed_graph_help_contrasts_with_seed_from_ttl() -> None:
    """The verb's own help says which of the two seeders it is."""
    runner = CliRunner()

    listing = runner.invoke(knowledge, ["--help"])
    assert listing.exit_code == 0, listing.output
    assert "seed-graph" in listing.output

    detail = runner.invoke(knowledge, ["seed-graph", "--help"])
    assert detail.exit_code == 0, detail.output
    flat = _flat(detail)
    assert "seeds the deployed graph store" in flat
    assert "seed-from-ttl builds an OKF document bundle" in flat


def test_seed_graph_seeds_an_empty_store(
    graph: _GraphStub, graph_ttl: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty store: bootstrap, import, then write the marker. In that order."""
    _patch_config(monkeypatch, {"port_host": 7699})
    monkeypatch.setenv("GRAPHDB_PASSWORD", "s3cret")

    result = CliRunner().invoke(knowledge, ["seed-graph", str(graph_ttl)])

    assert result.exit_code == 0, result.output
    assert "seeded" in _flat(result).lower()
    assert graph.connection == ("bolt://localhost:7699", "neo4j", "s3cret")
    assert graph.imported_text == _TTL_TEXT
    assert graph.written_marker == _TTL_SHA256

    ordered = [call for call in graph.calls if call in {"bootstrap", "import_ttl", "write_marker"}]
    assert ordered == ["bootstrap", "import_ttl", "write_marker"]


def test_seed_graph_reports_unchanged_on_a_matching_marker(
    graph: _GraphStub, graph_ttl: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A marker holding this TTL's digest means there is nothing to do."""
    _patch_config(monkeypatch, {})
    graph.marker = _TTL_SHA256
    graph.resources = 8114

    result = CliRunner().invoke(knowledge, ["seed-graph", str(graph_ttl)])

    assert result.exit_code == 0, result.output
    assert "unchanged" in _flat(result).lower()
    assert "import_ttl" not in graph.calls
    assert "write_marker" not in graph.calls
    assert "bootstrap" not in graph.calls


def test_seed_graph_refuses_a_differing_marker_without_force(
    graph: _GraphStub, graph_ttl: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store holding some other corpus is left alone and named as such."""
    _patch_config(monkeypatch, {})
    graph.marker = "0" * 64
    graph.resources = 12

    result = CliRunner().invoke(knowledge, ["seed-graph", str(graph_ttl)])

    assert result.exit_code != 0
    flat = _flat(result)
    assert "--force" in flat
    assert "differ" in flat.lower()
    assert "import_ttl" not in graph.calls
    assert "write_marker" not in graph.calls
    assert "wipe" not in graph.calls
    assert "Traceback" not in result.output


def test_seed_graph_refuses_an_unmanaged_partial_store_without_force(
    graph: _GraphStub, graph_ttl: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resource nodes with no marker: a crashed import or an adopted store."""
    _patch_config(monkeypatch, {})
    graph.marker = None
    graph.resources = 4200

    result = CliRunner().invoke(knowledge, ["seed-graph", str(graph_ttl)])

    assert result.exit_code != 0
    flat = _flat(result)
    assert "--force" in flat
    assert "4200" in flat
    assert "import_ttl" not in graph.calls
    assert "write_marker" not in graph.calls


def test_seed_graph_refuses_a_config_drifted_store_without_force(
    graph: _GraphStub, graph_ttl: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bootstrap drift lands in the differs state and surfaces its own message."""
    _patch_config(monkeypatch, {})
    graph.bootstrap_result = BootstrapResult(
        status=BootstrapStatus.DIFFERS, differing_keys=("handleMultival",)
    )

    result = CliRunner().invoke(knowledge, ["seed-graph", str(graph_ttl)])

    assert result.exit_code != 0
    flat = _flat(result)
    assert "graph config differs" in flat
    assert "handleMultival" in flat
    assert "--force" in flat
    assert "import_ttl" not in graph.calls
    assert "write_marker" not in graph.calls


def test_seed_graph_force_wipes_then_bootstraps_then_imports(
    graph: _GraphStub, graph_ttl: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--force replaces whatever is there, in the one order that works."""
    _patch_config(monkeypatch, {})
    graph.marker = "0" * 64
    graph.resources = 12

    result = CliRunner().invoke(knowledge, ["seed-graph", "--force", str(graph_ttl)])

    assert result.exit_code == 0, result.output
    assert "overwritten" in _flat(result).lower()
    assert graph.calls == [
        "open_session",
        "wipe",
        "bootstrap",
        "import_ttl",
        "write_marker",
    ]
    assert graph.written_marker == _TTL_SHA256


def test_seed_graph_leaves_no_marker_when_the_import_fails(
    graph: _GraphStub, graph_ttl: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A half-applied import must stay detectable, so no marker is written."""
    _patch_config(monkeypatch, {})
    graph.import_result = ImportResult(
        termination_status="KO",
        triples_loaded=3000,
        triples_parsed=8114,
        extra_info="bad syntax on line 41",
    )

    result = CliRunner().invoke(knowledge, ["seed-graph", str(graph_ttl)])

    assert result.exit_code != 0
    assert "write_marker" not in graph.calls
    flat = _flat(result)
    assert "bad syntax on line 41" in flat
    assert "Traceback" not in result.output


def test_seed_graph_resolves_a_relative_ttl_path_against_the_config_directory(
    graph: _GraphStub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The configured path is anchored at config.yml, never at the process CWD.

    Both directories hold a file at the same relative path, so a CWD-anchored
    resolution reads the decoy and this test fails on the imported payload
    rather than on a wrapped line of output.
    """
    project = tmp_path / "project"
    (project / "services" / "graphdb").mkdir(parents=True)
    (project / "config.yml").write_text("services:\n  graphdb: {}\n", encoding="utf-8")
    (project / "services" / "graphdb" / "als.ttl").write_text(_TTL_TEXT, encoding="utf-8")

    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "services" / "graphdb").mkdir(parents=True)
    (elsewhere / "services" / "graphdb" / "als.ttl").write_text(
        "# the decoy next to the process CWD\n", encoding="utf-8"
    )

    _patch_config(monkeypatch, {"ttl_path": "./services/graphdb/als.ttl"})
    monkeypatch.setenv("OSPREY_CONFIG", str(project / "config.yml"))
    monkeypatch.chdir(elsewhere)

    result = CliRunner().invoke(knowledge, ["seed-graph"])

    assert result.exit_code == 0, result.output
    assert graph.imported_text == _TTL_TEXT
    assert graph.written_marker == _TTL_SHA256


def test_seed_graph_typed_argument_stays_shell_relative(
    graph: _GraphStub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A TTL path typed on the command line resolves against the shell's CWD.

    Only the configured ``ttl_path`` gets the config-directory anchor; a typed
    argument behaves like any filename handed to a CLI. The decoy sits at the
    same relative path under the config directory, so a config-anchored
    resolution of the argument fails on the imported payload.
    """
    configured = tmp_path / "configured"
    configured.mkdir()
    (configured / "config.yml").write_text("services:\n  graphdb: {}\n", encoding="utf-8")
    (configured / "als.ttl").write_text("# the decoy next to config.yml\n", encoding="utf-8")

    shell_cwd = tmp_path / "shell"
    shell_cwd.mkdir()
    (shell_cwd / "als.ttl").write_text(_TTL_TEXT, encoding="utf-8")

    _patch_config(monkeypatch, {})
    monkeypatch.setenv("OSPREY_CONFIG", str(configured / "config.yml"))
    monkeypatch.chdir(shell_cwd)

    result = CliRunner().invoke(knowledge, ["seed-graph", "als.ttl"])

    assert result.exit_code == 0, result.output
    assert graph.imported_text == _TTL_TEXT
    assert graph.written_marker == _TTL_SHA256


def test_seed_graph_unreadable_ttl_is_a_clean_error(
    graph: _GraphStub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A TTL that cannot be decoded surfaces as a message, not a traceback."""
    _patch_config(monkeypatch, {})
    bad = tmp_path / "not_utf8.ttl"
    bad.write_bytes(b"\xff\xfe\x00 not a turtle file")

    result = CliRunner().invoke(knowledge, ["seed-graph", str(bad)])

    assert result.exit_code != 0
    assert "Cannot read" in _flat(result)
    assert "Traceback" not in result.output
    assert "import_ttl" not in graph.calls


def test_seed_graph_without_a_ttl_anywhere_is_a_usage_error(
    graph: _GraphStub, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No argument and no configured path names the config key."""
    _patch_config(monkeypatch, {})

    result = CliRunner().invoke(knowledge, ["seed-graph"])

    assert result.exit_code != 0
    assert "services.graphdb.ttl_path" in _flat(result)
    assert graph.calls == []


def test_seed_graph_missing_driver_is_a_clean_error(
    graph: _GraphStub, graph_ttl: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absent neo4j driver names the dependency and prints no traceback."""
    _patch_config(monkeypatch, {})
    graph.open_error = ImportError(
        "neo4j is required to reach the graph store; repair the install with: "
        "pip install 'osprey-framework'"
    )

    result = CliRunner().invoke(knowledge, ["seed-graph", str(graph_ttl)])

    assert result.exit_code != 0
    assert "neo4j" in _flat(result)
    assert "Traceback" not in result.output


def test_seed_graph_auth_failure_names_the_password_variable(
    graph: _GraphStub, graph_ttl: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rejected credential points at GRAPHDB_PASSWORD, not at the driver."""
    exceptions = pytest.importorskip("neo4j.exceptions")
    _patch_config(monkeypatch, {})
    graph.open_error = exceptions.AuthError("The client is unauthorized due to authentication.")

    result = CliRunner().invoke(knowledge, ["seed-graph", str(graph_ttl)])

    assert result.exit_code != 0
    assert "GRAPHDB_PASSWORD" in _flat(result)
    assert "Traceback" not in result.output


def test_seed_graph_unreachable_store_is_a_clean_error(
    graph: _GraphStub, graph_ttl: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreachable store names the address that was dialed."""
    exceptions = pytest.importorskip("neo4j.exceptions")
    _patch_config(monkeypatch, {"port_host": 7699})
    graph.open_error = exceptions.ServiceUnavailable("Unable to retrieve routing information")

    result = CliRunner().invoke(knowledge, ["seed-graph", str(graph_ttl)])

    assert result.exit_code != 0
    assert "bolt://localhost:7699" in _flat(result)
    assert "Traceback" not in result.output
