"""Tests for ``osprey knowledge build-index``.

Covers:
- A bare run takes both paths from config: the corpus from
  ``services.graphdb.ttl_path`` and the index from
  ``services.graphdb.index_path``, both resolved against the render's own
  ``config.yml`` directory rather than the process CWD.
- ``--ttl`` and ``--output`` override each of those, and a typed path stays
  shell-relative like every other filename an operator types.
- The output parent is created, so a first build into a render that has no
  ``data/channel_databases`` yet writes rather than refusing.
- A project with no corpus configured is refused in one sentence, with the
  exit code of a state that is wrong rather than of a command that was
  mistyped.
- The corpus failures an operator can actually hit -- a path naming nothing, a
  file that is not valid Turtle, and a file that is not UTF-8 -- are one
  legible line each and never a traceback.
- A malformed ``services.graphdb.index_path`` and an output parent that cannot
  be created are refused the same way.
- The digest the run prints is the corpus checksum the graph store's seed
  marker carries, truncated to the prefix the health row shows.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.knowledge_cmd import knowledge
from tests.services.channel_finder.graph_index import corpora

#: What ``SUBCLASS_CHAIN`` holds: three bindings under one device, the pruned
#: chain Quadrupole/Magnet/AcceleratorDevice, two signals, one section, and one
#: channel per binding.
EXPECTED_COUNTS = "3 bindings, 1 devices, 3 classes, 2 signals, 1 sections, 3 channels."

#: How many characters of the corpus checksum the verb prints, matching the
#: prefix the ``channel_finder_search_index`` health row shows.
DIGEST_PREFIX_LEN = 12


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _patch_config(monkeypatch: pytest.MonkeyPatch, block: object) -> None:
    """Make ``services.graphdb`` read as *block* wherever the verb looks it up."""

    def _get_config_value(path: str, default: object = None, config_path: object = None) -> object:
        if path == "services.graphdb":
            return block
        if path == "services.graphdb.ttl_path":
            return block.get("ttl_path", default) if isinstance(block, dict) else default
        return default

    monkeypatch.setattr("osprey.utils.config.get_config_value", _get_config_value)


def _flat(result: object) -> str:
    """Rich wraps at the console width, so compare on whitespace-collapsed text."""
    return " ".join(result.output.split())  # type: ignore[attr-defined]


@pytest.fixture()
def render(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A rendered project holding a corpus, published through ``OSPREY_CONFIG``.

    The process CWD is deliberately somewhere else, so a resolution anchored on
    it rather than on ``config.yml`` fails these tests instead of passing by
    coincidence.
    """
    build = tmp_path / "build"
    (build / "data").mkdir(parents=True)
    (build / "config.yml").write_text("services:\n  graphdb: {}\n", encoding="utf-8")
    (build / "data" / "demo.ttl").write_text(corpora.SUBCLASS_CHAIN, encoding="utf-8")

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setenv("OSPREY_CONFIG", str(build / "config.yml"))
    monkeypatch.chdir(elsewhere)
    return build


# ---------------------------------------------------------------------------
# Defaults from config
# ---------------------------------------------------------------------------


def test_build_index_takes_both_paths_from_config(
    render: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare run reads the configured corpus and writes the configured index."""
    _patch_config(monkeypatch, {"ttl_path": "./data/demo.ttl"})

    result = CliRunner().invoke(knowledge, ["build-index"])

    assert result.exit_code == 0, result.output
    index = render / "data" / "channel_databases" / "graph.duckdb"
    assert index.is_file(), "the shipped default index_path, under the render"
    flat = _flat(result)
    assert f"Wrote {index}." in flat
    assert EXPECTED_COUNTS in flat


def test_build_index_honours_a_configured_index_path(
    render: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``index_path`` moves the index, and is resolved against the render."""
    _patch_config(
        monkeypatch,
        {"ttl_path": "./data/demo.ttl", "index_path": "./var/search/graph.duckdb"},
    )

    result = CliRunner().invoke(knowledge, ["build-index"])

    assert result.exit_code == 0, result.output
    assert (render / "var" / "search" / "graph.duckdb").is_file()


def test_build_index_creates_the_output_parent(
    render: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A render with no channel-database directory yet still gets an index."""
    _patch_config(monkeypatch, {"ttl_path": "./data/demo.ttl"})
    assert not (render / "data" / "channel_databases").exists()

    result = CliRunner().invoke(knowledge, ["build-index"])

    assert result.exit_code == 0, result.output
    assert (render / "data" / "channel_databases" / "graph.duckdb").is_file()


# ---------------------------------------------------------------------------
# Explicit paths
# ---------------------------------------------------------------------------


def test_build_index_explicit_ttl_and_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both flags win over config, and neither needs a render to be there."""
    _patch_config(monkeypatch, None)
    ttl = tmp_path / "corpus.ttl"
    ttl.write_text(corpora.SUBCLASS_CHAIN, encoding="utf-8")
    output = tmp_path / "out" / "index.duckdb"

    result = CliRunner().invoke(
        knowledge, ["build-index", "--ttl", str(ttl), "--output", str(output)]
    )

    assert result.exit_code == 0, result.output
    assert output.is_file()
    flat = _flat(result)
    assert f"Wrote {output}." in flat
    assert f"Built from {ttl}" in flat


def test_build_index_typed_ttl_stays_shell_relative(
    render: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A typed corpus resolves against the CWD, not against config.yml.

    The decoy sits at the same relative path under the render, so a
    config-anchored reading of the argument indexes the wrong file and this
    test fails on the digest rather than on a wrapped line of output.
    """
    from osprey.services.facility_knowledge.seeder.graph_seeder import ttl_sha256

    _patch_config(monkeypatch, {"ttl_path": "./data/demo.ttl"})
    (render / "typed.ttl").write_text(corpora.BOTH_EDGES, encoding="utf-8")

    shell_cwd = tmp_path / "elsewhere"
    (shell_cwd / "typed.ttl").write_text(corpora.SUBCLASS_CHAIN, encoding="utf-8")
    monkeypatch.chdir(shell_cwd)

    result = CliRunner().invoke(knowledge, ["build-index", "--ttl", "typed.ttl"])

    assert result.exit_code == 0, result.output
    digest = ttl_sha256(corpora.SUBCLASS_CHAIN)
    assert digest[:DIGEST_PREFIX_LEN] in _flat(result)


# ---------------------------------------------------------------------------
# The digest
# ---------------------------------------------------------------------------


def test_build_index_prints_the_seed_marker_digest_prefix(
    render: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The printed prefix is the store's seed digest, so the two can be compared."""
    from osprey.services.facility_knowledge.seeder.graph_seeder import ttl_sha256

    _patch_config(monkeypatch, {"ttl_path": "./data/demo.ttl"})
    expected = ttl_sha256(corpora.SUBCLASS_CHAIN)[:DIGEST_PREFIX_LEN]

    result = CliRunner().invoke(knowledge, ["build-index"])

    assert result.exit_code == 0, result.output
    assert f"corpus sha256 {expected}." in _flat(result)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_build_index_refuses_when_no_corpus_is_configured(
    render: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ttl_path and no --ttl: one sentence naming the key and the verb."""
    from osprey.deployment.graphdb_service import GRAPHDB_BUILD_INDEX_COMMAND

    _patch_config(monkeypatch, {})

    result = CliRunner().invoke(knowledge, ["build-index"])

    assert result.exit_code == 1
    assert (
        "set services.graphdb.ttl_path to the corpus this store was seeded from, "
        f"then run {GRAPHDB_BUILD_INDEX_COMMAND}"
    ) in _flat(result)
    assert not (render / "data" / "channel_databases").exists()


def test_build_index_missing_corpus_is_a_clean_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A --ttl naming nothing is one line with the path on it."""
    _patch_config(monkeypatch, None)
    absent = tmp_path / "not-there.ttl"
    output = tmp_path / "index.duckdb"

    result = CliRunner().invoke(
        knowledge, ["build-index", "--ttl", str(absent), "--output", str(output)]
    )

    assert result.exit_code == 1
    assert f"There is no corpus at {absent}." in _flat(result)
    assert "Traceback" not in result.output
    assert not output.exists()


def test_build_index_invalid_turtle_is_a_clean_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corpus that will not parse surfaces the builder's message, not a traceback."""
    _patch_config(monkeypatch, None)
    ttl = tmp_path / "broken.ttl"
    ttl.write_text(corpora.INVALID_TURTLE, encoding="utf-8")
    output = tmp_path / "index.duckdb"

    result = CliRunner().invoke(
        knowledge, ["build-index", "--ttl", str(ttl), "--output", str(output)]
    )

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    flat = _flat(result)
    assert "Wrote" not in flat
    assert "The corpus is not valid Turtle:" in flat, "the builder's own message, verbatim"
    assert f"Corpus: {ttl}." in flat, "names which file failed to parse"


def test_build_index_non_utf8_corpus_is_a_clean_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corpus that is not UTF-8 is one legible line naming the path, not a traceback."""
    _patch_config(monkeypatch, None)
    ttl = tmp_path / "binary.ttl"
    ttl.write_bytes(b"\xff\xfe\x00\x01not-utf8")
    output = tmp_path / "index.duckdb"

    result = CliRunner().invoke(
        knowledge, ["build-index", "--ttl", str(ttl), "--output", str(output)]
    )

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    flat = _flat(result)
    assert f"Cannot read the corpus at {ttl}" in flat
    assert not output.exists()


def test_build_index_uncreatable_output_parent_is_a_clean_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parent directory that cannot be created is refused, not a traceback."""
    _patch_config(monkeypatch, None)
    ttl = tmp_path / "corpus.ttl"
    ttl.write_text(corpora.SUBCLASS_CHAIN, encoding="utf-8")

    readonly = tmp_path / "readonly"
    readonly.mkdir()
    readonly.chmod(0o555)
    if os.access(readonly, os.W_OK):  # pragma: no cover - root, or a permissionless FS
        pytest.skip("the chmod did not take effect (running as root?)")
    output = readonly / "sub" / "index.duckdb"

    try:
        result = CliRunner().invoke(
            knowledge, ["build-index", "--ttl", str(ttl), "--output", str(output)]
        )
    finally:
        readonly.chmod(0o755)

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert f"Cannot create {output.parent}" in _flat(result)


def test_build_index_refuses_a_malformed_index_path(
    render: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-string services.graphdb.index_path is refused by name, not a traceback."""
    _patch_config(monkeypatch, {"ttl_path": "./data/demo.ttl", "index_path": 7})

    result = CliRunner().invoke(knowledge, ["build-index"])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "Cannot use services.graphdb.index_path" in _flat(result)


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------


def test_build_index_help_names_its_defaults_and_the_restart() -> None:
    """The verb's help says where each path comes from and what to restart."""
    result = CliRunner().invoke(knowledge, ["build-index", "--help"])

    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    assert "services.graphdb.ttl_path" in flat
    assert "services.graphdb.index_path" in flat
    assert "restart it to read a rebuilt one" in flat


def test_build_index_is_listed_among_the_knowledge_verbs() -> None:
    """The group's listing carries the verb, so it is discoverable."""
    result = CliRunner().invoke(knowledge, ["--help"])

    assert result.exit_code == 0, result.output
    assert "build-index" in result.output
