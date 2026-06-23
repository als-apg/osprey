"""Tests for the ``osprey knowledge`` CLI command group.

Covers:
- ``knowledge --help`` lists regen-index, validate, and seed-from-ttl.
- ``knowledge regen-index <bundle>`` writes index files and is idempotent.
- ``knowledge validate <bundle>`` exits 0 on a clean bundle; exits 1 and
  collects ALL failures (broken concept doc + broken index.md in one run).
- ``knowledge seed-from-ttl <ttl> <bundle>`` writes stubs on first run,
  is idempotent on re-run, skips diffs without --force, overwrites with
  --force, and exits cleanly when rdflib is absent.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.knowledge_cmd import knowledge

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
