"""``osprey mml emit``: the command that turns a checked mapping into artifacts.

Pins the command surface the emit services cannot: the full chain on one
fixture writes every artifact where the deployment reads it, a re-run leaves
every non-DuckDB file byte-identical, the channel database lands at both the
flat path and ``tiers/tier3/`` when the deployment stages tiers, the demo tier
siblings and untouched demo knowledge pages refuse the run before any write
with one ``rm`` line, a mapping missing a signal group's direction refuses,
and the Turtle corpus opens with its mapping provenance.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from unittest import mock

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.main import cli

pytest.importorskip("linkml_runtime")

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "mml"
PACKAGED_DATA = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "osprey"
    / "templates"
    / "apps"
    / "control_assistant"
    / "data"
)
PACKAGED_KNOWLEDGE = PACKAGED_DATA / "facility_knowledge"
PACKAGED_TIERS = PACKAGED_DATA / "channel_databases" / "tiers"

DEMO_TIER_SIBLINGS = (
    "data/channel_databases/tiers/tier1/in_context.json",
    "data/channel_databases/tiers/tier3/hierarchical.json",
    "data/channel_databases/tiers/tier3/in_context.json",
)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A deployment repo with the ``paired`` fixture imported and mapped."""
    root = tmp_path / "deploy"
    root.mkdir()
    (root / "profile.yml").write_text("name: scratch\n", encoding="utf-8")
    monkeypatch.chdir(root)
    for name in ("quokka.ring.ao.json", "quokka.ring.ad.json"):
        shutil.copy(FIXTURES / "paired" / name, root / name)
    result = CliRunner().invoke(
        cli, ["mml", "import", "quokka.ring.ao.json"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    shutil.copy(FIXTURES / "paired" / "mapping.yaml", root / "data" / "mml" / "mapping.yaml")
    return root


def _emit(*args: str):
    return CliRunner().invoke(cli, ["mml", "emit", *args], catch_exceptions=False)


def _token(repo: Path) -> str:
    mapping = yaml.safe_load((repo / "data" / "mml" / "mapping.yaml").read_text(encoding="utf-8"))
    return mapping["facility"]["token"]


def _artifacts(repo: Path) -> dict[str, bytes]:
    """Every emitted non-DuckDB file under ``data/``, keyed by relative path."""
    data = repo / "data"
    return {
        path.relative_to(repo).as_posix(): path.read_bytes()
        for path in sorted(data.rglob("*"))
        if path.is_file()
        and path.suffix != ".duckdb"
        and not path.relative_to(data).as_posix().startswith("mml/")
    }


def _rm_lines(output: str) -> list[str]:
    return [line for line in output.splitlines() if line.startswith("rm ")]


def _stage_tiers(repo: Path) -> Path:
    tiers = repo / "data" / "channel_databases" / "tiers"
    shutil.copytree(PACKAGED_TIERS, tiers)
    return tiers


class TestFullChain:
    def test_writes_every_artifact(self, repo: Path) -> None:
        result = _emit()

        assert "Traceback" not in result.output
        assert result.exit_code == 0, result.output
        token = _token(repo)
        data = repo / "data"
        db_path = data / "channel_databases" / "middle_layer.json"
        expected = (
            db_path,
            data / "ontology" / f"{token}.yaml",
            data / "facility_ontology.json",
            data / "facility_knowledge" / "facility.md",
            data / "facility_knowledge" / "index.md",
            data / f"{token}.ttl",
        )
        for path in expected:
            assert path.is_file(), f"{path} was not written"
        assert list((data / "facility_knowledge" / "families").glob("*.md"))
        assert not (data / "channel_databases" / "middle_layer.duckdb").exists()
        assert not (data / "channel_databases" / "tiers").exists()

        text = db_path.read_text(encoding="utf-8")
        assert text.endswith("\n")
        assert json.loads(text)

        # One line per artifact; the knowledge bundle is one artifact.
        for path in (*expected[:3], expected[-1]):
            assert str(path) in result.output
        assert str(data / "facility_knowledge") in result.output
        assert "osprey build" in result.output.strip().splitlines()[-1]

    def test_ttl_first_line_names_mapping_provenance(self, repo: Path) -> None:
        assert _emit().exit_code == 0
        ttl = repo / "data" / f"{_token(repo)}.ttl"

        lines = ttl.read_text(encoding="utf-8").splitlines()

        assert lines[0] == "# osprey:direction-source mapping"
        assert lines[1].startswith("# exporter=")
        assert lines[2].startswith("# ao_sha256=")
        assert lines[3].startswith("# mapping_sha256=")

    def test_rerun_is_byte_identical(self, repo: Path) -> None:
        assert _emit().exit_code == 0
        first = _artifacts(repo)

        result = _emit()

        assert result.exit_code == 0, result.output
        assert _artifacts(repo) == first

    def test_duckdb_flag_writes_default_path(self, repo: Path) -> None:
        pytest.importorskip("duckdb")

        result = _emit("--duckdb")

        assert result.exit_code == 0, result.output
        duck = repo / "data" / "channel_databases" / "middle_layer.duckdb"
        assert duck.is_file()
        assert "middle_layer.duckdb" in result.output

    def test_duckdb_flag_accepts_a_path(self, repo: Path) -> None:
        pytest.importorskip("duckdb")

        result = _emit("--duckdb", "custom.duckdb")

        assert result.exit_code == 0, result.output
        assert (repo / "custom.duckdb").is_file()
        assert not (repo / "data" / "channel_databases" / "middle_layer.duckdb").exists()


class TestTiers:
    def test_demo_siblings_refuse_naming_each(self, repo: Path) -> None:
        _stage_tiers(repo)

        result = _emit()

        assert "Traceback" not in result.output
        assert result.exit_code != 0
        lines = _rm_lines(result.output)
        assert len(lines) == 1, result.output
        for sibling in DEMO_TIER_SIBLINGS:
            assert sibling in lines[0]
        assert "tier3/middle_layer.json" not in lines[0]
        assert not (repo / "data" / "channel_databases" / "middle_layer.json").exists()
        assert not (repo / "data" / f"{_token(repo)}.ttl").exists()

    def test_dual_write_under_tiers(self, repo: Path) -> None:
        tiers = _stage_tiers(repo)
        for sibling in DEMO_TIER_SIBLINGS:
            (repo / sibling).unlink()

        result = _emit()

        assert result.exit_code == 0, result.output
        flat = repo / "data" / "channel_databases" / "middle_layer.json"
        tiered = tiers / "tier3" / "middle_layer.json"
        assert tiered.read_bytes() == flat.read_bytes()
        assert tiers.is_dir()
        assert str(Path("tiers") / "tier3" / "middle_layer.json") in result.output


class TestDemoKnowledge:
    def _stage_devices(self, repo: Path) -> Path:
        bundle = repo / "data" / "facility_knowledge"
        bundle.mkdir(parents=True, exist_ok=True)
        shutil.copy(PACKAGED_KNOWLEDGE / "index.md", bundle / "index.md")
        shutil.copytree(PACKAGED_KNOWLEDGE / "devices", bundle / "devices")
        return bundle

    def test_untouched_directory_refuses_with_rm_r(self, repo: Path) -> None:
        self._stage_devices(repo)

        result = _emit()

        assert "Traceback" not in result.output
        assert result.exit_code != 0
        lines = _rm_lines(result.output)
        assert len(lines) == 1, result.output
        assert lines[0].startswith("rm -r data/facility_knowledge/devices")
        assert "devices/bpm.md" not in lines[0]
        assert not (repo / "data" / "channel_databases" / "middle_layer.json").exists()
        assert not (repo / "data" / "facility_knowledge" / "facility.md").exists()

    def test_root_index_alone_passes(self, repo: Path) -> None:
        bundle = repo / "data" / "facility_knowledge"
        bundle.mkdir(parents=True)
        shutil.copy(PACKAGED_KNOWLEDGE / "index.md", bundle / "index.md")

        result = _emit()

        assert result.exit_code == 0, result.output

    def test_mixed_directory_names_individual_files(self, repo: Path) -> None:
        bundle = self._stage_devices(repo)
        own = bundle / "devices" / "ion-pump.md"
        own.write_text(own.read_text(encoding="utf-8") + "\nEdited here.\n", encoding="utf-8")

        result = _emit()

        assert result.exit_code != 0
        lines = _rm_lines(result.output)
        assert len(lines) == 1, result.output
        assert "data/facility_knowledge/devices/bpm.md" in lines[0]
        assert "data/facility_knowledge/devices/index.md" in lines[0]
        assert "ion-pump.md" not in lines[0]
        assert "rm -r" not in lines[0]

    def test_edited_copy_passes(self, repo: Path) -> None:
        bundle = self._stage_devices(repo)
        for page in (bundle / "devices").glob("*.md"):
            page.write_text(page.read_text(encoding="utf-8") + "\nEdited.\n", encoding="utf-8")

        result = _emit()

        assert "Traceback" not in result.output
        assert result.exit_code == 0, result.output

    def test_after_removal_every_concept_resolves(self, repo: Path) -> None:
        from osprey.services.facility_knowledge.okf.bundle import OKFBundle

        bundle = self._stage_devices(repo)
        assert _emit().exit_code != 0
        shutil.rmtree(bundle / "devices")

        result = _emit()

        assert result.exit_code == 0, result.output
        concepts = OKFBundle(bundle).list_concepts()
        assert concepts
        for entry in concepts:
            assert (bundle / f"{entry.concept_id}.md").is_file(), entry.concept_id
        assert "devices" not in (bundle / "index.md").read_text(encoding="utf-8")

    def test_both_refusals_report_together(self, repo: Path) -> None:
        _stage_tiers(repo)
        self._stage_devices(repo)

        result = _emit()

        assert result.exit_code != 0
        lines = _rm_lines(result.output)
        assert len(lines) == 1, result.output
        assert lines[0].startswith("rm -r data/facility_knowledge/devices")
        for sibling in DEMO_TIER_SIBLINGS:
            assert sibling in lines[0]


class TestRefusals:
    def test_missing_direction_group_refuses(self, repo: Path) -> None:
        mapping_path = repo / "data" / "mml" / "mapping.yaml"
        document = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
        removed = next(iter(document["directions"]))
        del document["directions"][removed]
        mapping_path.write_text(
            yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )

        result = _emit()

        assert "Traceback" not in result.output
        assert result.exit_code != 0
        assert f"directions.{removed}" in result.output
        assert "map --check" in result.output
        assert not (repo / "data" / "channel_databases" / "middle_layer.json").exists()

    def test_missing_mapping_refuses(self, repo: Path) -> None:
        (repo / "data" / "mml" / "mapping.yaml").unlink()

        result = _emit()

        assert result.exit_code != 0
        assert "Traceback" not in result.output
        assert "map --init" in result.output

    def test_missing_import_refuses(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        root = tmp_path / "empty"
        root.mkdir()
        (root / "profile.yml").write_text("name: scratch\n", encoding="utf-8")
        monkeypatch.chdir(root)

        result = _emit()

        assert result.exit_code != 0
        assert "Traceback" not in result.output
        assert "osprey mml import" in result.output

    def test_missing_knowledge_extra_refuses_before_anything(self, repo: Path) -> None:
        _stage_tiers(repo)
        with mock.patch.dict(sys.modules, {"linkml_runtime": None}):
            result = _emit()

        assert result.exit_code != 0
        assert "knowledge" in result.output
        assert not _rm_lines(result.output)
