"""``osprey mml emit``: the command that turns a checked mapping into artifacts.

Pins the command surface the emit services cannot: the full chain on one
fixture writes every artifact where the deployment reads it, a re-run leaves
every non-DuckDB file byte-identical, the channel database lands at both the
flat path and ``tiers/tier3/`` when the deployment stages tiers, the demo tier
siblings and untouched demo knowledge pages refuse the run before any write
with one ``rm`` line, a mapping missing a signal group's direction refuses,
and the Turtle corpus opens with its mapping provenance.

The refusals ``map --check`` owns are pinned here too: an unanswered judgment,
an answer the export cannot carry and an answer naming a family the export
does not have each stop the run before the first file, and a direction is
required of the judged grain, so a field a judgment creates needs one. The
mapping of the ``repo`` fixture is filled the way the map tests fill a
skeleton, so no case here rests on which judgments a committed fixture
happens to answer.

The DuckDB collapse report is pinned on two repos, because it counts the
bindings the judgments settle rather than the ones the export holds: the
``repo`` fixture keeps its shared PV whole and the report names both owners,
while ``wrapped_repo`` hands that family's shared PV to one device and the PV
leaves the report, the database and the SQL surface together. Both repos are
imported through the command, so the report is measured against the artifacts
of the same run.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from unittest import mock

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.main import cli
from tests.cli.test_mml_map import _fill

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
    _write_mapping(root, _fill(_document(root)))
    return root


@pytest.fixture
def nsls2_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A deployment repo with the two-system ``nsls2`` fixture imported and mapped.

    Its mapping answers a row beyond the devices with a field of its own, the
    one shape a judged direction and an impossible field name need.
    """
    root = tmp_path / "nsls2"
    root.mkdir()
    (root / "profile.yml").write_text("name: scratch\n", encoding="utf-8")
    monkeypatch.chdir(root)
    for path in sorted((FIXTURES / "nsls2").glob("*.json")):
        shutil.copy(path, root / path.name)
    result = CliRunner().invoke(
        cli,
        ["mml", "import", "nsls2.ltb.ao.json", "nsls2.storagering.ao.json"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    shutil.copy(FIXTURES / "nsls2" / "mapping.yaml", root / "data" / "mml" / "mapping.yaml")
    return root


@pytest.fixture
def wrapped_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A deployment repo with the ``wrapped`` fixture imported and mapped.

    Its committed mapping hands the family's shared PV to one of the two
    devices that name it, the answer the collapse report has to follow.
    """
    root = tmp_path / "wrapped"
    root.mkdir()
    (root / "profile.yml").write_text("name: scratch\n", encoding="utf-8")
    monkeypatch.chdir(root)
    shutil.copy(FIXTURES / "wrapped" / "export.json", root / "export.json")
    result = CliRunner().invoke(
        cli, ["mml", "import", "export.json", "--system", "INJ"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    shutil.copy(FIXTURES / "wrapped" / "mapping.yaml", root / "data" / "mml" / "mapping.yaml")
    return root


def _emit(*args: str):
    return CliRunner().invoke(cli, ["mml", "emit", *args], catch_exceptions=False)


def _check(*args: str):
    return CliRunner().invoke(cli, ["mml", "map", "--check", *args], catch_exceptions=False)


def _document(repo: Path) -> dict:
    return yaml.safe_load((repo / "data" / "mml" / "mapping.yaml").read_text(encoding="utf-8"))


def _write_mapping(repo: Path, document: dict) -> None:
    (repo / "data" / "mml" / "mapping.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _problem_lines(output: str, key: str) -> list[str]:
    return [line for line in output.splitlines() if line.startswith(f"{key}: ")]


def _judgment_lines(output: str) -> list[str]:
    return [line for line in output.splitlines() if line.startswith("judgments.")]


def _assert_wrote_nothing(repo: Path) -> None:
    """Assert emit left the deployment without a single artifact."""
    data = repo / "data"
    assert not (data / "channel_databases").exists()
    assert not (data / "facility_knowledge").exists()
    assert not (data / "facility_ontology.json").exists()
    assert not list(data.glob("*.ttl"))


def _token(repo: Path) -> str:
    return _document(repo)["facility"]["token"]


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


def _repeated_pv(export: Path) -> str:
    """The one PV a fixture export names twice, read off the export itself."""
    names = [
        name
        for family in json.loads(export.read_text(encoding="utf-8"))["ao"].values()
        for field in family.values()
        if isinstance(field, dict)
        for name in field.get("ChannelNames", [])
    ]
    repeated = {name for name in names if name and names.count(name) > 1}
    assert len(repeated) == 1, repeated
    return repeated.pop()


def _collapse_counts(output: str) -> tuple[int, int]:
    """The binding and row counts the collapse report's first sentence claims."""
    match = re.search(
        r"of (\d+) bindings share a PV with another, "
        r"so the DuckDB channels table holds (\d+) rows",
        output,
    )
    assert match, output
    return int(match.group(1)), int(match.group(2))


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


class TestCollapseReport:
    """Which bindings the DuckDB collapse report counts, and how it names them."""

    def test_shared_pv_owners_are_counted_from_one(self, repo: Path) -> None:
        """The owners of a shared PV are named by device ordinal, not by index."""
        pytest.importorskip("duckdb")

        result = _emit("--duckdb")

        assert result.exit_code == 0, result.output
        shared = [line.strip() for line in result.output.splitlines() if " is bound by " in line]
        assert shared == ["QK:R12:HCM:RB is bound by RING.HCM.Monitor[1], RING.HCM.Monitor[2]."], (
            result.output
        )

    def test_an_owned_shared_pv_is_not_reported(self, wrapped_repo: Path) -> None:
        """A shared PV one device owns is that device's binding and nothing else's."""
        pytest.importorskip("duckdb")
        pv = _repeated_pv(FIXTURES / "wrapped" / "export.json")

        result = _emit("--duckdb")

        assert result.exit_code == 0, result.output
        assert pv not in result.output
        assert not [line for line in result.output.splitlines() if " is bound by " in line]
        database = wrapped_repo / "data" / "channel_databases" / "middle_layer.json"
        assert database.read_text(encoding="utf-8").count(f'"{pv}"') == 1

    def test_a_broadcast_row_still_collapses(self, wrapped_repo: Path) -> None:
        """An answered family does not silence the report the rest of the export earns."""
        pytest.importorskip("duckdb")

        result = _emit("--duckdb")

        assert result.exit_code == 0, result.output
        detail = [line.strip() for line in result.output.splitlines() if " broadcasts one " in line]
        assert detail == ["INJ.QM.Setpoint broadcasts one ChannelNames entry to every device."], (
            result.output
        )

    def test_the_row_count_is_the_channels_table(self, wrapped_repo: Path) -> None:
        """The rows the report claims the SQL surface holds are the rows it holds."""
        duckdb = pytest.importorskip("duckdb")

        result = _emit("--duckdb")

        assert result.exit_code == 0, result.output
        bindings, rows = _collapse_counts(result.output)
        assert bindings > rows
        connection = duckdb.connect(
            str(wrapped_repo / "data" / "channel_databases" / "middle_layer.duckdb"), read_only=True
        )
        try:
            held = connection.execute("SELECT count(*) FROM channels").fetchone()[0]
        finally:
            connection.close()
        assert held == rows


class TestRefusals:
    def test_missing_direction_group_refuses(self, repo: Path) -> None:
        document = _document(repo)
        removed = next(iter(document["directions"]))
        del document["directions"][removed]
        _write_mapping(repo, document)

        result = _emit()

        assert "Traceback" not in result.output
        assert result.exit_code != 0
        assert f"directions.{removed}" in result.output
        assert "map --check" in result.output
        _assert_wrote_nothing(repo)

    def test_judged_field_needs_a_direction(self, nsls2_repo: Path) -> None:
        """A field a judgment creates is a signal group the directions must name."""
        document = _document(nsls2_repo)
        del document["directions"]["DCCT.Lifetime"]
        _write_mapping(nsls2_repo, document)

        result = _emit()

        assert "Traceback" not in result.output
        assert result.exit_code != 0
        assert _problem_lines(result.output, "directions.DCCT.Lifetime")
        assert "map --check" in result.output
        _assert_wrote_nothing(nsls2_repo)

    def test_null_judgment_refuses(self, repo: Path) -> None:
        """A slot the reviewer left open stops the run at that slot's key."""
        document = _document(repo)
        document["judgments"]["HCM"]["shared_pvs"] = None
        _write_mapping(repo, document)

        result = _emit()

        assert "Traceback" not in result.output
        assert result.exit_code != 0
        assert _judgment_lines(result.output) == ["judgments.HCM.shared_pvs: must not be null"], (
            result.output
        )
        assert _judgment_lines(result.output) == _judgment_lines(_check().output)
        assert "map --check" in result.output
        _assert_wrote_nothing(repo)

    def test_answer_naming_an_absent_family_refuses(self, repo: Path) -> None:
        """The pre-flight is as strict as the check, down to an invented family."""
        document = _document(repo)
        document["judgments"]["NOPE"] = {"shared_pvs": "keep_all"}
        _write_mapping(repo, document)

        result = _emit()

        assert "Traceback" not in result.output
        assert result.exit_code != 0
        assert len(_problem_lines(result.output, "judgments.NOPE")) == 1, result.output
        assert _judgment_lines(result.output) == _judgment_lines(_check().output)
        _assert_wrote_nothing(repo)

    def test_impossible_answer_refuses_in_the_checks_words(self, nsls2_repo: Path) -> None:
        """An answer the export cannot carry refuses, as ``map --check`` refuses it."""
        document = _document(nsls2_repo)
        rows = document["judgments"]["DCCT"]["rows_beyond_devices"]["Monitor"]
        signal = next(iter(rows))
        rows[signal] = {"field": "Monitor"}
        _write_mapping(nsls2_repo, document)

        result = _emit()

        assert "Traceback" not in result.output
        assert result.exit_code != 0
        key = f"judgments.DCCT.rows_beyond_devices.Monitor[{signal}]"
        assert len(_problem_lines(result.output, key)) == 1, result.output
        assert _judgment_lines(result.output) == _judgment_lines(_check().output)
        assert "map --check" in result.output
        _assert_wrote_nothing(nsls2_repo)

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
