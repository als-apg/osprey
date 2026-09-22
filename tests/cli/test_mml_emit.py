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

The virtual-accelerator lane has its own class. A 2.0 tree gets five more
files, the three the lane owns whole opening with the provenance stamp and the
bindings naming the digest of the deck the same run saved; a 1.0 tree gets one
line saying the lane was skipped, not a single file, and the ring it was
serving removed, because a harvest that describes no machine leaves none to
serve. Its refusals stop the run with nothing written: a deck that was never
imported, a mapping deciding nothing about the exported virtual accelerator, an
answer the deck refuses, a hand-authored starting-state file that carries no
stamp, and a band a person's own ``channel_limits.json`` already states
differently.

Scenario bundles are held against the machine the deployment serves, which is
the document the simulation resolves them against: the one this run is about to
write where it writes one, the one already on the tree where it does not, and
no check at all on a deployment with no machine, which stops at boot for want
of it whatever its scenarios say. A bundle naming a channel that machine does
not carry is refused, and so is one the simulation could not read; a bundle it
can resolve is kept whoever wrote it, and an empty directory is kept.

Four more classes ask of that lane what a file on a served tree is for. Every
2.0 export the repo commits runs the whole lane and re-runs it byte-identically,
discovered from the fixtures rather than listed here, so a tree committed later
is covered the day it lands. Each of the five documents is then read back
through the code that reads it in production -- ``at.load_lattice``,
``load_bindings``, ``parse_machine`` beside ``load_machine_json_channels``,
``load_machine_state_candidate_addresses`` and the limits validator -- and
across documents, because the bindings, the seed and the write bands only
describe one machine if they name the same addresses. A facility's own write
bands survive the lane whole: what it never stamped comes back exactly as it
was, what it stamped is re-derived without losing the keys the lane does not
own, and the addresses of the channel database are not swept into a file the
facility already keeps. And every refusal leaves the tree byte-for-byte as it
found it, measured over the whole tree rather than a list of names.


The DuckDB collapse report is pinned on two repos, because it counts the
bindings the judgments settle rather than the ones the export holds: the
``repo`` fixture keeps its shared PV whole and the report names both owners,
while ``wrapped_repo`` hands that family's shared PV to one device and the PV
leaves the report, the database and the SQL surface together. Both repos are
imported through the command, so the report is measured against the artifacts
of the same run.
"""

from __future__ import annotations

import hashlib
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
from osprey.services.virtual_accelerator.bindings import (
    READOUT_KEYS,
    BindingsDocument,
    load_bindings,
    parse_bindings,
    setpoints,
)
from osprey.services.virtual_accelerator.manifest.loaders import (
    load_machine_json_channels,
    load_machine_state_candidate_addresses,
)
from osprey.services.virtual_accelerator.manifest.paths import ManifestPaths
from osprey.services.virtual_accelerator.model.bindings import build_action_variables
from osprey.simulation.machine import parse_machine
from osprey_connectors.control_system.limits_validator import LimitsValidator
from tests.cli.test_mml_map import _fill
from tests.templates.mml_export_contract import EXPORTER_VERSION

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
    # Everything but the mapping: the decks travel with the export, because
    # the 2.0 siblings are checked against the ring they were sampled over.
    for path in sorted((FIXTURES / "nsls2").iterdir()):
        if path.is_file() and path.name != "mapping.yaml":
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

    @pytest.mark.usefixtures("repo")
    def test_shared_pv_owners_are_counted_from_one(self) -> None:
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

    @pytest.mark.usefixtures("wrapped_repo")
    def test_a_broadcast_row_still_collapses(self) -> None:
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


#: The 2.0 export, whose ``ao.json`` pulls in the ``va`` sibling and the deck it
#: was sampled over -- the one committed tree the virtual-accelerator lane runs on.
SYNTHETIC = FIXTURES / "synthetic" / "quokka.sr.ao.json"

#: The system that export imports as, and the deck named after it.
SYNTHETIC_SYSTEM = "SR"

#: One coupled setpoint of that export, whose family the lane bands from its
#: own ``Range``: the address a facility's own band can disagree on.
SYNTHETIC_BANDED = "QK:QF:1:CUR:SP"

#: Where the lane's artifacts land, relative to the repo root. The deck and the
#: bindings sit under ``simulation/``, which is what the build copies into the
#: served tree; the machine-state view and the write bands are read from
#: ``data/`` itself.
VA_ARTIFACTS = (
    "data/simulation/lattice.json",
    "data/simulation/va_bindings.json",
    "data/simulation/machine.json",
    "data/machine_state_channels.json",
    "data/channel_limits.json",
)

#: The three documents the lane owns whole, which open with the provenance
#: stamp. ``lattice.json`` is deliberately not one of them: it stays a pure pyAT
#: document, and the bindings' ``lattice_sha256`` is its "was this emitted?".
VA_STAMPED = (
    "data/simulation/va_bindings.json",
    "data/simulation/machine.json",
    "data/machine_state_channels.json",
)

#: What one line says on a tree the lane has nothing to do on.
VA_SKIPPED = (
    "VA lane skipped: data/mml/va.json is not in the tree; "
    "re-export with mml_export 2.0 to enable it"
)


def _two_zero_tree(root: Path, fixture: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Import one committed 2.0 fixture export into ``root`` and answer its mapping.

    The mapping is the skeleton ``--init`` writes, every slot filled the way the
    map tests fill one, so the virtual-accelerator block carries the verdicts the
    export proposes and an answer for each of its open slots.
    """
    exports = sorted(fixture.glob("*.ao.json"))
    assert exports, f"{fixture.name} commits no export"
    root.mkdir(parents=True)
    (root / "profile.yml").write_text("name: scratch\n", encoding="utf-8")
    monkeypatch.chdir(root)
    result = CliRunner().invoke(
        cli, ["mml", "import", *(str(path) for path in exports)], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    init = CliRunner().invoke(cli, ["mml", "map", "--init"], catch_exceptions=False)
    assert init.exit_code == 0, init.output
    _write_mapping(root, _fill(_document(root)))
    return root


#: The fixture trees a 2.0 export commits, discovered rather than listed: the
#: virtual accelerator of an export lives in its ``*.va.json`` sibling, so a
#: directory that carries one is a tree this lane can run whole. A tree
#: committed later joins every case below without a name being typed here.
TWO_ZERO_TREES = tuple(
    sorted(
        directory.name
        for directory in FIXTURES.iterdir()
        if directory.is_dir() and any(directory.glob("*.va.json"))
    )
)


@pytest.fixture
def va_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A deployment repo with the 2.0 ``synthetic`` export imported and mapped."""
    return _two_zero_tree(tmp_path / "va", SYNTHETIC.parent, monkeypatch)


@pytest.fixture(params=TWO_ZERO_TREES)
def two_zero_repo(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """One repo per committed 2.0 fixture tree, imported and mapped."""
    return _two_zero_tree(tmp_path / request.param, FIXTURES / request.param, monkeypatch)


def _mark_the_deck(repo: Path, *names: str) -> None:
    """Put a monitor-type element of each name on this tree's imported deck.

    Where a facility's deck marks its girders, ours has drifts, and nothing
    the export states reaches one -- so the elements land there and every
    stated position stays where the export states it.
    """
    import at

    from osprey.services.mml.loaders.mat import load_lattice

    deck = repo / "data" / "mml" / "lattice" / f"{SYNTHETIC_SYSTEM}.mat"
    ring = load_lattice(deck)
    free = [index for index, element in enumerate(ring) if isinstance(element, at.Drift)]
    assert len(free) >= len(names), "the deck holds too few unbound elements"
    for index, name in zip(free, names, strict=False):
        ring[index] = at.Monitor(name)
    at.save_mat(ring, str(deck), mat_key="THERING")


def _take_the_cavity_off_the_deck(repo: Path) -> None:
    """Leave this tree's deck without a cavity, as a real facility exports one.

    The Middle Layer holds the radio frequency at such a facility, so the deck
    it saves has no cavity in it and the mapping is decided again against the
    deck it now is. The cavity is the last element, so no position the export
    states moves.
    """
    import at

    from osprey.services.mml.loaders.mat import load_lattice

    deck = repo / "data" / "mml" / "lattice" / f"{SYNTHETIC_SYSTEM}.mat"
    ring = load_lattice(deck)
    assert isinstance(ring[-1], at.RFCavity), "the synthetic deck no longer ends in its cavity"
    del ring[-1]
    at.save_mat(ring, str(deck), mat_key="THERING")
    again = CliRunner().invoke(cli, ["mml", "map", "--init", "--force"], catch_exceptions=False)
    assert again.exit_code == 0, again.output
    _write_mapping(repo, _fill(_document(repo)))


def _built_cavity(repo: Path) -> dict:
    """The one cavity the emitted deck carries, as the saved document states it."""
    document = json.loads(
        (repo / "data" / "simulation" / "lattice.json").read_text(encoding="utf-8")
    )
    cavities = [
        element
        for element in document["elements"]
        if "RFCavity" in (element.get("__class__"), element.get("Class"))
    ]
    assert len(cavities) == 1, f"the served deck carries {len(cavities)} cavities"
    return cavities[0]


def _leave_the_deck_no_monitor(repo: Path) -> None:
    """Latch the monitor families and give every monitor on the deck one name.

    What a facility hands over when its export could sample no beam-position
    family at all: nothing reads the monitors, they share a name, and the
    marker conversion takes the last one on the deck.
    """
    import at

    from osprey.services.mml.loaders.mat import load_lattice

    document = _document(repo)
    for name in ("BPMx", "BPMy"):
        document["virtual_accelerator"]["families"][name] = {
            "verdict": "latch",
            "reason": "getpvmodel answered for no device of it",
        }
    _write_mapping(repo, document)

    deck = repo / "data" / "mml" / "lattice" / f"{SYNTHETIC_SYSTEM}.mat"
    ring = load_lattice(deck)
    for element in ring:
        if isinstance(element, at.Monitor):
            element.FamName = "BPM"
    at.save_mat(ring, str(deck), mat_key="THERING")


def _va_files(repo: Path) -> dict[str, bytes]:
    """Every virtual-accelerator artifact on the tree, keyed by relative path."""
    return {name: (repo / name).read_bytes() for name in VA_ARTIFACTS if (repo / name).is_file()}


def _assert_wrote_no_va(repo: Path, *, except_for: str | None = None) -> None:
    """No artifact of the lane's is on the tree, bar one the facility wrote itself."""
    for name in VA_ARTIFACTS:
        if name == except_for:
            continue
        assert not (repo / name).is_file(), f"{name} was written"


def _first_key(repo: Path, name: str) -> str:
    return next(iter(json.loads((repo / name).read_text(encoding="utf-8"))))


#: An address no fixture export names, so a scenario asking for it is asking
#: for a channel no emit here serves.
ABSENT_CHANNEL = "SR:RF:CAVITY:01:TEMPERATURE:RB"


def _write_scenario(repo: Path, name: str, channel: str, *, key: str = "archiver") -> Path:
    """Plant one scenario bundle naming *channel*, through ``key``."""
    bundle = repo / "data" / "simulation" / "scenarios" / name
    bundle.mkdir(parents=True)
    path = bundle / "scenario.json"
    spec: dict = {"description": name}
    if key == "archiver":
        spec["archiver"] = [{"channel": channel, "events": []}]
    else:
        spec["overrides"] = {channel: 1.0}
    path.write_text(json.dumps(spec) + "\n", encoding="utf-8")
    return path


def _stale_scenario(repo: Path, name: str) -> Path:
    return _write_scenario(repo, name, ABSENT_CHANNEL)


def _machine_channels(repo: Path) -> list[str]:
    """The channels of the machine on this tree, which a scenario is resolved against."""
    document = json.loads(
        (repo / "data" / "simulation" / "machine.json").read_text(encoding="utf-8")
    )
    return list(document["channels"])


def _database_only_addresses(repo: Path) -> list[str]:
    """Addresses the emitted database carries that the served machine does not.

    The machine seeds one channel per nominal the export states, so it is a
    strict subset of the database -- and the difference is where a scenario can
    name something that reads as a channel of this facility and still has
    nothing on the machine to resolve against.
    """
    from osprey.cli.mml_cmd import _channel_addresses

    database = json.loads(
        (repo / "data" / "channel_databases" / "middle_layer.json").read_text(encoding="utf-8")
    )
    served = set(_machine_channels(repo))
    return [address for address in _channel_addresses(database) if address not in served]


def _emitted_tree() -> None:
    """Emit once, so the documents this tree's scenarios are judged against exist."""
    result = _emit()
    assert result.exit_code == 0, result.output


def _served_scenario(repo: Path, name: str) -> Path:
    """Plant a scenario on a channel the served machine really carries.

    The address is read off the machine of an emit of this very tree, which is
    the document the simulation resolves a scenario against, so the bundle is
    stated against the machine it will meet rather than against a name typed
    here.
    """
    _emitted_tree()
    channels = _machine_channels(repo)
    assert channels, "the emitted machine carries no channel"
    return _write_scenario(repo, name, channels[0])


class TestVirtualAcceleratorLane:
    """``emit`` on a 2.0 tree: the five files a served virtual accelerator boots from."""

    def test_the_va_lane_writes_every_artifact_the_served_tree_reads(self, va_repo: Path) -> None:
        result = _emit()

        assert "Traceback" not in result.output
        assert result.exit_code == 0, result.output
        for name in VA_ARTIFACTS:
            assert (va_repo / name).is_file(), f"{name} was not written"
            assert name.rsplit("/", 1)[-1] in result.output
        assert "osprey build" in result.output.strip().splitlines()[-1]

    def test_the_va_documents_the_lane_owns_open_with_the_provenance_stamp(
        self, va_repo: Path
    ) -> None:
        assert _emit().exit_code == 0

        for name in VA_STAMPED:
            assert _first_key(va_repo, name) == "_provenance", name
        # The deck is a pyAT document, not one of ours to stamp.
        assert "_provenance" not in json.loads(
            (va_repo / "data/simulation/lattice.json").read_text(encoding="utf-8")
        )

    def test_the_va_bindings_name_the_digest_of_the_lattice_this_run_wrote(
        self, va_repo: Path
    ) -> None:
        assert _emit().exit_code == 0

        deck = (va_repo / "data/simulation/lattice.json").read_bytes()
        bindings = json.loads(
            (va_repo / "data/simulation/va_bindings.json").read_text(encoding="utf-8")
        )
        assert bindings["lattice_sha256"] == hashlib.sha256(deck).hexdigest()
        assert bindings["system"] == SYNTHETIC_SYSTEM
        assert bindings["bindings"], "the export couples families and none was bound"

    def test_the_run_says_how_much_of_the_machine_the_model_drives(self, va_repo: Path) -> None:
        """One line for the question a reviewer asks of any install: how much of it?"""
        result = _emit()

        assert result.exit_code == 0, result.output
        block = _document(va_repo)["virtual_accelerator"]["families"]
        driven = sum(1 for family in block.values() if family["verdict"] == "couple")
        assert f"{driven} families driven, {len(block) - driven} standing still." in result.output

    def test_the_run_names_a_conversion_it_cut_back_and_the_span_it_kept(
        self, va_repo: Path
    ) -> None:
        """A conversion narrowed in silence is a conversion nobody checked.

        The exported table is bent back on itself here, which is what a
        facility's own polynomial does out at the edge of a sampled band.
        """
        path = va_repo / "data" / "mml" / "va.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        family = document[SYNTHETIC_SYSTEM]["families"]["QF"]
        devices = len(family["device_list"])
        family["Monitor"]["monitor_inverse"] = {
            "kind": "table",
            "grid": [[-1000.0, 0.0, 1000.0, 500.0]] * devices,
            "values": [[-1.0, 0.0, 1.0, 0.5]] * devices,
        }
        path.write_text(json.dumps(document), encoding="utf-8")

        result = _emit()

        assert result.exit_code == 0, result.output
        assert (
            f"QF monitor_inverse: {devices} device tables kept the stretch around the "
            "nominal, between -1000 and 1000." in result.output
        )

    def test_the_run_names_the_repeated_monitors_it_served_as_markers(self, va_repo: Path) -> None:
        """A facility marks girder ends with the monitor type and one name.

        Nothing reads them, the model refuses a deck that names two monitors
        alike, and the served deck carries them as markers -- which a reviewer
        reads here rather than by diffing two decks.
        """
        _mark_the_deck(va_repo, "GE", "GS", "GE", "GS", "GS")

        result = _emit()

        assert result.exit_code == 0, result.output
        assert (
            "5 monitor-type elements no family reads share a name; "
            "served as plain markers: GE (2), GS (3)." in result.output
        )

    def test_the_run_names_the_cavity_it_built_for_a_deck_that_carries_none(
        self, va_repo: Path
    ) -> None:
        """A deck with no cavity solves at fixed energy until one is built.

        The frequency the cavity is built at is the deck's own and the one the
        export states is the real ring's, so the line states both: they are
        the same number to the figures the facility quotes it to, and the
        difference is what would put the served beam off momentum.
        """
        _take_the_cavity_off_the_deck(va_repo)

        result = _emit()

        assert result.exit_code == 0, result.output
        built = _built_cavity(va_repo)
        assert built["HarmNumber"] == 40
        assert built["Voltage"] == 3.0e6
        assert (
            "RF: the deck holds no cavity, so the served ring carries one on harmonic 40 "
            f"at 3e+06 V, built at the deck's own {built['Frequency']:.9g} Hz where the "
            "export states 516883548 Hz; the served model solves 6D." in result.output
        )

    @pytest.mark.usefixtures("va_repo")
    def test_a_deck_that_carries_its_own_cavity_is_served_no_built_one(self) -> None:
        result = _emit()

        assert result.exit_code == 0, result.output
        assert "the deck holds no cavity" not in result.output

    def test_latching_the_family_serves_no_cavity_and_asks_for_no_voltage(
        self, va_repo: Path
    ) -> None:
        """The reviewer's other way out: stand the family still and answer nothing.

        A latched family drives nothing, so there is no cavity to state a
        voltage for -- and the check has to agree with the run about that,
        or the only way past the question would be to delete it.
        """
        _take_the_cavity_off_the_deck(va_repo)
        document = _document(va_repo)
        family = document["virtual_accelerator"]["families"]["RF"]
        family["values"]["voltage"]["answer"] = None
        family["verdict"] = "latch"
        family["reason"] = "the reviewer serves this ring no cavity"
        _write_mapping(va_repo, document)

        checked = _check()
        result = _emit()

        assert checked.exit_code == 0, checked.output
        assert result.exit_code == 0, result.output
        assert "the deck holds no cavity" not in result.output
        served = json.loads(
            (va_repo / "data" / "simulation" / "lattice.json").read_text(encoding="utf-8")
        )
        assert not [
            element
            for element in served["elements"]
            if "RFCavity" in (element.get("__class__"), element.get("Class"))
        ]

    def test_the_run_says_when_the_served_system_reads_no_position_at_all(
        self, va_repo: Path
    ) -> None:
        """A model with no monitor left measures no orbit and answers no response.

        The count of converted elements does not say it: a reviewer would have
        to know how many monitors the deck held to read that off.
        """
        _leave_the_deck_no_monitor(va_repo)

        result = _emit()

        assert result.exit_code == 0, result.output
        assert (
            "4 monitor-type elements no family reads share a name; "
            "served as plain markers: BPM (4)." in result.output
        )
        assert f"{SYNTHETIC_SYSTEM} is served reading no beam position anywhere." in result.output

    def test_the_run_says_nothing_of_the_kind_while_one_monitor_is_left(
        self, va_repo: Path
    ) -> None:
        _mark_the_deck(va_repo, "GE", "GE")

        result = _emit()

        assert result.exit_code == 0, result.output
        assert "served as plain markers: GE (2)." in result.output
        assert "reads no beam position" not in result.output

    def test_a_rerun_leaves_every_va_artifact_byte_identical(self, va_repo: Path) -> None:
        assert _emit().exit_code == 0
        first = _va_files(va_repo)

        result = _emit()

        assert result.exit_code == 0, result.output
        assert _va_files(va_repo) == first

    def test_a_one_zero_export_skips_the_va_lane_in_one_line(self, repo: Path) -> None:
        result = _emit()

        assert result.exit_code == 0, result.output
        assert VA_SKIPPED in result.output
        _assert_wrote_no_va(repo)

    def test_a_one_zero_export_removes_the_ring_the_tree_was_serving(self, repo: Path) -> None:
        # The harvest answers the channel set, and a ring it did not describe
        # would be served over those new names -- some other machine, addressed
        # as if it were this one. So the deck and the bindings go, and the
        # removal is reported rather than silent.
        simulation = repo / "data" / "simulation"
        simulation.mkdir(parents=True)
        for name in ("lattice.json", "va_bindings.json"):
            (simulation / name).write_text("{}\n", encoding="utf-8")

        result = _emit()

        assert result.exit_code == 0, result.output
        assert "data/simulation/lattice.json" in result.output
        assert "data/simulation/va_bindings.json" in result.output
        assert not (simulation / "lattice.json").exists()
        assert not (simulation / "va_bindings.json").exists()

    def test_a_second_one_zero_emit_has_no_ring_left_to_remove(self, repo: Path) -> None:
        simulation = repo / "data" / "simulation"
        simulation.mkdir(parents=True)
        (simulation / "lattice.json").write_text("{}\n", encoding="utf-8")
        assert _emit().exit_code == 0
        first = _artifacts(repo)

        result = _emit()

        assert result.exit_code == 0, result.output
        assert "Removed" not in result.output
        assert _artifacts(repo) == first

    def test_a_one_zero_emit_judges_scenarios_by_the_machine_already_on_the_tree(
        self, repo: Path
    ) -> None:
        # This run writes no machine, so the one the deployment serves is the
        # one already there -- and a scenario it can resolve is a scenario that
        # boots, whatever the new channel database says.
        machine = repo / "data" / "simulation" / "machine.json"
        machine.parent.mkdir(parents=True)
        machine.write_text(json.dumps({"channels": {ABSENT_CHANNEL: {"value": 1.0}}}) + "\n")
        _write_scenario(repo, "demo-thermal", ABSENT_CHANNEL)

        result = _emit()

        assert result.exit_code == 0, result.output
        assert (repo / "data" / "simulation" / "scenarios" / "demo-thermal").is_dir()

    def test_a_one_zero_emit_refuses_a_scenario_that_machine_cannot_resolve(
        self, repo: Path
    ) -> None:
        machine = repo / "data" / "simulation" / "machine.json"
        machine.parent.mkdir(parents=True)
        machine.write_text(json.dumps({"channels": {}}) + "\n")
        _write_scenario(repo, "demo-thermal", ABSENT_CHANNEL)

        refused = _emit()

        assert refused.exit_code != 0
        assert f"data/simulation/scenarios/demo-thermal names {ABSENT_CHANNEL}" in refused.output
        _assert_wrote_nothing(repo)

    def test_a_deployment_with_no_machine_says_nothing_about_its_scenarios(
        self, repo: Path
    ) -> None:
        # A simulation handed no machine stops for want of it, so there is
        # nothing a scenario could be judged against and nothing to say.
        _write_scenario(repo, "demo-thermal", ABSENT_CHANNEL)

        result = _emit()

        assert result.exit_code == 0, result.output
        assert (repo / "data" / "simulation" / "scenarios" / "demo-thermal").is_dir()

    @pytest.mark.usefixtures("repo")
    def test_a_tree_with_no_ring_of_its_own_says_nothing_about_one(self) -> None:
        result = _emit()

        assert result.exit_code == 0, result.output
        assert "Removed" not in result.output

    def test_a_va_json_for_no_imported_system_is_not_called_absent(self, repo: Path) -> None:
        # The file is in the tree; what it has no block for is a system this
        # deployment imported, which sends a reader to the import rather than
        # back to MATLAB.
        va_path = repo / "data" / "mml" / "va.json"
        va_path.parent.mkdir(parents=True, exist_ok=True)
        va_path.write_text('{"NOSUCH": {}}\n', encoding="utf-8")

        result = _emit()

        assert result.exit_code == 0, result.output
        assert (
            "VA lane skipped: data/mml/va.json carries no virtual accelerator "
            "for an imported system" in result.output
        )
        _assert_wrote_no_va(repo)

    def test_a_mapping_deciding_a_va_the_tree_no_longer_carries_refuses(
        self, va_repo: Path
    ) -> None:
        # The block is written from an export and answered against it, so a
        # mapping that carries one and an export that does not is a tree the
        # served machine has no files to boot from.
        (va_repo / "data" / "mml" / "va.json").unlink()

        result = _emit()

        assert result.exit_code != 0
        assert "Traceback" not in result.output
        assert "va.json" in result.output
        _assert_wrote_nothing(va_repo)
        _assert_wrote_no_va(va_repo)

    def test_a_deployment_serving_its_own_va_still_emits_a_one_zero_tree(self, repo: Path) -> None:
        # A preset's demo ring is served from its own deck, not the export's,
        # so the profile's block says nothing about this lane.
        (repo / "profile.yml").write_text(
            "name: scratch\nvirtual_accelerator:\n  port: 5064\n", encoding="utf-8"
        )

        result = _emit()

        assert result.exit_code == 0, result.output
        assert VA_SKIPPED in result.output
        _assert_wrote_no_va(repo)

    def test_a_deck_outside_the_tree_refuses_the_va_run(self, va_repo: Path) -> None:
        (va_repo / "data" / "mml" / "lattice" / f"{SYNTHETIC_SYSTEM}.mat").unlink()

        result = _emit()

        assert result.exit_code != 0
        assert "Traceback" not in result.output
        assert f"data/mml/lattice/{SYNTHETIC_SYSTEM}.mat" in result.output
        _assert_wrote_nothing(va_repo)
        _assert_wrote_no_va(va_repo)

    def test_a_mapping_without_a_block_refuses_the_va_export(self, va_repo: Path) -> None:
        document = _document(va_repo)
        del document["virtual_accelerator"]
        _write_mapping(va_repo, document)

        result = _emit()

        assert result.exit_code != 0
        assert "map --init" in result.output
        _assert_wrote_nothing(va_repo)
        _assert_wrote_no_va(va_repo)

    def test_an_unanswered_va_slot_stops_the_emit_before_any_write(self, va_repo: Path) -> None:
        document = _document(va_repo)
        document["virtual_accelerator"]["families"]["SEPTUM"]["slot"]["answer"] = None
        _write_mapping(va_repo, document)

        result = _emit()

        assert result.exit_code != 0
        assert _problem_lines(result.output, "virtual_accelerator.families.SEPTUM.slot.answer")
        _assert_wrote_nothing(va_repo)
        _assert_wrote_no_va(va_repo)

    def test_an_answer_the_deck_refuses_stops_the_va_emit(self, va_repo: Path) -> None:
        # SEPTUM's element is a drift, which carries no multipole at all: the
        # answer is refused against the deck, exactly as ``map --check`` refuses it.
        document = _document(va_repo)
        document["virtual_accelerator"]["families"]["SEPTUM"]["slot"]["answer"] = (
            "strength:PolynomB[9]"
        )
        _write_mapping(va_repo, document)

        result = _emit()

        assert result.exit_code != 0
        assert _problem_lines(result.output, "virtual_accelerator.families.SEPTUM.slot.answer")
        assert "takes no PolynomB[9]" in result.output
        _assert_wrote_nothing(va_repo)
        _assert_wrote_no_va(va_repo)

    def test_an_unstamped_machine_json_refuses_the_va_run_with_one_rm_line(
        self, va_repo: Path
    ) -> None:
        hand_written = va_repo / "data" / "simulation" / "machine.json"
        hand_written.parent.mkdir(parents=True, exist_ok=True)
        hand_written.write_text('{"channels": {}}\n', encoding="utf-8")

        result = _emit()

        assert result.exit_code != 0
        assert "Traceback" not in result.output
        assert _rm_lines(result.output) == ["rm data/simulation/machine.json"]
        assert hand_written.read_text(encoding="utf-8") == '{"channels": {}}\n'
        _assert_wrote_nothing(va_repo)

    def test_a_scenario_naming_a_channel_the_emit_does_not_serve_is_refused(
        self, va_repo: Path
    ) -> None:
        # A scenario is stated against a machine, and emit answers which
        # channels the machine has: one asking for a channel this run does not
        # write stops the simulation at boot, so it is refused by name, with
        # the channel it asks for, and the bundle beside it whose channels the
        # emit does serve is left alone.
        served = _served_scenario(va_repo, "orbit-step")
        stale = _stale_scenario(va_repo, "rf-thermal")

        refused = _emit()

        assert refused.exit_code != 0
        assert "Traceback" not in refused.output
        assert f"data/simulation/scenarios/rf-thermal names {ABSENT_CHANNEL}" in refused.output
        assert _rm_lines(refused.output) == ["rm -r data/simulation/scenarios/rf-thermal"]
        assert stale.is_file()
        assert served.is_file()

        shutil.rmtree(stale.parent)
        result = _emit()

        assert result.exit_code == 0, result.output
        assert served.is_file()

    def test_an_empty_scenario_directory_is_kept(self, va_repo: Path) -> None:
        # Nothing in it is stated against anything, so there is nothing for the
        # emitted channel set to have gone stale against.
        scenarios = va_repo / "data" / "simulation" / "scenarios"
        scenarios.mkdir(parents=True)

        result = _emit()

        assert result.exit_code == 0, result.output
        assert scenarios.is_dir()

    def test_a_scenario_over_a_channel_only_the_database_has_is_refused(
        self, va_repo: Path
    ) -> None:
        # The machine seeds one channel per nominal the export states, so the
        # database is the wider document of the two. A scenario is resolved
        # against the machine, so an address that is in the database and not on
        # the machine -- a readback, typically -- is one the simulation would
        # stop on however well it reads.
        _emitted_tree()
        absent = _database_only_addresses(va_repo)
        assert absent, "this export seeds every address the database carries"
        _write_scenario(va_repo, "readback-probe", absent[0], key="overrides")

        refused = _emit()

        assert refused.exit_code != 0
        assert "Traceback" not in refused.output
        assert f"data/simulation/scenarios/readback-probe names {absent[0]}" in refused.output

    def test_a_bundle_the_simulation_could_not_read_is_refused_by_name(self, va_repo: Path) -> None:
        # The simulation stops at boot on a directory under scenarios/ with no
        # scenario.json in it, so emit stops on the tree it can already see it
        # in rather than leaving it to a dead container.
        bundle = va_repo / "data" / "simulation" / "scenarios" / "no-json"
        bundle.mkdir(parents=True)

        refused = _emit()

        assert refused.exit_code != 0
        assert "Traceback" not in refused.output
        assert "data/simulation/scenarios/no-json has no scenario.json" in refused.output
        assert _rm_lines(refused.output) == ["rm -r data/simulation/scenarios/no-json"]

    def test_a_refused_scenario_stops_the_emit_before_any_write(self, va_repo: Path) -> None:
        # The channel set a scenario is held against is built before the
        # refusal and written after it, so a tree refused over a scenario is
        # the tree the command was handed.
        _stale_scenario(va_repo, "rf-thermal")

        refused = _emit()

        assert refused.exit_code != 0
        _assert_wrote_nothing(va_repo)
        _assert_wrote_no_va(va_repo)

    def test_a_hand_banded_setpoint_refuses_the_va_lane_and_names_the_address(
        self, va_repo: Path
    ) -> None:
        # The band the facility wrote itself disagrees with the one this
        # setpoint's family is banded from, and is the last thing the lane can
        # learn: every text is rendered before it is known. So it is the one
        # refusal that proves the rest -- the deck included -- reaches the tree
        # only once the whole lane holds, and reaches it again on no re-run.
        theirs = {"min_value": -1.5, "max_value": 1.5, "writable": True}
        limits = va_repo / "data" / "channel_limits.json"
        limits.parent.mkdir(parents=True, exist_ok=True)
        limits.write_text(json.dumps({SYNTHETIC_BANDED: theirs}, indent=2) + "\n", encoding="utf-8")
        facility = limits.read_bytes()

        result = _emit()

        assert result.exit_code != 0
        assert "Traceback" not in result.output
        assert SYNTHETIC_BANDED in result.output
        assert limits.read_bytes() == facility
        _assert_wrote_no_va(va_repo, except_for="data/channel_limits.json")

        again = _emit()

        assert again.exit_code != 0
        assert limits.read_bytes() == facility
        _assert_wrote_no_va(va_repo, except_for="data/channel_limits.json")


#: The address of the one entry below planted by a facility that this export
#: knows nothing about: it is not in the channel database, so a lane that swept
#: the database into a file it does not own would still leave it alone, while a
#: lane that rewrote the file whole would drop it.
FACILITY_ONLY = "ZZ:OTHER:1:CUR:SP"


def _json(repo: Path, name: str) -> dict:
    return json.loads((repo / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stamp(repo: Path) -> str:
    """The provenance one emit of this tree renders, computed from its inputs."""
    return (
        f"exporter={EXPORTER_VERSION} "
        f"ao_sha256={_sha256(repo / 'data' / 'mml' / 'ao.json')} "
        f"mapping_sha256={_sha256(repo / 'data' / 'mml' / 'mapping.yaml')}"
    )


def _tree(repo: Path) -> dict[str, bytes]:
    """Every file on the deployment, keyed by relative path."""
    return {
        path.relative_to(repo).as_posix(): path.read_bytes()
        for path in sorted(repo.rglob("*"))
        if path.is_file()
    }


def _bindings(repo: Path):
    return load_bindings(repo / "data" / "simulation" / "va_bindings.json")


def _without_readout(repo: Path) -> dict:
    """The emitted bindings with every readout block taken back out."""
    body = _json(repo, "data/simulation/va_bindings.json")
    body["bindings"] = [
        {key: value for key, value in binding.items() if key != "readout"}
        for binding in body["bindings"]
    ]
    return body


class _Solved:
    """A solved orbit for the tree's monitors, keyed by element name.

    Each monitor sits at a place of its own, and its two planes differ, so a
    reading that came from the wrong element or the wrong axis is a different
    number rather than the same one.
    """

    def __init__(self, document: BindingsDocument) -> None:
        elements = sorted(
            {binding.element for binding in document.bindings if binding.kind == "monitor"}
        )
        self.last_solution = {
            name: (1.0e-4 * (position + 1), -2.0e-4 * (position + 1))
            for position, name in enumerate(elements)
        }


def _served_readings(document: BindingsDocument, orbit: _Solved) -> dict[str, float]:
    """What the IOC would publish for every monitor the document binds.

    Built through the model layer's own factories, so this is the served
    reading rather than an arithmetic stand-in for one.
    """
    factories = build_action_variables(document)
    return {
        binding.setpoint_address: factories[binding.setpoint_address](
            {"address": binding.setpoint_address},
            name=binding.setpoint_address,
            read_only=True,
            default_validation_config="none",
            default_value=0.0,
            value_range=None,
            unit="mm",
        )._get(orbit)
        for binding in document.bindings
        if binding.kind == "monitor"
    }


def _again(document: BindingsDocument, address: str, reading: float) -> float:
    """The reading a consumer would publish having applied the readout twice.

    ``real = gain * (raw - offset)`` is the facility's own algebra, offset
    first; a number it states nothing for corrects by nothing.
    """
    binding = next(entry for entry in document.bindings if entry.setpoint_address == address)
    readout = binding.readout
    gain = 1.0 if readout is None or readout.gain is None else readout.gain
    offset = 0.0 if readout is None or readout.offset is None else readout.offset
    return gain * (reading - offset)


class TestEveryCommittedTwoZeroTree:
    """The whole lane, on every 2.0 export the repo commits."""

    def test_at_least_one_two_zero_export_is_committed(self) -> None:
        # Every other case in this class is parametrised over that discovery,
        # so an empty one would pass the file in silence rather than fail it.
        assert TWO_ZERO_TREES, "no fixture export carries a *.va.json sibling"

    def test_the_lane_writes_the_five_artifacts_a_served_tree_boots_from(
        self, two_zero_repo: Path
    ) -> None:
        result = _emit()

        assert "Traceback" not in result.output
        assert result.exit_code == 0, result.output
        assert sorted(_va_files(two_zero_repo)) == sorted(VA_ARTIFACTS)

    def test_a_second_emit_leaves_every_artifact_byte_identical(self, two_zero_repo: Path) -> None:
        assert _emit().exit_code == 0
        first = _tree(two_zero_repo)

        result = _emit()

        assert result.exit_code == 0, result.output
        # The whole tree, not just the lane's five: an emit of unchanged inputs
        # is what a deployment re-runs, and a byte that moves is a diff a
        # facility has to read.
        assert _tree(two_zero_repo) == first

    def test_the_stamp_names_the_exporter_the_frozen_contract_spells(
        self, two_zero_repo: Path
    ) -> None:
        assert _emit().exit_code == 0

        for name in VA_STAMPED:
            assert _json(two_zero_repo, name)["_provenance"] == _stamp(two_zero_repo)


class TestEachDocumentThroughItsOwnReader:
    """Every file the lane writes, read back by the code that reads it in production."""

    def test_the_deck_loads_through_pyat_and_holds_the_elements_the_bindings_name(
        self, two_zero_repo: Path
    ) -> None:
        import at

        assert _emit().exit_code == 0

        ring = at.load_lattice(two_zero_repo / "data" / "simulation" / "lattice.json")
        document = _bindings(two_zero_repo)
        assert len(ring) > 0
        assert ring.energy == pytest.approx(document.energy_gev * 1e9)
        names = {element.FamName for element in ring}
        bound = {binding.element for binding in document.bindings if binding.element}
        bound |= {part.element for binding in document.bindings for part in binding.slices}
        assert bound, "no binding names an element of the deck"
        assert bound <= names, sorted(bound - names)

    def test_the_bindings_load_through_their_reader_and_name_this_runs_deck(
        self, two_zero_repo: Path
    ) -> None:
        assert _emit().exit_code == 0

        document = _bindings(two_zero_repo)
        deck = (two_zero_repo / "data" / "simulation" / "lattice.json").read_bytes()
        assert document.lattice_sha256 == hashlib.sha256(deck).hexdigest()
        assert document.provenance == _stamp(two_zero_repo)
        assert setpoints(document), "the export couples families and none is writable"

    def test_the_starting_state_parses_through_the_simulation_reader(
        self, two_zero_repo: Path
    ) -> None:
        assert _emit().exit_code == 0

        path = two_zero_repo / "data" / "simulation" / "machine.json"
        model = parse_machine(json.loads(path.read_text(encoding="utf-8")), path)
        channels = load_machine_json_channels(path)
        assert set(channels) == set(model.channels)
        assert "_provenance" not in channels
        assert "nominal" in model.scenarios
        # The one cross-document claim that makes it a starting state: every
        # address the served machine can be written on has a value to start at.
        assert set(setpoints(_bindings(two_zero_repo))) <= set(channels)

    def test_the_machine_state_view_loads_through_the_manifest_reader(
        self, two_zero_repo: Path
    ) -> None:
        assert _emit().exit_code == 0

        paths = ManifestPaths(two_zero_repo / "data")
        candidates = load_machine_state_candidate_addresses(paths)
        document = json.loads(paths.machine_state_channels.read_text(encoding="utf-8"))
        assert candidates == [key for key in document if not key.startswith("_")]
        assert candidates, "the export has monitor-only families and the view names none"
        assert all(
            document[address]["label"] and document[address]["group"] for address in candidates
        )

    def test_the_write_bands_load_through_the_limits_validator(self, two_zero_repo: Path) -> None:
        assert _emit().exit_code == 0

        limits = two_zero_repo / "data" / "channel_limits.json"
        database, raw = LimitsValidator._load_limits_database(str(limits))
        assert set(database) == set(raw)
        # Exactly the addresses the bindings drive may be written, and the file
        # this lane created states every one of its entries itself.
        assert LimitsValidator.writable_addresses(limits) == frozenset(
            setpoints(_bindings(two_zero_repo))
        )
        assert all(entry["_provenance"] == _stamp(two_zero_repo) for entry in raw.values())
        # A blank device slot stays blank in the channel database rather than
        # compacting the list; a blank is not an address and never gets a band.
        assert all(address.strip() for address in raw)

    def test_the_write_bands_carry_no_stamp_of_their_own_above_the_entries(
        self, two_zero_repo: Path
    ) -> None:
        # The file is shared with the facility, so the lane states each entry it
        # owns and never the document.
        assert _emit().exit_code == 0

        document = _json(two_zero_repo, "data/channel_limits.json")
        assert [key for key in document if key.startswith("_")] == []

    def test_the_state_view_can_name_a_monitor_the_starting_state_holds_no_value_for(
        self, va_repo: Path
    ) -> None:
        # Pinned as it stands rather than asserted as a rule: the machine-state
        # view lists the export's monitor-only families, while the starting
        # state seeds the channels the export carries a nominal for, and SEPTUM
        # is refused beside its facts -- it reaches the view without a value.
        # The build-time manifest reconciles the two and publishes the split
        # under ``_metadata.machine_state_reconciliation`` (manifest/build.py),
        # which is where an address outside the served set is meant to show up.
        assert _emit().exit_code == 0

        view = _json(va_repo, "data/machine_state_channels.json")
        channels = load_machine_json_channels(va_repo / "data" / "simulation" / "machine.json")
        named = [key for key in view if not key.startswith("_")]
        assert "QK:SEPTUM:1:CUR:RB" in named
        assert "QK:SEPTUM:1:CUR:RB" not in channels


class TestTheMonitorReadoutReachesTheBindingsAndChangesNothing:
    """The export's per-device reading calibration, carried and not applied.

    The numbers are the facility's own -- what it already applied before it
    published the reading -- and they are carried so that a readout-error
    model has something to perturb around. Nothing serves from them today,
    and their gain and offset are inside the conversions beside them, so a
    served reading has to be the same byte whether they are there or not.
    """

    def _monitors(self, repo: Path) -> dict[str, list]:
        """Every monitor binding of the tree, in document order, by family."""
        found: dict[str, list] = {}
        for binding in _bindings(repo).bindings:
            if binding.kind == "monitor":
                found.setdefault(binding.family, []).append(binding)
        return found

    def _exported(self, repo: Path) -> dict[str, dict]:
        """Each family's exported readout block, by family."""
        document = _json(repo, "data/mml/va.json")[SYNTHETIC_SYSTEM]["families"]
        return {
            family: body["Monitor"]["readout"]
            for family, body in document.items()
            if isinstance(body.get("Monitor"), dict) and "readout" in body["Monitor"]
        }

    def test_nothing_but_a_reading_carries_one(self, two_zero_repo: Path) -> None:
        # The rule as it stands on every committed export, whose readouts are
        # all on families that publish a reading. What pins the emitter's
        # choice is a driven family that states one, and that case is built
        # against ``emit_bindings`` in tests/services/mml/test_emit_va.py.
        assert _emit().exit_code == 0

        assert all(
            binding.readout is None
            for binding in _bindings(two_zero_repo).bindings
            if binding.kind != "monitor"
        )

    def test_every_device_of_a_calibrated_family_carries_its_own_numbers(
        self, va_repo: Path
    ) -> None:
        assert _emit().exit_code == 0

        exported = self._exported(va_repo)
        assert exported, "the export calibrates no reading"
        monitors = self._monitors(va_repo)
        for family, readout in exported.items():
            bound = monitors[family]
            for key, column in readout.items():
                assert [getattr(binding.readout, key) for binding in bound] == column

    def test_a_correction_the_export_never_states_is_absent_not_neutral(
        self, va_repo: Path
    ) -> None:
        assert _emit().exit_code == 0

        exported = self._exported(va_repo)
        monitors = self._monitors(va_repo)
        for family, readout in exported.items():
            for binding in monitors[family]:
                assert binding.readout.stated == tuple(
                    key for key in READOUT_KEYS if key in readout
                )
        # The rule above is only worth stating because a family exercises it:
        # one of them leaves a correction unstated, and it stays unstated.
        assert any(set(readout) != set(READOUT_KEYS) for readout in exported.values())

    def test_the_served_reading_is_the_same_with_the_readout_and_without_it(
        self, va_repo: Path
    ) -> None:
        """The whole of what the carried numbers are allowed to do today.

        Both documents are built by the production reader from the tree this
        run emitted, and read at one solved orbit, so what is compared is the
        reading the IOC would publish.
        """
        assert _emit().exit_code == 0

        document = _bindings(va_repo)
        stripped = parse_bindings(_without_readout(va_repo))
        orbit = _Solved(_bindings(va_repo))
        carried = _served_readings(document, orbit)
        assert carried, "the tree binds no monitor"
        assert carried == _served_readings(stripped, orbit)

    def test_the_stripped_document_differs_in_nothing_but_the_readout(self, va_repo: Path) -> None:
        # The comparison above is only an identity of readings; this says the
        # readout is additive, so no other served fact moved with it.
        assert _emit().exit_code == 0

        body = _json(va_repo, "data/simulation/va_bindings.json")
        stripped = _without_readout(va_repo)
        assert body != stripped
        for carried, bare in zip(body["bindings"], stripped["bindings"], strict=True):
            assert {key: value for key, value in carried.items() if key != "readout"} == bare

    def test_applying_the_carried_gain_and_offset_again_would_move_the_reading(
        self, va_repo: Path
    ) -> None:
        """The guard that makes the identity above worth asserting.

        A facility whose corrections all happened to change nothing would pass
        it while a consumer double-counted them in silence.
        """
        assert _emit().exit_code == 0

        document = _bindings(va_repo)
        orbit = _Solved(document)
        served = _served_readings(document, orbit)
        moved = [
            address
            for address, reading in served.items()
            if _again(document, address, reading) != pytest.approx(reading, rel=1e-9)
        ]
        assert moved, "no carried correction would change a reading if applied twice"


class TestTheFacilitysOwnWriteBandsSurviveTheLane:
    """``channel_limits.json`` is shared, so the lane edits its own entries only."""

    def _plant(self, repo: Path, document: dict) -> Path:
        limits = repo / "data" / "channel_limits.json"
        limits.parent.mkdir(parents=True, exist_ok=True)
        limits.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        return limits

    def test_everything_the_lane_does_not_own_comes_back_exactly_as_it_was(
        self, va_repo: Path
    ) -> None:
        theirs = {
            "_comment": "written by the facility, not by osprey",
            "defaults": {"writable": True, "confirm": False},
            FACILITY_ONLY: {"min_value": -3.0, "max_value": 3.0, "confirm": True},
        }
        limits = self._plant(va_repo, theirs)

        result = _emit()

        assert result.exit_code == 0, result.output
        merged = json.loads(limits.read_text(encoding="utf-8"))
        assert {key: merged[key] for key in theirs} == theirs
        # Metadata first, then ``defaults``, then the addresses sorted.
        assert list(merged)[:2] == ["_comment", "defaults"]
        addresses = [key for key in merged if not key.startswith("_") and key != "defaults"]
        assert addresses == sorted(addresses)
        # Only the addresses the bindings name are added: sweeping the channel
        # database into a facility's file would newly block writes it allows.
        document = _bindings(va_repo)
        owned = {binding.setpoint_address for binding in document.bindings}
        owned |= {
            binding.readback_address for binding in document.bindings if binding.readback_address
        }
        assert set(addresses) - {FACILITY_ONLY} <= owned
        assert set(setpoints(document)) <= set(addresses)

    def test_a_band_this_lane_stamped_before_is_re_derived_and_keeps_the_rest(
        self, va_repo: Path
    ) -> None:
        assert _emit().exit_code == 0
        band = _json(va_repo, "data/channel_limits.json")[SYNTHETIC_BANDED]

        stale = dict(band, min_value=-99.0, max_value=99.0, confirm=True, max_step=0.5)
        stale["_provenance"] = "exporter=mml_export 2.0.0 ao_sha256=old mapping_sha256=old"
        self._plant(va_repo, {SYNTHETIC_BANDED: stale})
        result = _emit()

        assert result.exit_code == 0, result.output
        again = _json(va_repo, "data/channel_limits.json")[SYNTHETIC_BANDED]
        assert again["min_value"] == band["min_value"]
        assert again["max_value"] == band["max_value"]
        assert again["_provenance"] == _stamp(va_repo)
        # The keys the lane does not own survive the re-derivation.
        assert again["confirm"] is True
        assert again["max_step"] == 0.5

    def test_an_unstamped_band_that_agrees_is_left_untouched_and_unstamped(
        self, va_repo: Path
    ) -> None:
        assert _emit().exit_code == 0
        band = _json(va_repo, "data/channel_limits.json")[SYNTHETIC_BANDED]
        theirs = {key: value for key, value in band.items() if key != "_provenance"}

        self._plant(va_repo, {SYNTHETIC_BANDED: theirs})
        result = _emit()

        assert result.exit_code == 0, result.output
        # Nothing to refuse and nothing to correct, so the entry is copied
        # across as it stands -- a stamp here would claim a band the lane did
        # not write.
        assert _json(va_repo, "data/channel_limits.json")[SYNTHETIC_BANDED] == theirs

    def test_a_merged_file_re_emits_byte_identically(self, va_repo: Path) -> None:
        limits = self._plant(
            va_repo,
            {
                "_comment": "written by the facility, not by osprey",
                FACILITY_ONLY: {"min_value": -3.0, "max_value": 3.0},
            },
        )
        assert _emit().exit_code == 0
        first = limits.read_bytes()

        result = _emit()

        assert result.exit_code == 0, result.output
        assert limits.read_bytes() == first


def _unanswer_a_slot(repo: Path) -> None:
    document = _document(repo)
    document["virtual_accelerator"]["families"]["SEPTUM"]["slot"]["answer"] = None
    _write_mapping(repo, document)


def _answer_what_the_deck_refuses(repo: Path) -> None:
    document = _document(repo)
    document["virtual_accelerator"]["families"]["SEPTUM"]["slot"]["answer"] = "strength:PolynomB[9]"
    _write_mapping(repo, document)


def _drop_the_deck(repo: Path) -> None:
    (repo / "data" / "mml" / "lattice" / f"{SYNTHETIC_SYSTEM}.mat").unlink()


def _drop_the_exported_va(repo: Path) -> None:
    (repo / "data" / "mml" / "va.json").unlink()


def _plant_an_unstamped_starting_state(repo: Path) -> None:
    hand_written = repo / "data" / "simulation" / "machine.json"
    hand_written.parent.mkdir(parents=True, exist_ok=True)
    hand_written.write_text('{"channels": {}}\n', encoding="utf-8")


def _plant_a_band_of_their_own(repo: Path) -> None:
    limits = repo / "data" / "channel_limits.json"
    limits.parent.mkdir(parents=True, exist_ok=True)
    limits.write_text(
        json.dumps({SYNTHETIC_BANDED: {"min_value": -1.5, "max_value": 1.5, "writable": True}})
        + "\n",
        encoding="utf-8",
    )


class TestNothingReachesTheTreeUntilEveryRefusalIsKnown:
    """What a refused emit leaves behind, measured over the whole deployment."""

    @pytest.mark.parametrize(
        "break_it",
        [
            _unanswer_a_slot,
            _answer_what_the_deck_refuses,
            _drop_the_deck,
            _drop_the_exported_va,
            _plant_an_unstamped_starting_state,
        ],
        ids=[
            "unanswered-slot",
            "answer-the-deck-refuses",
            "deck-never-imported",
            "export-without-a-virtual-accelerator",
            "unstamped-starting-state",
        ],
    )
    def test_a_pre_flight_refusal_writes_nothing_at_all(self, va_repo: Path, break_it) -> None:
        break_it(va_repo)
        before = _tree(va_repo)

        result = _emit()

        assert result.exit_code != 0
        assert "Traceback" not in result.output
        # The whole tree, not a list of names: every one of these is decided
        # before the first lane writes, so nothing of any lane is on the tree.
        assert _tree(va_repo) == before

    def test_a_band_refusal_keeps_the_va_off_the_tree_but_not_the_lanes_before_it(
        self, va_repo: Path
    ) -> None:
        # A band a facility states itself is the one refusal that cannot be
        # reached in the pre-flight: it is known only once all five VA
        # documents are rendered, and by then the channel database, the
        # ontology, the knowledge pages and the corpus of the same run are on
        # the tree. So the VA lane withholds its own five files and says
        # exactly that, and what the lanes before it wrote stays written.
        _plant_a_band_of_their_own(va_repo)
        before = _tree(va_repo)

        result = _emit()

        assert result.exit_code != 0
        assert "Traceback" not in result.output
        _assert_wrote_no_va(va_repo, except_for="data/channel_limits.json")
        written = set(_tree(va_repo)) - set(before)
        assert written, "the lanes before the VA lane wrote nothing at all"
        assert not written & set(VA_ARTIFACTS)
        assert "no virtual-accelerator file was written" in result.output

    def test_a_second_refused_run_leaves_the_tree_the_first_one_left(self, va_repo: Path) -> None:
        _plant_a_band_of_their_own(va_repo)
        assert _emit().exit_code != 0
        before = _tree(va_repo)

        result = _emit()

        assert result.exit_code != 0
        assert _tree(va_repo) == before


class TestTheProvenanceOfOneEmitRun:
    """One run, one stamp: the inputs it read and the deck it saved."""

    def test_one_stamp_names_the_inputs_of_the_run_that_wrote_them(self, va_repo: Path) -> None:
        assert _emit().exit_code == 0

        stamps = {_json(va_repo, name)["_provenance"] for name in VA_STAMPED}
        stamps |= {
            entry["_provenance"] for entry in _json(va_repo, "data/channel_limits.json").values()
        }
        assert stamps == {_stamp(va_repo)}

    def test_a_mapping_the_run_reads_differently_restamps_every_document(
        self, va_repo: Path
    ) -> None:
        assert _emit().exit_code == 0
        first = _stamp(va_repo)
        deck = (va_repo / "data" / "simulation" / "lattice.json").read_bytes()

        mapping = va_repo / "data" / "mml" / "mapping.yaml"
        mapping.write_text(
            mapping.read_text(encoding="utf-8") + "\n# a note the facility keeps here\n",
            encoding="utf-8",
        )
        result = _emit()

        assert result.exit_code == 0, result.output
        second = _stamp(va_repo)
        assert second != first
        assert _json(va_repo, "data/simulation/machine.json")["_provenance"] == second
        # The deck did not change, so neither does the digest the bindings
        # carry: it is recorded off the text this run rendered, not off the
        # stamp it is written beside.
        assert (va_repo / "data" / "simulation" / "lattice.json").read_bytes() == deck
        assert _bindings(va_repo).lattice_sha256 == hashlib.sha256(deck).hexdigest()
