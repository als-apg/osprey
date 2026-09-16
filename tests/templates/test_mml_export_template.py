"""The shipped MATLAB Middle Layer exporter and its README.

No MATLAB runs here. The script is held to its written contract instead: it is
pullable into a deployment, it encodes with the options that keep non-finite
values out of ``null``, it writes the paired file names and ``_export`` keys the
importer reads, and a document in exactly that dialect imports with no flags.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from click.testing import CliRunner
from tests.cli.test_scaffold_ci import named_commands, unresolvable

from osprey.cli.main import cli
from osprey.services.mml.loaders.json_any import load_json
from osprey.services.mml.normalize import normalize_family
from osprey.services.mml.systems import resolve_system

REPO_ROOT = Path(__file__).resolve().parents[2]
MML_DIR = REPO_ROOT / "src" / "osprey" / "templates" / "apps" / "control_assistant" / "data" / "mml"
EXPORTER = MML_DIR / "mml_export.m"
README = MML_DIR / "README.md"
PAIRED_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mml" / "paired" / "quokka.ring.ao.json"

PULL_LINE = "osprey scaffold pull control-assistant:data/mml/mml_export.m"


@pytest.fixture
def exporter_source() -> str:
    return EXPORTER.read_text(encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A deployment repo: a ``profile.yml`` marker and nothing else."""
    root = tmp_path / "deployment"
    root.mkdir()
    (root / "profile.yml").write_text("", encoding="utf-8")
    return root


def _pull(repo: Path, target: str):
    return CliRunner().invoke(cli, ["scaffold", "pull", target, "--repo", str(repo)])


# ---------------------------------------------------------------------------
# Pulling
# ---------------------------------------------------------------------------


def test_pulling_the_directory_lands_both_files(repo: Path) -> None:
    result = _pull(repo, "control-assistant:data/mml")

    assert result.exit_code == 0, result.output
    for source in (EXPORTER, README):
        landed = repo / "data" / "mml" / source.name
        assert landed.is_file(), result.output
        assert landed.read_bytes() == source.read_bytes()


def test_pulling_the_script_alone_lands_only_the_script(repo: Path) -> None:
    result = _pull(repo, "control-assistant:data/mml/mml_export.m")

    assert result.exit_code == 0, result.output
    assert (repo / "data" / "mml" / "mml_export.m").read_bytes() == EXPORTER.read_bytes()
    assert not (repo / "data" / "mml" / "README.md").exists()


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------


def test_readme_names_the_pull_line_and_it_resolves() -> None:
    text = README.read_text(encoding="utf-8")

    assert PULL_LINE in text
    scaffold_chains = {chain for chain in named_commands(text) if chain[:1] == ("scaffold",)}
    assert scaffold_chains
    assert [unresolvable(chain) for chain in scaffold_chains if unresolvable(chain)] == []


def test_readme_states_the_one_command_usage_and_the_file_pair() -> None:
    text = README.read_text(encoding="utf-8")

    assert "```matlab\nmml_export\n```" in text
    assert "<machine>.<submachine>.ao.json" in text
    assert "<machine>.<submachine>.ad.json" in text
    assert "osprey mml import" in text


# ---------------------------------------------------------------------------
# Script contract
# ---------------------------------------------------------------------------


def test_script_reads_the_live_ao_and_ad(exporter_source: str) -> None:
    assert re.search(r"^function\s+files\s*=\s*mml_export\(", exporter_source, re.M)
    assert re.search(r"\bAO\s*=\s*getao\b", exporter_source)
    assert re.search(r"\bAD\s*=\s*getad\b", exporter_source)


def test_every_jsonencode_call_keeps_non_finite_out_of_null(exporter_source: str) -> None:
    calls = re.findall(r"jsonencode\(([^;]*)\)", exporter_source)

    assert calls, "the script must encode with jsonencode"
    for call in calls:
        assert re.search(r"'ConvertInfAndNaN'\s*,\s*false", call), call


def test_script_applies_the_normalisation_rules(exporter_source: str) -> None:
    assert "func2str(" in exporter_source
    assert "deblank(" in exporter_source
    assert "'$fn'" in exporter_source
    for spelling in ("'Inf'", "'-Inf'", "'NaN'"):
        assert spelling in exporter_source
    assert "'Handles'" in exporter_source


def test_script_writes_the_paired_file_names(exporter_source: str) -> None:
    assert "'.ao.json'" in exporter_source
    assert "'.ad.json'" in exporter_source


def test_export_block_names_version_matlab_machine_submachine_timestamp(
    exporter_source: str,
) -> None:
    block = re.search(r"export\s*=\s*struct\((.*?)\);", exporter_source, re.S)

    assert block is not None
    keys = re.findall(r"'(\w+)'\s*,", block.group(1))
    assert keys == ["exporter", "matlab", "machine", "submachine", "timestamp"]
    assert '"_export"' in exporter_source


def test_exporter_version_matches_the_paired_fixture(exporter_source: str) -> None:
    """The committed paired fixture is the shipped exporter's form."""
    version = re.search(r"EXPORTER_VERSION\s*=\s*'([^']+)'", exporter_source)
    fixture = json.loads(PAIRED_FIXTURE.read_text(encoding="utf-8"))

    assert version is not None
    assert version.group(1) == fixture["_export"]["exporter"]


# ---------------------------------------------------------------------------
# Dialect: what the script writes imports with no flags
# ---------------------------------------------------------------------------

#: One family spelled exactly as ``mml_export.m`` encodes it: a handle as
#: ``{"$fn", "file"}``, an N-row matrix with non-finite entries as rows of mixed
#: numbers and strings, char-matrix rows deblanked, logicals already 0/1.
_EXPORTED_AO = """{"_export":{"exporter":"mml_export 1.0.0","matlab":"24.2.0.1 (R2024b)",\
"machine":"Quokka","submachine":"BOOSTER","timestamp":"2026-01-01T00:00:00"},\
"QF":{"FamilyName":"QF","MemberOf":["QF","Magnet"],"DeviceList":[[1,1],[1,2]],\
"Status":[1,0],"CommonNames":["qf-1","qf-2"],"Position":[1.5,2.5],\
"Setpoint":{"MemberOf":["QF","Setpoint"],"Mode":"Simulator","DataType":"Scalar",\
"ChannelNames":["QK:B:QF1:SP",""],"Units":"Hardware","HWUnits":"A","PhysicsUnits":"1/m^2",\
"Range":[["-Inf","Inf"],[0,"NaN"]],\
"HW2PhysicsFcn":{"$fn":"amp2k","file":"/mml/quokka/amp2k.m"}}}}"""

_EXPORTED_AD = """{"_export":{"exporter":"mml_export 1.0.0","matlab":"24.2.0.1 (R2024b)",\
"machine":"Quokka","submachine":"BOOSTER","timestamp":"2026-01-01T00:00:00"},\
"Machine":"Quokka","SubMachine":"BOOSTER","Energy":"Inf"}"""


def test_the_exported_dialect_imports_with_no_flags(tmp_path: Path) -> None:
    ao_path = tmp_path / "quokka.booster.ao.json"
    ao_path.write_text(_EXPORTED_AO, encoding="utf-8")
    (tmp_path / "quokka.booster.ad.json").write_text(_EXPORTED_AD, encoding="utf-8")

    loaded = load_json(ao_path)

    assert loaded.system_keyed is False
    assert "_export" not in loaded.ao
    assert loaded.export["submachine"] == "BOOSTER"
    assert loaded.ad["SubMachine"] == "BOOSTER"
    assert resolve_system(loaded, None) == "BOOSTER"


def test_the_exported_dialect_is_already_canonical(tmp_path: Path) -> None:
    """Normalising the exporter's output changes nothing but blank slots."""
    ao_path = tmp_path / "quokka.booster.ao.json"
    ao_path.write_text(_EXPORTED_AO, encoding="utf-8")

    body = load_json(ao_path).ao["QF"]
    normalized = normalize_family(body)

    setpoint = normalized["Setpoint"]
    assert setpoint["HW2PhysicsFcn"] == {"$fn": "amp2k", "file": "/mml/quokka/amp2k.m"}
    assert setpoint["Range"] == [["-Inf", "Inf"], [0, "NaN"]]
    assert setpoint["ChannelNames"] == ["QK:B:QF1:SP", None]
    assert {k: v for k, v in normalized.items() if k != "Setpoint"} == {
        k: v for k, v in body.items() if k != "Setpoint"
    }
