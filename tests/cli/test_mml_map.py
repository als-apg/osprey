"""``osprey mml map``: the command surface around the mapping services.

Pins what the services cannot: that exactly one of ``--init``/``--check`` runs,
that ``--init`` writes ``data/mml/mapping.yaml`` from the imported export and
never overwrites a reviewed file without ``--force``, and that ``--check``
names every problem as ``<key>: <message>``, reports the derived counts as a
warning, and exits non-zero on any problem -- including a file that is not a
structurally valid mapping.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.main import cli
from osprey.services.mml.mapping import parse_mapping
from osprey.services.mml.mapping.branches import packaged_classes

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "mml"


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A deployment repo with the ``wrapped`` fixture imported as system ``INJ``."""
    root = tmp_path / "deploy"
    root.mkdir()
    (root / "profile.yml").write_text("name: scratch\n", encoding="utf-8")
    monkeypatch.chdir(root)
    shutil.copy(FIXTURES / "wrapped" / "export.json", root / "flat.json")
    result = CliRunner().invoke(
        cli, ["mml", "import", "flat.json", "--system", "INJ"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    return root


def _map(*args: str):
    return CliRunner().invoke(cli, ["mml", "map", *args], catch_exceptions=False)


def _mapping_path(repo: Path) -> Path:
    return repo / "data" / "mml" / "mapping.yaml"


def _load(repo: Path) -> dict:
    return yaml.safe_load(_mapping_path(repo).read_text(encoding="utf-8"))


def _write(repo: Path, document: dict) -> None:
    _mapping_path(repo).write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _fill(document: dict) -> dict:
    """Fill every unfilled slot of a skeleton so it passes the check."""
    packaged = packaged_classes()
    document["facility"]["token"] = "quokka"
    # The fixture carries no AD, so facility and system prose start null.
    for entry in (document["facility"], *document["systems"].values()):
        if entry["description"] is None:
            entry["description"] = "Filled by the reviewer."
            entry["provenance"] = "stated"
    for family in document["families"].values():
        if "class" not in family:
            continue
        cls = family["class"]
        if cls in packaged:
            family["branch"] = None
        else:
            family["branch"] = sorted(packaged)[0]
    for slot in document["directions"].values():
        if slot["direction"] is None:
            slot["direction"] = "read"
    return document


class TestVerbSelection:
    def test_neither_verb_is_a_usage_error(self, repo: Path) -> None:
        result = _map()

        assert result.exit_code == 2
        assert "--init" in result.output and "--check" in result.output

    def test_both_verbs_is_a_usage_error(self, repo: Path) -> None:
        result = _map("--init", "--check")

        assert result.exit_code == 2
        assert not _mapping_path(repo).exists()

    def test_force_without_init_is_a_usage_error(self, repo: Path) -> None:
        assert _map("--init").exit_code == 0

        result = _map("--check", "--force")

        assert result.exit_code == 2

    def test_no_derived_without_check_is_a_usage_error(self, repo: Path) -> None:
        result = _map("--init", "--no-derived")

        assert result.exit_code == 2
        assert not _mapping_path(repo).exists()


class TestInit:
    def test_writes_a_skeleton_that_parses(self, repo: Path) -> None:
        result = _map("--init")

        assert result.exit_code == 0, result.output
        assert "Traceback" not in result.output
        document = _load(repo)
        mapping = parse_mapping(document)
        assert "INJ" in mapping.systems
        assert {"BPM", "QM", "GUN"} <= set(mapping.families)
        assert "mapping.yaml" in result.output

    def test_skeleton_is_readable_block_yaml(self, repo: Path) -> None:
        _map("--init")

        text = _mapping_path(repo).read_text(encoding="utf-8")
        assert text.startswith("facility:")
        assert "{" not in text.split("\n", 1)[0]
        assert "directions:" in text

    def test_refuses_to_overwrite_without_force(self, repo: Path) -> None:
        assert _map("--init").exit_code == 0
        _mapping_path(repo).write_text("reviewed: true\n", encoding="utf-8")

        result = _map("--init")

        assert result.exit_code != 0
        assert "Traceback" not in result.output
        assert "--force" in result.output
        assert _mapping_path(repo).read_text(encoding="utf-8") == "reviewed: true\n"

    def test_force_overwrites(self, repo: Path) -> None:
        assert _map("--init").exit_code == 0
        _mapping_path(repo).write_text("reviewed: true\n", encoding="utf-8")

        result = _map("--init", "--force")

        assert result.exit_code == 0, result.output
        assert "facility" in _load(repo)

    def test_without_an_import_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "empty"
        root.mkdir()
        (root / "profile.yml").write_text("name: scratch\n", encoding="utf-8")
        monkeypatch.chdir(root)

        result = _map("--init")

        assert result.exit_code != 0
        assert "Traceback" not in result.output
        assert "mml import" in result.output
        assert not (root / "data" / "mml" / "mapping.yaml").exists()


class TestCheck:
    def test_fresh_skeleton_names_each_unfilled_slot(self, repo: Path) -> None:
        _map("--init")

        result = _map("--check")

        assert result.exit_code != 0
        assert "Traceback" not in result.output
        lines = result.output.splitlines()
        assert any(line.strip().startswith("facility.token: ") for line in lines), result.output
        # The On/OnControl pair has no MemberOf tags, so its vote is undecided.
        assert any(line.strip().startswith("directions.QM.On.direction: ") for line in lines)

    def test_filled_mapping_passes_and_warns_derived_counts(self, repo: Path) -> None:
        _map("--init")
        _write(repo, _fill(_load(repo)))

        result = _map("--check")

        assert result.exit_code == 0, result.output
        assert "Traceback" not in result.output
        warning = next(line for line in result.output.splitlines() if "derived" in line.lower())
        # BPM.X, QM.Setpoint/Monitor/On/OnControl; GUN.Monitor has no channel.
        assert "5 directions" in warning
        assert "3 family" in warning and "0 facility" in warning

    def test_no_derived_reports_every_derived_slot(self, repo: Path) -> None:
        _map("--init")
        _write(repo, _fill(_load(repo)))

        result = _map("--check", "--no-derived")

        assert result.exit_code != 0
        assert "directions.BPM.X.provenance: is derived" in result.output

    def test_structural_error_is_named_by_key(self, repo: Path) -> None:
        _map("--init")
        document = _load(repo)
        document["directions"]["BPM.X"]["direction"] = "sideways"
        _write(repo, document)

        result = _map("--check")

        assert result.exit_code != 0
        assert "Traceback" not in result.output
        assert "directions.BPM.X" in result.output

    def test_invalid_yaml_is_refused(self, repo: Path) -> None:
        _map("--init")
        _mapping_path(repo).write_text("facility: [unclosed\n", encoding="utf-8")

        result = _map("--check")

        assert result.exit_code != 0
        assert "Traceback" not in result.output
        assert "mapping.yaml" in result.output

    def test_non_mapping_document_is_refused(self, repo: Path) -> None:
        _map("--init")
        _mapping_path(repo).write_text("- a\n- b\n", encoding="utf-8")

        result = _map("--check")

        assert result.exit_code != 0
        assert "Traceback" not in result.output

    def test_missing_mapping_points_at_init(self, repo: Path) -> None:
        result = _map("--check")

        assert result.exit_code != 0
        assert "Traceback" not in result.output
        assert "--init" in result.output
