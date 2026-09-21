"""``osprey mml import``: the command surface around the import services.

Pins the parts the services cannot: how ``--system`` values are paired with
inputs (so a token can never land on the wrong input), the fixed ``data/mml/``
output location inside the deployment repo, the one-line summary, and that the
command module keeps the heavy loaders out of its import graph.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.main import cli

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "mml"


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A minimal deployment repo (a ``profile.yml`` marker) as the cwd."""
    root = tmp_path / "deploy"
    root.mkdir()
    (root / "profile.yml").write_text("name: scratch\n", encoding="utf-8")
    monkeypatch.chdir(root)
    return root


def _paired_as_sr(repo: Path) -> None:
    shutil.copy(FIXTURES / "paired" / "quokka.ring.ao.json", repo / "sr.ao.json")
    shutil.copy(FIXTURES / "paired" / "quokka.ring.ad.json", repo / "sr.ad.json")


def _flat(repo: Path, name: str = "flat.json") -> None:
    shutil.copy(FIXTURES / "wrapped" / "export.json", repo / name)


def _invoke(*args: str):
    return CliRunner().invoke(cli, ["mml", "import", *args], catch_exceptions=False)


class TestMixedInputs:
    def test_system_keyed_by_ad_plus_flat_with_path_token(self, repo: Path) -> None:
        _paired_as_sr(repo)
        _flat(repo)

        result = _invoke("sr.ao.json", "flat.json", "--system", "flat.json=LTB")

        assert result.exit_code == 0, result.output
        assert "Traceback" not in result.output
        out = repo / "data" / "mml"
        for name in ("ao.json", "ad.json", "PROFILE.md"):
            assert (out / name).is_file(), name
        ao = json.loads((out / "ao.json").read_text(encoding="utf-8"))
        assert ao["_import_order"] == ["RING", "LTB"]
        assert {"RING", "LTB"} <= set(ao)
        ad = json.loads((out / "ad.json").read_text(encoding="utf-8"))
        assert "RING" in ad
        profile = (out / "PROFILE.md").read_text(encoding="utf-8")
        assert "RING" in profile and "LTB" in profile

    def test_summary_is_one_line_naming_the_four_counts(self, repo: Path) -> None:
        _paired_as_sr(repo)
        _flat(repo)

        result = _invoke("sr.ao.json", "flat.json", "--system", "flat.json=LTB")

        lines = [line for line in result.output.splitlines() if line.strip()]
        assert len(lines) == 1, result.output
        summary = lines[0]
        assert "2 systems" in summary
        for word in ("families", "distinct PVs", "undecided"):
            assert word in summary

    def test_writes_to_repo_root_when_run_from_a_subdirectory(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _flat(repo)
        sub = repo / "incoming"
        sub.mkdir()
        monkeypatch.chdir(sub)

        result = _invoke("../flat.json", "--system", "LTB")

        assert result.exit_code == 0, result.output
        assert (repo / "data" / "mml" / "ao.json").is_file()
        assert not (sub / "data").exists()


class TestSystemFlagPairing:
    def test_bare_token_with_one_input(self, repo: Path) -> None:
        _flat(repo)

        result = _invoke("flat.json", "--system", "INJ")

        assert result.exit_code == 0, result.output
        ao = json.loads((repo / "data" / "mml" / "ao.json").read_text(encoding="utf-8"))
        assert ao["_import_order"] == ["INJ"]

    def test_path_token_with_one_input(self, repo: Path) -> None:
        _flat(repo)

        result = _invoke("flat.json", "--system", "flat.json=INJ")

        assert result.exit_code == 0, result.output

    def test_bare_token_with_several_inputs_is_a_usage_error(self, repo: Path) -> None:
        _paired_as_sr(repo)
        _flat(repo)

        result = _invoke("sr.ao.json", "flat.json", "--system", "LTB")

        assert result.exit_code == 2
        assert "PATH=TOKEN" in result.output
        assert not (repo / "data" / "mml").exists()

    def test_path_naming_no_input_is_a_usage_error(self, repo: Path) -> None:
        _paired_as_sr(repo)
        _flat(repo)

        result = _invoke("sr.ao.json", "flat.json", "--system", "other.json=LTB")

        assert result.exit_code == 2
        assert "other.json" in result.output

    def test_two_tokens_for_one_input_is_a_usage_error(self, repo: Path) -> None:
        _paired_as_sr(repo)
        _flat(repo)

        result = _invoke(
            "sr.ao.json", "flat.json", "--system", "flat.json=LTB", "--system", "flat.json=BTS"
        )

        assert result.exit_code == 2

    def test_two_bare_tokens_with_one_input_is_a_usage_error(self, repo: Path) -> None:
        _flat(repo)

        result = _invoke("flat.json", "--system", "A", "--system", "B")

        assert result.exit_code == 2

    def test_empty_token_is_a_usage_error(self, repo: Path) -> None:
        _flat(repo)

        result = _invoke("flat.json", "--system", "flat.json=")

        assert result.exit_code == 2

    def test_path_matches_by_location_not_spelling(self, repo: Path) -> None:
        _paired_as_sr(repo)
        _flat(repo)

        result = _invoke("sr.ao.json", "flat.json", "--system", "./flat.json=LTB")

        assert result.exit_code == 0, result.output


class TestRefusals:
    def test_unsupported_suffix_is_a_usage_error(self, repo: Path) -> None:
        (repo / "export.txt").write_text("{}", encoding="utf-8")

        result = _invoke("export.txt", "--system", "X")

        assert result.exit_code == 2
        assert ".json" in result.output and ".mat" in result.output

    def test_outside_a_repo_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loose = tmp_path / "loose"
        loose.mkdir()
        shutil.copy(FIXTURES / "wrapped" / "export.json", loose / "flat.json")
        monkeypatch.chdir(loose)

        result = _invoke("flat.json", "--system", "INJ")

        assert result.exit_code != 0
        assert "Traceback" not in result.output
        assert not (loose / "data").exists()

    def test_service_usage_error_propagates_without_traceback(self, repo: Path) -> None:
        _flat(repo, "a.json")
        _flat(repo, "b.json")

        result = _invoke("a.json", "b.json", "--system", "a.json=X", "--system", "b.json=X")

        assert result.exit_code == 2
        assert "Traceback" not in result.output


def test_command_module_keeps_the_loaders_out_of_its_import_graph() -> None:
    code = (
        "import sys, osprey.cli.mml_cmd\n"
        "heavy = [m for m in sys.modules if m.startswith('osprey.services.mml')]\n"
        "print(heavy)\n"
        "sys.exit(1 if heavy else 0)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
