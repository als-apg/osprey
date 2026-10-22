"""``osprey ariel`` reads the config of the deployment it is typed in.

A deployment repo keeps its render at ``build/config.yml`` and its operator
stands at the repo root, or somewhere inside it. The ``ariel`` verbs must find
that render the way every other host verb does, with no ``CONFIG_FILE`` export.
"""

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

import osprey_connectors.config as config_module
from osprey.cli.ariel import ariel_group

_VOCABULARY = """
concepts:
  - canonical: timing system
    kind: acronym
    forms: ["ts"]
"""


@pytest.fixture
def deployment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A built deployment repo whose ARIEL vocabulary path is repo-relative."""
    (tmp_path / "profile.yml").write_text("name: demo\n")
    (tmp_path / "vocab.yml").write_text(_VOCABULARY)
    build = tmp_path / "build"
    build.mkdir()
    (build / "config.yml").write_text(
        yaml.safe_dump({"ariel": {"vocabulary": {"enabled": True, "path": "vocab.yml"}}})
    )
    (tmp_path / "data" / "raw").mkdir(parents=True)

    monkeypatch.delenv("CONFIG_FILE", raising=False)
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    monkeypatch.setattr(config_module, "_default_config", None)
    monkeypatch.setattr(config_module, "_default_configurable", None)
    monkeypatch.setattr(config_module, "_config_cache", {})
    return tmp_path


@pytest.mark.parametrize("stance", [".", "data/raw"], ids=["repo-root", "subdirectory"])
def test_configured_vocabulary_is_found_from_inside_the_repo(
    deployment: Path, stance: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(deployment / stance)

    result = CliRunner().invoke(ariel_group, ["vocab-check"])

    assert result.exit_code == 0, result.output
    assert "Vocabulary OK: 1 concept" in result.output


def test_status_reads_the_render_from_the_repo_root(
    deployment: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(deployment)
    seen: dict = {}

    async def _fake_status(config_dict, config_dir=None):
        seen["config"] = config_dict
        seen["config_dir"] = config_dir
        return {"status": "error", "message": "database down", "vocabulary": None}

    monkeypatch.setattr("osprey.services.ariel_search.cli_operations.get_status", _fake_status)

    result = CliRunner().invoke(ariel_group, ["status"])

    assert "No config.yml found" not in result.output
    assert seen["config"]["vocabulary"]["path"] == "vocab.yml"
    assert Path(seen["config_dir"]) == deployment / "build"


def test_the_anchor_does_not_outlive_the_command(deployment: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(deployment)

    CliRunner().invoke(ariel_group, ["vocab-check"])

    assert config_module.default_config_path() is None


def test_an_ambient_config_file_does_not_redirect_the_command(
    deployment: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
):
    """A CONFIG_FILE naming another deployment must not change which one acts."""
    other = tmp_path_factory.mktemp("other") / "config.yml"
    vocab = other.parent / "elsewhere.yml"
    vocab.write_text(
        _VOCABULARY
        + '  - canonical: beam position monitor\n    kind: acronym\n    forms: ["bpm"]\n'
    )
    other.write_text(
        yaml.safe_dump({"ariel": {"vocabulary": {"enabled": True, "path": str(vocab)}}})
    )
    monkeypatch.setenv("CONFIG_FILE", str(other))
    monkeypatch.chdir(deployment)

    result = CliRunner().invoke(ariel_group, ["vocab-check"])

    assert result.exit_code == 0, result.output
    assert "Vocabulary OK: 1 concept" in result.output
