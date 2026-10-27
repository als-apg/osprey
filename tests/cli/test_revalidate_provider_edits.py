"""``osprey init --revalidate`` treats ``providers.yml`` the way it treats ``profile.yml``.

It never changes a value the repo's catalog states, an edited entry OSPREY
ships included. It adds the packaged entries the file lacks and the packaged
keys an existing entry lacks. ``osprey validate`` names each place a packaged
entry differs from what this OSPREY ships, as preset drift.
"""

from __future__ import annotations

import re
from pathlib import Path

from click.testing import CliRunner

from osprey.cli.init_cmd import init
from osprey.cli.validate_cmd import validate
from osprey.profiles.providers import PROVIDERS_FILENAME
from tests.fixtures.lifecycle_repo import build_exemplar_repo, preserved_environ

PROJECT = "als-exemplar"
PACKAGED_OLLAMA_URL = "    base_url: ${OLLAMA_HOST:-http://localhost:11434}\n"
FACILITY_OLLAMA_URL = (
    "    # Our GPU host serves Ollama; every local-inference consumer reads it here.\n"
    "    base_url: http://doudna:11434\n"
)
PACKAGED_CBORG_HEALTH = "    health_model: claude-haiku-4-5\n"


def _revalidate(repo: Path) -> None:
    with preserved_environ():
        result = CliRunner().invoke(init, [str(repo), "--revalidate"], catch_exceptions=False)
    assert result.exit_code == 0, result.output


def _repo(tmp_path: Path, old: str, new: str) -> Path:
    repo = build_exemplar_repo(tmp_path / PROJECT)
    catalog = repo / PROVIDERS_FILENAME
    text = catalog.read_text(encoding="utf-8")
    assert text.count(old) == 1, old
    catalog.write_text(text.replace(old, new), encoding="utf-8")
    return repo


def test_an_edited_packaged_value_and_its_comment_survive_revalidate(tmp_path: Path) -> None:
    repo = _repo(tmp_path, PACKAGED_OLLAMA_URL, FACILITY_OLLAMA_URL)
    catalog = repo / PROVIDERS_FILENAME
    before = catalog.read_text(encoding="utf-8")

    _revalidate(repo)
    first = catalog.read_text(encoding="utf-8")
    assert first == before
    assert FACILITY_OLLAMA_URL in first

    _revalidate(repo)
    assert catalog.read_text(encoding="utf-8") == first


def test_a_packaged_key_the_file_lacks_is_added(tmp_path: Path) -> None:
    repo = _repo(tmp_path, PACKAGED_CBORG_HEALTH, "")
    catalog = repo / PROVIDERS_FILENAME
    before = catalog.read_text(encoding="utf-8")

    _revalidate(repo)

    text = catalog.read_text(encoding="utf-8")
    start = text.index("  cborg:\n")
    cborg = text[start : start + re.search(r"\n  [a-z]", text[start + 1 :]).start() + 2]
    assert PACKAGED_CBORG_HEALTH in cborg
    # Nothing else moved: the only change is the restored line.
    assert text.replace(PACKAGED_CBORG_HEALTH, "", 1) == before


def test_validate_names_the_differing_entry_and_key(tmp_path: Path) -> None:
    repo = _repo(tmp_path, PACKAGED_OLLAMA_URL, FACILITY_OLLAMA_URL)

    with preserved_environ():
        result = CliRunner().invoke(validate, ["--repo", str(repo)])

    # A value the facility turned is a note, as it is in profile.yml.
    assert result.exit_code == 0, result.output
    assert "providers.ollama.base_url" in result.output
    assert "http://doudna:11434" in result.output
