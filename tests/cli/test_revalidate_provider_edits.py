"""``osprey init --revalidate`` treats ``providers.yml`` the way it treats ``profile.yml``.

It never changes a value the repo's catalog states, an edited entry OSPREY
ships included. It adds the packaged entries and keys OSPREY added since the
repo last took the catalog, and never one the repo deleted: the saved copy
under ``.osprey/preset-base/`` holds it, so its absence is on purpose.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from click.testing import CliRunner

from osprey.cli.build_profile_updates import PRESET_BASE_DIR
from osprey.cli.init_cmd import init
from osprey.cli.validate_cmd import validate
from osprey.profiles.providers import PROVIDERS_FILENAME
from tests.fixtures.lifecycle_repo import (
    build_exemplar_repo,
    packaged_providers_yml,
    preserved_environ,
)

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


def _older_catalog(repo: Path, edit) -> None:
    """Edit the saved copy of the catalog into what an older OSPREY shipped."""
    path = repo / PRESET_BASE_DIR / PROVIDERS_FILENAME
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    edit(document["providers"])
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def test_a_packaged_key_osprey_added_since_is_added(tmp_path: Path) -> None:
    repo = _repo(tmp_path, PACKAGED_CBORG_HEALTH, "")
    _older_catalog(repo, lambda providers: providers["cborg"].pop("health_model"))
    catalog = repo / PROVIDERS_FILENAME
    before = catalog.read_text(encoding="utf-8")

    _revalidate(repo)

    text = catalog.read_text(encoding="utf-8")
    start = text.index("  cborg:\n")
    cborg = text[start : start + re.search(r"\n  [a-z]", text[start + 1 :]).start() + 2]
    assert PACKAGED_CBORG_HEALTH in cborg
    # Nothing else moved: the only change is the restored line.
    assert text.replace(PACKAGED_CBORG_HEALTH, "", 1) == before


def test_a_packaged_entry_the_file_deleted_is_not_added_back(tmp_path: Path) -> None:
    from osprey.cli.profile_cmd import _catalog_entry_blocks

    block = _catalog_entry_blocks(packaged_providers_yml())["openrouter"]
    repo = _repo(tmp_path, block, "")
    catalog = repo / PROVIDERS_FILENAME
    before = catalog.read_text(encoding="utf-8")

    _revalidate(repo)

    assert catalog.read_text(encoding="utf-8") == before
    with preserved_environ():
        result = CliRunner().invoke(validate, ["--repo", str(repo)])
    assert result.exit_code == 0, result.output
    assert "openrouter" not in result.output


def test_a_packaged_key_the_file_deleted_is_not_added_back(tmp_path: Path) -> None:
    repo = _repo(tmp_path, PACKAGED_CBORG_HEALTH, "")
    catalog = repo / PROVIDERS_FILENAME
    before = catalog.read_text(encoding="utf-8")

    _revalidate(repo)
    _revalidate(repo)

    assert catalog.read_text(encoding="utf-8") == before


def test_a_new_key_under_an_entry_the_file_deleted_is_not_added(tmp_path: Path) -> None:
    from osprey.cli.profile_cmd import _catalog_entry_blocks

    block = _catalog_entry_blocks(packaged_providers_yml())["cborg"]
    repo = _repo(tmp_path, block, "")
    _older_catalog(repo, lambda providers: providers["cborg"].pop("health_model"))
    catalog = repo / PROVIDERS_FILENAME
    before = catalog.read_text(encoding="utf-8")

    _revalidate(repo)

    assert catalog.read_text(encoding="utf-8") == before


def test_validate_accepts_an_edited_entry_without_listing_it(tmp_path: Path) -> None:
    repo = _repo(tmp_path, PACKAGED_OLLAMA_URL, FACILITY_OLLAMA_URL)

    with preserved_environ():
        result = CliRunner().invoke(validate, ["--repo", str(repo)])

    # The catalog is the facility's to edit, as profile.yml is: a value it
    # turned is neither refused nor listed.
    assert result.exit_code == 0, result.output
    assert "providers.ollama.base_url" not in result.output
