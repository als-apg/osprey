"""Provider catalog loading, validation, and hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from osprey.errors import BuildProfileError
from osprey.profiles.providers import (
    PROVIDERS_FILENAME,
    ProviderCatalog,
    compute_providers_hash,
    load_provider_catalog,
    packaged_catalog_path,
)

# The ten providers the control-assistant app template shipped under
# `api.providers`. Frozen here so a dropped entry is a test failure, not a
# silently smaller catalog.
EXPECTED_PROVIDERS = {
    "als-apg",
    "amsc-i2",
    "anthropic",
    "argo",
    "cborg",
    "ds4",
    "google",
    "ollama",
    "openai",
    "stanford",
}

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_catalog(directory: Path, entries: dict) -> Path:
    path = directory / PROVIDERS_FILENAME
    path.write_text(yaml.safe_dump({"providers": entries}), encoding="utf-8")
    return path


class TestPackagedCatalog:
    def test_packaged_file_ships_beside_the_presets(self):
        path = packaged_catalog_path()
        assert path.is_file()
        assert path.parent.name == "profiles"

    def test_packaged_catalog_holds_the_ten_providers(self):
        catalog = load_provider_catalog(None)
        assert set(catalog.entries) == EXPECTED_PROVIDERS
        assert catalog.source == "packaged"
        assert catalog.path == packaged_catalog_path()

    def test_every_packaged_entry_carries_base_url_and_tier_models(self):
        catalog = load_provider_catalog(None)
        for name, entry in catalog.entries.items():
            assert entry["base_url"], name
            assert set(entry["models"]) == {"haiku", "sonnet", "opus"}, name

    def test_packaged_values_match_the_app_template_block(self):
        """The catalog is a literal lift of the template's `api.providers`."""
        template = REPO_ROOT / "src/osprey/templates/apps/control_assistant/config.yml.j2"
        if not template.is_file():  # deleted once the conversion lands
            pytest.skip("app template already removed")
        source = template.read_text(encoding="utf-8")
        # The providers block is plain YAML — no Jinja — so it parses as-is.
        block = source.split("\napi:\n", 1)[1].split("\ncontainer_runtime:", 1)[0]
        rendered = yaml.safe_load("api:\n" + block)["api"]["providers"]
        assert load_provider_catalog(None).entries == rendered

    def test_catalog_carries_no_jinja(self):
        text = packaged_catalog_path().read_text(encoding="utf-8")
        assert "{{" not in text
        assert "{%" not in text


class TestRepoCatalogReplacesPackaged:
    def test_repo_file_wins(self, tmp_path):
        _write_catalog(tmp_path, {"only-one": {"base_url": "https://example.test/v1"}})
        catalog = load_provider_catalog(tmp_path)
        assert catalog.source == "repo"
        assert set(catalog.entries) == {"only-one"}

    def test_repo_file_replaces_rather_than_merges(self, tmp_path):
        _write_catalog(tmp_path, {"only-one": {"base_url": "https://example.test/v1"}})
        assert "cborg" not in load_provider_catalog(tmp_path).entries

    def test_absent_repo_file_falls_back_to_packaged(self, tmp_path):
        catalog = load_provider_catalog(tmp_path)
        assert catalog.source == "packaged"
        assert set(catalog.entries) == EXPECTED_PROVIDERS

    def test_catalog_is_a_frozen_record(self, tmp_path):
        catalog = load_provider_catalog(tmp_path)
        assert isinstance(catalog, ProviderCatalog)
        with pytest.raises(Exception):
            catalog.source = "repo"  # type: ignore[misc]


class TestOptionalKeys:
    def test_base_url_alone_is_a_valid_entry(self, tmp_path):
        _write_catalog(tmp_path, {"bare": {"base_url": "https://example.test/v1"}})
        entry = load_provider_catalog(tmp_path).entries["bare"]
        assert entry == {"base_url": "https://example.test/v1"}

    def test_api_protocol_and_api_key_and_models_accepted(self, tmp_path):
        _write_catalog(
            tmp_path,
            {
                "gw": {
                    "base_url": "https://example.test",
                    "api_key": "${GW_KEY}",
                    "api_protocol": "anthropic",
                    "models": {"haiku": "a", "sonnet": "b", "opus": "c"},
                }
            },
        )
        assert load_provider_catalog(tmp_path).entries["gw"]["api_protocol"] == "anthropic"

    def test_unknown_entry_key_passes_through(self, tmp_path):
        _write_catalog(tmp_path, {"gw": {"base_url": "https://example.test", "timeout": 30}})
        assert load_provider_catalog(tmp_path).entries["gw"]["timeout"] == 30


class TestValidationRefusals:
    def _refuses(self, tmp_path, text: str) -> str:
        (tmp_path / PROVIDERS_FILENAME).write_text(text, encoding="utf-8")
        with pytest.raises(BuildProfileError) as exc:
            load_provider_catalog(tmp_path)
        message = str(exc.value)
        assert PROVIDERS_FILENAME in message
        return message

    def test_non_mapping_document_refused(self, tmp_path):
        assert "mapping" in self._refuses(tmp_path, "- one\n- two\n")

    def test_missing_providers_key_refused(self, tmp_path):
        assert "providers" in self._refuses(tmp_path, "something_else: {}\n")

    def test_non_mapping_providers_refused(self, tmp_path):
        assert "mapping" in self._refuses(tmp_path, "providers:\n  - cborg\n")

    def test_non_mapping_entry_refused_naming_the_key(self, tmp_path):
        assert "providers.cborg" in self._refuses(tmp_path, "providers:\n  cborg: https://x\n")

    def test_entry_without_base_url_refused_naming_the_key(self, tmp_path):
        message = self._refuses(tmp_path, "providers:\n  gw:\n    api_key: k\n")
        assert "providers.gw" in message
        assert "base_url" in message

    def test_empty_base_url_refused(self, tmp_path):
        message = self._refuses(tmp_path, "providers:\n  gw:\n    base_url: '   '\n")
        assert "providers.gw.base_url" in message

    def test_non_mapping_models_refused(self, tmp_path):
        message = self._refuses(
            tmp_path, "providers:\n  gw:\n    base_url: https://x\n    models: haiku\n"
        )
        assert "providers.gw.models" in message

    def test_unparseable_yaml_refused_naming_the_file(self, tmp_path):
        assert "Cannot read" in self._refuses(tmp_path, "providers:\n  gw: [unclosed\n")

    def test_missing_file_refused_naming_the_file(self, tmp_path):
        with pytest.raises(BuildProfileError) as exc:
            compute_providers_hash(tmp_path / PROVIDERS_FILENAME)
        assert PROVIDERS_FILENAME in str(exc.value)


class TestProvidersHash:
    def test_hash_spelling_matches_compute_preset_hash(self, tmp_path):
        entries = {"gw": {"base_url": "https://example.test"}}
        path = _write_catalog(tmp_path, entries)
        canonical = json.dumps(entries, sort_keys=True, default=str)
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert compute_providers_hash(path) == f"sha256:{expected}"

    def test_comment_edit_does_not_move_the_hash(self, tmp_path):
        path = tmp_path / PROVIDERS_FILENAME
        path.write_text("providers:\n  gw:\n    base_url: https://x\n", encoding="utf-8")
        before = compute_providers_hash(path)
        path.write_text(
            "# a fresh header comment\nproviders:\n  gw:\n    base_url: https://x\n",
            encoding="utf-8",
        )
        assert compute_providers_hash(path) == before

    def test_changed_entry_moves_the_hash(self, tmp_path):
        path = tmp_path / PROVIDERS_FILENAME
        path.write_text("providers:\n  gw:\n    base_url: https://x\n", encoding="utf-8")
        before = compute_providers_hash(path)
        path.write_text("providers:\n  gw:\n    base_url: https://y\n", encoding="utf-8")
        assert compute_providers_hash(path) != before

    def test_packaged_catalog_hashes(self):
        assert compute_providers_hash(packaged_catalog_path()).startswith("sha256:")


class TestPackaging:
    def test_catalog_is_not_excluded_from_the_wheel(self):
        """Hatchling packages everything under src/osprey bar the exclude list."""
        text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert 'packages = ["src/osprey"]' in text
        excluded = text.split("exclude = [", 1)[1].split("]", 1)[0]
        assert "profiles" not in excluded
