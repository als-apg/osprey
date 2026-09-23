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


#: The smallest valid entry: an endpoint, the ids it serves, and its default.
_ENTRY = {"base_url": "https://example.test/v1", "default_model": "m-1", "models": ["m-1"]}


def _entry(**extra) -> dict:
    return {**_ENTRY, **extra}


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

    def test_every_packaged_entry_lists_its_models_and_names_a_served_default(self):
        catalog = load_provider_catalog(None)
        for name, entry in catalog.entries.items():
            assert entry["base_url"], name
            assert isinstance(entry["models"], list) and entry["models"], name
            assert entry["default_model"] in entry["models"], name
            if "health_model" in entry:
                assert entry["health_model"] in entry["models"], name

    def test_no_packaged_entry_names_a_model_by_a_tier_word(self):
        catalog = load_provider_catalog(None)
        for name, entry in catalog.entries.items():
            assert not isinstance(entry["models"], dict), name
            for word in ("haiku", "sonnet", "opus"):
                assert word not in entry["models"], name
                assert entry["default_model"] != word, name

    def test_catalog_carries_no_jinja(self):
        text = packaged_catalog_path().read_text(encoding="utf-8")
        assert "{{" not in text
        assert "{%" not in text


class TestRepoCatalogReplacesPackaged:
    def test_repo_file_wins(self, tmp_path):
        _write_catalog(tmp_path, {"only-one": _entry()})
        catalog = load_provider_catalog(tmp_path)
        assert catalog.source == "repo"
        assert set(catalog.entries) == {"only-one"}

    def test_repo_file_replaces_rather_than_merges(self, tmp_path):
        _write_catalog(tmp_path, {"only-one": _entry()})
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
    def test_endpoint_models_and_default_alone_are_a_valid_entry(self, tmp_path):
        _write_catalog(tmp_path, {"bare": _entry()})
        entry = load_provider_catalog(tmp_path).entries["bare"]
        assert entry == _ENTRY

    def test_every_optional_key_accepted(self, tmp_path):
        _write_catalog(
            tmp_path,
            {
                "gw": {
                    "base_url": "https://example.test",
                    "api_key": "${GW_KEY}",
                    "api_protocol": "anthropic",
                    "default_model": "claude-sonnet-5",
                    "health_model": "claude-haiku-4-5",
                    "models": ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"],
                    "claude_code_aliases": {"opus": "claude-opus-5"},
                }
            },
        )
        entry = load_provider_catalog(tmp_path).entries["gw"]
        assert entry["api_protocol"] == "anthropic"
        assert entry["claude_code_aliases"] == {"opus": "claude-opus-5"}

    def test_unknown_entry_key_passes_through(self, tmp_path):
        _write_catalog(tmp_path, {"gw": _entry(timeout=30)})
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

    def test_misspelled_api_protocol_refused(self, tmp_path):
        """The value decides whether a translation hop is inserted, so a
        misspelling is refused where the catalog is read rather than read as
        "not Anthropic" further downstream."""
        message = self._refuses(
            tmp_path,
            "providers:\n  gw:\n    base_url: https://x\n    api_protocol: Anthropic\n",
        )
        assert "providers.gw.api_protocol" in message
        assert "anthropic" in message and "openai" in message

    def test_both_accepted_api_protocols_load(self, tmp_path):
        for protocol in ("anthropic", "openai"):
            _write_catalog(tmp_path, {"gw": _entry(api_protocol=protocol)})
            assert load_provider_catalog(tmp_path).entries["gw"]["api_protocol"] == protocol

    def _refuses_entry(self, tmp_path, **entry) -> str:
        return self._refuses(tmp_path, yaml.safe_dump({"providers": {"gw": entry}}))

    def test_models_that_are_not_a_list_refused(self, tmp_path):
        for models in ("m-1", {"haiku": "m-1"}, [], ["m-1", ""], None):
            message = self._refuses_entry(
                tmp_path, base_url="https://x", default_model="m-1", models=models
            )
            assert "providers.gw.models" in message, models
            assert "list" in message

    def test_default_model_outside_the_list_refused(self, tmp_path):
        message = self._refuses_entry(
            tmp_path, base_url="https://x", default_model="m-2", models=["m-1"]
        )
        assert "providers.gw.default_model" in message
        assert "m-1" in message

    def test_missing_default_model_refused(self, tmp_path):
        message = self._refuses_entry(tmp_path, base_url="https://x", models=["m-1"])
        assert "providers.gw.default_model" in message

    def test_health_model_outside_the_list_refused(self, tmp_path):
        message = self._refuses_entry(tmp_path, **_entry(health_model="m-9"))
        assert "providers.gw.health_model" in message

    def test_alias_map_with_a_foreign_key_refused(self, tmp_path):
        message = self._refuses_entry(tmp_path, **_entry(claude_code_aliases={"fast": "m-1"}))
        assert "providers.gw.claude_code_aliases" in message
        assert "haiku, sonnet, opus" in message

    def test_alias_map_naming_an_unserved_id_refused(self, tmp_path):
        message = self._refuses_entry(tmp_path, **_entry(claude_code_aliases={"haiku": "m-9"}))
        assert "providers.gw.claude_code_aliases.haiku" in message

    def test_alias_map_that_is_not_a_mapping_refused(self, tmp_path):
        message = self._refuses_entry(tmp_path, **_entry(claude_code_aliases=["m-1"]))
        assert "providers.gw.claude_code_aliases" in message

    def test_unparseable_yaml_refused_naming_the_file(self, tmp_path):
        assert "Cannot read" in self._refuses(tmp_path, "providers:\n  gw: [unclosed\n")

    def test_missing_file_refused_naming_the_file(self, tmp_path):
        with pytest.raises(BuildProfileError) as exc:
            compute_providers_hash(tmp_path / PROVIDERS_FILENAME)
        assert PROVIDERS_FILENAME in str(exc.value)


class TestProvidersHash:
    def test_hash_spelling_matches_compute_preset_hash(self, tmp_path):
        entries = {"gw": _entry()}
        path = _write_catalog(tmp_path, entries)
        canonical = json.dumps(entries, sort_keys=True, default=str)
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert compute_providers_hash(path) == f"sha256:{expected}"

    def test_comment_edit_does_not_move_the_hash(self, tmp_path):
        path = tmp_path / PROVIDERS_FILENAME
        body = "providers:\n  gw:\n    base_url: https://x\n    default_model: m\n    models: [m]\n"
        path.write_text(body, encoding="utf-8")
        before = compute_providers_hash(path)
        path.write_text("# a fresh header comment\n" + body, encoding="utf-8")
        assert compute_providers_hash(path) == before

    def test_changed_entry_moves_the_hash(self, tmp_path):
        path = tmp_path / PROVIDERS_FILENAME
        body = (
            "providers:\n  gw:\n    base_url: https://{}\n    default_model: m\n    models: [m]\n"
        )
        path.write_text(body.format("x"), encoding="utf-8")
        before = compute_providers_hash(path)
        path.write_text(body.format("y"), encoding="utf-8")
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
