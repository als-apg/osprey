"""Unit tests for :mod:`osprey.registry.export`.

Covers the JSON export contract: connector serialisation shape, metadata
counts, the return-only (no ``output_dir``) path, and the on-disk write path
including directory creation and a valid round-trippable JSON payload.
"""

from __future__ import annotations

import json

import pytest

from osprey.registry.base import ConnectorRegistration, RegistryConfig
from osprey.registry.export import export_registry_to_json


def _config(*connectors: ConnectorRegistration) -> RegistryConfig:
    return RegistryConfig(connectors=list(connectors))


def _connector(name: str, ctype: str = "control_system") -> ConnectorRegistration:
    return ConnectorRegistration(
        name=name,
        connector_type=ctype,
        module_path=f"pkg.{name}",
        class_name=f"{name.title()}Connector",
        description=f"{name} connector",
    )


class TestExportRegistryToJson:
    def test_returns_connectors_and_metadata_keys(self):
        data = export_registry_to_json(_config(_connector("epics")), registries={})
        assert set(data) == {"connectors", "metadata"}

    def test_connector_serialisation_shape(self):
        data = export_registry_to_json(_config(_connector("mock")), registries={})
        (entry,) = data["connectors"]
        assert entry == {
            "name": "mock",
            "connector_type": "control_system",
            "description": "mock connector",
            "module_path": "pkg.mock",
            "class_name": "MockConnector",
        }

    def test_metadata_total_matches_connector_count(self):
        data = export_registry_to_json(
            _config(_connector("a"), _connector("b", "archiver")), registries={}
        )
        assert data["metadata"]["total_connectors"] == 2
        assert len(data["connectors"]) == 2
        assert data["metadata"]["registry_version"] == "1.0"
        # exported_at is an ISO timestamp string.
        assert isinstance(data["metadata"]["exported_at"], str)
        assert "T" in data["metadata"]["exported_at"]

    def test_empty_config_yields_zero_connectors(self):
        data = export_registry_to_json(_config(), registries={})
        assert data["connectors"] == []
        assert data["metadata"]["total_connectors"] == 0

    def test_no_output_dir_writes_nothing(self, tmp_path):
        export_registry_to_json(_config(_connector("epics")), registries={})
        # Nothing should have been created anywhere under tmp_path.
        assert list(tmp_path.iterdir()) == []

    def test_writes_file_to_output_dir(self, tmp_path):
        out = tmp_path / "exports"
        data = export_registry_to_json(
            _config(_connector("epics")), registries={}, output_dir=str(out)
        )
        export_file = out / "registry_export.json"
        assert export_file.exists()
        on_disk = json.loads(export_file.read_text(encoding="utf-8"))
        assert on_disk == data

    def test_creates_nested_output_dir(self, tmp_path):
        out = tmp_path / "deep" / "nested" / "dir"
        export_registry_to_json(_config(_connector("epics")), registries={}, output_dir=str(out))
        assert (out / "registry_export.json").exists()

    def test_registries_arg_does_not_affect_output(self):
        # The reserved-but-unused ``registries`` dict must not alter the payload.
        cfg = _config(_connector("epics"))
        a = export_registry_to_json(cfg, registries={})
        b = export_registry_to_json(cfg, registries={"anything": {"x": 1}})
        assert a["connectors"] == b["connectors"]

    def test_save_failure_propagates(self, tmp_path, monkeypatch):
        # A write failure must not be swallowed -- it re-raises after logging.
        import osprey.registry.export as export_mod

        def _boom(*_a, **_k):
            raise OSError("disk full")

        monkeypatch.setattr(export_mod, "open", _boom, raising=False)
        with pytest.raises(OSError, match="disk full"):
            export_registry_to_json(
                _config(_connector("epics")), registries={}, output_dir=str(tmp_path / "o")
            )
