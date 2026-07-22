"""Unit tests for :mod:`osprey.registry.loader`.

These exercise the loader/merge layer directly (not via ``RegistryManager``):

* ``build_merged_configuration`` -- framework-only, standalone, extend, and
  the ``None``-returning-provider guard.
* ``load_registry_from_module`` -- success plus the no-provider /
  multiple-provider / import-error branches and the ``component_name`` labels.
* ``load_registry_from_path`` -- filesystem guards, provider discovery, and the
  ``src/`` ``sys.path`` detection.
* the pure merge helpers (``merge_named_registrations``,
  ``apply_framework_exclusions``, ``merge_application_with_override``).

The autouse ``reset_state_between_tests`` fixture (tests/conftest.py) resets the
global registry/config caches; the loader functions under test are otherwise
pure, and the ``load_registry_from_path`` tests restore ``sys.path`` /
``sys.modules`` themselves.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest

from osprey.errors import RegistryError
from osprey.registry.base import (
    ArielSearchModuleRegistration,
    ConnectorRegistration,
    ExtendedRegistryConfig,
    ProviderRegistration,
    RegistryConfig,
    RegistryConfigProvider,
    ServiceRegistration,
)
from osprey.registry.loader import (
    apply_framework_exclusions,
    build_merged_configuration,
    load_registry_from_module,
    load_registry_from_path,
    merge_application_with_override,
    merge_named_registrations,
)


@pytest.fixture
def restore_sys_state():
    """Snapshot/restore ``sys.path`` and the ``_dynamic_registry`` module slot.

    ``load_registry_from_path`` mutates both as a side effect; without this a
    leaked ``sys.path`` entry or module object could bleed into sibling tests.
    """
    saved_path = list(sys.path)
    saved_mod = sys.modules.get("_dynamic_registry")
    try:
        yield
    finally:
        sys.path[:] = saved_path
        if saved_mod is None:
            sys.modules.pop("_dynamic_registry", None)
        else:
            sys.modules["_dynamic_registry"] = saved_mod


def _write_registry(tmp_path, body: str, *, app_name: str = "myapp"):
    """Write a ``registry.py`` under ``tmp_path/<app_name>/`` and return its path."""
    app_dir = tmp_path / app_name
    app_dir.mkdir(parents=True, exist_ok=True)
    registry_file = app_dir / "registry.py"
    registry_file.write_text(body)
    return registry_file


# ---------------------------------------------------------------------------
# build_merged_configuration
# ---------------------------------------------------------------------------


class TestBuildMergedConfiguration:
    def test_framework_only_when_path_is_none(self):
        config, excluded = build_merged_configuration(None)
        assert isinstance(config, RegistryConfig)
        # The framework registry always ships connectors (mock, epics, ...).
        assert len(config.connectors) > 0
        assert excluded == []

    def test_empty_string_path_is_framework_only(self):
        # Empty string is falsy -> same branch as None.
        config, excluded = build_merged_configuration("")
        assert isinstance(config, RegistryConfig)
        assert excluded == []

    def test_standalone_config_bypasses_framework(self, tmp_path, restore_sys_state):
        registry_file = _write_registry(
            tmp_path,
            """
from osprey.registry import RegistryConfigProvider, RegistryConfig
from osprey.registry.base import ConnectorRegistration

class StandaloneProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return RegistryConfig(connectors=[
            ConnectorRegistration(
                name="only_one",
                connector_type="control_system",
                module_path="x.y",
                class_name="Z",
                description="d",
            )
        ])
""",
        )
        config, excluded = build_merged_configuration(str(registry_file))
        # Standalone: framework connectors are NOT merged in.
        assert [c.name for c in config.connectors] == ["only_one"]
        assert excluded == []

    def test_extend_mode_merges_framework(self, tmp_path, restore_sys_state):
        registry_file = _write_registry(
            tmp_path,
            """
from osprey.registry import RegistryConfigProvider
from osprey.registry.base import ExtendedRegistryConfig, ConnectorRegistration

class ExtendingProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return ExtendedRegistryConfig(connectors=[
            ConnectorRegistration(
                name="app_extra",
                connector_type="control_system",
                module_path="x.y",
                class_name="Z",
                description="d",
            )
        ])
""",
        )
        framework_only, _ = build_merged_configuration(None)
        merged, excluded = build_merged_configuration(str(registry_file))
        names = {c.name for c in merged.connectors}
        assert "app_extra" in names
        # Framework connectors are preserved alongside the added one.
        assert len(merged.connectors) == len(framework_only.connectors) + 1
        assert excluded == []

    def test_provider_returning_none_raises(self, tmp_path, restore_sys_state):
        registry_file = _write_registry(
            tmp_path,
            """
from osprey.registry import RegistryConfigProvider

class NoneProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return None
""",
        )
        with pytest.raises(RegistryError, match="returned None"):
            build_merged_configuration(str(registry_file))

    def test_missing_file_wrapped_in_registry_error(self, tmp_path):
        missing = tmp_path / "nope" / "registry.py"
        with pytest.raises(RegistryError, match="Failed to load registry from"):
            build_merged_configuration(str(missing))


# ---------------------------------------------------------------------------
# load_registry_from_module
# ---------------------------------------------------------------------------


class _GoodProvider(RegistryConfigProvider):
    def get_registry_config(self) -> RegistryConfig:
        return RegistryConfig(
            connectors=[
                ConnectorRegistration(
                    name="from_module",
                    connector_type="control_system",
                    module_path="x.y",
                    class_name="Z",
                    description="d",
                )
            ]
        )


class _SecondProvider(RegistryConfigProvider):
    def get_registry_config(self) -> RegistryConfig:
        return RegistryConfig()


def _fake_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


class TestLoadRegistryFromModule:
    def test_loads_builtins_framework_registry(self):
        config = load_registry_from_module("osprey.registry.builtins")
        assert isinstance(config, RegistryConfig)
        assert len(config.connectors) > 0

    def test_single_provider_discovered(self, monkeypatch):
        fake = _fake_module("fake_reg", Provider=_GoodProvider)
        monkeypatch.setattr(importlib, "import_module", lambda _p: fake)
        config = load_registry_from_module("some.module")
        assert [c.name for c in config.connectors] == ["from_module"]

    def test_no_provider_raises(self, monkeypatch):
        fake = _fake_module("fake_reg", NotAProvider=object)
        monkeypatch.setattr(importlib, "import_module", lambda _p: fake)
        with pytest.raises(RegistryError, match="No RegistryConfigProvider"):
            load_registry_from_module("some.module")

    def test_multiple_providers_raises(self, monkeypatch):
        fake = _fake_module("fake_reg", A=_GoodProvider, B=_SecondProvider)
        monkeypatch.setattr(importlib, "import_module", lambda _p: fake)
        with pytest.raises(RegistryError, match="Multiple RegistryConfigProvider"):
            load_registry_from_module("some.module")

    def test_import_error_labels_module(self):
        with pytest.raises(RegistryError, match="module totally.bogus.pkg"):
            load_registry_from_module("totally.bogus.pkg")

    def test_import_error_labels_application(self):
        # applications.<name>.registry -> "<name> application" in the message.
        with pytest.raises(RegistryError, match="myapp application"):
            load_registry_from_module("applications.myapp.registry")

    def test_provider_exception_wrapped(self, monkeypatch):
        class _Boom(RegistryConfigProvider):
            def get_registry_config(self):
                raise ValueError("kaboom")

        fake = _fake_module("fake_reg", Provider=_Boom)
        monkeypatch.setattr(importlib, "import_module", lambda _p: fake)
        with pytest.raises(RegistryError, match="Failed to load"):
            load_registry_from_module("some.module")


# ---------------------------------------------------------------------------
# load_registry_from_path
# ---------------------------------------------------------------------------


class TestLoadRegistryFromPath:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(RegistryError, match="Registry file not found"):
            load_registry_from_path(str(tmp_path / "absent.py"))

    def test_directory_raises(self, tmp_path):
        a_dir = tmp_path / "a_directory"
        a_dir.mkdir()
        with pytest.raises(RegistryError, match="not a file"):
            load_registry_from_path(str(a_dir))

    def test_valid_standalone_returns_config(self, tmp_path, restore_sys_state):
        registry_file = _write_registry(
            tmp_path,
            """
from osprey.registry import RegistryConfigProvider, RegistryConfig

class P(RegistryConfigProvider):
    def get_registry_config(self):
        return RegistryConfig()
""",
        )
        config = load_registry_from_path(str(registry_file))
        assert isinstance(config, RegistryConfig)
        assert not isinstance(config, ExtendedRegistryConfig)

    def test_extended_config_type_preserved(self, tmp_path, restore_sys_state):
        registry_file = _write_registry(
            tmp_path,
            """
from osprey.registry import RegistryConfigProvider
from osprey.registry.base import ExtendedRegistryConfig

class P(RegistryConfigProvider):
    def get_registry_config(self):
        return ExtendedRegistryConfig()
""",
        )
        config = load_registry_from_path(str(registry_file))
        assert isinstance(config, ExtendedRegistryConfig)

    def test_no_provider_raises(self, tmp_path, restore_sys_state):
        registry_file = _write_registry(tmp_path, "x = 1\n")
        with pytest.raises(RegistryError, match="No RegistryConfigProvider"):
            load_registry_from_path(str(registry_file))

    def test_multiple_providers_raises(self, tmp_path, restore_sys_state):
        registry_file = _write_registry(
            tmp_path,
            """
from osprey.registry import RegistryConfigProvider, RegistryConfig

class P1(RegistryConfigProvider):
    def get_registry_config(self):
        return RegistryConfig()

class P2(RegistryConfigProvider):
    def get_registry_config(self):
        return RegistryConfig()
""",
        )
        with pytest.raises(RegistryError, match="Multiple RegistryConfigProvider"):
            load_registry_from_path(str(registry_file))

    def test_syntax_error_wrapped(self, tmp_path, restore_sys_state):
        registry_file = _write_registry(tmp_path, "def broken(:\n")
        with pytest.raises(RegistryError, match="Failed to load Python module"):
            load_registry_from_path(str(registry_file))

    def test_provider_get_config_error_wrapped(self, tmp_path, restore_sys_state):
        registry_file = _write_registry(
            tmp_path,
            """
from osprey.registry import RegistryConfigProvider

class P(RegistryConfigProvider):
    def get_registry_config(self):
        raise RuntimeError("bad config")
""",
        )
        with pytest.raises(RegistryError, match="Failed to instantiate or get config"):
            load_registry_from_path(str(registry_file))

    def test_src_layout_added_to_sys_path(self, tmp_path, restore_sys_state):
        # project/src/app/registry.py -> project/src is added to sys.path.
        src_dir = tmp_path / "project" / "src"
        registry_file = src_dir / "app" / "registry.py"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(
            """
from osprey.registry import RegistryConfigProvider, RegistryConfig

class P(RegistryConfigProvider):
    def get_registry_config(self):
        return RegistryConfig()
"""
        )
        load_registry_from_path(str(registry_file))
        assert str(src_dir.resolve()) in sys.path


# ---------------------------------------------------------------------------
# merge_named_registrations
# ---------------------------------------------------------------------------


class TestMergeNamedRegistrations:
    def _mod(self, name: str) -> ArielSearchModuleRegistration:
        return ArielSearchModuleRegistration(name=name, module_path=f"m.{name}", description=name)

    def test_appends_new_entries(self):
        merged = [self._mod("a")]
        merge_named_registrations(merged, [self._mod("b")], "search module", "app")
        assert [m.name for m in merged] == ["a", "b"]

    def test_override_replaces_by_name(self):
        original = self._mod("a")
        merged = [original]
        replacement = ArielSearchModuleRegistration(
            name="a", module_path="m.override", description="override"
        )
        merge_named_registrations(merged, [replacement], "search module", "app")
        assert len(merged) == 1
        assert merged[0].module_path == "m.override"

    def test_mutates_in_place(self):
        merged: list = []
        returned = merge_named_registrations(merged, [self._mod("a")], "search module", "app")
        assert returned is None
        assert len(merged) == 1


# ---------------------------------------------------------------------------
# apply_framework_exclusions
# ---------------------------------------------------------------------------


class TestApplyFrameworkExclusions:
    def _config_with_connectors(self, *names: str) -> RegistryConfig:
        return RegistryConfig(
            connectors=[
                ConnectorRegistration(
                    name=n,
                    connector_type="control_system",
                    module_path="x.y",
                    class_name="Z",
                    description="d",
                )
                for n in names
            ]
        )

    def test_removes_named_component(self):
        merged = self._config_with_connectors("keep", "drop")
        excluded_providers: list = []
        apply_framework_exclusions(merged, {"connectors": ["drop"]}, "app", excluded_providers)
        assert [c.name for c in merged.connectors] == ["keep"]
        assert excluded_providers == []

    def test_provider_exclusions_are_deferred(self):
        merged = RegistryConfig()
        excluded_providers: list = []
        apply_framework_exclusions(
            merged, {"providers": ["anthropic", "openai"]}, "app", excluded_providers
        )
        # Provider names are recorded for later introspection, not removed here.
        assert excluded_providers == ["anthropic", "openai"]

    def test_empty_exclusion_list_is_noop(self):
        merged = self._config_with_connectors("a", "b")
        apply_framework_exclusions(merged, {"connectors": []}, "app", [])
        assert [c.name for c in merged.connectors] == ["a", "b"]

    def test_unknown_component_type_is_skipped(self):
        merged = self._config_with_connectors("a")
        # Unknown type must not raise -- it is logged and ignored.
        apply_framework_exclusions(merged, {"widgets": ["x"]}, "app", [])
        assert [c.name for c in merged.connectors] == ["a"]


# ---------------------------------------------------------------------------
# merge_application_with_override
# ---------------------------------------------------------------------------


class TestMergeApplicationWithOverride:
    def _framework(self) -> RegistryConfig:
        return RegistryConfig(
            services=[
                ServiceRegistration(
                    name="svc", module_path="f.svc", class_name="S", description="d"
                )
            ],
            providers=[ProviderRegistration(module_path="f.prov", class_name="P", name="fp")],
            connectors=[
                ConnectorRegistration(
                    name="conn",
                    connector_type="control_system",
                    module_path="f.conn",
                    class_name="C",
                    description="d",
                )
            ],
        )

    def test_service_override_by_name(self):
        merged = self._framework()
        app = RegistryConfig(
            services=[
                ServiceRegistration(
                    name="svc", module_path="app.svc", class_name="S2", description="override"
                )
            ]
        )
        merge_application_with_override(merged, app, "app", [])
        assert len(merged.services) == 1
        assert merged.services[0].module_path == "app.svc"

    def test_provider_added_when_key_differs(self):
        merged = self._framework()
        app = RegistryConfig(
            providers=[ProviderRegistration(module_path="app.prov", class_name="AP", name="ap")]
        )
        merge_application_with_override(merged, app, "app", [])
        keys = {(p.module_path, p.class_name) for p in merged.providers}
        assert ("f.prov", "P") in keys
        assert ("app.prov", "AP") in keys

    def test_provider_override_on_same_key(self):
        merged = self._framework()
        # Same (module_path, class_name) key -> override, not duplicate.
        app = RegistryConfig(
            providers=[ProviderRegistration(module_path="f.prov", class_name="P", name="renamed")]
        )
        merge_application_with_override(merged, app, "app", [])
        matching = [p for p in merged.providers if (p.module_path, p.class_name) == ("f.prov", "P")]
        assert len(matching) == 1
        assert matching[0].name == "renamed"

    def test_connector_override_and_add(self):
        merged = self._framework()
        app = RegistryConfig(
            connectors=[
                ConnectorRegistration(
                    name="conn",
                    connector_type="control_system",
                    module_path="app.conn",
                    class_name="C2",
                    description="override",
                ),
                ConnectorRegistration(
                    name="newconn",
                    connector_type="archiver",
                    module_path="app.new",
                    class_name="N",
                    description="new",
                ),
            ]
        )
        merge_application_with_override(merged, app, "app", [])
        by_name = {c.name: c for c in merged.connectors}
        assert by_name["conn"].module_path == "app.conn"
        assert "newconn" in by_name
        assert len(merged.connectors) == 2

    def test_framework_exclusions_applied_during_merge(self):
        merged = self._framework()
        app = RegistryConfig(framework_exclusions={"connectors": ["conn"]})
        merge_application_with_override(merged, app, "app", [])
        assert merged.connectors == []

    def test_provider_exclusion_accumulated_during_merge(self):
        merged = self._framework()
        excluded_providers: list = []
        app = RegistryConfig(framework_exclusions={"providers": ["fp"]})
        merge_application_with_override(merged, app, "app", excluded_providers)
        assert excluded_providers == ["fp"]
