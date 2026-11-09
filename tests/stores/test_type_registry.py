"""Extension types: loading, validation, viewer containment, lazy load."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from osprey.stores import type_registry as tr
from osprey.stores.artifact_forms import form_of
from osprey.stores.type_registry import (
    ExtensionType,
    get_extension_types,
    load_extension_types,
    viewer_path,
)


@pytest.fixture(autouse=True)
def _fresh_registry():
    tr.reset_extension_types()
    yield
    tr.reset_extension_types()


def _cfg(**types_):
    return {"artifact_server": {"extension_types": types_}}


def _load(root: Path, validate: bool = False, eps=(), **types_):
    return load_extension_types(
        validate=validate, config=_cfg(**types_), project_root=root, entry_points=list(eps)
    )


class _EP:
    """Stub entry point: loads ``obj`` and names ``module`` as its home."""

    def __init__(self, name, obj, module="osprey.stores"):
        self.name = name
        self._obj = obj
        self.module = module
        self.value = f"{module}:x"

    def load(self):
        if isinstance(self._obj, Exception):
            raise self._obj
        return self._obj


def _entry(**kw) -> types.SimpleNamespace:
    """Stub entry: form_of reads attributes only, so ``kind`` needs no store field."""
    base = {
        "id": "a1",
        "artifact_type": "json",
        "title": "t",
        "tool_source": "",
        "mime_type": "",
        "metadata": {},
    }
    base.update(kw)
    return types.SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# Happy path and form_of integration
# ---------------------------------------------------------------------------


class TestLoad:
    def test_config_declaration_loads(self, tmp_path):
        reg = _load(tmp_path, bmlog={"label": "Beam Log", "base_form": "tables"})
        assert reg == {"bmlog": ExtensionType("bmlog", "Beam Log", "tables", None)}

    def test_form_of_uses_registry_for_kind(self, tmp_path):
        reg = _load(tmp_path, bmlog={"label": "Beam Log", "base_form": "documents"})
        entry = _entry(kind="bmlog")
        assert form_of(entry, reg) == "documents"
        # Without the registry the intrinsic mapping (json -> data) applies.
        assert form_of(entry) == "data"
        assert form_of(_entry(kind="unknown"), reg) == "data"

    def test_entry_point_declaration_loads(self, tmp_path):
        ep = _EP("orbit", {"label": "Orbit Fit", "base_form": "plots"})
        reg = _load(tmp_path, eps=[ep])
        assert reg["orbit"] == ExtensionType("orbit", "Orbit Fit", "plots", None)

    def test_entry_point_callable_and_extension_type(self, tmp_path):
        eps = [
            _EP("a", lambda: {"label": "Alpha", "base_form": "tables"}),
            _EP("ignored-name", ExtensionType("b", "Beta", "images")),
        ]
        reg = _load(tmp_path, eps=eps)
        assert set(reg) == {"a", "b"}
        assert reg["b"].base_form == "images"

    def test_explicit_config_does_not_install(self, tmp_path):
        tr._EXTENSION_TYPES = {}
        _load(tmp_path, x={"label": "X Type", "base_form": "data"})
        assert tr._EXTENSION_TYPES == {}

    def test_categories_not_loaded(self, tmp_path, caplog):
        before = set(tr.CATEGORIES)
        cfg = {"artifact_server": {"categories": {"beam": {"label": "B", "color": "#000000"}}}}
        with caplog.at_level(logging.WARNING, logger=tr.__name__):
            reg = load_extension_types(config=cfg, project_root=tmp_path, entry_points=[])
        assert reg == {}
        assert set(tr.CATEGORIES) == before
        assert "extension_types" in caplog.text


# ---------------------------------------------------------------------------
# Bad declarations: runtime skip vs build raise
# ---------------------------------------------------------------------------

BAD_DECLARATIONS = {
    "unknown_base_form": {"bad": {"label": "Bad", "base_form": "spreadsheets"}},
    "runs_base_form": {"bad": {"label": "Bad", "base_form": "runs"}},
    "label_collides_base_label": {"bad": {"label": "Plots", "base_form": "plots"}},
    "label_collides_base_key_case": {"bad": {"label": "TABLES", "base_form": "tables"}},
    "missing_label": {"bad": {"base_form": "tables"}},
    "spec_not_mapping": {"bad": "tables"},
    "key_is_base_form": {"plots": {"label": "My Plots", "base_form": "plots"}},
}


class TestBadDeclarations:
    @pytest.mark.parametrize("name", sorted(BAD_DECLARATIONS))
    def test_runtime_logs_and_skips(self, tmp_path, caplog, name):
        decl = dict(BAD_DECLARATIONS[name])
        decl["good"] = {"label": "Good One", "base_form": "tables"}
        with caplog.at_level(logging.WARNING, logger=tr.__name__):
            reg = _load(tmp_path, **decl)
        assert set(reg) == {"good"}
        assert caplog.records, "a skipped declaration must be logged"

    @pytest.mark.parametrize("name", sorted(BAD_DECLARATIONS))
    def test_build_raises_naming_key(self, tmp_path, name):
        decl = BAD_DECLARATIONS[name]
        (key,) = decl
        with pytest.raises(ValueError, match=repr(key)):
            _load(tmp_path, validate=True, **decl)

    def test_label_collides_with_other_type(self, tmp_path):
        decl = {
            "one": {"label": "Beam Log", "base_form": "tables"},
            "two": {"label": "beam log", "base_form": "documents"},
        }
        assert set(_load(tmp_path, **decl)) == {"one"}
        with pytest.raises(ValueError, match="'two'"):
            _load(tmp_path, validate=True, **decl)

    def test_duplicate_key_across_config_and_entry_point(self, tmp_path):
        ep = _EP("dup", {"label": "Dup EP", "base_form": "plots"})
        decl = {"dup": {"label": "Dup Cfg", "base_form": "tables"}}
        reg = _load(tmp_path, eps=[ep], **decl)
        assert reg["dup"].label == "Dup Cfg"
        with pytest.raises(ValueError, match="'dup'"):
            _load(tmp_path, validate=True, eps=[ep], **decl)

    def test_entry_point_that_fails_to_load(self, tmp_path):
        ep = _EP("boom", RuntimeError("nope"))
        assert _load(tmp_path, eps=[ep]) == {}
        with pytest.raises(ValueError, match="'boom'"):
            _load(tmp_path, validate=True, eps=[ep])

    def test_categories_raise_under_validate(self, tmp_path):
        cfg = {"artifact_server": {"categories": {"x": {"label": "X", "color": "#000000"}}}}
        with pytest.raises(ValueError, match="extension_types"):
            load_extension_types(validate=True, config=cfg, project_root=tmp_path, entry_points=[])

    def test_extension_types_not_a_mapping(self, tmp_path):
        cfg = {"artifact_server": {"extension_types": ["a"]}}
        assert load_extension_types(config=cfg, project_root=tmp_path, entry_points=[]) == {}
        with pytest.raises(ValueError, match="extension_types"):
            load_extension_types(validate=True, config=cfg, project_root=tmp_path, entry_points=[])


# ---------------------------------------------------------------------------
# Viewer resolution
# ---------------------------------------------------------------------------


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "repo"
    (root / "viewers").mkdir(parents=True)
    (root / "viewers" / "beam.js").write_text("export function render() {}\n")
    (root / "viewers" / "beam.mjs").write_text("export function render() {}\n")
    (root / "viewers" / "beam.txt").write_text("not a module\n")
    outside = tmp_path / "outside.js"
    outside.write_text("export function render() {}\n")
    (root / "viewers" / "link.js").symlink_to(outside)
    return root


def _with_viewer(viewer):
    return {"beam": {"label": "Beam Log", "base_form": "tables", "viewer": viewer}}


class TestViewer:
    @pytest.mark.parametrize(
        "rel", ["viewers/beam.js", "viewers/beam.mjs", "./viewers/../viewers/beam.js"]
    )
    def test_inside_resolves(self, project, rel):
        reg = _load(project, **_with_viewer(rel))
        assert reg["beam"].viewer == (project / "viewers" / Path(rel).name).resolve()

    @pytest.mark.parametrize(
        "case", ["dotdot", "absolute_outside", "symlink_out", "missing", "wrong_suffix"]
    )
    def test_bad_viewer(self, project, case, caplog):
        outside = project.parent / "outside.js"
        rel = {
            "dotdot": "../outside.js",
            "absolute_outside": str(outside),
            "symlink_out": "viewers/link.js",
            "missing": "viewers/nope.js",
            "wrong_suffix": "viewers/beam.txt",
        }[case]
        with caplog.at_level(logging.WARNING, logger=tr.__name__):
            reg = _load(project, **_with_viewer(rel))
        # The type stays; only the viewer is dropped.
        assert reg["beam"] == ExtensionType("beam", "Beam Log", "tables", None)
        assert "viewer" in caplog.text
        with pytest.raises(ValueError, match="'beam'"):
            _load(project, validate=True, **_with_viewer(rel))

    def test_entry_point_viewer_is_package_resource(self, tmp_path, monkeypatch):
        pkg = tmp_path / "site" / "facpkg"
        (pkg / "viewers").mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "types.py").write_text("")
        (pkg / "viewers" / "orbit.mjs").write_text("export function render() {}\n")
        (tmp_path / "site" / "elsewhere.js").write_text("")
        monkeypatch.syspath_prepend(str(tmp_path / "site"))
        eps = [
            _EP(
                "orbit",
                {"label": "Orbit", "base_form": "plots", "viewer": "viewers/orbit.mjs"},
                module="facpkg.types",
            ),
            _EP(
                "esc",
                {"label": "Esc", "base_form": "plots", "viewer": "../elsewhere.js"},
                module="facpkg.types",
            ),
        ]
        reg = _load(tmp_path, eps=eps)
        assert reg["orbit"].viewer == (pkg / "viewers" / "orbit.mjs").resolve()
        assert reg["esc"].viewer is None

    def test_viewer_path_reads_process_registry(self, project, monkeypatch):
        monkeypatch.setattr(
            tr,
            "_read_config_block",
            lambda: (_cfg(**_with_viewer("viewers/beam.js"))["artifact_server"], project),
        )
        monkeypatch.setattr(tr, "_entry_point_declarations", lambda eps, errors: [])
        assert viewer_path("beam") == (project / "viewers" / "beam.js").resolve()
        assert viewer_path("nope") is None


# ---------------------------------------------------------------------------
# Lazy load
# ---------------------------------------------------------------------------


class TestLazyLoad:
    def test_get_extension_types_loads_once(self, monkeypatch):
        calls = []

        def fake_block():
            calls.append(1)
            return {"extension_types": {"x": {"label": "X Kind", "base_form": "data"}}}, None

        monkeypatch.setattr(tr, "_read_config_block", fake_block)
        monkeypatch.setattr(tr, "_entry_point_declarations", lambda eps, errors: [])
        assert set(get_extension_types()) == {"x"}
        assert set(get_extension_types()) == {"x"}
        assert calls == [1]

    def test_config_failure_yields_empty_registry(self, monkeypatch):
        def boom():
            raise RuntimeError("no config")

        monkeypatch.setattr(tr, "_read_config_block", boom)
        monkeypatch.setattr(tr, "_entry_point_declarations", lambda eps, errors: [])
        assert get_extension_types() == {}

    def test_fresh_process_loads_on_first_use(self, tmp_path):
        repo = tmp_path / "repo"
        (repo / "build").mkdir(parents=True)
        (repo / "viewers").mkdir()
        (repo / "viewers" / "beam.js").write_text("export function render() {}\n")
        config = repo / "build" / "config.yml"
        config.write_text(
            f"project_root: {repo}\n"
            "artifact_server:\n"
            "  extension_types:\n"
            "    beam:\n"
            "      label: Beam Log\n"
            "      base_form: tables\n"
            "      viewer: viewers/beam.js\n"
        )
        script = (
            "import json\n"
            "from osprey.stores.artifact_forms import form_of\n"
            "from types import SimpleNamespace\n"
            "from osprey.stores import type_registry as tr\n"
            "assert tr._EXTENSION_TYPES is None\n"
            "e = SimpleNamespace(artifact_type='json', kind='beam', metadata={})\n"
            "print(json.dumps({'form': form_of(e, tr.get_extension_types()),"
            " 'viewer': str(tr.viewer_path('beam'))}))\n"
        )
        env = {**os.environ, "OSPREY_CONFIG": str(config)}
        out = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        result = json.loads(out.stdout.strip().splitlines()[-1])
        assert result == {
            "form": "tables",
            "viewer": str((repo / "viewers" / "beam.js").resolve()),
        }
