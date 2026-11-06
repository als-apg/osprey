"""``osprey build`` validates and persists ``artifact_server.extension_types``.

A profile declares extension types as ``{key: {label, base_form, viewer?}}``.
The build runs them, with the installed ``osprey.artifact_types`` entry points,
through :func:`osprey.stores.type_registry.load_extension_types` in validate
mode, so every bad declaration is refused before anything renders. A viewer is
a path relative to the profile's ``project/`` mirror, which lands on the
project root. The retired ``categories`` block is refused with a pointer to
``extension_types``.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.build_persistence import _persist_artifact_server
from osprey.cli.build_profile import BuildProfile
from osprey.cli.build_profile_emit import emit_standalone_profile_yaml
from osprey_connectors.errors import BuildProfileError

pytestmark = pytest.mark.usefixtures("facility_data_tree")


class _EP:
    """Stub entry point loading ``obj``, homed in an importable module."""

    def __init__(self, name: str, obj: Any, module: str = "osprey.stores") -> None:
        self.name = name
        self._obj = obj
        self.module = module
        self.value = f"{module}:x"

    def load(self) -> Any:
        return self._obj


@pytest.fixture
def no_entry_points(monkeypatch: pytest.MonkeyPatch) -> None:
    """No installed extension types, so a test sees only its own profile."""
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda **_kw: [])


def _profile(**artifact_server: Any) -> BuildProfile:
    return BuildProfile(name="T", data="data", artifact_server=artifact_server)


def _errors(profile: BuildProfile, profile_dir: Path) -> list[str]:
    with pytest.raises(BuildProfileError) as exc:
        profile.validate(profile_dir)
    header, _, body = str(exc.value).partition(":\n  - ")
    assert header == "Build profile validation failed"
    return body.split("\n  - ")


def _viewer(profile_dir: Path, rel: str = "viewers/beam.js") -> str:
    path = profile_dir / "project" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("export default function render() {}\n")
    return rel


# --- accepted ---------------------------------------------------------------


@pytest.mark.usefixtures("no_entry_points")
def test_valid_extension_types_with_viewer_pass(tmp_path: Path) -> None:
    rel = _viewer(tmp_path)
    _profile(
        port=9000,
        extension_types={
            "beam_scan": {"label": "Beam scan", "base_form": "plots", "viewer": rel},
            "lattice": {"label": "Lattice", "base_form": "tables"},
        },
    ).validate(tmp_path)


# --- rejected ---------------------------------------------------------------


@pytest.mark.usefixtures("no_entry_points")
@pytest.mark.parametrize(
    ("types_", "needle"),
    [
        pytest.param(
            {
                "a": {"label": "Scan", "base_form": "plots"},
                "b": {"label": "scan", "base_form": "tables"},
            },
            "'b'",
            id="duplicate-label",
        ),
        pytest.param(
            {"mine": {"label": "Plots", "base_form": "plots"}},
            "'mine'",
            id="label-is-base-form",
        ),
        pytest.param(
            {"plots": {"label": "My plots", "base_form": "plots"}},
            "'plots'",
            id="key-is-base-form",
        ),
        pytest.param(
            {"weird": {"label": "Weird", "base_form": "holograms"}},
            "'weird'",
            id="unknown-base-form",
        ),
        pytest.param(
            {"runlike": {"label": "Run-like", "base_form": "runs"}},
            "'runlike'",
            id="runs-is-not-a-base-form-to-conform-to",
        ),
    ],
)
def test_bad_declarations_are_refused_naming_the_key(
    tmp_path: Path, types_: dict[str, Any], needle: str
) -> None:
    errors = _errors(_profile(extension_types=types_), tmp_path)
    assert any(needle in e and "extension type" in e for e in errors), errors


@pytest.mark.usefixtures("no_entry_points")
def test_missing_viewer_is_refused(tmp_path: Path) -> None:
    (tmp_path / "project").mkdir()
    errors = _errors(
        _profile(
            extension_types={
                "scan": {"label": "Scan", "base_form": "plots", "viewer": "viewers/nope.js"}
            }
        ),
        tmp_path,
    )
    assert any("'scan'" in e and "missing" in e for e in errors), errors


@pytest.mark.usefixtures("no_entry_points")
def test_viewer_outside_project_dir_is_refused(tmp_path: Path) -> None:
    (tmp_path / "project").mkdir()
    (tmp_path / "escape.js").write_text("export default 1;\n")
    errors = _errors(
        _profile(
            extension_types={
                "scan": {"label": "Scan", "base_form": "plots", "viewer": "../escape.js"}
            }
        ),
        tmp_path,
    )
    assert any("'scan'" in e and "outside" in e for e in errors), errors


@pytest.mark.usefixtures("no_entry_points")
def test_symlinked_viewer_leaving_the_tree_is_refused(tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere" / "v.js"
    outside.parent.mkdir()
    outside.write_text("export default 1;\n")
    link = tmp_path / "project" / "viewers" / "v.js"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)
    errors = _errors(
        _profile(
            extension_types={
                "scan": {"label": "Scan", "base_form": "plots", "viewer": "viewers/v.js"}
            }
        ),
        tmp_path,
    )
    assert any("'scan'" in e and "outside" in e for e in errors), errors


@pytest.mark.usefixtures("no_entry_points")
def test_non_module_viewer_is_refused(tmp_path: Path) -> None:
    rel = _viewer(tmp_path, "viewers/beam.txt")
    errors = _errors(
        _profile(extension_types={"scan": {"label": "Scan", "base_form": "plots", "viewer": rel}}),
        tmp_path,
    )
    assert any("'scan'" in e and ".js/.mjs" in e for e in errors), errors


def test_entry_point_collision_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eps = [_EP("pkg_scan", {"label": "Scan", "base_form": "plots"})]
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda **_kw: eps)
    errors = _errors(
        _profile(extension_types={"scan": {"label": "Scan", "base_form": "tables"}}),
        tmp_path,
    )
    assert any("'pkg_scan'" in e and "collides" in e for e in errors), errors


def test_entry_point_key_duplicate_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eps = [_EP("scan", {"label": "Other scan", "base_form": "plots"})]
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda **_kw: eps)
    errors = _errors(
        _profile(extension_types={"scan": {"label": "Scan", "base_form": "tables"}}),
        tmp_path,
    )
    assert any("'scan'" in e and "already declared" in e for e in errors), errors


@pytest.mark.usefixtures("no_entry_points")
def test_categories_is_refused_with_migration_message(tmp_path: Path) -> None:
    errors = _errors(
        _profile(categories={"optics": {"label": "Optics", "color": "#4C9AFF"}}), tmp_path
    )
    hits = [e for e in errors if "categories" in e]
    assert len(hits) == 1, errors
    assert "extension_types" in hits[0]


@pytest.mark.usefixtures("no_entry_points")
def test_unknown_artifact_server_key_still_refused(tmp_path: Path) -> None:
    errors = _errors(_profile(colour="red"), tmp_path)
    assert any("artifact_server.colour" in e for e in errors), errors


@pytest.mark.usefixtures("no_entry_points")
def test_extension_types_must_be_a_mapping(tmp_path: Path) -> None:
    errors = _errors(_profile(extension_types=["scan"]), tmp_path)
    assert any("artifact_server.extension_types" in e for e in errors), errors


# --- persisted ----------------------------------------------------------------


def test_persist_writes_extension_types_with_declared_viewer(tmp_path: Path) -> None:
    (tmp_path / "config.yml").write_text("artifact_server:\n  host: 127.0.0.1\n  port: 8080\n")
    _persist_artifact_server(
        tmp_path,
        {
            "port": 9001,
            "extension_types": {
                "beam_scan": {
                    "label": "Beam scan",
                    "base_form": "plots",
                    "viewer": "viewers/beam.js",
                },
                "lattice": {"label": "Lattice", "base_form": "tables"},
            },
        },
    )
    block = yaml.safe_load((tmp_path / "config.yml").read_text())["artifact_server"]
    assert block["host"] == "127.0.0.1"
    assert block["port"] == 9001
    assert block["extension_types"] == {
        "beam_scan": {"label": "Beam scan", "base_form": "plots", "viewer": "viewers/beam.js"},
        "lattice": {"label": "Lattice", "base_form": "tables"},
    }
    assert "categories" not in block


# --- emitted ------------------------------------------------------------------


@pytest.mark.parametrize("preset", ["hello-world", "demo"])
def test_emitted_profile_offers_extension_types_not_categories(preset: str) -> None:
    text = emit_standalone_profile_yaml(preset, (), "Emitted", project_name="emitted")
    assert "categories:" not in text
    assert "category=" not in text
    assert "extension_types" in text


def test_bundled_presets_name_no_categories() -> None:
    presets = Path(__file__).resolve().parents[2] / "src/osprey/profiles/presets"
    for path in sorted(presets.glob("*.yml")):
        assert "artifact_server.categories" not in path.read_text(), path.name
