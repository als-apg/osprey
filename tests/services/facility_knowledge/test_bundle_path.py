"""The control-assistant preset's knowledge bundle lives in the facility tree.

The preset's ``facility_knowledge.bundle_path`` names ``data/facility/knowledge``,
the packaged template ships its pages there, and every reader of the key lands
on that directory: the bundle resolver, the mount entitlement and the
in-container mount target.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from osprey.deployment.web_terminals.personas import config_needs_facility_bundle
from osprey.deployment.web_terminals.render import _container_bundle_dir
from osprey.services.facility_knowledge.bundle_path import resolve_bundle_path

_SRC = Path(__file__).resolve().parents[3] / "src" / "osprey"
_PRESET = _SRC / "profiles" / "presets" / "control-assistant.yml"
_PACKAGED_DATA = _SRC / "templates" / "apps" / "control_assistant" / "data"

BUNDLE_PATH = "data/facility/knowledge"


def _preset_bundle_path() -> str:
    preset = yaml.safe_load(_PRESET.read_text(encoding="utf-8"))
    return preset["config"]["facility_knowledge.bundle_path"]


def test_the_preset_names_the_bundle_in_the_facility_tree() -> None:
    assert _preset_bundle_path() == BUNDLE_PATH


def test_the_packaged_pages_sit_where_the_preset_points() -> None:
    bundle = _PACKAGED_DATA.parent / _preset_bundle_path()
    assert (bundle / "index.md").is_file()
    assert sorted(path.name for path in bundle.iterdir() if path.is_dir()) == [
        "devices",
        "physics",
        "procedures",
        "references",
        "subsystems",
    ]
    assert not (_PACKAGED_DATA / "facility_knowledge").exists()


def test_the_resolver_lands_on_the_moved_bundle(tmp_path: Path) -> None:
    render = tmp_path / "build"
    render.mkdir()
    (render / "config.yml").write_text(f"project_root: {tmp_path}\n", encoding="utf-8")

    assert resolve_bundle_path(_preset_bundle_path(), render) == tmp_path / BUNDLE_PATH


def test_the_demo_mounts_the_moved_bundle() -> None:
    config = {"facility_knowledge": {"bundle_path": _preset_bundle_path()}}

    assert config_needs_facility_bundle(config)
    assert _container_bundle_dir(config, "/app/demo") == f"/app/demo/{BUNDLE_PATH}"
