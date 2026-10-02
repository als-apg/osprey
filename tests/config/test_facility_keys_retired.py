"""The facility's display name is the build's facility identity, nowhere else.

The dispatcher dashboard shows the identity's name like every other surface,
so neither a config key nor a profile key can name the facility a second time.
The resurrection guard lists each retired config key and proves no shipped
surface spells it again.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from ruamel.yaml import YAML

from osprey.cli.build_cmd import _inject_dispatch
from osprey.cli.build_profile_load import _parse_profile
from osprey.cli.build_profile_schema import DispatchConfig
from osprey.errors import BuildProfileError

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "src" / "osprey" / "profiles" / "config_key_manifest.yml"
PRESETS = REPO_ROOT / "src" / "osprey" / "profiles" / "presets"
RETIRED_CONFIG_KEYS = ("facility.name", "facility.timezone")


@pytest.mark.parametrize("key", RETIRED_CONFIG_KEYS)
def test_each_retired_key_is_on_the_resurrection_list(key: str) -> None:
    """A retired key is listed as deleted, carries orphan sites and has no live row."""
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))

    assert key in manifest["deleted"]
    assert manifest["orphan_sites"].get(key), f"{key} has no orphan site"
    assert key not in manifest["keys"]


def test_the_resurrection_guard_is_green() -> None:
    """No preset, template, reader or reference page spells a retired key."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_config_keys.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("preset", sorted(PRESETS.glob("*.yml")), ids=lambda p: p.stem)
def test_no_preset_spells_a_retired_key(preset: Path) -> None:
    """Neither a live nor a commented preset line names a retired key."""
    text = preset.read_text(encoding="utf-8")

    for key in RETIRED_CONFIG_KEYS:
        assert key not in text, f"{preset.name} still spells {key}"


def test_a_profile_naming_the_dispatch_facility_name_is_refused() -> None:
    """The dashboard's name is the identity's; a profile cannot override it."""
    with pytest.raises(BuildProfileError, match="'facility_name'"):
        _parse_profile({"name": "x", "dispatch": {"facility_name": "ERF"}})


def test_the_dispatch_block_has_no_facility_name_field() -> None:
    """The profile schema carries no second spelling of the facility's name."""
    assert "facility_name" not in DispatchConfig.__dataclass_fields__


def _rendered_dispatcher_name(tmp_path: Path, *, identity: dict | None, project_name: str) -> str:
    """Inject a bundled dispatch block and return the dispatcher's facility name."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    with open(project_path / "config.yml", "w") as fh:
        YAML().dump({"project_name": project_name, "deployed_services": []}, fh)
    if identity is not None:
        document = {"schema": "osprey.facility.facility/1", "identity": identity}
        (project_path / "facility.json").write_text(json.dumps(document), encoding="utf-8")

    _inject_dispatch(
        DispatchConfig(triggers="tutorial_triggers.yml"),
        profile_dir=profile_dir,
        project_path=project_path,
    )

    with open(project_path / "config.yml") as fh:
        config = YAML().load(fh)
    return config["services"]["event_dispatcher"]["facility_name"]


def test_the_dispatcher_shows_the_identity_name(tmp_path: Path) -> None:
    """The rendered dispatcher name is the facility identity's ``name``."""
    name = _rendered_dispatcher_name(
        tmp_path,
        identity={"code": "erf", "name": "Example Research Facility"},
        project_name="my-assistant",
    )

    assert name == "Example Research Facility"


def test_the_dispatcher_falls_back_to_the_project_name(tmp_path: Path) -> None:
    """An identity that names no display name shows the project name."""
    name = _rendered_dispatcher_name(
        tmp_path, identity={"code": "erf"}, project_name="my-assistant"
    )

    assert name == "my-assistant"
