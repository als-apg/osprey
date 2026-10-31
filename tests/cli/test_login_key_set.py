"""A login file carries a positively declared key set.

A login (``tiers/<name>.yml``) is one access level of the deployment's agent:
it names what that card shows and how it writes, never a fact of the
deployment. Its key set is written out by name rather than derived from the
root set, so a deployment fact added to the root schema does not leak onto
logins by subtraction. Every deployment fact is refused on a login, naming
``profile.yml`` as its home, while every login ``osprey init`` materialises and
every key the bundled layer files spell still loads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.build_profile_load import (
    LOGIN_DEPLOYMENT_FACTS,
    RENDER_KEYS,
    ROOT_KIND_KEYS,
    ROOT_ONLY_KEYS,
    TIER_KIND_KEYS,
    load_profile_document,
    reject_unknown_kind_keys,
)
from osprey.cli.profile_root import ProfileKind
from osprey.utils.workspace import LAYER_DIR_KINDS
from osprey_connectors.errors import BuildProfileError

_PRESETS_DIR = Path(__file__).resolve().parents[2] / "src" / "osprey" / "profiles" / "presets"

#: The deployment facts a login may never spell, as the login contract names them.
_FACTS = frozenset(
    {
        "connector",
        "provider",
        "port_base",
        "dispatch",
        "bluesky",
        "bluesky_web",
        "virtual_accelerator",
        "nextcloud_bridge",
        "gchat_bridge",
        "teams_bridge",
        "va_archiver",
        "artifact_server",
        "requires_osprey_version",
        "osprey_install",
        "python_env",
    }
)

#: The bundled layer files that define a whole agent beside the deployment's own.
_PRESET_APP_STEMS = ("logbook", "knowledge", "channel-finder")


def _write_yaml(path: Path, body: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


def _preset_layer_file(stem: str) -> Path:
    """The bundled layer file named *stem*, in whichever layer directory holds it."""
    found = [
        _PRESETS_DIR / dirname / f"{stem}.yml"
        for dirname in LAYER_DIR_KINDS
        if (_PRESETS_DIR / dirname / f"{stem}.yml").is_file()
    ]
    assert found, f"no bundled layer file {stem}.yml under {sorted(LAYER_DIR_KINDS)}"
    return found[0]


def test_the_facts_are_the_named_deployment_facts() -> None:
    assert LOGIN_DEPLOYMENT_FACTS == _FACTS


def test_the_login_set_is_the_root_set_minus_deployment_keys_and_facts() -> None:
    assert TIER_KIND_KEYS == (ROOT_KIND_KEYS - ROOT_ONLY_KEYS - _FACTS) | RENDER_KEYS
    assert {"model", "panel_presets", "provenance"} <= TIER_KIND_KEYS
    assert RENDER_KEYS <= TIER_KIND_KEYS


@pytest.mark.parametrize("key", sorted(_FACTS | ROOT_ONLY_KEYS | {"services", "data"}))
def test_no_deployment_fact_is_admitted_on_a_login(key: str) -> None:
    assert key not in TIER_KIND_KEYS
    with pytest.raises(BuildProfileError) as excinfo:
        reject_unknown_kind_keys({"name": "ro", key: {}}, ProfileKind.TIER, "tiers/ro.yml")

    assert f"{key!r} (legal only in the deployment's profile.yml)" in str(excinfo.value)


def test_a_login_picks_a_model_but_never_the_provider() -> None:
    assert "model" in TIER_KIND_KEYS
    reject_unknown_kind_keys({"name": "ro", "model": "m"}, ProfileKind.TIER, "tiers/ro.yml")

    with pytest.raises(BuildProfileError) as excinfo:
        reject_unknown_kind_keys(
            {"name": "ro", "provider": "cborg"}, ProfileKind.TIER, "tiers/ro.yml"
        )

    message = str(excinfo.value)
    assert "tiers/ro.yml" in message
    assert "'provider' (legal only in the deployment's profile.yml)" in message


@pytest.mark.parametrize("stem", _PRESET_APP_STEMS)
def test_every_key_a_preset_app_file_spells_is_admitted_on_a_login(stem: str) -> None:
    raw = yaml.safe_load(_preset_layer_file(stem).read_text(encoding="utf-8"))

    assert set(raw) <= TIER_KIND_KEYS, set(raw) - TIER_KIND_KEYS
    reject_unknown_kind_keys(raw, ProfileKind.TIER, f"tiers/{stem}.yml")


@pytest.mark.usefixtures("facility_data_tree")
@pytest.mark.parametrize("tier", ["readonly", "readwrite", "admin"])
def test_an_init_materialised_login_loads(tmp_path: Path, tier: str) -> None:
    from osprey.cli.build_profile_emit import emit_layer_delta_yaml, emit_standalone_profile_yaml

    (tmp_path / "profile.yml").write_text(
        emit_standalone_profile_yaml(
            "control-assistant", ("data=data",), "Site", project_name="site"
        ),
        encoding="utf-8",
    )
    text = emit_layer_delta_yaml(
        preset_name=f"tiers/{tier}", profile_name="Site", profile_filename=f"tiers/{tier}.yml"
    )
    layer = tmp_path / "tiers" / f"{tier}.yml"
    layer.parent.mkdir()
    layer.write_text(text, encoding="utf-8")

    assert "provenance" in yaml.safe_load(text)
    loaded = load_profile_document(layer)

    assert loaded.profile.provenance is not None
    assert loaded.profile.provenance.preset == tier
