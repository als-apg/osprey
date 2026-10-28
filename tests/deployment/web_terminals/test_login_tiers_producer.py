"""The ``control_system.login_tiers`` producer (C-LT) and the sign-in predicate.

Each map is built from a resolved root config plus per-tier renders staged under
a layer root, exactly as a build in flight stages them, and every output is
checked against the package's shape fixture, read by path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.deployment.web_terminals import login_tiers as login_tiers_module
from osprey.deployment.web_terminals.login_tiers import build_login_tiers
from osprey.deployment.web_terminals.roster import sign_in_decides_tier

SHAPE_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "packages/osprey-connectors/tests/osprey_connectors_tests/_harness/fixtures"
    / "login_tiers_shape.json"
)

#: Each tier's staged posture.
TIER_POSTURE = {"readonly": "never", "readwrite": "allowed", "admin": "allowed"}


def _assert_shape(mapping: Any) -> None:
    shape = json.loads(SHAPE_FIXTURE.read_text(encoding="utf-8"))
    allowed_values = shape["value_shape"]["values"]
    assert isinstance(mapping, dict)
    for key, value in mapping.items():
        assert isinstance(key, str) and key
        assert value in allowed_values


def _stage(layer_root: Path, tiers: dict[str, str] = TIER_POSTURE) -> None:
    for tier, writes in tiers.items():
        target = layer_root / f"proj-{tier}"
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.yml").write_text(
            yaml.safe_dump({"control_system": {"writes": writes}}), encoding="utf-8"
        )


def _config(
    users: list[Any],
    *,
    oidc_claim: bool = False,
    default_tier: str | None = "readwrite",
    root_writes: str = "allowed",
) -> dict[str, Any]:
    web_terminals: dict[str, Any] = {
        "enabled": True,
        "users": users,
        "renders": {
            tier: {"kind": "tier", "project": f"proj-{tier}", "project_path": f"build/proj-{tier}"}
            for tier in TIER_POSTURE
        },
        "authorization": {
            "roles": {"reader": {"tier": "readonly"}, "operator": {"tier": "readwrite"}},
        },
    }
    if default_tier is not None:
        web_terminals["default_tier"] = default_tier
    if oidc_claim:
        web_terminals["auth"] = {"method": "oidc"}
        web_terminals["authorization"]["claims"] = {
            "claim": "groups",
            "map": {"ops": "operator", "readers": "reader"},
        }
    return {
        "control_system": {"writes": root_writes},
        "modules": {"web_terminals": web_terminals},
    }


@pytest.fixture
def staged(tmp_path: Path) -> Path:
    layer_root = tmp_path / "stage"
    _stage(layer_root)
    return layer_root


def _set_account(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Make the account running the build *name*."""
    monkeypatch.setattr("osprey_connectors.identity.acting_identity", lambda: name)


@pytest.fixture(autouse=True)
def _default_build_account(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_account(monkeypatch, "builder")


def _build(config: dict[str, Any], tmp_path: Path, layer_root: Path) -> dict[str, str]:
    result = build_login_tiers(config, tmp_path, layer_root=layer_root)
    _assert_shape(result)
    return result


# --- the sign-in predicate -------------------------------------------------------------


def test_sign_in_decides_a_roleless_card_under_oidc_with_a_claim() -> None:
    web_terminals = _config([], oidc_claim=True)["modules"]["web_terminals"]
    assert sign_in_decides_tier({"name": "ann"}, web_terminals) is True
    assert sign_in_decides_tier({"name": "ann", "tier": "admin"}, web_terminals) is True
    assert sign_in_decides_tier("ann", web_terminals) is True
    assert sign_in_decides_tier({"name": "ann", "role": "reader"}, web_terminals) is False


def test_sign_in_decides_nothing_without_oidc_or_without_a_claim() -> None:
    no_oidc = _config([])["modules"]["web_terminals"]
    assert sign_in_decides_tier({"name": "ann"}, no_oidc) is False
    oidc_no_claim = dict(no_oidc, auth={"method": "oidc"})
    assert sign_in_decides_tier({"name": "ann"}, oidc_no_claim) is False


# --- contents and absences -------------------------------------------------------------


def test_sign_in_decided_card_is_absent_despite_default_tier(tmp_path: Path, staged: Path) -> None:
    config = _config([{"name": "ann"}], oidc_claim=True, default_tier="readwrite")
    assert "ann" not in _build(config, tmp_path, staged)


def test_the_same_card_with_an_explicit_role_is_mapped(tmp_path: Path, staged: Path) -> None:
    config = _config(
        [{"name": "ann", "role": "reader"}, {"name": "bo", "role": "operator"}],
        oidc_claim=True,
    )
    result = _build(config, tmp_path, staged)
    assert result["ann"] == "never"
    assert result["bo"] == "allowed"


def test_cards_map_to_their_own_render_tier_without_sign_in(tmp_path: Path, staged: Path) -> None:
    config = _config(
        [
            {"name": "ann", "tier": "readonly"},
            {"name": "bo"},
            "cy",
            {"name": "di", "role": "reader"},
        ],
        default_tier="readwrite",
    )
    result = _build(config, tmp_path, staged)
    assert result["ann"] == "never"
    assert result["bo"] == "allowed"
    assert result["cy"] == "allowed"
    assert result["di"] == "never"


def test_keys_are_exact_logins(tmp_path: Path, staged: Path) -> None:
    config = _config([{"name": "Ann.Smith", "tier": "readonly"}])
    result = _build(config, tmp_path, staged)
    assert "Ann.Smith" in result
    assert "ann.smith" not in result


def test_a_card_with_no_tier_is_absent(tmp_path: Path, staged: Path) -> None:
    config = _config([{"name": "ann"}], default_tier=None)
    assert "ann" not in _build(config, tmp_path, staged)


def test_a_card_whose_role_does_not_resolve_is_absent(tmp_path: Path, staged: Path) -> None:
    config = _config(
        [{"name": "ann", "role": "nobody"}, {"name": "bo", "role": "reader", "tier": "admin"}]
    )
    result = _build(config, tmp_path, staged)
    assert "ann" not in result
    assert "bo" not in result


def test_an_unreadable_render_leaves_the_card_absent(tmp_path: Path, staged: Path) -> None:
    (staged / "proj-readonly" / "config.yml").write_text("control_system: [unclosed\n")
    (staged / "proj-admin" / "config.yml").unlink()
    config = _config(
        [{"name": "ann", "tier": "readonly"}, {"name": "bo", "tier": "admin"}, {"name": "cy"}],
        root_writes="allowed",
    )
    result = _build(config, tmp_path, staged)
    assert "ann" not in result
    assert "bo" not in result
    assert result["cy"] == "allowed"


def test_renders_are_read_from_the_layer_root_not_the_build_zone(tmp_path: Path) -> None:
    published = tmp_path / "build"
    _stage(published, {"readonly": "allowed"})
    layer_root = tmp_path / "stage"
    _stage(layer_root, {"readonly": "never"})
    config = _config([{"name": "ann", "tier": "readonly"}])
    assert _build(config, tmp_path, layer_root)["ann"] == "never"


# --- the build account -----------------------------------------------------------------


def test_build_account_that_is_a_readonly_roster_login_is_never(
    tmp_path: Path, staged: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_account(monkeypatch, "ann")
    config = _config([{"name": "ann", "tier": "readonly"}], root_writes="allowed")
    assert _build(config, tmp_path, staged)["ann"] == "never"


def test_build_account_that_is_a_sign_in_decided_card_is_absent(
    tmp_path: Path, staged: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_account(monkeypatch, "ann")
    config = _config([{"name": "ann"}], oidc_claim=True, root_writes="allowed")
    assert "ann" not in _build(config, tmp_path, staged)


def test_build_account_that_is_an_unreadable_card_is_absent(
    tmp_path: Path, staged: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_account(monkeypatch, "ann")
    (staged / "proj-readonly" / "config.yml").unlink()
    config = _config([{"name": "ann", "tier": "readonly"}], root_writes="allowed")
    assert "ann" not in _build(config, tmp_path, staged)


def test_build_account_naming_no_card_is_mapped_at_the_root_tier(
    tmp_path: Path, staged: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_account(monkeypatch, "builder")
    config = _config([{"name": "ann", "tier": "readonly"}], root_writes="never")
    result = _build(config, tmp_path, staged)
    assert result["builder"] == "never"
    assert result["ann"] == "never"


def test_unknown_identity_is_never_mapped(
    tmp_path: Path, staged: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_account(monkeypatch, "unknown")
    config = _config([{"name": "ann", "tier": "readonly"}])
    result = _build(config, tmp_path, staged)
    assert "unknown" not in result
    assert set(result) == {"ann"}


def test_a_build_account_the_tree_refuses_is_never_mapped(
    tmp_path: Path, staged: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_account(monkeypatch, "Not A Dir/../x")
    result = _build(_config([]), tmp_path, staged)
    assert result == {}


def test_single_user_maps_the_build_account_at_the_root_tier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_account(monkeypatch, "solo")
    allowed = {"control_system": {"writes": "allowed"}}
    assert _build(allowed, tmp_path, tmp_path / "stage") == {"solo": "allowed"}
    never = {"control_system": {"writes": "never"}}
    assert _build(never, tmp_path, tmp_path / "stage") == {"solo": "never"}


def test_the_producer_writes_nothing(tmp_path: Path, staged: Path) -> None:
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    _build(_config([{"name": "ann", "tier": "readonly"}]), tmp_path, staged)
    after = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    assert before == after


def test_rendered_key_name_matches_the_shape_fixture() -> None:
    shape = json.loads(SHAPE_FIXTURE.read_text(encoding="utf-8"))
    assert shape["location"] == f"control_system.{login_tiers_module.LOGIN_TIERS_KEY}"
