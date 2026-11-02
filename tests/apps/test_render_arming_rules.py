"""The per-login arming rules: what an application armed on a login must declare."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from osprey.apps.errors import AppManifestError
from osprey.apps.manifest import AppManifest, parse_manifest
from osprey.apps.rules import AppSelection, LoginSelection, Reserved, validate

EMPTY = Reserved(services=frozenset(), panels=frozenset(), servers=frozenset(), names=frozenset())

#: An application with one write tool and no container driving the target.
DIAL: dict[str, Any] = {
    "name": "dial",
    "containers": {"dial": {"runtime": "osprey", "port": 8200}},
    "tools": {
        "module": "dial.tools",
        "read": ["read_dial"],
        "approval": [{"tool": "set_dial", "prompt": "Set {{ input.value }}"}],
    },
}


def _app(drives_target: bool = False, *, approval: bool = True, name: str = "dial") -> AppManifest:
    raw = copy.deepcopy(DIAL)
    raw["name"] = name
    raw["containers"] = {name: {"runtime": "osprey", "port": 8200, "drives_target": drives_target}}
    raw["tools"]["module"] = f"{name}.tools"
    if not approval:
        raw["tools"]["approval"] = []
    return parse_manifest(raw, app=name)


def _login(
    tier: str,
    *,
    shown: bool = True,
    app: str = "dial",
    shared_cards: tuple[str, ...] = (),
) -> LoginSelection:
    return LoginSelection(
        audience="shared" if shared_cards else "person",
        apps={app: AppSelection(shown=shown)},
        tier=tier,  # type: ignore[arg-type]
        shared_cards=shared_cards,
    )


def _refused(manifests: list[AppManifest], logins: dict[str, LoginSelection]) -> AppManifestError:
    with pytest.raises(AppManifestError) as info:
        validate(manifests, logins=logins, reserved=EMPTY)
    return info.value


# --- arming -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tier", "shown", "armed"),
    [
        ("never", True, False),
        ("asked", True, True),
        ("allowed", True, True),
        ("asked", False, False),
        ("allowed", False, False),
    ],
)
def test_a_login_arms_an_application_it_shows_under_a_writing_tier(
    tier: str, shown: bool, armed: bool
) -> None:
    assert _login(tier, shown=shown).arms("dial") is armed
    assert not _login(tier, shown=shown).arms("other")


# --- target -----------------------------------------------------------------------------------


def test_a_never_login_selecting_an_application_with_write_tools_passes() -> None:
    validate([_app()], logins={"readonly": _login("never")}, reserved=EMPTY)


@pytest.mark.parametrize("tier", ["asked", "allowed"])
def test_an_armed_application_with_write_tools_needs_a_container_driving_the_target(
    tier: str,
) -> None:
    err = _refused([_app()], {"operator": _login(tier)})
    assert [v.rule for v in err.violations] == ["armed_drives_target"]
    assert err.app == "dial" and err.field == "containers"
    assert "operator" in err.message and "drives_target: true" in err.message


def test_one_refusal_per_arming_login() -> None:
    err = _refused([_app()], {"a": _login("allowed"), "b": _login("never"), "c": _login("asked")})
    assert [v.rule for v in err.violations] == ["armed_drives_target"] * 2
    assert "login a " in err.violations[0].message and "login c " in err.violations[1].message


def test_an_armed_application_reaching_the_target_passes() -> None:
    validate([_app(drives_target=True)], logins={"operator": _login("allowed")}, reserved=EMPTY)


def test_an_armed_application_without_write_tools_needs_no_target() -> None:
    validate([_app(approval=False)], logins={"operator": _login("allowed")}, reserved=EMPTY)


def test_a_hidden_application_is_not_armed() -> None:
    validate([_app()], logins={"operator": _login("allowed", shown=False)}, reserved=EMPTY)


# --- shared audience --------------------------------------------------------------------------


@pytest.mark.parametrize("tier", ["asked", "allowed"])
def test_a_shared_login_arming_an_application_is_refused_naming_cards_and_app(tier: str) -> None:
    shared = _login(tier, shared_cards=("control-room", "night-shift"))
    err = _refused([_app(drives_target=True)], {"ops": shared})
    assert [v.rule for v in err.violations] == ["shared_login_tier"]
    assert err.app == "" and err.field == "logins.ops"
    assert "control-room" in err.message and "night-shift" in err.message
    assert "dial" in err.message


def test_a_shared_login_names_every_application_it_arms() -> None:
    shared = LoginSelection(
        audience="shared",
        apps={"dial": AppSelection(shown=True), "knob": AppSelection(shown=True)},
        tier="allowed",
        shared_cards=("control-room",),
    )
    apps = [_app(drives_target=True), _app(drives_target=True, name="knob")]
    err = _refused(apps, {"ops": shared})
    assert [v.rule for v in err.violations] == ["shared_login_tier"] * 2
    assert "dial" in err.violations[0].message and "knob" in err.violations[1].message


def test_a_shared_login_selecting_no_application_is_left_to_the_advisory() -> None:
    shared = _login("allowed", shown=False, shared_cards=("control-room",))
    validate([_app(drives_target=True)], logins={"ops": shared}, reserved=EMPTY)
    validate([], logins={"ops": shared}, reserved=EMPTY)


def test_a_sealed_shared_login_passes() -> None:
    sealed = _login("never", shared_cards=("control-room",))
    validate([_app()], logins={"ops": sealed}, reserved=EMPTY)


def test_every_arming_violation_is_collected_into_one_refusal() -> None:
    shared = _login("allowed", shared_cards=("control-room",))
    err = _refused([_app()], {"ops": shared, "operator": _login("asked")})
    assert sorted(v.rule for v in err.violations) == [
        "armed_drives_target",
        "armed_drives_target",
        "shared_login_tier",
    ]
