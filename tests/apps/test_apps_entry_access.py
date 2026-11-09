"""``access:`` on a login's ``apps.<name>`` entry: who sees and opens the panel.

The entry is authored in the words a shared roster card's ``access`` uses and
read by the same parser. The build resolves it as a shared card's set is
expanded at render — ``self``, ``roster`` and ``tier:`` spelled out to the
identities the sidecar forwards — and projects the result onto the login's
``web.panels.<id>.access`` as ``user:``/``domain:`` members only. The terminal
compares the forwarded subject against that set: ``GET /api/panels`` leaves out
a panel the caller may not open, and the panel proxy answers it exactly as an
unpublished panel. An entry without the key is every user of the login, and the
tier alone decides writes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.build_profile_load import load_profile_document, reject_unknown_kind_keys
from osprey.cli.profile_root import ProfileKind
from osprey_connectors.errors import BuildProfileError

SUBJECT_HEADER = "X-Osprey-Auth-Subject"

_MANIFEST = {"name": "knob", "containers": {"knob": {"runtime": "osprey"}}}

_PANEL_MANIFEST = {
    "name": "dial",
    "containers": {"dial": {"runtime": "osprey"}},
    "panel": {"label": "Dial", "container": "dial", "entry": "/"},
}


def _write_yaml(path: Path, body: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


def _repo(tmp_path: Path, root: dict[str, Any] | None = None) -> Path:
    _write_yaml(tmp_path / "apps" / "knob" / "app.yml", _MANIFEST)
    _write_yaml(
        tmp_path / "profile.yml",
        {"name": "site", "data": "data", "apps": {"knob": {}}, **(root or {})},
    )
    (tmp_path / "data").mkdir(exist_ok=True)
    return tmp_path


# ---------------------------------------------------------------------------
# Schema: the login spells it, the one parser reads it
# ---------------------------------------------------------------------------


def test_a_login_may_spell_access() -> None:
    reject_unknown_kind_keys(
        {"name": "ro", "apps": {"knob": {"shown": True, "access": ["user:a@example.org"]}}},
        ProfileKind.TIER,
        "tiers/ro.yml",
    )


def test_a_login_entry_carries_its_access(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    layer = _write_yaml(
        repo / "tiers" / "ro.yml",
        {"name": "ro", "apps": {"knob": {"access": ["tier:admin", "user:x@example.org"]}}},
    )

    profile = load_profile_document(layer).profile

    assert profile.apps["knob"].access == ["tier:admin", "user:x@example.org"]


def test_an_entry_without_access_carries_none(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    layer = _write_yaml(repo / "tiers" / "ro.yml", {"name": "ro", "apps": {"knob": {}}})

    assert load_profile_document(layer).profile.apps["knob"].access is None


@pytest.mark.parametrize(
    ("access", "named"), [(["group:ops"], "'group:ops'"), ("everyone", "'everyone'"), ([], "empty")]
)
def test_an_unreadable_access_is_refused_naming_login_app_and_value(
    tmp_path: Path, access: Any, named: str
) -> None:
    repo = _repo(tmp_path)
    layer = _write_yaml(
        repo / "tiers" / "ro.yml", {"name": "ro", "apps": {"knob": {"access": access}}}
    )

    with pytest.raises(BuildProfileError) as excinfo:
        load_profile_document(layer)

    message = str(excinfo.value)
    assert "login 'ro' apps.knob.access" in message
    assert named in message


def test_a_refusal_is_the_parsers_one_sentence_naming_the_login_entry(tmp_path: Path) -> None:
    _write_yaml(
        tmp_path / "apps" / "knob-example" / "app.yml",
        {**_MANIFEST, "name": "knob-example"},
    )
    _write_yaml(
        tmp_path / "profile.yml", {"name": "site", "data": "data", "apps": {"knob-example": {}}}
    )
    (tmp_path / "data").mkdir()
    layer = _write_yaml(
        tmp_path / "tiers" / "readonly.yml",
        {"name": "readonly", "apps": {"knob-example": {"access": ["group:ops"]}}},
    )

    with pytest.raises(BuildProfileError) as excinfo:
        load_profile_document(layer)

    (refusal,) = [line for line in str(excinfo.value).splitlines() if "group:ops" in line]
    assert refusal.lstrip(" -").startswith("login 'readonly' apps.knob-example.access")
    assert "modules.web_terminals.shared" not in str(excinfo.value)
