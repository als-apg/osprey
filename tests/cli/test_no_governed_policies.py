"""Nothing shipped spells a per-tool policy for a tool the write switch governs.

The slider on the control-target chip decides for a governed tool, so an
``approval.tools.<tool>`` line for one would state a second answer nobody reads.
The files are loaded as YAML rather than grepped: a comment may name the tool,
and only a key is a policy. The preset's dotted spelling and a render's nested
one are the same key here.

The stanza tests live beside the policies because the same edit moves both: the
presets dropped the lines, spelled the new ceiling, and the frozen stanza table
has to keep finding the prose above it.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey import cli as osprey_cli
from osprey.cli import init_revalidate
from osprey.cli.build_profile_resolve import resolve_build_profile
from osprey.cli.init_revalidate import (
    WRITE_STANZAS,
    WriteStanza,
    _preset_stanza,
    _rewrite_write_prose,
    _stanza_hash,
)
from osprey.registry.mcp import governed_write_tools, resolve_servers

REPO_ROOT = Path(__file__).resolve().parents[2]
PRESETS_DIR = Path(osprey_cli.__file__).resolve().parents[1] / "profiles" / "presets"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "explicit_config"
EXEMPLAR_DIR = REPO_ROOT / "tests" / "deployment" / "goldens" / "exemplar-profile"

#: The two lines the bundled presets carried until the slider took them over.
RETIRED_POLICIES = frozenset({"channel_write", "execute"})

_POLICY_KEY = re.compile(r"(?:^|\.)approval\.tools\.([^.]+)$")


def _dotted_keys(node: Any, prefix: str = "") -> Iterator[str]:
    """Every key of *node*, dotted from the root, whichever spelling the file uses."""
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path
            yield from _dotted_keys(value, path)
    elif isinstance(node, list):
        for item in node:
            yield from _dotted_keys(item, prefix)


def _policies(path: Path) -> set[str]:
    """The tools *path* states an ``approval.tools`` policy for."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    matches = (_POLICY_KEY.search(key) for key in _dotted_keys(document))
    return {match.group(1) for match in matches if match}


def _spelled(root: Path, tools: frozenset[str]) -> dict[str, list[str]]:
    """Each YAML file under *root* stating a policy for one of *tools*, with the tools."""
    found: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*.yml")):
        hits = sorted(_policies(path) & tools)
        if hits:
            found[str(path.relative_to(root))] = hits
    return found


def _preset_servers(preset: str) -> dict[str, Any]:
    """The preset's ``claude_code.servers`` overrides, with every opt-in server on."""
    profile, _ = resolve_build_profile(None, preset=preset)
    servers: dict[str, Any] = {}
    prefix = "claude_code.servers."
    for key, value in profile.config.items():
        if key.startswith(prefix):
            name, _, leaf = key[len(prefix) :].partition(".")
            servers.setdefault(name, {})[leaf] = value
    for name in ("bluesky", "phoebus", "ariel"):
        servers[name] = {"enabled": True}
    return servers


@pytest.fixture(scope="module")
def governed() -> frozenset[str]:
    """The bare names of the tools the switch decides for, from the one producer."""
    tools: set[str] = set()
    for preset in ("hello-world", "control-assistant"):
        servers = resolve_servers(
            {"servers": _preset_servers(preset)},
            {"event_dispatcher_wired": True, "phoebus_agent_access": "read_write"},
        )
        tools.update(tool.rsplit("__", 1)[-1] for tool in governed_write_tools(servers))
    return frozenset(tools)


def test_the_scan_sees_both_spellings_and_the_set_is_not_vacuous(
    tmp_path: Path, governed: frozenset[str]
) -> None:
    assert RETIRED_POLICIES <= governed

    dotted = tmp_path / "dotted.yml"
    dotted.write_text("config:\n  approval.tools.channel_write: always\n", encoding="utf-8")
    nested = tmp_path / "nested.yml"
    nested.write_text("approval:\n  tools:\n    execute: selective\n", encoding="utf-8")

    assert _spelled(tmp_path, governed) == {
        "dotted.yml": ["channel_write"],
        "nested.yml": ["execute"],
    }


def test_the_scan_ignores_a_policy_in_a_comment(tmp_path: Path, governed: frozenset[str]) -> None:
    quiet = tmp_path / "quiet.yml"
    quiet.write_text(
        "config:\n  # approval.tools.channel_write: always\n  approval.enabled: true\n",
        encoding="utf-8",
    )
    assert _spelled(tmp_path, governed) == {}


def test_a_non_governed_policy_survives_the_retirement() -> None:
    """The scan's other half: the presets still state policies, just not these."""
    policies = _policies(PRESETS_DIR / "control-assistant.yml")
    assert {"channel_read", "archiver_read", "setup_patch"} <= policies


def test_no_shipped_preset_tier_or_app_states_a_governed_policy(
    governed: frozenset[str],
) -> None:
    assert sorted(PRESETS_DIR.rglob("*.yml"))
    assert _spelled(PRESETS_DIR, governed) == {}


def test_the_exemplar_deployment_states_no_governed_policy(governed: frozenset[str]) -> None:
    assert _spelled(EXEMPLAR_DIR, governed) == {}


@pytest.mark.xfail(
    strict=True,
    reason=(
        "These files are the frozen render at the baseline sha. PROPOSAL's criterion "
        "says they are re-frozen; the plan's ruling says they never are, and "
        "CELL_DELTAS declares the two moved leaves instead. Ruling owed: drop this "
        "test, or re-freeze and drop this mark."
    ),
)
def test_no_explicit_config_fixture_states_a_governed_policy(governed: frozenset[str]) -> None:
    assert _spelled(FIXTURES_DIR, governed) == {}


def test_the_frozen_fixtures_spell_only_the_two_retired_policies(
    governed: frozenset[str],
) -> None:
    """All the frozen render owes the slider is what CELL_DELTAS will declare gone."""
    spelled = _spelled(FIXTURES_DIR, governed)
    assert spelled
    assert {tool for tools in spelled.values() for tool in tools} == RETIRED_POLICIES


# ── The stanza table ────────────────────────────────────────────────────────


@pytest.mark.parametrize("stanza", WRITE_STANZAS, ids=lambda s: f"{s.preset}:{s.anchor[:24]}")
def test_every_stanza_hash_matches_its_preset(stanza: WriteStanza) -> None:
    current = _preset_stanza(stanza)
    assert current is not None
    assert _stanza_hash(current) == stanza.new_hash


def test_the_two_stanzas_on_the_tier_line_look_for_the_shipped_spelling() -> None:
    moved = [s for s in WRITE_STANZAS if s.anchor == "control_system.writes: allowed"]
    assert sorted(s.preset for s in moved) == ["control-assistant", "tiers/readwrite"]
    assert {s.preset_anchor for s in moved} == {"control_system.writes: asked"}
    assert not any(s.preset_anchor for s in WRITE_STANZAS if s not in moved)


def test_a_stanza_without_its_preset_anchor_no_longer_resolves() -> None:
    """Proves the field carries weight: the operator's line left the preset."""
    stanza = next(s for s in WRITE_STANZAS if s.preset == "control-assistant" and s.preset_anchor)
    assert _preset_stanza(stanza._replace(preset_anchor=None)) is None


def test_revalidate_replaces_the_prose_and_leaves_the_operators_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The operator keeps ``allowed``: the anchor matches their file, not the preset."""
    shipped = next(s for s in WRITE_STANZAS if s.preset == "control-assistant" and s.preset_anchor)
    old = ["# The previous release's words.\n", "# Two lines of them.\n"]
    stanza = shipped._replace(old_lines=len(old), old_hash=_stanza_hash(old))
    monkeypatch.setattr(init_revalidate, "WRITE_STANZAS", (stanza,))

    after = _rewrite_write_prose(
        "".join(old) + "control_system.writes: allowed\n", "profile.yml", [], rewrote=False
    )

    lines = after.splitlines(keepends=True)
    assert lines[-1] == "control_system.writes: allowed\n"
    assert lines[:-1] == _preset_stanza(stanza)
