"""The terminal's one reader of its render: :class:`TerminalConfigView`.

Every gate, the tier and the control-context owner ask the same view, so they
see one parse (``${VAR}`` expanded, a non-mapping refused) and one failure
answer — an unreadable config arms nothing.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from osprey.interfaces.web_terminal import config_view
from osprey.interfaces.web_terminal.config_view import (
    TIER_UNKNOWN,
    UNREADABLE,
    PrivilegeGates,
    TerminalConfigView,
    terminal_config,
)
from osprey_connectors.dotenv import ENV_CHAIN_APPLIED_ENV
from tests.interfaces.web_terminal._started_app import terminal_client

KEY = "web.terminal"

GATE_KEYS = {
    "web": {
        "terminal": False,
        "config_panel": {"enabled": False},
        "scaffold_gallery": {"write_enabled": False},
    },
    "control_system": {"writes": "asked"},
}


def _write(path: Path, document: object) -> Path:
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def _view(tmp_path: Path, case: str) -> TerminalConfigView:
    """The view for one row of the semantics table."""
    config_file = tmp_path / "config.yml"
    if case == "no-file":
        return TerminalConfigView(None)
    if case == "closed":
        return terminal_config(SimpleNamespace(state=SimpleNamespace()))
    if case == "broken-yaml":
        config_file.write_text("web: [unclosed\n", encoding="utf-8")
    elif case == "not-a-mapping":
        config_file.write_text("- web\n- terminal\n", encoding="utf-8")
    elif case == "gone":
        pass
    elif case == "absent-key":
        _write(config_file, {"project_name": "p"})
    elif case == "uninterpretable":
        _write(
            config_file,
            {
                "web": {
                    "terminal": ["not", "a", "flag"],
                    "config_panel": {"enabled": {"nested": 1}},
                    "scaffold_gallery": {"write_enabled": 7},
                }
            },
        )
    return TerminalConfigView(config_file)


# ---- the semantics table, one row per case and method ----

#: case -> (flag(KEY, True), tier, unreadable is the path, gates)
SEMANTICS = {
    "no-file": (True, TIER_UNKNOWN, False, PrivilegeGates(True, True)),
    "broken-yaml": (False, TIER_UNKNOWN, True, None),
    "not-a-mapping": (False, TIER_UNKNOWN, True, None),
    "gone": (False, TIER_UNKNOWN, True, None),
    "absent-key": (True, "never", False, PrivilegeGates(True, True)),
    "uninterpretable": (True, "never", False, PrivilegeGates(True, True)),
    "closed": (False, TIER_UNKNOWN, False, PrivilegeGates(False, False)),
}


@pytest.mark.parametrize("case", sorted(SEMANTICS))
def test_flag(tmp_path, case):
    assert _view(tmp_path, case).flag(KEY, True) is SEMANTICS[case][0]


@pytest.mark.parametrize("case", sorted(SEMANTICS))
def test_tier(tmp_path, case):
    assert _view(tmp_path, case).tier("alice") == SEMANTICS[case][1]


@pytest.mark.parametrize("case", sorted(SEMANTICS))
def test_unreadable(tmp_path, case):
    view = _view(tmp_path, case)
    expected = view.path if SEMANTICS[case][2] else None
    assert view.unreadable == expected


@pytest.mark.parametrize("case", sorted(SEMANTICS))
def test_gates(tmp_path, case):
    view = _view(tmp_path, case)
    expected = SEMANTICS[case][3] or PrivilegeGates(False, False, view.path)
    assert view.gates() == expected


@pytest.mark.parametrize("case", sorted(SEMANTICS))
def test_render(tmp_path, case):
    rendered = _view(tmp_path, case).render()
    if SEMANTICS[case][2] or case in ("no-file", "closed"):
        assert rendered is UNREADABLE
    else:
        assert isinstance(rendered, dict)


def test_a_flag_absent_with_a_false_default_takes_the_default(tmp_path):
    view = _view(tmp_path, "absent-key")
    assert view.flag(KEY, False) is False
    assert TerminalConfigView(None).flag(KEY, False) is False


def test_a_readable_file_answers_its_keys(tmp_path):
    view = TerminalConfigView(_write(tmp_path / "config.yml", GATE_KEYS))

    assert view.flag(KEY, True) is False
    assert view.tier("alice") == "asked"
    assert view.gates() == PrivilegeGates(False, False)
    assert view.unreadable is None


def test_the_view_on_app_state_is_the_one_answered(tmp_path):
    view = TerminalConfigView(tmp_path / "config.yml")
    assert terminal_config(SimpleNamespace(state=SimpleNamespace(terminal_config=view))) is view


# ---- the module is a leaf ----


def test_config_view_imports_neither_routes_nor_app():
    probe = (
        "import sys\n"
        "import osprey.interfaces.web_terminal.config_view\n"
        "bad = [m for m in ('osprey.interfaces.web_terminal.routes',"
        " 'osprey.interfaces.web_terminal.app') if m in sys.modules]\n"
        "print(','.join(bad))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == ""


# ---- freshness ----


def test_one_re_render_is_seen_by_render_flag_and_tier(tmp_path):
    config_file = _write(tmp_path / "config.yml", {"web": {"terminal": True}})
    view = TerminalConfigView(config_file)
    assert view.flag(KEY, True) is True
    assert view.tier("alice") == "never"

    replacement = _write(tmp_path / "config.yml.new", GATE_KEYS)
    os.replace(replacement, config_file)

    assert view.render()["control_system"] == {"writes": "asked"}
    assert view.flag(KEY, True) is False
    assert view.tier("alice") == "asked"


def test_an_unreadable_file_is_never_memoized(tmp_path):
    config_file = tmp_path / "config.yml"
    config_file.write_text("web: [unclosed\n", encoding="utf-8")
    view = TerminalConfigView(config_file)
    assert view.render() is UNREADABLE

    config_file.write_text(yaml.safe_dump({"web": {"terminal": False}}), encoding="utf-8")

    assert view.unreadable is None
    assert view.flag(KEY, True) is False


# ---- one parse: ${VAR} ----


def test_a_defaulted_variable_gives_its_default_tier(tmp_path, monkeypatch):
    monkeypatch.delenv("OSPREY_TEST_TIER", raising=False)
    config_file = tmp_path / "config.yml"
    config_file.write_text("control_system:\n  writes: ${OSPREY_TEST_TIER:-asked}\n")

    assert TerminalConfigView(config_file).tier("alice") == "asked"


def test_an_unset_variable_gives_never(tmp_path, monkeypatch):
    monkeypatch.delenv("OSPREY_TEST_TIER", raising=False)
    config_file = tmp_path / "config.yml"
    config_file.write_text("control_system:\n  writes: ${OSPREY_TEST_TIER}\n")

    assert TerminalConfigView(config_file).tier("alice") == "never"


def test_a_re_parse_leaves_the_environment_alone(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("OSPREY_TEST_DOTENV_ONLY=from-dotenv\n", encoding="utf-8")
    monkeypatch.delenv("OSPREY_TEST_DOTENV_ONLY", raising=False)
    before = dict(os.environ)

    config_view.rendered_config(_write(tmp_path / "config.yml", GATE_KEYS))

    assert dict(os.environ) == before


# ---- the lifespan ----


def test_a_variable_set_only_in_dotenv_reaches_the_lifespan_flag(
    workspace_dir, tmp_path, monkeypatch
):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    # Recorded, so the value the lifespan's dotenv load leaves behind is undone.
    monkeypatch.setenv("OSPREY_TEST_WEB_TERMINAL", "true")
    monkeypatch.delenv("OSPREY_TEST_WEB_TERMINAL")
    monkeypatch.delenv(ENV_CHAIN_APPLIED_ENV, raising=False)
    (project / ".env").write_text("OSPREY_TEST_WEB_TERMINAL=false\n", encoding="utf-8")
    config_file = project / "config.yml"
    config_file.write_text("web:\n  terminal: ${OSPREY_TEST_WEB_TERMINAL}\n")

    with terminal_client(workspace_dir, config_path=config_file) as client:
        view = terminal_config(client.app)
        assert view.path == config_file.resolve()
        assert view.flag(KEY, True) is False
        assert client.app.state.web_terminal_offered is False


# ---- the watchdogs block ----


def test_the_watchdogs_block_is_read_through_the_view(tmp_path, monkeypatch):
    from osprey.interfaces.web_terminal import app as app_module

    seen = []
    monkeypatch.setattr(app_module, "load_watchdogs", lambda config, _dir: seen.append(config))
    monkeypatch.setenv("OSPREY_TEST_WATCHDOG", "expanded")
    config_file = tmp_path / "config.yml"
    config_file.write_text("watchdogs:\n  note: ${OSPREY_TEST_WATCHDOG}\n")
    state = SimpleNamespace()

    app_module._start_watchdogs(SimpleNamespace(state=state), TerminalConfigView(config_file))

    assert seen == [{"watchdogs": {"note": "expanded"}}]
    assert state.watchdog_runner is None


def test_an_unreadable_render_starts_no_watchdog(tmp_path, monkeypatch):
    from osprey.interfaces.web_terminal import app as app_module

    monkeypatch.setattr(
        app_module, "load_watchdogs", lambda *_: pytest.fail("an unreadable render arms nothing")
    )
    config_file = tmp_path / "config.yml"
    config_file.write_text("watchdogs: [unclosed\n", encoding="utf-8")
    state = SimpleNamespace()

    app_module._start_watchdogs(SimpleNamespace(state=state), TerminalConfigView(config_file))

    assert state.watchdog_runner is None
    assert state.session_hold_root is None
