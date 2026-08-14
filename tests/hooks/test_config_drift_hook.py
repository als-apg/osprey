"""Tests for the osprey_config_drift.py SessionStart hook.

The hook warns (never blocks) when config.yml has drifted from the rendered
.claude/settings.json. Two layers of coverage:

* The pure helpers (``_writes_enabled_in_config``, ``_channel_write_denied``,
  ``_detect_drift``) are imported and called directly, one test per documented
  return branch. The hook is stdlib-only and imports nothing at module scope, so
  a plain import is safe — it is still written inside the test bodies so that
  importing this file stays free of side effects (tests/README.md §5).
* The end-to-end behaviour goes through the shared ``hook_runner_raw`` harness,
  which runs the hook as a subprocess with a curated environment. The hook reads
  its project root from ``CLAUDE_PROJECT_DIR``; that variable is on the harness
  allowlist and is set per test with ``monkeypatch.setenv``, so no value from the
  parent process can reach the subprocess.
"""

from __future__ import annotations

import json
import os
import stat

import pytest
import yaml

HOOK_SCRIPT = "osprey_config_drift.py"

# Spelled out here rather than imported: the tests state the contract the hook is
# supposed to implement, independently of the constant the hook happens to use.
CHANNEL_WRITE = "mcp__controls__channel_write"

# Channel-write states a rendered settings.json can be in.
DENIED = "denied"
ALLOWED = "allowed"
NO_PERMISSIONS = "no-permissions"


def _hook():
    """Import the drift hook module.

    Called from inside test bodies rather than at module scope so that importing
    this test file does nothing but define functions.
    """
    import osprey.templates.claude_code.claude.hooks.osprey_config_drift as hook

    return hook


class _RaisingPath:
    """A path that exists but whose reads and stats fail.

    Real filesystems produce this: the file can be unlinked, or its mount can go
    away, between the ``exists()`` probe and the ``stat()`` call that follows.
    """

    def exists(self) -> bool:
        return True

    def read_text(self, encoding: str = "utf-8") -> str:
        raise OSError("vanished")

    def stat(self):
        raise OSError("vanished")


# --------------------------------------------------------------------------- #
# Project fixtures
# --------------------------------------------------------------------------- #


def _make_project(project_dir, *, writes_enabled: bool | None, channel_write: str):
    """Write config.yml + .claude/settings.json in a chosen (possibly drifted) state.

    ``writes_enabled=None`` omits the key entirely; ``channel_write=NO_PERMISSIONS``
    renders a settings.json with no ``permissions`` block at all.
    """
    (project_dir / ".claude").mkdir(parents=True, exist_ok=True)
    config_path = project_dir / "config.yml"
    settings_path = project_dir / ".claude" / "settings.json"

    lines = ["control_system:", "  type: mock"]
    if writes_enabled is not None:
        lines.append(f"  writes_enabled: {'true' if writes_enabled else 'false'}")
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if channel_write == NO_PERMISSIONS:
        settings = {"hooks": {}}
    else:
        deny = ["Bash", "Edit"]
        if channel_write == DENIED:
            deny.append(CHANNEL_WRITE)
        settings = {"permissions": {"deny": deny}}
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return config_path, settings_path


def _set_mtimes(config_path, settings_path, *, config_newer: bool):
    """Pin mtimes so the mtime drift signal is deterministic."""
    base = 1_700_000_000
    if config_newer:
        os.utime(settings_path, (base, base))
        os.utime(config_path, (base + 100, base + 100))
    else:
        os.utime(config_path, (base, base))
        os.utime(settings_path, (base + 100, base + 100))


@pytest.fixture
def run_drift(hook_runner_raw, monkeypatch):
    """Run the drift hook as a SessionStart subprocess against a project dir."""

    def run(project_dir):
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(project_dir))
        return hook_runner_raw(
            HOOK_SCRIPT,
            tool_name=None,
            tool_input=None,
            stdin_override=json.dumps({"hook_event_name": "SessionStart", "source": "startup"}),
        )

    return run


# --------------------------------------------------------------------------- #
# _writes_enabled_in_config — regex parse of the kill-switch
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("control_system:\n  writes_enabled: true\n", True),
        ("control_system:\n  writes_enabled: false\n", False),
        # The regex is case-insensitive, so a hand-typed capital still reads.
        ("control_system:\n  WRITES_ENABLED: TRUE\n", True),
        # No key at all — undeterminable, not "false".
        ("control_system:\n  type: mock\n", None),
        # Commented out: `^\s*` refuses to skip the '#', so it does not count.
        ("control_system:\n  # writes_enabled: true\n", None),
        # Only the literals true/false are accepted; YAML's other booleans are not.
        ("control_system:\n  writes_enabled: yes\n", None),
    ],
    ids=["true", "false", "uppercase", "absent", "commented-out", "yaml-yes"],
)
def test_writes_enabled_in_config_parses(tmp_path, text, expected):
    config_path = tmp_path / "config.yml"
    config_path.write_text(text, encoding="utf-8")

    assert _hook()._writes_enabled_in_config(config_path) is expected


def test_writes_enabled_in_config_is_none_when_unreadable(tmp_path):
    # OSError (here: no such file) means undeterminable, never a guessed default.
    assert _hook()._writes_enabled_in_config(tmp_path / "nope.yml") is None


# --------------------------------------------------------------------------- #
# _channel_write_denied — JSON read of permissions.deny
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (json.dumps({"permissions": {"deny": ["Bash", CHANNEL_WRITE]}}), True),
        (json.dumps({"permissions": {"deny": ["Bash"]}}), False),
        (json.dumps({"permissions": {"allow": ["Bash"]}}), False),  # deny defaults to []
        (json.dumps({"hooks": {}}), None),  # no permissions block
        (json.dumps({"permissions": None}), None),  # hand-edited null
        (json.dumps({"permissions": {"deny": CHANNEL_WRITE}}), None),  # deny not a list
        (json.dumps(["not", "a", "mapping"]), None),
        ("{ this is not json", None),
    ],
    ids=[
        "denied",
        "allowed",
        "deny-key-absent",
        "permissions-absent",
        "permissions-null",
        "deny-not-a-list",
        "top-level-not-a-mapping",
        "malformed-json",
    ],
)
def test_channel_write_denied(tmp_path, payload, expected):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(payload, encoding="utf-8")

    assert _hook()._channel_write_denied(settings_path) is expected


def test_channel_write_denied_is_none_when_file_missing(tmp_path):
    assert _hook()._channel_write_denied(tmp_path / "settings.json") is None


# --------------------------------------------------------------------------- #
# _detect_drift — branch order and fail-open
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("missing", ["config", "settings"])
def test_detect_drift_is_none_when_an_artifact_is_missing(tmp_path, missing):
    # A project that was never built is not "drifted" — there is nothing to compare.
    config_path, settings_path = _make_project(tmp_path, writes_enabled=True, channel_write=DENIED)
    (config_path if missing == "config" else settings_path).unlink()

    assert _hook()._detect_drift(config_path, settings_path) is None


def test_detect_drift_fails_open_when_stat_raises():
    # Both signals blow up on IO: the hook returns no message rather than guessing.
    assert _hook()._detect_drift(_RaisingPath(), _RaisingPath()) is None


def test_detect_drift_prefers_the_targeted_check_over_mtime(tmp_path):
    # Both signals fire at once; the specific one wins so the operator is told
    # *what* drifted, not merely that something did.
    config_path, settings_path = _make_project(tmp_path, writes_enabled=True, channel_write=DENIED)
    _set_mtimes(config_path, settings_path, config_newer=True)

    message = _hook()._detect_drift(config_path, settings_path)
    assert "still blocks writes" in message
    assert "changed since" not in message


# --------------------------------------------------------------------------- #
# Behavioural drift matrix (subprocess)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("writes_enabled", "channel_write", "expected_fragment"),
    [
        # config wants writes, artifacts still deny them → the #244 case.
        (True, DENIED, "still blocks writes"),
        (True, ALLOWED, None),
        (True, NO_PERMISSIONS, None),  # deny state undeterminable → no claim
        (False, DENIED, None),
        # Safety-critical direction: config locked writes out, artifacts did not.
        (False, ALLOWED, "still permits writes"),
        (False, NO_PERMISSIONS, None),
        (None, DENIED, None),  # config state undeterminable → no claim
        (None, ALLOWED, None),
        (None, NO_PERMISSIONS, None),
    ],
    ids=[
        "on-denied-drift",
        "on-allowed-sync",
        "on-nopermissions-silent",
        "off-denied-sync",
        "off-allowed-drift",
        "off-nopermissions-silent",
        "absent-denied-silent",
        "absent-allowed-silent",
        "absent-nopermissions-silent",
    ],
)
def test_drift_matrix(tmp_path, run_drift, writes_enabled, channel_write, expected_fragment):
    config_path, settings_path = _make_project(
        tmp_path, writes_enabled=writes_enabled, channel_write=channel_write
    )
    # Settings newer than config, so the mtime signal never fires and every
    # warning below is attributable to the targeted writes_enabled check.
    _set_mtimes(config_path, settings_path, config_newer=False)

    code, out, err = run_drift(tmp_path)

    assert code == 0
    if expected_fragment is None:
        assert err.strip() == ""
        assert out.strip() == ""
    else:
        assert expected_fragment in err
        assert expected_fragment in json.loads(out)["hookSpecificOutput"]["additionalContext"]


def test_warns_on_writes_mismatch(tmp_path, run_drift):
    # config says writes enabled, but artifacts still deny the write → #244 case.
    config_path, settings_path = _make_project(tmp_path, writes_enabled=True, channel_write=DENIED)
    # Even with settings newer (mtime in sync), the targeted check still fires.
    _set_mtimes(config_path, settings_path, config_newer=False)

    code, out, err = run_drift(tmp_path)
    assert code == 0
    assert "writes_enabled: true" in err
    # The remedy the hint names, not the word it happens to use for it.
    assert "osprey build" in err
    # Also surfaced as SessionStart additionalContext on stdout.
    payload = json.loads(out)
    assert "writes_enabled: true" in payload["hookSpecificOutput"]["additionalContext"]


def test_warns_when_config_disables_writes_but_agent_permits(tmp_path, run_drift):
    # The safety-critical direction: config.yml turned writes OFF, but the stale
    # artifacts still permit them — the agent could write hardware the operator
    # believes is locked out.
    config_path, settings_path = _make_project(
        tmp_path, writes_enabled=False, channel_write=ALLOWED
    )
    # Settings newer than config: the targeted check must fire on its own.
    _set_mtimes(config_path, settings_path, config_newer=False)

    code, out, err = run_drift(tmp_path)
    assert code == 0
    assert "writes_enabled: false" in err
    assert "permits writes" in err
    assert "osprey build" in err
    payload = json.loads(out)
    assert "permits writes" in payload["hookSpecificOutput"]["additionalContext"]


def test_warns_on_mtime_drift(tmp_path, run_drift):
    # writes_enabled is consistent, but config.yml was edited after the last build.
    config_path, settings_path = _make_project(tmp_path, writes_enabled=False, channel_write=DENIED)
    _set_mtimes(config_path, settings_path, config_newer=True)

    code, _out, err = run_drift(tmp_path)
    assert code == 0
    assert "changed since" in err
    assert "osprey build" in err


def test_warning_is_mirrored_to_stderr_and_session_context(tmp_path, run_drift):
    # The transcript line and the injected context must be the same warning, so
    # the agent cannot relay something the operator does not also see.
    config_path, settings_path = _make_project(tmp_path, writes_enabled=True, channel_write=DENIED)
    _set_mtimes(config_path, settings_path, config_newer=False)

    code, out, err = run_drift(tmp_path)
    assert code == 0
    payload = json.loads(out)
    assert payload["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert payload["hookSpecificOutput"]["additionalContext"] == err.strip()
    assert err.strip().startswith("⚠")


def test_warns_on_config_that_pyyaml_cannot_parse(tmp_path, run_drift):
    # The hook must not depend on a YAML parser: under a raw `claude` launch the
    # interpreter running it may have no PyYAML at all. A config broken elsewhere
    # in the file still yields the kill-switch line by regex.
    (tmp_path / ".claude").mkdir(parents=True)
    config_path = tmp_path / "config.yml"
    settings_path = tmp_path / ".claude" / "settings.json"
    broken = "control_system:\n  writes_enabled: true\n  channels: [unclosed\n\t- tab-indented\n"
    config_path.write_text(broken, encoding="utf-8")
    settings_path.write_text(
        json.dumps({"permissions": {"deny": [CHANNEL_WRITE]}}), encoding="utf-8"
    )
    _set_mtimes(config_path, settings_path, config_newer=False)

    with pytest.raises(yaml.YAMLError):
        yaml.safe_load(broken)

    code, out, err = run_drift(tmp_path)
    assert code == 0
    assert "still blocks writes" in err
    assert "still blocks writes" in json.loads(out)["hookSpecificOutput"]["additionalContext"]


# --------------------------------------------------------------------------- #
# Fail-open paths (subprocess)
# --------------------------------------------------------------------------- #


def test_fails_open_when_artifacts_missing(tmp_path, run_drift):
    # No config.yml / settings.json at all → silent, never blocks the session.
    code, out, err = run_drift(tmp_path)
    assert code == 0
    assert err.strip() == ""
    assert out.strip() == ""


def test_fails_open_on_malformed_permissions(tmp_path, run_drift):
    # Hand-edited settings.json with `permissions: null` must not crash the targeted
    # check; it falls back to the mtime signal (here pinned in-sync → silent).
    (tmp_path / ".claude").mkdir(parents=True)
    config_path = tmp_path / "config.yml"
    settings_path = tmp_path / ".claude" / "settings.json"
    config_path.write_text("control_system:\n  writes_enabled: true\n", encoding="utf-8")
    settings_path.write_text(json.dumps({"permissions": None}), encoding="utf-8")
    _set_mtimes(config_path, settings_path, config_newer=False)

    code, out, err = run_drift(tmp_path)
    assert code == 0
    assert err.strip() == ""
    assert out.strip() == ""


@pytest.mark.skipif(
    os.name != "posix" or os.geteuid() == 0,
    reason="needs POSIX permissions and a non-root user to make a file unreadable",
)
def test_fails_open_when_settings_unreadable(tmp_path, run_drift):
    # An unreadable settings.json would look like "writes permitted" to a hook
    # that treated a failed read as absence. It must stay silent instead: the
    # deny list is undeterminable, so there is nothing honest to warn about.
    config_path, settings_path = _make_project(
        tmp_path, writes_enabled=False, channel_write=ALLOWED
    )
    _set_mtimes(config_path, settings_path, config_newer=False)
    os.chmod(settings_path, 0o000)

    code, out, err = run_drift(tmp_path)
    os.chmod(settings_path, stat.S_IRUSR | stat.S_IWUSR)

    assert code == 0
    assert err.strip() == ""
    assert out.strip() == ""


@pytest.mark.parametrize("stdin", ["", "{nope", "[]"], ids=["empty", "invalid-json", "wrong-shape"])
def test_stdin_is_never_read(tmp_path, hook_runner_raw, monkeypatch, stdin):
    # SessionStart hands the hook a JSON payload it does not use: the project
    # root comes from CLAUDE_PROJECT_DIR and everything else from the two files.
    # A closed pipe, a truncated write and a payload of the wrong JSON type all
    # have to produce the same warning as a well-formed one — malformed stdin
    # must neither crash the hook nor silence a real drift.
    config_path, settings_path = _make_project(tmp_path, writes_enabled=True, channel_write=DENIED)
    _set_mtimes(config_path, settings_path, config_newer=False)
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path))

    code, out, err = hook_runner_raw(
        HOOK_SCRIPT,
        tool_name=None,
        tool_input=None,
        stdin_override=stdin,
    )

    assert code == 0
    assert "Traceback" not in err
    assert "still blocks writes" in json.loads(out)["hookSpecificOutput"]["additionalContext"]
