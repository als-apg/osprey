"""Tests for the osprey_config_drift.py SessionStart hook.

The hook warns (never blocks) when config.yml has drifted from the rendered
.claude/settings.json. Two layers of coverage:

* The pure helpers (``_tier_in_config``, ``_kill_switch_tools``,
  ``_write_tools_denied``, ``_detect_drift``) are imported and called directly,
  one test per documented return branch. The hook is stdlib-only and imports nothing at module scope, so
  a plain import is safe — it is still written inside the test bodies so that
  importing this file stays free of side effects (tests/README.md §5).
* Parity with the renderer: a real ``never`` / ``allowed`` render's
  ``hook_config.json`` and ``permissions.deny`` are fed to the hook, so the set
  it compares is the set the kill switch actually denies.
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

# What the same tool is called on an `extends: controls` clone. A deployment
# whose controls server is a clone denies THIS and nothing named `controls`, so
# a hook keyed on the framework literal claims nothing about its posture while
# looking like it had checked.
CLONE_CHANNEL_WRITE = "mcp__ring__channel_write"

# A per-call-gated write tool a project appends via `control_system.write_tools`.
# The renderer folds it into `write_tools` but the kill switch never denies it, so
# a hook that expects to find it in `permissions.deny` sees a partial overlap and
# abstains — on every deployment that sets the key.
SITE_CUSTOM_WRITE = "mcp__site__custom_write"

# A facility server's declared write tool, from the render's facility_write_tools.
FACILITY_TOOL = "mcp__plc__set_value"

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

    def read_text(self, encoding: str = "utf-8") -> str:  # noqa: ARG002 - Path.read_text takes encoding by keyword
        raise OSError("vanished")

    def stat(self):
        raise OSError("vanished")


# --------------------------------------------------------------------------- #
# Project fixtures
# --------------------------------------------------------------------------- #


def _make_project(
    project_dir,
    *,
    writes_enabled: bool | None = None,
    channel_write: str,
    writes: str | None = None,
    write_tool: str = CHANNEL_WRITE,
    extras: tuple[str, ...] = (),
):
    """Write config.yml + .claude/settings.json in a chosen (possibly drifted) state.

    ``writes`` is the tier (``control_system.writes``) spelled verbatim;
    ``writes_enabled`` is the retired flat leaf. ``None`` omits a key entirely;
    ``channel_write=NO_PERMISSIONS`` renders a settings.json with no
    ``permissions`` block at all.
    """
    lines = [
        "control_system:",
        "  type: virtual_accelerator",
        "  connector:",
        "    virtual_accelerator:",
        "      serving: in_process",
    ]
    if writes is not None:
        lines.append(f"  writes: {writes}")
    if writes_enabled is not None:
        lines.append(f"  writes_enabled: {'true' if writes_enabled else 'false'}")
    return _make_project_from_text(
        project_dir,
        "\n".join(lines) + "\n",
        channel_write=channel_write,
        write_tool=write_tool,
        extras=extras,
    )


def _make_project_from_text(
    project_dir,
    config_text: str,
    *,
    channel_write: str,
    write_tool: str = CHANNEL_WRITE,
    extras: tuple[str, ...] = (),
    facility: tuple[str, ...] = (),
):
    """Same, for a config.yml whose exact text matters (shapes a bool cannot express).

    ``write_tool`` names the deployment's own write tool: the framework's by
    default, a clone's when the test is about a clone. It is written into both
    ``hook_config.json`` (where the hook reads the set from) and
    ``permissions.deny`` (where the kill switch puts it).

    ``extras`` names the tools the project appended via
    ``control_system.write_tools``: the renderer folds them into ``write_tools``
    and repeats them under ``control_system_write_tools``, and the kill switch
    leaves them out of ``permissions.deny``.

    ``facility`` names the render's ``facility_write_tools`` (hook matchers): the
    gated rules land in ``write_tools`` and the ``never`` kill switch denies each,
    a whole-server ``mcp__<srv>__.*`` as ``mcp__<srv>``.
    """
    (project_dir / ".claude" / "hooks").mkdir(parents=True, exist_ok=True)
    config_path = project_dir / "config.yml"
    settings_path = project_dir / ".claude" / "settings.json"
    hook_config_path = project_dir / ".claude" / "hooks" / "hook_config.json"

    config_path.write_text(config_text, encoding="utf-8")
    hook_config_path.write_text(
        json.dumps(
            {
                "write_tools": [write_tool, *facility, *extras],
                "control_system_write_tools": list(extras),
                "facility_write_tools": list(facility),
            }
        ),
        encoding="utf-8",
    )

    if channel_write == NO_PERMISSIONS:
        settings = {"hooks": {}}
    else:
        deny = ["Bash", "Edit"]
        if channel_write == DENIED:
            deny.append(write_tool)
            deny.extend(_as_permission(m) for m in facility)
        settings = {"permissions": {"deny": deny}}
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return config_path, settings_path


def _as_permission(matcher: str) -> str:
    """A facility hook matcher as the kill switch denies it."""
    return matcher[: -len("__.*")] if matcher.endswith("__.*") else matcher


# Retired leaves only: a flat `false` beside a connector block's `true`. No tier.
MIXED_POSTURE_CONFIG = """\
control_system:
  type: virtual_accelerator
  writes_enabled: false
  connector:
    virtual_accelerator:
      serving: in_process
      writes_enabled: true
"""

# Retired leaves only, the second carrying no value at all. No tier.
BARE_SECOND_KEY_CONFIG = """\
control_system:
  type: virtual_accelerator
  writes_enabled: true
  connector:
    virtual_accelerator:
      serving: in_process
      writes_enabled:
"""


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
# _tier_in_config — regex parse of the tier, scoped to control_system
# --------------------------------------------------------------------------- #

# A facility server that declares its writes. Its `writes:` list sits under
# `claude_code.servers`, not under `control_system`, and says nothing about the
# tier: a section-blind `writes:` match would read it as one.
FACILITY_WRITES_LIST_CONFIG = """\
claude_code:
  servers:
    plc:
      command: node
      writes:
        - set_value
      permissions:
        ask: [set_value]
control_system:
  type: epics
  writes: {tier}
"""


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("control_system:\n  writes: allowed\n", (True, "control_system.writes: allowed")),
        ("control_system:\n  writes: never\n", (False, "control_system.writes: never")),
        # Quotes do not change the string.
        ("control_system:\n  writes: 'allowed'\n", (True, "control_system.writes: allowed")),
        ('control_system:\n  writes: "never"\n', (False, "control_system.writes: never")),
        # A trailing comment is not part of the value.
        (
            "control_system:\n  writes: allowed  # armed for MD\n",
            (True, "control_system.writes: allowed"),
        ),
        # Only the literal `allowed` arms, as in `tier_writes`: a capital, a
        # boolean, any other word all answer never.
        ("control_system:\n  writes: Allowed\n", (False, "control_system.writes: Allowed")),
        ("control_system:\n  writes: true\n", (False, "control_system.writes: true")),
        # The tier decides; a retired leaf beside it is not read.
        (
            "control_system:\n  writes: never\n  writes_enabled: true\n",
            (False, "control_system.writes: never"),
        ),
        (
            "control_system:\n  writes_enabled: false\n  writes: allowed\n",
            (True, "control_system.writes: allowed"),
        ),
    ],
    ids=[
        "allowed",
        "never",
        "single-quoted",
        "double-quoted",
        "trailing-comment",
        "capital-is-never",
        "boolean-is-never",
        "tier-beats-retired-true",
        "tier-beats-retired-false",
    ],
)
def test_tier_in_config_parses(tmp_path, text, expected):
    config_path = tmp_path / "config.yml"
    config_path.write_text(text, encoding="utf-8")

    assert _hook()._tier_in_config(config_path) == expected


@pytest.mark.parametrize(
    "text",
    [
        # No key at all — undeterminable, not "never".
        "control_system:\n  type: virtual_accelerator\n  connector:\n    virtual_accelerator:\n"
        "      serving: in_process\n",
        # Commented out: a comment line is not a key.
        "control_system:\n  # writes: allowed\n",
        # The tier stated twice: YAML's answer is not the regex's to guess.
        "control_system:\n  writes: never\n  writes: allowed\n",
        # A value the regex cannot read for certain.
        "control_system:\n  writes:\n",
        "control_system:\n  writes: ${OSPREY_TIER}\n",
        "control_system:\n  writes: *tier\n",
        "control_system:\n  writes: [allowed]\n",
        # Retired leaves alone state no tier, as the kill switch reads them: the
        # flat leaf, a connector block's, and both together all abstain.
        "control_system:\n  writes_enabled: true\n",
        "control_system:\n  writes_enabled: false\n",
        "control_system:\n  writes_enabled: false\n  connector:\n    epics:\n"
        "      writes_enabled: true\n",
        "control_system:\n  connector:\n    epics:\n      writes_enabled: false\n",
        # Keys outside control_system are never the tier.
        "writes: allowed\nother:\n  writes: allowed\n  writes_enabled: true\n",
        # Nested deeper than the section's own child: not the tier.
        "control_system:\n  connector:\n    epics:\n      writes: allowed\n",
        # A key inside a sequence entry is not a mapping path.
        "control_system:\n  limits:\n    - channel: X\n      writes_enabled: true\n",
        # A key-looking line inside a block scalar is text.
        "control_system:\n  notes: |\n    writes: allowed\n",
        # A flow mapping is not walked.
        "control_system: {writes: allowed}\n",
    ],
    ids=[
        "absent",
        "commented-out",
        "duplicated",
        "bare",
        "env-var",
        "alias",
        "flow-sequence",
        "retired-flat-true",
        "retired-flat-false",
        "retired-connector-true",
        "retired-connector-false",
        "outside-the-section",
        "too-deep",
        "inside-a-list",
        "inside-a-block-scalar",
        "flow-mapping",
    ],
)
def test_tier_in_config_abstains(tmp_path, text):
    config_path = tmp_path / "config.yml"
    config_path.write_text(text, encoding="utf-8")

    assert _hook()._tier_in_config(config_path) is None


@pytest.mark.parametrize(
    ("tier", "expected"), [("allowed", True), ("never", False)], ids=["allowed", "never"]
)
def test_tier_in_config_ignores_a_facility_servers_writes_list(tmp_path, tier, expected):
    """A facility `writes:` list above the section never reads as the tier."""
    config_path = tmp_path / "config.yml"
    config_path.write_text(FACILITY_WRITES_LIST_CONFIG.format(tier=tier), encoding="utf-8")

    assert _hook()._tier_in_config(config_path) == (expected, f"control_system.writes: {tier}")


def test_tier_in_config_is_none_with_only_a_facility_writes_list(tmp_path):
    """With the tier line gone, the facility list is all that says `writes:`."""
    config_path = tmp_path / "config.yml"
    text = FACILITY_WRITES_LIST_CONFIG.format(tier="never").replace("  writes: never\n", "")
    config_path.write_text(text, encoding="utf-8")

    assert _hook()._tier_in_config(config_path) is None


def test_tier_in_config_is_none_when_unreadable(tmp_path):
    # OSError (here: no such file) means undeterminable, never a guessed default.
    assert _hook()._tier_in_config(tmp_path / "nope.yml") is None


# --------------------------------------------------------------------------- #
# _kill_switch_tools — which tools this render's kill switch denies
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (json.dumps({"write_tools": [CLONE_CHANNEL_WRITE]}), (CLONE_CHANNEL_WRITE,)),
        (
            json.dumps({"write_tools": [CHANNEL_WRITE, CLONE_CHANNEL_WRITE]}),
            (CHANNEL_WRITE, CLONE_CHANNEL_WRITE),
        ),
        # A whole facility-custom server names no tool that could be in deny.
        (json.dumps({"write_tools": [CHANNEL_WRITE, "mcp__site__.*"]}), (CHANNEL_WRITE,)),
        # A mixed read/write tool keeps its reads with writes off, so it is not
        # denied outright and says nothing about the posture.
        (
            json.dumps(
                {
                    "write_tools": [CHANNEL_WRITE, "mcp__bluesky__queue_add"],
                    "mixed_read_write_tools": ["mcp__bluesky__queue_add"],
                }
            ),
            (CHANNEL_WRITE,),
        ),
        # A tool the project appended via `control_system.write_tools` is gated
        # per call, not by the kill switch, so it is never in deny either.
        (
            json.dumps(
                {
                    "write_tools": [CHANNEL_WRITE, SITE_CUSTOM_WRITE],
                    "control_system_write_tools": [SITE_CUSTOM_WRITE],
                }
            ),
            (CHANNEL_WRITE,),
        ),
        # A render from before the key existed: nothing to subtract, same answer
        # as that render's hook gave.
        (
            json.dumps({"write_tools": [CHANNEL_WRITE, CLONE_CHANNEL_WRITE]}),
            (CHANNEL_WRITE, CLONE_CHANNEL_WRITE),
        ),
        # The facility set the never kill switch denies: a declared tool exactly
        # (already in write_tools, so not repeated) and a list-less server whole,
        # as the permission rule `mcp__<srv>` the deny carries.
        (
            json.dumps(
                {
                    "write_tools": [CHANNEL_WRITE, FACILITY_TOOL, "mcp__sitectl__.*"],
                    "facility_write_tools": [FACILITY_TOOL, "mcp__sitectl__.*"],
                }
            ),
            (CHANNEL_WRITE, FACILITY_TOOL, "mcp__sitectl"),
        ),
        # Facility entries are denied whatever the other lists say, so neither
        # exclusion list subtracts them.
        (
            json.dumps(
                {
                    "write_tools": [CHANNEL_WRITE],
                    "facility_write_tools": [FACILITY_TOOL],
                    "control_system_write_tools": [FACILITY_TOOL],
                }
            ),
            (CHANNEL_WRITE, FACILITY_TOOL),
        ),
        # A facility-only render: the framework tools are disabled.
        (
            json.dumps({"write_tools": [], "facility_write_tools": ["mcp__rawctl__.*"]}),
            ("mcp__rawctl",),
        ),
        (
            json.dumps({"write_tools": [CHANNEL_WRITE], "facility_write_tools": "mcp__x__.*"}),
            (CHANNEL_WRITE,),
        ),
        (
            json.dumps({"write_tools": [CHANNEL_WRITE], "facility_write_tools": [1, None]}),
            (CHANNEL_WRITE,),
        ),
        (json.dumps({"write_tools": []}), ()),
        (json.dumps({"write_tools": [1, None]}), ()),
    ],
    ids=[
        "clone-only",
        "framework-and-clone",
        "wildcard-dropped",
        "mixed-dropped",
        "extras-dropped",
        "extras-key-absent",
        "facility-exact-and-whole-server",
        "facility-never-excluded",
        "facility-only",
        "facility-not-a-list",
        "facility-junk",
        "empty",
        "junk",
    ],
)
def test_kill_switch_tools(tmp_path, payload, expected):
    hook_config_path = tmp_path / "hook_config.json"
    hook_config_path.write_text(payload, encoding="utf-8")

    assert _hook()._kill_switch_tools(hook_config_path) == expected


@pytest.mark.parametrize(
    "payload",
    [None, "{ this is not json", json.dumps(["not", "a", "mapping"]), json.dumps({})],
    ids=["file-missing", "malformed-json", "top-level-not-a-mapping", "key-absent"],
)
def test_kill_switch_tools_falls_back_to_the_framework_tool(tmp_path, payload):
    """A render without a readable hook_config still gets the old check.

    Falling back to nothing would silently retire the check on exactly the
    deployments least likely to notice.
    """
    hook_config_path = tmp_path / "hook_config.json"
    if payload is not None:
        hook_config_path.write_text(payload, encoding="utf-8")

    assert _hook()._kill_switch_tools(hook_config_path) == (CHANNEL_WRITE,)


# --------------------------------------------------------------------------- #
# _write_tools_denied — JSON read of permissions.deny
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
def test_write_tools_denied(tmp_path, payload, expected):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(payload, encoding="utf-8")

    assert _hook()._write_tools_denied(settings_path, (CHANNEL_WRITE,)) is expected


def test_write_tools_denied_is_none_when_file_missing(tmp_path):
    assert _hook()._write_tools_denied(tmp_path / "settings.json", (CHANNEL_WRITE,)) is None


def test_write_tools_denied_abstains_on_an_empty_tool_set(tmp_path):
    """Nothing to compare is not "writes are allowed"."""
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"permissions": {"deny": []}}), encoding="utf-8")

    assert _hook()._write_tools_denied(settings_path, ()) is None


def test_write_tools_denied_abstains_on_a_partial_deny(tmp_path):
    """The kill switch renders the whole set together, so half of it is not a posture.

    A hand-edited settings.json is what produces this, and the mtime check is
    the signal for that — warning about the write posture would name the wrong
    control.
    """
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({"permissions": {"deny": [CHANNEL_WRITE]}}), encoding="utf-8"
    )

    assert _hook()._write_tools_denied(settings_path, (CHANNEL_WRITE, CLONE_CHANNEL_WRITE)) is None


# --------------------------------------------------------------------------- #
# _detect_drift — branch order and fail-open
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("missing", ["config", "settings"])
def test_detect_drift_is_none_when_an_artifact_is_missing(tmp_path, missing):
    # A project that was never built is not "drifted" — there is nothing to compare.
    config_path, settings_path = _make_project(tmp_path, writes="allowed", channel_write=DENIED)
    (config_path if missing == "config" else settings_path).unlink()

    hook_config_path = tmp_path / ".claude" / "hooks" / "hook_config.json"
    assert _hook()._detect_drift(config_path, settings_path, hook_config_path) is None


def test_detect_drift_fails_open_when_stat_raises():
    # Both signals blow up on IO: the hook returns no message rather than guessing.
    assert _hook()._detect_drift(_RaisingPath(), _RaisingPath(), _RaisingPath()) is None


def test_detect_drift_prefers_the_targeted_check_over_mtime(tmp_path):
    # Both signals fire at once; the specific one wins so the operator is told
    # *what* drifted, not merely that something did.
    config_path, settings_path = _make_project(tmp_path, writes="allowed", channel_write=DENIED)
    _set_mtimes(config_path, settings_path, config_newer=True)

    hook_config_path = tmp_path / ".claude" / "hooks" / "hook_config.json"
    message = _hook()._detect_drift(config_path, settings_path, hook_config_path)
    assert "still blocks writes" in message
    assert "changed since" not in message


def test_detect_drift_sees_a_clone_the_framework_literal_would_miss(tmp_path):
    """The #244 case on a deployment whose controls server is an `extends` clone.

    Its kill switch denies `mcp__ring__channel_write`; nothing named `controls`
    appears anywhere in the render. Probing the framework literal read that as
    "not denied", so a config that had turned writes back ON matched a settings
    file that still blocked them, and the operator was told nothing.
    """
    config_path, settings_path = _make_project(
        tmp_path, writes="allowed", channel_write=DENIED, write_tool=CLONE_CHANNEL_WRITE
    )
    _set_mtimes(config_path, settings_path, config_newer=False)
    hook_config_path = tmp_path / ".claude" / "hooks" / "hook_config.json"

    message = _hook()._detect_drift(config_path, settings_path, hook_config_path)
    assert message is not None
    assert "still blocks writes" in message


# --------------------------------------------------------------------------- #
# Behavioural drift matrix (subprocess)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("writes", "channel_write", "expected_fragment"),
    [
        # config wants writes, artifacts still deny them → the #244 case.
        ("allowed", DENIED, "still blocks writes"),
        ("allowed", ALLOWED, None),
        ("allowed", NO_PERMISSIONS, None),  # deny state undeterminable → no claim
        ("never", DENIED, None),
        # Safety-critical direction: config locked writes out, artifacts did not.
        ("never", ALLOWED, "still permits writes"),
        ("never", NO_PERMISSIONS, None),
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
def test_drift_matrix(tmp_path, run_drift, writes, channel_write, expected_fragment):
    config_path, settings_path = _make_project(tmp_path, writes=writes, channel_write=channel_write)
    # Settings newer than config, so the mtime signal never fires and every
    # warning below is attributable to the targeted tier check.
    _set_mtimes(config_path, settings_path, config_newer=False)

    code, out, err = run_drift(tmp_path)

    assert code == 0
    if expected_fragment is None:
        assert err.strip() == ""
        assert out.strip() == ""
    else:
        assert expected_fragment in err
        assert expected_fragment in json.loads(out)["hookSpecificOutput"]["additionalContext"]


@pytest.mark.parametrize("channel_write", [DENIED, ALLOWED], ids=["denied", "allowed"])
@pytest.mark.parametrize(
    "config_text",
    [MIXED_POSTURE_CONFIG, BARE_SECOND_KEY_CONFIG],
    ids=["mixed-retired-leaves", "bare-second-leaf"],
)
def test_retired_leaves_alone_claim_nothing(tmp_path, run_drift, config_text, channel_write):
    # A config that states no tier, only retired `writes_enabled` leaves. The kill
    # switch reads no retired leaf, so neither does this hook: it abstains against
    # a render that denies and one that permits alike, and check (a) is left to
    # cover the config.
    config_path, settings_path = _make_project_from_text(
        tmp_path, config_text, channel_write=channel_write
    )
    # Settings newer than config, so any output here is the writes check, not mtime.
    _set_mtimes(config_path, settings_path, config_newer=False)

    code, out, err = run_drift(tmp_path)

    assert code == 0
    assert err.strip() == ""
    assert out.strip() == ""


@pytest.mark.parametrize(
    ("writes", "channel_write", "expected_fragment"),
    [
        ("allowed", DENIED, "still blocks writes"),
        ("allowed", ALLOWED, None),
        ("never", DENIED, None),
        # Safety-critical direction: the tier forbids writes, the render does not.
        ("never", ALLOWED, "still permits writes"),
        ("never", NO_PERMISSIONS, None),
        # Not the literal `allowed`: the kill switch denied, so the render agrees.
        ("Allowed", DENIED, None),
        ("Allowed", ALLOWED, "still permits writes"),
    ],
    ids=[
        "allowed-denied-drift",
        "allowed-allowed-sync",
        "never-denied-sync",
        "never-allowed-drift",
        "never-nopermissions-silent",
        "capital-denied-sync",
        "capital-allowed-drift",
    ],
)
def test_tier_drift_matrix(tmp_path, run_drift, writes, channel_write, expected_fragment):
    # A retired leaf saying the opposite rides along: the tier alone decides.
    config_path, settings_path = _make_project(
        tmp_path,
        writes=writes,
        writes_enabled=writes != "allowed",
        channel_write=channel_write,
    )
    _set_mtimes(config_path, settings_path, config_newer=False)

    code, out, err = run_drift(tmp_path)

    assert code == 0
    if expected_fragment is None:
        assert err.strip() == ""
        assert out.strip() == ""
    else:
        assert expected_fragment in err
        assert f"control_system.writes: {writes}" in err
        assert expected_fragment in json.loads(out)["hookSpecificOutput"]["additionalContext"]


@pytest.mark.parametrize(
    ("tier", "channel_write", "expected_fragment"),
    [
        ("never", DENIED, None),
        ("never", ALLOWED, "still permits writes"),
        ("allowed", DENIED, "still blocks writes"),
        ("allowed", ALLOWED, None),
    ],
    ids=["never-sync", "never-drift", "allowed-drift", "allowed-sync"],
)
def test_a_facility_servers_writes_list_and_tools_count(
    tmp_path, run_drift, tier, channel_write, expected_fragment
):
    """A deployment with a declaring facility server and a list-less one.

    The config's facility `writes:` list must not be read as the tier, and the
    facility tools the never kill switch denies — the declared one exactly, the
    list-less server whole as `mcp__<srv>` — are part of the compared set, so a
    render that denies them reads as `denied`, not as a partial overlap.
    """
    config_path, settings_path = _make_project_from_text(
        tmp_path,
        FACILITY_WRITES_LIST_CONFIG.format(tier=tier),
        channel_write=channel_write,
        facility=(FACILITY_TOOL, "mcp__sitectl__.*"),
        extras=(SITE_CUSTOM_WRITE,),
    )
    _set_mtimes(config_path, settings_path, config_newer=False)

    code, out, err = run_drift(tmp_path)

    assert code == 0
    if expected_fragment is None:
        assert err.strip() == ""
        assert out.strip() == ""
    else:
        assert expected_fragment in err
        assert f"control_system.writes: {tier}" in err


def test_a_render_that_denies_only_the_framework_tool_is_not_a_posture(tmp_path, run_drift):
    """Facility tools left out of a deny the framework tool is in: a hand edit.

    The never kill switch denies the facility set together with the framework
    set, so a deny holding one and not the other is the partial overlap the
    targeted check abstains on.
    """
    config_path, settings_path = _make_project_from_text(
        tmp_path,
        FACILITY_WRITES_LIST_CONFIG.format(tier="allowed"),
        channel_write=DENIED,
        facility=(FACILITY_TOOL,),
    )
    settings_path.write_text(
        json.dumps({"permissions": {"deny": ["Bash", CHANNEL_WRITE]}}), encoding="utf-8"
    )
    _set_mtimes(config_path, settings_path, config_newer=False)

    code, out, err = run_drift(tmp_path)

    assert code == 0
    assert err.strip() == ""
    assert out.strip() == ""


def test_warns_on_writes_mismatch(tmp_path, run_drift):
    # config says writes enabled, but artifacts still deny the write → #244 case.
    config_path, settings_path = _make_project(tmp_path, writes="allowed", channel_write=DENIED)
    # Even with settings newer (mtime in sync), the targeted check still fires.
    _set_mtimes(config_path, settings_path, config_newer=False)

    code, out, err = run_drift(tmp_path)
    assert code == 0
    assert "control_system.writes: allowed" in err
    # The remedy the hint names, not the word it happens to use for it.
    assert "osprey build" in err
    # Also surfaced as SessionStart additionalContext on stdout.
    payload = json.loads(out)
    assert "control_system.writes: allowed" in payload["hookSpecificOutput"]["additionalContext"]


def test_warns_on_writes_mismatch_when_the_project_appends_write_tools(tmp_path, run_drift):
    # Same drift, on a deployment that sets `control_system.write_tools`. Those
    # extras are gated per call and never enter `permissions.deny`, so counting
    # them as kill-switch tools makes every such deployment look like a
    # hand-edited settings.json and the check goes quiet — the signal is lost
    # exactly where the config says writes are on and the artifacts still block.
    config_path, settings_path = _make_project(
        tmp_path, writes="allowed", channel_write=DENIED, extras=(SITE_CUSTOM_WRITE,)
    )
    _set_mtimes(config_path, settings_path, config_newer=False)

    code, out, err = run_drift(tmp_path)

    assert code == 0
    assert "control_system.writes: allowed" in err
    assert "blocks writes" in err
    payload = json.loads(out)
    assert "blocks writes" in payload["hookSpecificOutput"]["additionalContext"]


def test_warns_when_config_disables_writes_but_agent_permits(tmp_path, run_drift):
    # The safety-critical direction: config.yml turned writes OFF, but the stale
    # artifacts still permit them — the agent could write hardware the operator
    # believes is locked out.
    config_path, settings_path = _make_project(tmp_path, writes="never", channel_write=ALLOWED)
    # Settings newer than config: the targeted check must fire on its own.
    _set_mtimes(config_path, settings_path, config_newer=False)

    code, out, err = run_drift(tmp_path)
    assert code == 0
    assert "control_system.writes: never" in err
    assert "permits writes" in err
    assert "osprey build" in err
    payload = json.loads(out)
    assert "permits writes" in payload["hookSpecificOutput"]["additionalContext"]


def test_warns_when_writes_are_off_and_permitted_with_appended_write_tools(tmp_path, run_drift):
    # The writes-off direction already warned with extras present — an empty
    # overlap is not a partial one — and must keep warning after the extras are
    # subtracted.
    config_path, settings_path = _make_project(
        tmp_path, writes="never", channel_write=ALLOWED, extras=(SITE_CUSTOM_WRITE,)
    )
    _set_mtimes(config_path, settings_path, config_newer=False)

    code, out, err = run_drift(tmp_path)

    assert code == 0
    assert "control_system.writes: never" in err
    assert "permits writes" in err


def test_warns_on_mtime_drift(tmp_path, run_drift):
    # The tier is consistent, but config.yml was edited after the last build.
    config_path, settings_path = _make_project(tmp_path, writes="never", channel_write=DENIED)
    _set_mtimes(config_path, settings_path, config_newer=True)

    code, _out, err = run_drift(tmp_path)
    assert code == 0
    assert "changed since" in err
    assert "osprey build" in err


def test_warning_is_mirrored_to_stderr_and_session_context(tmp_path, run_drift):
    # The transcript line and the injected context must be the same warning, so
    # the agent cannot relay something the operator does not also see.
    config_path, settings_path = _make_project(tmp_path, writes="allowed", channel_write=DENIED)
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
    broken = "control_system:\n  writes: allowed\n  channels: [unclosed\n\t- tab-indented\n"
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
    config_path.write_text("control_system:\n  writes: allowed\n", encoding="utf-8")
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
    config_path, settings_path = _make_project(tmp_path, writes="never", channel_write=ALLOWED)
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
    config_path, settings_path = _make_project(tmp_path, writes="allowed", channel_write=DENIED)
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


# --------------------------------------------------------------------------- #
# Parity with the renderer — the compared set is the set the kill switch denies
# --------------------------------------------------------------------------- #

_HOOK_CONFIG_TEMPLATE = "claude_code/claude/hooks/hook_config.json.j2"


def _render(tmp_path, *, writes: str):
    """Render a control-assistant project's hook_config and deny at ``writes``.

    The servers are a declaring facility server, a list-less writes_check one
    (with and without approval) and Bluesky; the project also appends a
    ``control_system.write_tools`` entry. Returns the paths of config.yml and of
    the rendered hook_config.json and settings.json.
    """
    from osprey.cli.templates import claude_code
    from tests.cli.test_killswitch_on_tier import _FACILITY, _project

    manager, project_dir, _config = _project(
        tmp_path,
        writes=writes,
        servers=_FACILITY,
        write_tools=["mcp__plc__get_value"],
    )
    ctx = claude_code.build_claude_code_context(
        manager.template_root, manager.jinja_env, project_dir, _config
    )
    hooks_dir = project_dir / ".claude" / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook_config_path = hooks_dir / "hook_config.json"
    hook_config_path.write_text(
        manager.jinja_env.get_template(_HOOK_CONFIG_TEMPLATE).render(**ctx), encoding="utf-8"
    )
    settings_path = project_dir / ".claude" / "settings.json"
    settings_path.write_text(
        json.dumps({"permissions": {"deny": claude_code._rendered_deny_list(ctx)}}),
        encoding="utf-8",
    )
    return project_dir / "config.yml", hook_config_path, settings_path, ctx


@pytest.mark.parametrize(
    ("writes", "edited", "fragment"),
    [("never", "allowed", "still blocks writes"), ("allowed", "never", "still permits writes")],
    ids=["never", "allowed"],
)
def test_the_compared_set_matches_the_rendered_kill_switch(tmp_path, writes, edited, fragment):
    """A real render agrees with the hook in both tiers — never a partial overlap.

    On ``never`` every tool the hook compares is in the rendered deny, and on
    ``allowed`` none is; a set that drifted from the kill switch would answer
    None (partial) and silence the warning on every such deployment.

    Then the #244 case end to end on the same render: a hand-edited tier warns.
    """
    hook = _hook()
    config_path, hook_config_path, settings_path, ctx = _render(tmp_path, writes=writes)
    tools = hook._kill_switch_tools(hook_config_path)

    assert "mcp__controls__channel_write" in tools
    assert "mcp__plc__set_value" in tools
    assert "mcp__sitectl" in tools and "mcp__rawctl" in tools
    # Mixed tools and the project's write_tools entry never count.
    assert "mcp__plc__get_value" not in tools
    assert not set(tools) & set(ctx["mixed_read_write_tools"])
    assert hook._write_tools_denied(settings_path, tools) is (writes == "never")
    assert hook._tier_in_config(config_path) == (
        writes == "allowed",
        f"control_system.writes: {writes}",
    )
    if writes == "never":
        assert set(tools) <= set(ctx["killswitch_deny"]) | set(ctx["deny_defaults"] or [])

    text = config_path.read_text(encoding="utf-8")
    assert text.count(f"writes: {writes}") == 1
    config_path.write_text(text.replace(f"writes: {writes}", f"writes: {edited}"), encoding="utf-8")
    _set_mtimes(config_path, settings_path, config_newer=False)

    message = hook._detect_drift(config_path, settings_path, hook_config_path)

    assert message is not None
    assert fragment in message
    assert f"control_system.writes: {edited}" in message
