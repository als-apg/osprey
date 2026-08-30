"""The hook's restated store rule against the module it restates.

``osprey_connectors.session_store`` is the canonical reader of the
per-(session, target) posture store; the PreToolUse hooks cannot import it —
they run outside the osprey venv — so
``osprey_target_state.effective_writes_for`` restates its four rules in stdlib
terms. Two implementations of one safety rule drift silently, and the drift is
only ever visible at a write, so this module is the table that pins them
together: for every cell of {ceiling} x {store value} x {readonly run} x
{target}, the hook's answer is the connector module's answer.

Both are exercised IN PROCESS. The connector module is imported here and only
here — a test may import what the code under test may not, and comparing the
hook against a second hand-written expectation would only pin the hook to this
file's opinion of the rule.

One difference between them is deliberate, stated in the hook's module
docstring and expressed literally by :func:`expected_answer` below: with no
resolvable target, ``session_store.effective_writes`` takes the UNION of the
deployment's configured targets (right for a roster describing a deployment)
while the hook takes the INTERSECTION (right for a gate, which must not be
handed the more permissive of two answers it cannot choose between). The store
half, the read-only half and every target-resolvable cell are identical, and
the intersection implies the union, so the two are pinned as::

    hook(None) == intersection AND session_store.effective_writes(..., None)

which is an equality, not a weakening: the only freedom it leaves the hook is
the ceiling it is deliberately stricter about.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest

import osprey.templates.claude_code.claude.hooks.osprey_target_state as reader
from osprey_connectors import session_store
from osprey_connectors.types import session_posture

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# the four axes
# ---------------------------------------------------------------------------

#: A deployment with one reachable target (``live``), armed.
ARMED_SINGLE = {"type": "epics", "writes_enabled": True}

#: The same deployment, unarmed.
UNARMED_SINGLE = {"type": "epics", "writes_enabled": False}

#: A deployment that states no posture anywhere — the shape every deployment had
#: before the per-type key existed. The hook calls that a third state (``None``);
#: neither implementation may read silence as permission.
SILENT_SINGLE = {"type": "epics"}

#: A switch-capable deployment armed for BOTH of its targets.
ARMED_BOTH = {
    "type": "epics",
    "writes_enabled": True,
    "connector": {
        "epics": {"prefix": "RING:"},
        "virtual_accelerator": {"prefix": "VA:"},
    },
}

#: A switch-capable deployment armed for its simulator and NOT for its ring —
#: the shape the per-type key exists for, and the one where the union and the
#: intersection over configured targets disagree.
MIXED = {
    "type": "epics",
    "writes_enabled": True,
    "connector": {
        "epics": {"prefix": "RING:", "writes_enabled": False},
        "virtual_accelerator": {"prefix": "VA:", "writes_enabled": True},
    },
}

#: A three-target deployment — ring, virtual accelerator and the live stand-in —
#: armed everywhere. ``standin`` is a machine in its own right, reachable only
#: through its own target, so a sweep without one never exercises the third slot
#: either reader resolves.
THREE_TARGETS_ARMED = {
    "type": "epics",
    "writes_enabled": True,
    "connector": {
        "epics": {"prefix": "RING:"},
        "virtual_accelerator": {"prefix": "VA:"},
        "live_standin": {"prefix": "SIM:"},
    },
}

#: The same three, with the ring alone disarmed: the shape where the union and
#: the intersection disagree AND a third target has to be folded into both.
THREE_TARGETS_MIXED = {
    "type": "epics",
    "writes_enabled": True,
    "connector": {
        "epics": {"prefix": "RING:", "writes_enabled": False},
        "virtual_accelerator": {"prefix": "VA:", "writes_enabled": True},
        "live_standin": {"prefix": "SIM:", "writes_enabled": True},
    },
}

#: ``(name, section)`` for the ceiling axis.
CEILINGS = [
    ("armed-single", ARMED_SINGLE),
    ("unarmed-single", UNARMED_SINGLE),
    ("silent-single", SILENT_SINGLE),
    ("armed-both", ARMED_BOTH),
    ("mixed", MIXED),
    ("three-targets-armed", THREE_TARGETS_ARMED),
    ("three-targets-mixed", THREE_TARGETS_MIXED),
]

#: ``(name, stored value)`` for the store axis. ``None`` means no entry at all.
#: The two bare strings are the legacy shapes the session-wide posture wrote
#: before targets existed; the unknown leaf and the unknown bare string are the
#: hand-edit and future-version shapes both readers must DROP rather than honour.
STORE_VALUES = [
    ("absent", None),
    ("legacy-bare-sandbox", "sandbox"),
    ("legacy-bare-writes", "writes"),
    ("unknown-bare-string", "locked"),
    ("live-sandboxed", {"live": "sandbox"}),
    ("va-sandboxed", {"va": "sandbox"}),
    ("standin-sandboxed", {"standin": "sandbox"}),
    ("both-sandboxed", {"live": "sandbox", "va": "sandbox"}),
    ("unknown-leaf", {"live": "locked"}),
    ("empty-map", {}),
]

#: The target axis. ``None`` is the session whose target could not be identified.
TARGETS = ["live", "va", "standin", None]

#: The session keys the table runs under. ``operator-`` keys are RETAINED by
#: both readers: their drop-on-restore rule belongs to the web server's startup
#: load alone, and an enforcement reader that dropped them would ignore a
#: narrowing that is live for the rest of that process's life.
SESSION_KEYS = ["4f1c2a7e-0000-4000-8000-000000000001", "operator-console-1"]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stamped_root(tmp_path, monkeypatch):
    """One agent-data root, stamped, so BOTH readers resolve the same store.

    The stamp is the anchor a session child carries; using it here is what makes
    the two implementations answer about one file rather than two. The unstamped
    derivation is pinned separately by the path test at the bottom.

    ``OSPREY_LAUNCH_POSTURE`` is cleared with the other two because it is a term
    the canonical module has and the hook deliberately does not — see
    :func:`test_the_launch_pin_is_the_one_term_the_hook_does_not_restate`. A
    stamp inherited from the environment this suite runs in would make the
    canonical reader refuse where the hook permits, in cells that are otherwise
    about the store.
    """
    root = tmp_path / "var" / "agent_data"
    (root / reader.STATE_DIR_NAME).mkdir(parents=True)
    monkeypatch.setenv(reader.AGENT_DATA_ROOT_ENV_VAR, str(root))
    monkeypatch.delenv(reader.EXECUTION_MODE_ENV_VAR, raising=False)
    monkeypatch.delenv(reader.POSTURE_SESSION_ENV_VAR, raising=False)
    monkeypatch.delenv(session_store.LAUNCH_POSTURE_ENV_VAR, raising=False)
    session_store.invalidate_cache()
    yield root
    session_store.invalidate_cache()


def write_store(root, payload):
    """Write the posture store both readers will read."""
    path = root / reader.STATE_DIR_NAME / reader.STORE_FILENAME
    path.write_text(json.dumps(payload), encoding="utf-8")
    session_store.invalidate_cache()


def expected_answer(section, session_key, target):
    """What the connector module says the hook must answer.

    Built from the canonical module's own public surface, never from a second
    hand-written copy of the rule. The intersection term is the one deliberate
    difference (see the module docstring); it is applied only where the session
    holds no target, which is the only place the two ceilings can differ.
    """
    canonical = session_store.effective_writes(section, session_key, target)
    if target is not None:
        return canonical
    intersection = all(session_posture(section).values())
    return bool(intersection and canonical)


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("readonly_run", [False, True], ids=["run-readwrite", "run-readonly"])
@pytest.mark.parametrize("session_key", SESSION_KEYS)
@pytest.mark.parametrize(
    "target", TARGETS, ids=["target-live", "target-va", "target-standin", "target-none"]
)
@pytest.mark.parametrize("store_name,store_value", STORE_VALUES)
@pytest.mark.parametrize("ceiling_name,section", CEILINGS)
def test_the_hook_answers_what_the_connector_module_answers(
    stamped_root,
    monkeypatch,
    ceiling_name,
    section,
    store_name,
    store_value,
    target,
    session_key,
    readonly_run,
):
    """Every cell of the truth table, both implementations, one answer."""
    # Arrange
    payload = {} if store_value is None else {session_key: store_value}
    # A second session's narrowing is always present, so a reader that ignored
    # the key would be caught rather than passing on an accident.
    payload["another-session"] = {"live": "sandbox", "va": "sandbox"}
    write_store(stamped_root, payload)
    monkeypatch.setenv(reader.POSTURE_SESSION_ENV_VAR, session_key)
    if readonly_run:
        monkeypatch.setenv(reader.EXECUTION_MODE_ENV_VAR, reader.SANDBOX_MODE)

    # Act
    answer = reader.effective_writes_for({}, section, target)

    # Assert
    assert answer is expected_answer(section, session_key, target)


@pytest.mark.parametrize(
    "target", TARGETS, ids=["target-live", "target-va", "target-standin", "target-none"]
)
@pytest.mark.parametrize("ceiling_name,section", CEILINGS)
def test_without_a_session_key_the_store_is_not_consulted(
    stamped_root, monkeypatch, ceiling_name, section, target
):
    """Rule 3's last clause, on both sides.

    Nothing addressed the session, so nothing narrowed it — even with a store
    that sandboxes every target under every key it holds. A reader that matched
    on the file rather than on the key would sandbox every CLI session on the
    host the moment one operator narrowed one web-terminal session.
    """
    # Arrange
    write_store(
        stamped_root,
        {
            "4f1c2a7e-0000-4000-8000-000000000001": "sandbox",
            "operator-console-1": {"live": "sandbox", "va": "sandbox"},
        },
    )
    monkeypatch.delenv(reader.POSTURE_SESSION_ENV_VAR, raising=False)

    # Act / Assert
    assert reader.effective_writes_for({}, section, target) is expected_answer(
        section, None, target
    )


@pytest.mark.parametrize("target", TARGETS)
def test_a_blank_session_key_is_no_session_key(stamped_root, monkeypatch, target):
    """Whitespace is not a key. Both readers strip before deciding."""
    # Arrange
    write_store(stamped_root, {"": "sandbox", "   ": "sandbox"})
    monkeypatch.setenv(reader.POSTURE_SESSION_ENV_VAR, "   ")

    # Act / Assert
    assert reader.effective_writes_for({}, ARMED_BOTH, target) is True
    assert reader.session_key() is None


def test_a_corrupt_store_is_an_empty_store_on_both_sides(stamped_root, monkeypatch):
    """A store nobody can repair from the browser must not wedge every write.

    Losing narrowings an operator can set again is the lesser harm, and it is
    the harm the canonical module chose; the hook may not choose the other one.
    """
    # Arrange
    path = stamped_root / reader.STATE_DIR_NAME / reader.STORE_FILENAME
    path.write_text('{"key": "sandbox"', encoding="utf-8")  # truncated
    session_store.invalidate_cache()
    monkeypatch.setenv(reader.POSTURE_SESSION_ENV_VAR, "key")

    # Act / Assert
    assert reader.parse_store(path.read_text(encoding="utf-8")) == {}
    assert reader.effective_writes_for({}, ARMED_BOTH, "live") is expected_answer(
        ARMED_BOTH, "key", "live"
    )
    assert reader.effective_writes_for({}, ARMED_BOTH, "live") is True


def test_a_non_string_key_is_dropped_by_both_parsers(stamped_root):
    """JSON cannot spell one, but ``parse_store`` also takes decoded objects."""
    raw = {"good": {"live": "sandbox"}, 7: {"live": "sandbox"}}
    assert reader.parse_store(raw) == session_store.parse_store(raw)
    assert reader.parse_store(raw) == {"good": {"live": "sandbox"}}


@pytest.mark.parametrize(
    "raw",
    [
        '{"k": "sandbox"}',
        '{"k": "writes"}',
        '{"k": {"live": "sandbox", "va": "writes"}}',
        '{"k": {"live": "sandbox"}, "operator-x": "sandbox"}',
        '{"k": []}',
        '{"k": null}',
        "[]",
        "not json at all",
        "",
    ],
    ids=[
        "bare-sandbox",
        "bare-writes",
        "mixed-map",
        "operator-key",
        "list-value",
        "null-value",
        "top-level-list",
        "garbage",
        "empty",
    ],
)
def test_parse_store_agrees_shape_for_shape(raw):
    """Rule 2, on the parser itself rather than through a lookup.

    The lookups above can agree by accident on a shape both drop for different
    reasons; this pins the surviving structure, which is what a future target
    name or a future posture value would change.
    """
    assert reader.parse_store(raw) == session_store.parse_store(raw)


def test_legacy_bare_sandbox_covers_the_whole_target_vocabulary():
    """The one shape whose meaning is not visible in the file.

    A bare ``"sandbox"`` narrowed the whole SESSION before targets existed, so
    it narrows every target — including one this deployment has not configured,
    which costs nothing and is what keeps the two parsers on one rule.
    """
    parsed = reader.parse_store({"k": "sandbox"})
    assert parsed == session_store.parse_store({"k": "sandbox"})
    assert set(parsed["k"]) == set(reader.CONTROL_TARGETS)


# ---------------------------------------------------------------------------
# one path, three resolvers
# ---------------------------------------------------------------------------


def test_the_stamped_store_path_is_one_path(tmp_path, monkeypatch):
    """Writer and both readers, stamped.

    A store the writer puts in one directory and a reader looks for in another
    is a narrowing that silently never applies — the failure mode this pin
    exists for, because nothing else about it looks wrong.
    """
    # Arrange
    root = tmp_path / "elsewhere" / "agent_data"
    monkeypatch.setenv(reader.AGENT_DATA_ROOT_ENV_VAR, str(root))
    session_store.invalidate_cache()

    # Act / Assert
    assert reader.store_path({}) == str(session_store.store_path())
    assert reader.legacy_store_path({}) == str(session_store.legacy_store_path())
    assert reader.resolve_state_dir({}) == str(session_store.state_dir())
    assert os.path.basename(reader.store_path({})) == session_store.STORE_FILENAME


def test_the_unstamped_store_path_is_one_path(tmp_path, monkeypatch):
    """Writer and both readers, with no stamp and only a config to go on.

    The connector module anchors on ``project_root`` from the config; the hook
    restates that with :func:`osprey_hook_log.get_repo_root`. They agree, which
    is what lets a hook running outside the venv read the file the web server
    wrote.
    """
    # Arrange
    from osprey_connectors.workspace import reset_config_cache

    config = tmp_path / "config.yml"
    config.write_text(f"project_root: {tmp_path}\ncontrol_system:\n  type: mock\n")
    monkeypatch.delenv(reader.AGENT_DATA_ROOT_ENV_VAR, raising=False)
    monkeypatch.setenv("OSPREY_CONFIG", str(config))
    monkeypatch.setenv("CONFIG_FILE", str(config))
    reset_config_cache()
    session_store.invalidate_cache()

    # Act
    hook_answer = reader.store_path({})
    canonical = session_store.store_path()

    # Assert
    try:
        assert canonical is not None
        assert os.path.realpath(hook_answer) == os.path.realpath(str(canonical))
        assert os.path.realpath(reader.legacy_store_path({})) == os.path.realpath(
            str(session_store.legacy_store_path())
        )
    finally:
        reset_config_cache()
        session_store.invalidate_cache()


def test_the_store_sits_beside_the_target_state_file(tmp_path, monkeypatch):
    """One directory answers "session state for the control targets".

    Co-siting is not cosmetic: the hook's ``posture_unknown`` uses a live record
    in that directory as the evidence that its DERIVED path found the right
    place, which only means anything while the two files live together.
    """
    root = tmp_path / "var" / "agent_data"
    monkeypatch.setenv(reader.AGENT_DATA_ROOT_ENV_VAR, str(root))

    assert os.path.dirname(reader.store_path({})) == reader.resolve_state_dir({})
    assert reader.resolve_state_dir({}) == os.path.join(str(root), reader.STATE_DIR_NAME)
    assert reader.STATE_DIR_NAME == session_store.STATE_DIR_NAME
    assert reader.STORE_FILENAME == session_store.STORE_FILENAME
    assert reader.AGENT_DATA_ROOT_ENV_VAR == session_store.AGENT_DATA_ROOT_ENV_VAR
    assert reader.POSTURE_SANDBOX == session_store.POSTURE_SANDBOX
    assert reader.POSTURE_WRITES == session_store.POSTURE_WRITES
    assert set(reader.VALID_POSTURES) == set(session_store.VALID_POSTURES)


def test_the_legacy_store_answers_only_while_the_new_one_is_absent(tmp_path, monkeypatch):
    """Rule 1's upgrade clause, both sides.

    A deployment upgrading with a sandboxed session live must not have that
    narrowing lifted the moment the code lands; once the web server persists the
    migrated shape, the old file stops answering rather than fighting the new one.
    """
    # Arrange
    root = tmp_path / "var" / "agent_data"
    (root / reader.STATE_DIR_NAME).mkdir(parents=True)
    monkeypatch.setenv(reader.AGENT_DATA_ROOT_ENV_VAR, str(root))
    monkeypatch.setenv(reader.POSTURE_SESSION_ENV_VAR, "k")
    (root / reader.STORE_FILENAME).write_text(json.dumps({"k": "sandbox"}), encoding="utf-8")
    session_store.invalidate_cache()

    # Act / Assert — legacy answers while the new file does not exist
    assert reader.read_session_store({}) == session_store.load_store()
    assert reader.effective_writes_for({}, ARMED_BOTH, "live") is False

    # Arrange — the migrated shape lands
    (root / reader.STATE_DIR_NAME / reader.STORE_FILENAME).write_text(
        json.dumps({"k": {"va": "sandbox"}}), encoding="utf-8"
    )
    session_store.invalidate_cache()

    # Act / Assert — and the old file stops answering
    assert reader.read_session_store({}) == session_store.load_store()
    assert reader.effective_writes_for({}, ARMED_BOTH, "live") is True
    assert reader.effective_writes_for({}, ARMED_BOTH, "va") is False


def test_the_target_blind_ceiling_is_deliberately_stricter(stamped_root, monkeypatch):
    """The one difference, isolated so it can never become an accident.

    ``MIXED`` arms the simulator and not the ring. Asked with no target, the
    canonical reader answers for a caller that holds none at all and takes the
    union — one of these machines is armed. The hook is a gate: it takes the
    intersection, because a call it cannot attribute to a machine must not be
    granted the more permissive of the two answers, and one of them is hardware.
    """
    # Arrange
    monkeypatch.setenv(reader.POSTURE_SESSION_ENV_VAR, SESSION_KEYS[0])
    write_store(stamped_root, {})

    # Assert — the two ceilings genuinely disagree on this section
    posture = session_posture(MIXED)
    assert any(posture.values()) is True
    assert all(posture.values()) is False

    # Act / Assert — and the hook takes the stricter one
    assert session_store.effective_writes(MIXED, SESSION_KEYS[0], None) is True
    assert reader.effective_writes_for({}, MIXED, None) is False

    # Where both ceilings agree, so do the two implementations.
    assert reader.effective_writes_for({}, ARMED_BOTH, None) is session_store.effective_writes(
        ARMED_BOTH, SESSION_KEYS[0], None
    )


# ---------------------------------------------------------------------------
# the union ceiling stays unreachable from a write path
# ---------------------------------------------------------------------------

#: Directories scanned for production calls of the canonical rule. Tests are
#: excluded: this file itself compares the two ceilings on purpose, and so may
#: any other test that documents the difference.
_PRODUCTION_ROOTS = ("packages", "src")


def _production_python_files():
    """Every non-test ``.py`` file under :data:`_PRODUCTION_ROOTS`."""
    repo_root = Path(__file__).resolve().parents[2]
    for root in _PRODUCTION_ROOTS:
        for path in (repo_root / root).rglob("*.py"):
            if "tests" in path.parts or "test" in path.parts:
                continue
            yield path


def _effective_writes_calls(path):
    """``(lineno, call)`` for every ``effective_writes(...)`` in *path*."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):  # pragma: no cover - an unreadable file is not a caller
        return
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "effective_writes":
            yield node.lineno, node
        elif isinstance(func, ast.Name) and func.id == "effective_writes":
            yield node.lineno, node


def _names_a_target(call):
    """Whether *call* hands ``effective_writes`` a target that is not ``None``.

    Positionally, ``target`` is the third argument. As a keyword it must not be
    the literal ``None`` — a caller spelling ``target=None`` has explicitly
    asked for the union ceiling, which is what this pin forbids on a write path.
    A keyword whose value is a variable passes: the guard cannot see through it,
    and treating that as a violation would only teach people to launder the
    argument through a local.
    """
    if len(call.args) >= 3:
        return True
    for keyword in call.keywords:
        if keyword.arg != "target":
            continue
        return not (isinstance(keyword.value, ast.Constant) and keyword.value.value is None)
    return False


def test_no_production_caller_takes_the_union_ceiling():
    """The one divergence is unreachable from any write path, and stays so.

    ``session_store.effective_writes`` with no target answers the UNION over the
    deployment's configured targets; the hook answers the intersection. That is
    safe only because nothing on a write path asks the question without a target
    — the roster and the popover do, and they describe rather than decide. This
    guard is what keeps it true: a future caller that dropped the target would
    silently write under the more permissive of two ceilings, and the hook
    denying the same call would look like the bug.

    It fails in CI rather than at a write, which is the whole point.
    """
    # Arrange / Act
    offenders = [
        f"{path}:{lineno}"
        for path in _production_python_files()
        for lineno, call in _effective_writes_calls(path)
        if not _names_a_target(call)
    ]

    # Assert
    assert offenders == [], (
        "these callers of session_store.effective_writes pass no target, so they "
        "take the UNION ceiling the hook deliberately does not: "
        + ", ".join(offenders)
        + ". Pass the target the write lands on, or move the call off the write path."
    )


def test_the_guard_can_see_a_violation():
    """A guard is only worth having if it would actually catch one."""
    module = ast.parse(
        "session_store.effective_writes(section, key)\n"
        "session_store.effective_writes(section, key, target=None)\n"
        "session_store.effective_writes(section, key, target)\n"
        "session_store.effective_writes(section, key, target=t)\n"
        "session_store.effective_writes(section, key, None, connector_type='epics')\n"
    )
    calls = [node for node in ast.walk(module) if isinstance(node, ast.Call)]
    assert [_names_a_target(call) for call in calls] == [False, False, True, True, True]


def test_the_guard_actually_scans_the_callers_that_exist():
    """Not vacuous: the sweep must be finding real calls to have an opinion.

    A guard that silently matched nothing — a moved directory, a renamed
    function — would pass forever while the thing it protects rots.
    """
    found = [
        f"{path.name}:{lineno}"
        for path in _production_python_files()
        for lineno, _call in _effective_writes_calls(path)
    ]
    assert found, "the reachability guard found no callers at all; its scan has gone stale"


# ---------------------------------------------------------------------------
# failure modes of the file itself
# ---------------------------------------------------------------------------


def test_an_unreadable_new_store_does_not_fall_back_to_the_legacy_one(tmp_path, monkeypatch):
    """Existence alone chooses the path — the rule the canonical reader uses.

    A new store that exists but cannot be read is an empty store on both sides.
    Falling back to the legacy file here would hand the hook a narrowing the
    canonical reader does not see, or lift one it does; the divergence would
    surface only as one layer refusing a write another allowed.
    """
    # Arrange
    root = tmp_path / "var" / "agent_data"
    (root / reader.STATE_DIR_NAME).mkdir(parents=True)
    monkeypatch.setenv(reader.AGENT_DATA_ROOT_ENV_VAR, str(root))
    monkeypatch.setenv(reader.POSTURE_SESSION_ENV_VAR, "k")
    (root / reader.STORE_FILENAME).write_text(json.dumps({"k": "sandbox"}), encoding="utf-8")
    new_store = root / reader.STATE_DIR_NAME / reader.STORE_FILENAME
    new_store.write_text(json.dumps({"k": {"va": "sandbox"}}), encoding="utf-8")
    new_store.chmod(0o000)
    session_store.invalidate_cache()

    # Act / Assert
    try:
        if os.access(new_store, os.R_OK):  # pragma: no cover - root can read anything
            pytest.skip("this user can read a mode-000 file; the fallback rule is untestable here")
        assert reader.read_session_store({}) == {}
        assert reader.read_session_store({}) == session_store.load_store()
        assert reader.effective_writes_for({}, ARMED_BOTH, "live") is True
    finally:
        new_store.chmod(0o600)
        session_store.invalidate_cache()


def test_the_launch_pin_is_the_one_term_the_hook_does_not_restate(stamped_root, monkeypatch):
    """The premise the fixture's third ``delenv`` rests on, pinned rather than assumed.

    ``OSPREY_LAUNCH_POSTURE`` is stamped by the executor into a sandbox child's
    environment and by nothing else, so no hook process ever carries one: the
    PreToolUse hooks run in the Claude Code process, above every sandbox. That
    is why the hook's restatement has three terms where the canonical module has
    four, and why the table above clears the stamp instead of teaching the hook
    about it.

    With the stamp set, the two answers legitimately diverge — which is what
    makes this an executor-only term rather than a drift. If a hook ever DID run
    somewhere the stamp exists, this test failing is the signal that the
    restatement has to grow the term.
    """
    # Arrange — nothing narrowed in the store at all; only the run is pinned.
    write_store(stamped_root, {})
    monkeypatch.setenv(reader.POSTURE_SESSION_ENV_VAR, "k")
    monkeypatch.setenv(session_store.LAUNCH_POSTURE_ENV_VAR, "live=sandbox")

    # Act / Assert — the canonical reader refuses, the hook permits.
    assert session_store.effective_writes(ARMED_BOTH, "k", "live") is False
    assert reader.effective_writes_for({}, ARMED_BOTH, "live") is True

    # And the divergence is the STAMP, not the target: every other cell agrees.
    assert session_store.effective_writes(ARMED_BOTH, "k", "va") is True
    assert reader.effective_writes_for({}, ARMED_BOTH, "va") is True


def test_an_undecodable_store_is_an_empty_store_on_both_sides(tmp_path, monkeypatch):
    """Nothing may raise on a store that is not UTF-8 — in a hook or anywhere.

    ``UnicodeDecodeError`` is a ``ValueError``, so it slips past the ``OSError``
    guard a file read normally carries, and the canonical reader used to let it
    propagate while the hook swallowed it. That was a real divergence and it was
    the wrong way round: a hook has no caller to hand an exception to (an
    unhandled one exits non-zero with no JSON, which PreToolUse reads as "no
    opinion" — the opposite of what an unreadable store must mean), and the
    in-process readers are on the write path, where one mis-encoded file would
    raise into every posture lookup. Both now answer the empty store, the same
    way both already answer it for a truncated one.
    """
    # Arrange
    root = tmp_path / "var" / "agent_data"
    (root / reader.STATE_DIR_NAME).mkdir(parents=True)
    monkeypatch.setenv(reader.AGENT_DATA_ROOT_ENV_VAR, str(root))
    monkeypatch.setenv(reader.POSTURE_SESSION_ENV_VAR, "k")
    # Cleared for the reason ``stamped_root`` clears it: this test builds its own
    # root rather than taking that fixture, and an inherited launch pin would
    # make the canonical reader refuse a cell that is about the file's encoding.
    monkeypatch.delenv(session_store.LAUNCH_POSTURE_ENV_VAR, raising=False)
    (root / reader.STATE_DIR_NAME / reader.STORE_FILENAME).write_bytes(
        b"\xff\xfe{\x00k\x00: sandbox}"
    )
    session_store.invalidate_cache()

    # Act / Assert — both answer, and the deployment ceiling stays in charge
    assert reader.read_session_store({}) == {}
    assert session_store.load_store() == {}
    assert reader.read_session_store({}) == session_store.load_store()
    assert reader.effective_writes_for({}, ARMED_BOTH, "live") is expected_answer(
        ARMED_BOTH, "k", "live"
    )
    assert reader.effective_writes_for({}, ARMED_BOTH, "live") is True
    session_store.invalidate_cache()
