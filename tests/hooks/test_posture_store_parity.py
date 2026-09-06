"""The hook's restated posture rule against the module it restates.

``osprey_connectors.posture_store`` is the canonical reader of the control
context's recorded posture; the PreToolUse hooks cannot import it — they run
outside the osprey venv — so ``osprey_target_state.effective_writes_for``
restates its rules in stdlib terms. Two implementations of one safety rule
drift silently, and the drift is only ever visible at a write, so this module
is the table that pins them together: for every cell of {ceiling} x {recorded
posture} x {readonly run} x {target}, the hook's answer is the connector
module's answer.

Both are exercised IN PROCESS. The connector module is imported here and only
here — a test may import what the code under test may not, and comparing the
hook against a second hand-written expectation would only pin the hook to this
file's opinion of the rule.

One difference between them is deliberate, stated in the hook's module
docstring and expressed literally by :func:`expected_answer` below: with no
resolvable target, ``posture_store.effective_writes`` takes the UNION of the
deployment's configured targets (right for a roster describing a deployment)
while the hook takes the INTERSECTION (right for a gate, which must not be
handed the more permissive of two answers it cannot choose between). The
recorded half, the read-only half and every target-resolvable cell are
identical, and the intersection implies the union, so the two are pinned as::

    hook(None) == intersection AND posture_store.effective_writes(..., None)

which is an equality, not a weakening: the only freedom it leaves the hook is
the ceiling it is deliberately stricter about.

The second thing pinned here is :func:`osprey_target_state.posture_unknown`,
whose fail-closed rule has no counterpart in the canonical module at all — a
process the web terminal never stamped is refused until the deployment's
record exists, and permitted on the deployment's own terms afterwards.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest

import osprey.templates.claude_code.claude.hooks.osprey_target_state as reader
from osprey_connectors import control_context, posture_store
from osprey_connectors.types import session_posture
from tests._control_context_fixtures import write_control_context, write_payload

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

#: ``(name, recorded posture)`` for the posture axis. The two bare strings are
#: the legacy shapes the session-wide posture wrote before targets existed; the
#: unknown leaf and the unknown bare string are the hand-edit and future-version
#: shapes both readers must DROP rather than honour.
POSTURES = [
    ("absent", {}),
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

#: The target axis. ``None`` is the call whose target could not be identified.
TARGETS = ["live", "va", "standin", None]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stamped_root(tmp_path, monkeypatch):
    """One agent-data root, stamped, so BOTH readers resolve the same record.

    The stamp is the anchor a session child carries; using it here is what makes
    the two implementations answer about one file rather than two. The unstamped
    derivation is pinned separately by the path test at the bottom.

    ``OSPREY_LAUNCH_POSTURE`` is cleared with the other two because it is a term
    the canonical module has and the hook deliberately does not — see
    :func:`test_the_launch_pin_is_the_one_term_the_hook_does_not_restate`. A
    stamp inherited from the environment this suite runs in would make the
    canonical reader refuse where the hook permits, in cells that are otherwise
    about the record.
    """
    root = tmp_path / "var" / "agent_data"
    (root / reader.STATE_DIR_NAME).mkdir(parents=True)
    monkeypatch.setenv(reader.AGENT_DATA_ROOT_ENV_VAR, str(root))
    monkeypatch.delenv(reader.EXECUTION_MODE_ENV_VAR, raising=False)
    monkeypatch.delenv(reader.POSTURE_SESSION_ENV_VAR, raising=False)
    monkeypatch.delenv(posture_store.LAUNCH_POSTURE_ENV_VAR, raising=False)
    posture_store.invalidate_cache()
    yield root
    posture_store.invalidate_cache()


def write_record(root, posture):
    """Write the record both readers will read, carrying *posture* verbatim."""
    return write_control_context(root, posture=posture)


def expected_answer(section, target):
    """What the connector module says the hook must answer.

    Built from the canonical module's own public surface, never from a second
    hand-written copy of the rule. The intersection term is the one deliberate
    difference (see the module docstring); it is applied only where the caller
    holds no target, which is the only place the two ceilings can differ.
    """
    canonical = posture_store.effective_writes(section, target)
    if target is not None:
        return canonical
    intersection = all(session_posture(section).values())
    return bool(intersection and canonical)


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("readonly_run", [False, True], ids=["run-readwrite", "run-readonly"])
@pytest.mark.parametrize(
    "target", TARGETS, ids=["target-live", "target-va", "target-standin", "target-none"]
)
@pytest.mark.parametrize("posture_name,posture", POSTURES)
@pytest.mark.parametrize("ceiling_name,section", CEILINGS)
def test_the_hook_answers_what_the_connector_module_answers(
    stamped_root,
    monkeypatch,
    ceiling_name,
    section,
    posture_name,
    posture,
    target,
    readonly_run,
):
    """Every cell of the truth table, both implementations, one answer."""
    # Arrange
    write_record(stamped_root, posture)
    if readonly_run:
        monkeypatch.setenv(reader.EXECUTION_MODE_ENV_VAR, reader.SANDBOX_MODE)

    # Act
    answer = reader.effective_writes_for({}, section, target)

    # Assert
    assert answer is expected_answer(section, target)


@pytest.mark.parametrize(
    "target", TARGETS, ids=["target-live", "target-va", "target-standin", "target-none"]
)
@pytest.mark.parametrize("ceiling_name,section", CEILINGS)
def test_the_narrowing_reaches_a_process_that_carries_no_session_key(
    stamped_root, monkeypatch, ceiling_name, section, target
):
    """The narrowing is the deployment's, so nothing has to be addressed.

    A bare ``claude``, a dispatch worker and a CLI run carry no session of their
    own. Each of them writes to the same machine an operator took away, so each
    of them reads the same record and is refused by it — on both sides.
    """
    # Arrange
    write_record(stamped_root, {"live": "sandbox", "va": "sandbox", "standin": "sandbox"})
    monkeypatch.delenv(reader.POSTURE_SESSION_ENV_VAR, raising=False)

    # Act / Assert
    assert reader.effective_writes_for({}, section, target) is expected_answer(section, target)
    assert reader.effective_writes_for({}, section, target) is False


def test_a_blank_session_key_is_no_session_key(stamped_root, monkeypatch):
    """Whitespace is not a key. It attributes a stamp; it narrows nothing."""
    # Arrange
    write_record(stamped_root, {})
    monkeypatch.setenv(reader.POSTURE_SESSION_ENV_VAR, "   ")

    # Act / Assert
    assert reader.session_key() is None
    assert reader.effective_writes_for({}, ARMED_BOTH, "live") is True


def test_a_corrupt_record_is_an_unnarrowed_record_on_both_sides(stamped_root):
    """A record nobody can repair from the browser must not wedge every write.

    Losing narrowings an operator can set again is the lesser harm, and it is
    the harm the canonical module chose; the hook may not choose the other one.
    """
    # Arrange
    path = control_context.record_path_under(stamped_root)
    path.write_text('{"schema": 1, "target": "live"', encoding="utf-8")  # truncated
    posture_store.invalidate_cache()

    # Act / Assert
    assert reader.read_record({}) is None
    assert reader.recorded_posture({}) == posture_store.recorded_posture() == {}
    assert reader.effective_writes_for({}, ARMED_BOTH, "live") is expected_answer(
        ARMED_BOTH, "live"
    )
    assert reader.effective_writes_for({}, ARMED_BOTH, "live") is True


def test_a_non_string_target_is_dropped_by_both_parsers():
    """JSON cannot spell one, but both parsers also take decoded objects."""
    raw = {"live": "sandbox", 7: "sandbox"}
    assert reader.parse_posture(raw) == control_context.parse_posture(raw)
    assert reader.parse_posture(raw) == {"live": "sandbox"}


@pytest.mark.parametrize(
    "raw",
    [
        "sandbox",
        "writes",
        "locked",
        {"live": "sandbox", "va": "writes"},
        {"live": "sandbox"},
        [],
        None,
        7,
        {},
    ],
    ids=[
        "bare-sandbox",
        "bare-writes",
        "unknown-bare-string",
        "mixed-map",
        "one-narrowing",
        "list-value",
        "null-value",
        "int-value",
        "empty-map",
    ],
)
def test_the_posture_parsers_agree_shape_for_shape(raw):
    """Rule 2, on the parser itself rather than through a lookup.

    The lookups above can agree by accident on a shape both drop for different
    reasons; this pins the surviving structure, which is what a future target
    name or a future posture value would change.
    """
    assert reader.parse_posture(raw) == control_context.parse_posture(raw)


def test_legacy_bare_sandbox_covers_the_whole_target_vocabulary():
    """The one shape whose meaning is not visible in the file.

    A bare ``"sandbox"`` narrowed the whole deployment before targets existed,
    so it narrows every target — including one this deployment has not
    configured, which costs nothing and is what keeps the two parsers on one
    rule.
    """
    parsed = reader.parse_posture("sandbox")
    assert parsed == control_context.parse_posture("sandbox")
    assert set(parsed) == set(reader.CONTROL_TARGETS)


# ---------------------------------------------------------------------------
# one path, three resolvers
# ---------------------------------------------------------------------------


def test_the_stamped_record_path_is_one_path(tmp_path, monkeypatch):
    """Writer and both readers, stamped.

    A record the owner puts in one directory and a reader looks for in another
    is a narrowing that silently never applies — the failure mode this pin
    exists for, because nothing else about it looks wrong.
    """
    # Arrange
    root = tmp_path / "elsewhere" / "agent_data"
    monkeypatch.setenv(reader.AGENT_DATA_ROOT_ENV_VAR, str(root))
    posture_store.invalidate_cache()

    # Act / Assert
    assert reader.record_path({}) == str(control_context.record_path())
    assert reader.resolve_state_dir({}) == str(posture_store.state_dir())
    assert os.path.basename(reader.record_path({})) == control_context.RECORD_FILENAME


def test_the_unstamped_record_path_is_one_path(tmp_path, monkeypatch):
    """Writer and both readers, with no stamp and only a config to go on.

    The connector module anchors on ``project_root`` from the config; the hook
    restates that with :func:`osprey_hook_log.get_repo_root`. They agree, which
    is what lets a hook running outside the venv read the file the web terminal
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
    posture_store.invalidate_cache()

    # Act
    hook_answer = reader.record_path({})
    canonical = control_context.record_path()

    # Assert
    try:
        assert canonical is not None
        assert os.path.realpath(hook_answer) == os.path.realpath(str(canonical))
    finally:
        reset_config_cache()
        posture_store.invalidate_cache()


def test_the_record_sits_beside_the_server_reports(tmp_path, monkeypatch):
    """One directory answers "control context for this deployment".

    Co-siting is not cosmetic: the hook's ``posture_unknown`` uses a readable
    record in that directory as the evidence that its DERIVED path found the
    right place, which only means anything while the record and the reports
    live together.
    """
    root = tmp_path / "var" / "agent_data"
    monkeypatch.setenv(reader.AGENT_DATA_ROOT_ENV_VAR, str(root))

    assert os.path.dirname(reader.record_path({})) == reader.resolve_state_dir({})
    assert reader.resolve_state_dir({}) == os.path.join(str(root), reader.STATE_DIR_NAME)
    assert reader.STATE_DIR_NAME == posture_store.STATE_DIR_NAME
    assert reader.RECORD_FILENAME == control_context.RECORD_FILENAME
    assert reader.REPORT_FILE_PREFIX == control_context.REPORT_FILE_PREFIX
    assert reader.REPORT_FILE_SUFFIX == control_context.REPORT_FILE_SUFFIX
    assert reader.AGENT_DATA_ROOT_ENV_VAR == posture_store.AGENT_DATA_ROOT_ENV_VAR
    assert reader.POSTURE_SANDBOX == posture_store.POSTURE_SANDBOX
    assert reader.POSTURE_WRITES == posture_store.POSTURE_WRITES
    assert set(reader.VALID_POSTURES) == set(posture_store.VALID_POSTURES)


def test_the_retired_per_session_store_is_read_by_neither_side(stamped_root):
    """The posture is the record's field now, and nothing else is a posture.

    ``session-postures.json`` is retired. A file left at that name narrows
    nothing on either side — a reader that still honoured it would apply one
    operator's stale, session-keyed narrowing on a deployment whose record says
    the machine is open.
    """
    # Arrange — the retired file says sandbox for everything; the record does not
    retired = stamped_root / reader.STATE_DIR_NAME / "session-postures.json"
    retired.write_text(json.dumps({"k": {"live": "sandbox", "va": "sandbox"}}), encoding="utf-8")
    write_record(stamped_root, {})

    # Act / Assert
    assert reader.recorded_posture({}) == posture_store.recorded_posture() == {}
    assert reader.effective_writes_for({}, ARMED_BOTH, "live") is True
    assert reader.effective_writes_for({}, ARMED_BOTH, "va") is True


def test_a_narrowing_in_the_record_holds_with_nothing_at_the_retired_location(stamped_root):
    """The safety property the retirement must not disturb.

    A target narrowed in the record stays narrowed when the retired location
    holds nothing at all — the absence of an old file is not a permissive
    default, on either side.
    """
    # Arrange
    write_record(stamped_root, {"va": "sandbox"})

    # Act / Assert
    assert not (stamped_root / reader.STATE_DIR_NAME / "session-postures.json").exists()
    assert reader.recorded_posture({}) == posture_store.recorded_posture() == {"va": "sandbox"}
    assert reader.effective_writes_for({}, ARMED_BOTH, "va") is False
    assert reader.effective_writes_for({}, ARMED_BOTH, "live") is True


def test_the_target_blind_ceiling_is_deliberately_stricter(stamped_root):
    """The one difference, isolated so it can never become an accident.

    ``MIXED`` arms the simulator and not the ring. Asked with no target, the
    canonical reader answers for a caller that holds none at all and takes the
    union — one of these machines is armed. The hook is a gate: it takes the
    intersection, because a call it cannot attribute to a machine must not be
    granted the more permissive of the two answers, and one of them is hardware.
    """
    # Arrange
    write_record(stamped_root, {})

    # Assert — the two ceilings genuinely disagree on this section
    posture = session_posture(MIXED)
    assert any(posture.values()) is True
    assert all(posture.values()) is False

    # Act / Assert — and the hook takes the stricter one
    assert posture_store.effective_writes(MIXED, None) is True
    assert reader.effective_writes_for({}, MIXED, None) is False

    # Where both ceilings agree, so do the two implementations.
    assert reader.effective_writes_for({}, ARMED_BOTH, None) is posture_store.effective_writes(
        ARMED_BOTH, None
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

    Positionally, ``target`` is the second argument. As a keyword it must not be
    the literal ``None`` — a caller spelling ``target=None`` has explicitly
    asked for the union ceiling, which is what this pin forbids on a write path.
    A keyword whose value is a variable passes: the guard cannot see through it,
    and treating that as a violation would only teach people to launder the
    argument through a local.
    """
    if len(call.args) >= 2:
        return True
    for keyword in call.keywords:
        if keyword.arg != "target":
            continue
        return not (isinstance(keyword.value, ast.Constant) and keyword.value.value is None)
    return False


def test_no_production_caller_takes_the_union_ceiling():
    """The one divergence is unreachable from any write path, and stays so.

    ``posture_store.effective_writes`` with no target answers the UNION over the
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
        "these callers of posture_store.effective_writes pass no target, so they "
        "take the UNION ceiling the hook deliberately does not: "
        + ", ".join(offenders)
        + ". Pass the target the write lands on, or move the call off the write path."
    )


def test_the_guard_can_see_a_violation():
    """A guard is only worth having if it would actually catch one."""
    module = ast.parse(
        "posture_store.effective_writes(section)\n"
        "posture_store.effective_writes(section, target=None)\n"
        "posture_store.effective_writes(section, target)\n"
        "posture_store.effective_writes(section, target=t)\n"
        "posture_store.effective_writes(section, None, connector_type='epics')\n"
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
# posture_unknown — the hook's own rule, with no counterpart to compare to
# ---------------------------------------------------------------------------


def test_an_unstamped_process_with_no_record_is_fail_closed(tmp_path, monkeypatch):
    """The bare-``claude`` case before any controls server has published.

    Nothing stamped this process, so the directory below was DERIVED from the
    framework default — and a project that moved ``agent_data.base_dir`` moved
    it out from under this reader. With no record there either, the evidence
    that the derivation found the right place is missing, and a readwrite call
    is refused until the first controls server writes one.
    """
    from osprey_connectors.workspace import reset_config_cache

    config = tmp_path / "config.yml"
    config.write_text(f"project_root: {tmp_path}\ncontrol_system:\n  type: mock\n")
    monkeypatch.delenv(reader.AGENT_DATA_ROOT_ENV_VAR, raising=False)
    monkeypatch.setenv("OSPREY_CONFIG", str(config))
    monkeypatch.setenv("CONFIG_FILE", str(config))
    reset_config_cache()

    try:
        assert reader.posture_unknown({}) is True
    finally:
        reset_config_cache()
        posture_store.invalidate_cache()


def test_an_unstamped_process_with_a_readable_record_is_answered_by_it(tmp_path, monkeypatch):
    """The retry that succeeds: the same bare ``claude``, one server later.

    A readable record at the derived path IS the evidence the derivation found
    the right directory, so the deployment's own terms decide from here on.
    """
    from osprey_connectors.workspace import reset_config_cache

    config = tmp_path / "config.yml"
    config.write_text(f"project_root: {tmp_path}\ncontrol_system:\n  type: mock\n")
    monkeypatch.delenv(reader.AGENT_DATA_ROOT_ENV_VAR, raising=False)
    monkeypatch.setenv("OSPREY_CONFIG", str(config))
    monkeypatch.setenv("CONFIG_FILE", str(config))
    reset_config_cache()

    try:
        write_control_context(tmp_path / "var" / "agent_data", posture={"live": "sandbox"})

        assert reader.posture_unknown({}) is False
        assert reader.effective_writes_for({}, ARMED_BOTH, "live") is False
        assert reader.effective_writes_for({}, ARMED_BOTH, "va") is True
    finally:
        reset_config_cache()
        posture_store.invalidate_cache()


def test_a_stamped_process_is_never_posture_unknown(stamped_root):
    """The stamp is the deployment handing the directory over.

    A stamped process was told where to look, so an empty directory there means
    exactly what it says — nothing is narrowed — and nothing is refused on the
    record's account.
    """
    assert reader.posture_unknown({}) is False

    write_record(stamped_root, {"live": "sandbox"})
    assert reader.posture_unknown({}) is False


# ---------------------------------------------------------------------------
# failure modes of the file itself
# ---------------------------------------------------------------------------


def test_an_unreadable_record_is_an_unnarrowed_record_on_both_sides(stamped_root):
    """A record that exists but cannot be read narrows nothing, on both sides.

    The divergence this pins would surface only as one layer refusing a write
    another allowed, which is the failure mode least likely to be noticed.
    """
    # Arrange
    path = write_record(stamped_root, {"va": "sandbox"})
    path.chmod(0o000)
    posture_store.invalidate_cache()

    # Act / Assert
    try:
        if os.access(path, os.R_OK):  # pragma: no cover - root can read anything
            pytest.skip("this user can read a mode-000 file; the rule is untestable here")
        assert reader.recorded_posture({}) == posture_store.recorded_posture() == {}
        assert reader.effective_writes_for({}, ARMED_BOTH, "va") is True
    finally:
        path.chmod(0o600)
        posture_store.invalidate_cache()


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
    # Arrange — nothing narrowed in the record at all; only the run is pinned.
    write_record(stamped_root, {})
    monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "live=sandbox")

    # Act / Assert — the canonical reader refuses, the hook permits.
    assert posture_store.effective_writes(ARMED_BOTH, "live") is False
    assert reader.effective_writes_for({}, ARMED_BOTH, "live") is True

    # And the divergence is the STAMP, not the target: every other cell agrees.
    assert posture_store.effective_writes(ARMED_BOTH, "va") is True
    assert reader.effective_writes_for({}, ARMED_BOTH, "va") is True


def test_an_undecodable_record_is_an_unnarrowed_record_on_both_sides(tmp_path, monkeypatch):
    """Nothing may raise on a record that is not UTF-8 — in a hook or anywhere.

    ``UnicodeDecodeError`` is a ``ValueError``, so it slips past the ``OSError``
    guard a file read normally carries, and the canonical reader used to let it
    propagate while the hook swallowed it. That was a real divergence and it was
    the wrong way round: a hook has no caller to hand an exception to (an
    unhandled one exits non-zero with no JSON, which PreToolUse reads as "no
    opinion" — the opposite of what an unreadable record must mean), and the
    in-process readers are on the write path, where one mis-encoded file would
    raise into every posture lookup. Both now answer the unnarrowed record, the
    same way both already answer it for a truncated one.
    """
    # Arrange
    root = tmp_path / "var" / "agent_data"
    (root / reader.STATE_DIR_NAME).mkdir(parents=True)
    monkeypatch.setenv(reader.AGENT_DATA_ROOT_ENV_VAR, str(root))
    # Cleared for the reason ``stamped_root`` clears it: this test builds its own
    # root rather than taking that fixture, and an inherited launch pin would
    # make the canonical reader refuse a cell that is about the file's encoding.
    monkeypatch.delenv(posture_store.LAUNCH_POSTURE_ENV_VAR, raising=False)
    monkeypatch.delenv(reader.EXECUTION_MODE_ENV_VAR, raising=False)
    control_context.record_path_under(root).write_bytes(b"\xff\xfe{\x00schema\x00: 1}")
    posture_store.invalidate_cache()

    # Act / Assert — both answer, and the deployment ceiling stays in charge
    assert reader.read_record({}) is None
    assert reader.recorded_posture({}) == posture_store.recorded_posture() == {}
    assert reader.effective_writes_for({}, ARMED_BOTH, "live") is expected_answer(
        ARMED_BOTH, "live"
    )
    assert reader.effective_writes_for({}, ARMED_BOTH, "live") is True
    posture_store.invalidate_cache()


def test_the_degradation_hatch_writes_a_record_neither_side_honours(stamped_root):
    """A payload that is not a record at all: no schema, no identity fields."""
    write_payload(control_context.record_path_under(stamped_root), {"posture": {"live": "sandbox"}})

    assert reader.read_record({}) is None
    assert reader.recorded_posture({}) == posture_store.recorded_posture() == {}
