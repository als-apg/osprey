"""The hook's stdlib record readers against the modules they restate.

``osprey_connectors.posture_store`` and ``osprey_connectors.control_context`` are
the canonical readers of the control-context record; the PreToolUse hooks cannot
import them — they run outside the osprey venv — so ``osprey_target_state``
restates the parts a hook needs in stdlib terms. Two implementations of one
reader drift silently, and the drift is only ever visible at a write, so this
module pins them together: the posture-field parsers agree shape for shape, the
four resolvers of the record's directory land on one path, both sides read a
corrupt, unreadable or undecodable record as unreadable, and the switch
predicate is decided alike.

The write decision itself is not compared here. ``write_verdict`` is the one
verdict on both sides, and ``tests/hooks/test_write_verdict_twin.py`` drives the
shared case table through the twin and the package row by row.

Both sides are exercised IN PROCESS. The connector modules are imported here and
only here — a test may import what the code under test may not.

The second thing pinned here is :func:`osprey_target_state.posture_unknown`,
whose fail-closed rule has no counterpart in the canonical module at all — a
process the web terminal never stamped is refused until the deployment's
record exists, and answered by that record afterwards.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest

import osprey.templates.claude_code.claude.hooks.osprey_target_state as reader
from osprey_connectors import control_context, posture_env, posture_store
from osprey_connectors.types import switch_capable
from osprey_connectors_tests._harness import (
    pin_identity,
    state_dir_under,
    write_control_context,
    write_payload,
)

#: Every machine the record can switch on.
ALL_ON = {"live": "on", "va": "on", "standin": "on"}


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stamped_root(tmp_path, monkeypatch):
    """One agent-data root, stamped, so BOTH readers resolve the same record.

    The stamp is the anchor a session child carries; using it here is what makes
    the two implementations answer about one file rather than two. The unstamped
    derivation is pinned separately by the path test at the bottom.

    ``OSPREY_LAUNCH_POSTURE`` is cleared with the other two because a stamp
    inherited from the environment this suite runs in would pin the canonical
    reader in tests that are otherwise about the record.

    The container bind is cleared for a blunter reason: it outranks the stamp on
    both sides, so a suite run inside a bound container would put every test
    here against that deployment's own record rather than the one the test
    wrote. The directory itself comes from :func:`state_dir_under`, which is
    where every resolver below lands, identity hop included.
    """
    root = tmp_path / "var" / "agent_data"
    monkeypatch.setenv(reader.AGENT_DATA_ROOT_ENV_VAR, str(root))
    monkeypatch.delenv(reader.CONTROL_CONTEXT_DIR_ENV_VAR, raising=False)
    monkeypatch.delenv(reader.EXECUTION_MODE_ENV_VAR, raising=False)
    monkeypatch.delenv(reader.POSTURE_SESSION_ENV_VAR, raising=False)
    monkeypatch.delenv(posture_store.LAUNCH_POSTURE_ENV_VAR, raising=False)
    state_dir_under(root).mkdir(parents=True)
    posture_store.invalidate_cache()
    yield root
    posture_store.invalidate_cache()


def write_record(root, posture, switch=None):
    """Write the record both readers will read: *switch*, and *posture* verbatim.

    The production writer stores the mirror of the switch in ``posture``, so an
    old narrowing is staged by overwriting that field afterwards — the shape a
    file written before the mirror still has on disk.
    """
    path = write_control_context(root, switch=switch)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["posture"] = posture
    write_payload(path, payload)
    posture_store.invalidate_cache()
    return path


#: Every stdlib-only hook that restates the read-only run, and the names it
#: spells the variable and its sandboxing value with. The hooks cannot import
#: ``osprey_connectors``, so each holds its own copy; these are the copies.
READ_ONLY_RESTATEMENTS = [
    ("osprey_target_state", "EXECUTION_MODE_ENV_VAR", "SANDBOX_MODE"),
    ("osprey_writes_check", "EXECUTION_MODE_ENV_VAR", "SANDBOX_MODE"),
    ("osprey_hook_log", "EXECUTION_MODE_ENV", "READONLY_MODE"),
]


@pytest.mark.parametrize(
    ("hook", "variable", "value"),
    READ_ONLY_RESTATEMENTS,
    ids=[r[0] for r in READ_ONLY_RESTATEMENTS],
)
def test_every_hook_spells_the_read_only_run_as_posture_env_does(
    hook_module, hook, variable, value
):
    """The variable and the value are the wire contract; a renamed one sandboxes nobody.

    :mod:`osprey_connectors.posture_env` owns them. A hook whose copy drifted
    would read an ordinary run as read-only, or a read-only run as writable,
    with nothing else noticing.
    """
    # The target-state reader is a library sibling rather than a hook entry
    # point, so it is imported as a module here rather than through the seam.
    module = reader if hook == "osprey_target_state" else hook_module(hook)

    assert getattr(module, variable) == posture_env.POSTURE_ENV_VAR
    assert getattr(module, value) == posture_env.SANDBOX_MODE


@pytest.mark.parametrize("raw", [None, "", "readonly", "readwrite", "READONLY", " readonly"])
def test_the_hook_answers_is_readonly_run_as_posture_env_does(monkeypatch, raw):
    """A value comparison on both sides, never presence: garbage sandboxes nothing."""
    if raw is None:
        monkeypatch.delenv(posture_env.POSTURE_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(posture_env.POSTURE_ENV_VAR, raw)

    assert reader.is_readonly_run() is posture_env.is_readonly_run()


def test_a_blank_session_key_is_no_session_key(stamped_root, monkeypatch):
    """Whitespace is not a key. It attributes a stamp; it narrows nothing."""
    # Arrange
    write_record(stamped_root, {}, ALL_ON)
    monkeypatch.setenv(reader.POSTURE_SESSION_ENV_VAR, "   ")

    # Act / Assert
    assert reader.session_key() is None


def test_a_corrupt_record_is_unreadable_on_both_sides(stamped_root):
    """A record that is there and cannot be parsed is unreadable, never missing.

    The canonical reader fails closed on an unreadable record — it cannot tell
    which machines the owner turned on — and the hook may not choose the other
    answer. The verdict that follows is the case table's ``record-corrupt``.
    """
    # Arrange
    path = control_context._record_path_under(stamped_root)
    path.write_text('{"schema": 1, "target": "live"', encoding="utf-8")  # truncated
    posture_store.invalidate_cache()

    # Act / Assert
    assert reader.read_record({}) is None
    assert reader.read_record_status({}) == (None, "unreadable")
    assert control_context.read_record_status(path=path)[1] == "unreadable"


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
# one path, four resolvers
# ---------------------------------------------------------------------------
# The posture store, the control-context module, the controls server's own
# ``target_state`` and the stdlib hook each resolve this directory for a caller
# that cannot import the others. Four ladders is four chances to disagree, and
# the disagreement is invisible: every reader answers "no record", which is a
# legitimate answer meaning "nothing is narrowed".


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
    config.write_text(
        f"project_root: {tmp_path}\ncontrol_system:\n  type: virtual_accelerator\n  connector:\n    virtual_accelerator:\n      serving: in_process\n"
    )
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


def test_a_staged_configs_foreign_root_moves_neither_reader(tmp_path, monkeypatch):
    """Writer and both readers, on a config that was staged somewhere else.

    A deployed service reads a config flattened onto the build host and mounted
    into the container, so its ``project_root`` names a directory on the host and
    nothing here. Both ladders drop that value and unwrap the config's own zone
    instead, which is the repo the service runs in — the case the row above
    cannot see, because the root it records is a directory that exists.

    This is the shape the whole module is for: a disagreement here is invisible
    at the seam, because a reader looking in a directory nobody writes answers
    "no record", which is a legitimate answer meaning "nothing is narrowed".
    """
    # Arrange
    from osprey_connectors.workspace import reset_config_cache

    render = tmp_path / "build"
    render.mkdir()
    config = render / "config.yml"
    config.write_text(
        "project_root: /home/runner/work/osprey/osprey/stack\ncontrol_system:\n  type: virtual_accelerator\n  connector:\n    virtual_accelerator:\n      serving: in_process\n"
    )
    monkeypatch.delenv(reader.AGENT_DATA_ROOT_ENV_VAR, raising=False)
    monkeypatch.setenv("OSPREY_CONFIG", str(config))
    monkeypatch.setenv("CONFIG_FILE", str(config))
    reset_config_cache()
    posture_store.invalidate_cache()

    # Act
    hook_record = reader.record_path({})
    hook_dir = reader.resolve_state_dir({})
    canonical = control_context.record_path()
    canonical_dir = posture_store.state_dir()

    # Assert
    try:
        assert canonical is not None
        assert os.path.realpath(hook_record) == os.path.realpath(str(canonical))
        assert os.path.realpath(hook_dir) == os.path.realpath(str(canonical_dir))
        assert os.path.realpath(hook_dir).startswith(os.path.realpath(str(tmp_path)))
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
    monkeypatch.delenv(reader.CONTROL_CONTEXT_DIR_ENV_VAR, raising=False)
    identity = pin_identity(monkeypatch)

    assert os.path.dirname(reader.record_path({})) == reader.resolve_state_dir({})
    assert reader.resolve_state_dir({}) == os.path.join(str(root), reader.STATE_DIR_NAME, identity)
    assert reader.STATE_DIR_NAME == posture_store.STATE_DIR_NAME
    assert reader.RECORD_FILENAME == control_context.RECORD_FILENAME
    assert reader.REPORT_FILE_PREFIX == control_context.REPORT_FILE_PREFIX
    assert reader.REPORT_FILE_SUFFIX == control_context.REPORT_FILE_SUFFIX
    assert reader.AGENT_DATA_ROOT_ENV_VAR == posture_store.AGENT_DATA_ROOT_ENV_VAR
    assert reader.POSTURE_SANDBOX == posture_store.POSTURE_SANDBOX
    assert reader.POSTURE_WRITES == posture_store.POSTURE_WRITES
    assert set(reader.VALID_POSTURES) == set(posture_store.VALID_POSTURES)


def test_the_container_bind_is_the_whole_directory_on_all_four_resolvers(tmp_path, monkeypatch):
    """One bind value, four resolvers, one directory — and no identity appended.

    A multi-user container mounts one operator's state at a fixed path and binds
    it; the bind has therefore already made the identity hop, and a resolver
    that appended the acting identity to it would sit one directory below the
    record its own web terminal writes. The stamp is set to a DIFFERENT root on
    purpose: each assertion then fails if its resolver consulted the stamp
    first, rather than merely agreeing with the others by accident.

    ``target_state`` is imported inside the test: it is the controls server's
    resolver rather than a hook-side one, and this is the only cell that needs it.
    """
    # Arrange
    from osprey.mcp_server.control_system import target_state

    bind = tmp_path / "bound-state"
    monkeypatch.setenv(reader.CONTROL_CONTEXT_DIR_ENV_VAR, str(bind))
    monkeypatch.setenv(reader.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path / "not-this-root"))
    pin_identity(monkeypatch)
    posture_store.invalidate_cache()

    # Act / Assert
    assert posture_store.state_dir() == bind
    assert control_context.state_dir() == bind
    assert target_state.state_dir() == bind
    assert reader.resolve_state_dir({}) == str(bind)
    assert control_context.record_path() == bind / control_context.RECORD_FILENAME
    assert reader.record_path({}) == str(bind / reader.RECORD_FILENAME)


def test_the_stamped_path_and_the_explicit_root_path_are_one_path(tmp_path, monkeypatch):
    """``record_path()`` and ``_record_path_under(agent_data_root())`` coincide.

    The two spellings serve different callers — one reads the environment, the
    other is handed a root by a test fixture or by a writer preparing a
    deployment — and both make the identity hop. They are pinned together
    because every suite downstream writes through the second and asserts through
    the first: if they parted, those suites would lay a record down in one
    directory and read "nothing is narrowed" from another.

    With no bind set, because the bind deliberately does NOT apply to the
    explicit-root spelling: a caller passing a root has said which tree it means.
    """
    # Arrange
    monkeypatch.delenv(reader.CONTROL_CONTEXT_DIR_ENV_VAR, raising=False)
    monkeypatch.setenv(reader.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path))
    pin_identity(monkeypatch)
    posture_store.invalidate_cache()

    # Act
    root = posture_store.agent_data_root()

    # Assert
    assert root == tmp_path
    assert control_context.record_path() == control_context._record_path_under(root)
    assert control_context.state_dir() == state_dir_under(root)
    assert reader.record_path({}) == str(control_context._record_path_under(root))


def test_the_retired_per_session_store_is_read_by_neither_side(stamped_root):
    """The posture is the record's field now, and nothing else is a posture.

    ``session-postures.json`` is retired. A file left at that name narrows
    nothing on either side — a reader that still honoured it would apply one
    operator's stale, session-keyed narrowing on a deployment whose record says
    the machine is open.
    """
    # Arrange — the retired file says sandbox for everything; the record does not
    retired = state_dir_under(stamped_root) / "session-postures.json"
    retired.write_text(json.dumps({"k": {"live": "sandbox", "va": "sandbox"}}), encoding="utf-8")
    write_record(stamped_root, {}, ALL_ON)

    # Act / Assert
    record, status = reader.read_record_status({})
    assert status == "ok"
    assert record["switch"] == ALL_ON
    assert reader.read_record({})["posture"] == {}


# ---------------------------------------------------------------------------
# every write-path verdict names its machine
# ---------------------------------------------------------------------------

#: Directories scanned for production calls of the canonical rule. Tests are
#: excluded: this file itself compares the two ceilings on purpose, and so may
#: any other test that documents the difference.
_PRODUCTION_ROOTS = ("packages", "src")

#: The posture store's write-path question, taking the target the write lands
#: on as its second argument. A caller that hands it no target has stopped
#: naming the machine it writes to.
_TARGETED_RULES = ("write_verdict",)


def _production_python_files():
    """Every non-test ``.py`` file under :data:`_PRODUCTION_ROOTS`."""
    repo_root = Path(__file__).resolve().parents[2]
    for root in _PRODUCTION_ROOTS:
        for path in (repo_root / root).rglob("*.py"):
            if "tests" in path.parts or "test" in path.parts:
                continue
            yield path


def _verdict_calls(path):
    """``(lineno, call)`` for every call of a :data:`_TARGETED_RULES` name in *path*."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):  # pragma: no cover - an unreadable file is not a caller
        return
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in _TARGETED_RULES:
            yield node.lineno, node
        elif isinstance(func, ast.Name) and func.id in _TARGETED_RULES:
            yield node.lineno, node


def _names_a_target(call):
    """Whether *call* hands ``write_verdict`` a target that is not ``None``.

    Positionally, ``target`` is the second argument. As a keyword it must not be
    the literal ``None`` — a caller spelling ``target=None`` has explicitly
    declined to name the machine, which is what this pin forbids on a write path.
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


def test_no_production_caller_decides_a_write_without_its_target():
    """Every production ``write_verdict`` call names the machine it writes to.

    It fails in CI rather than at a write, which is the whole point.
    """
    # Arrange / Act
    offenders = [
        f"{path}:{lineno}"
        for path in _production_python_files()
        for lineno, call in _verdict_calls(path)
        if not _names_a_target(call)
    ]

    # Assert
    assert offenders == [], (
        "these callers of posture_store.write_verdict pass no target, so they "
        "decide a write without naming its machine: "
        + ", ".join(offenders)
        + ". Pass the target the write lands on, or move the call off the write path."
    )


def test_the_guard_can_see_a_violation():
    """A guard is only worth having if it would actually catch one."""
    module = ast.parse(
        "posture_store.write_verdict(section)\n"
        "posture_store.write_verdict(section, target=None)\n"
        "posture_store.write_verdict(section, target)\n"
        "posture_store.write_verdict(section, target=t)\n"
        "posture_store.write_verdict(section, None, owner='alice')\n"
    )
    calls = [node for node in ast.walk(module) if isinstance(node, ast.Call)]
    assert [_names_a_target(call) for call in calls] == [False, False, True, True, True]


def test_the_guard_actually_scans_the_callers_that_exist():
    """Not vacuous: the sweep must be finding real calls to have an opinion.

    A guard that silently matched nothing — a moved directory, a renamed
    function — would pass forever while the thing it protects rots.
    """
    repo_root = Path(__file__).resolve().parents[2]
    found = [
        (path.relative_to(repo_root).parts[0], path.name)
        for path in _production_python_files()
        for _lineno, _call in _verdict_calls(path)
    ]
    assert found, "the reachability guard found no callers at all; its scan has gone stale"
    # Both roots are really scanned: the package's own write path and the
    # framework's connector write tool each ask the verdict.
    assert ("packages", "_pool.py") in found, found
    assert ("src", "channel_write.py") in found, found


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
    config.write_text(
        f"project_root: {tmp_path}\ncontrol_system:\n  type: virtual_accelerator\n  connector:\n    virtual_accelerator:\n      serving: in_process\n"
    )
    monkeypatch.delenv(reader.AGENT_DATA_ROOT_ENV_VAR, raising=False)
    monkeypatch.setenv("OSPREY_CONFIG", str(config))
    monkeypatch.setenv("CONFIG_FILE", str(config))
    reset_config_cache()

    try:
        assert reader.posture_unknown({}) is True
    finally:
        reset_config_cache()
        posture_store.invalidate_cache()


def test_an_unstamped_hook_and_the_store_derive_one_root_under_a_relocated_base_dir(
    tmp_path, monkeypatch
):
    """With no stamp, the hook and the store both follow ``agent_data.base_dir``.

    Paths are compared resolved: the framework side resolves its answer, and a
    temporary directory can sit under a symlink.
    """
    from osprey_connectors.workspace import reset_config_cache

    config = tmp_path / "config.yml"
    config.write_text(f"project_root: {tmp_path}\nagent_data:\n  base_dir: relocated/agent_data\n")
    monkeypatch.delenv(reader.AGENT_DATA_ROOT_ENV_VAR, raising=False)
    monkeypatch.delenv(reader.CONTROL_CONTEXT_DIR_ENV_VAR, raising=False)
    monkeypatch.setenv("OSPREY_CONFIG", str(config))
    monkeypatch.setenv("CONFIG_FILE", str(config))
    reset_config_cache()
    posture_store.invalidate_cache()

    try:
        hook_root = Path(reader.agent_data_root({})).resolve()
        expected = (tmp_path / "relocated/agent_data").resolve()

        assert hook_root == posture_store.agent_data_root().resolve() == expected
        assert Path(reader.resolve_state_dir({})).resolve().is_relative_to(expected)
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
    config.write_text(
        f"project_root: {tmp_path}\ncontrol_system:\n  type: virtual_accelerator\n  connector:\n    virtual_accelerator:\n      serving: in_process\n"
    )
    monkeypatch.delenv(reader.AGENT_DATA_ROOT_ENV_VAR, raising=False)
    monkeypatch.setenv("OSPREY_CONFIG", str(config))
    monkeypatch.setenv("CONFIG_FILE", str(config))
    reset_config_cache()

    try:
        write_control_context(tmp_path / "var" / "agent_data", switch={"va": "on"})

        assert reader.posture_unknown({}) is False
        record, status = reader.read_record_status({})
        assert status == "ok"
        assert record["switch"] == {"va": "on"}
    finally:
        reset_config_cache()
        posture_store.invalidate_cache()


def test_a_stamped_process_is_never_posture_unknown(stamped_root):
    """The stamp is the deployment handing the directory over.

    A stamped process was told where to look, so an empty directory there means
    exactly what it says — no record, every switch off — and is not refused as
    an unknown posture on top of that.
    """
    assert reader.posture_unknown({}) is False

    write_record(stamped_root, {"live": "sandbox"})
    assert reader.posture_unknown({}) is False


# ---------------------------------------------------------------------------
# failure modes of the file itself
# ---------------------------------------------------------------------------


def test_an_unreadable_record_is_unreadable_on_both_sides(stamped_root):
    """A record that exists but cannot be read is unreadable, on both sides.

    Even one whose switch was on: an unreadable record cannot say so. The
    divergence this pins would surface only as one layer refusing a write
    another allowed, which is the failure mode least likely to be noticed.
    """
    # Arrange
    path = write_record(stamped_root, {}, ALL_ON)
    path.chmod(0o000)
    posture_store.invalidate_cache()

    # Act / Assert
    try:
        if os.access(path, os.R_OK):  # pragma: no cover - root can read anything
            pytest.skip("this user can read a mode-000 file; the rule is untestable here")
        assert reader.read_record_status({}) == (None, "unreadable")
        assert control_context.read_record_status(path=path)[1] == "unreadable"
    finally:
        path.chmod(0o600)
        posture_store.invalidate_cache()


def test_an_undecodable_record_is_unreadable_on_both_sides(tmp_path, monkeypatch):
    """Nothing may raise on a record that is not UTF-8 — in a hook or anywhere.

    ``UnicodeDecodeError`` is a ``ValueError``, so it slips past the ``OSError``
    guard a file read normally carries, and the canonical reader used to let it
    propagate while the hook swallowed it. That was a real divergence and it was
    the wrong way round: a hook has no caller to hand an exception to (an
    unhandled one exits non-zero with no JSON, which PreToolUse reads as "no
    opinion" — the opposite of what an unreadable record must mean), and the
    in-process readers are on the write path, where one mis-encoded file would
    raise into every posture lookup. Both now answer without raising, and the
    answer is the one both give a truncated record: unreadable, so refused.
    """
    # Arrange
    root = tmp_path / "var" / "agent_data"
    monkeypatch.setenv(reader.AGENT_DATA_ROOT_ENV_VAR, str(root))
    # Cleared for the reason ``stamped_root`` clears them: this test builds its
    # own root rather than taking that fixture, and an inherited launch pin would
    # make the canonical reader refuse a cell that is about the file's encoding,
    # while an inherited bind would point both readers at another deployment.
    monkeypatch.delenv(posture_store.LAUNCH_POSTURE_ENV_VAR, raising=False)
    monkeypatch.delenv(reader.EXECUTION_MODE_ENV_VAR, raising=False)
    monkeypatch.delenv(reader.CONTROL_CONTEXT_DIR_ENV_VAR, raising=False)
    state_dir_under(root).mkdir(parents=True)
    control_context._record_path_under(root).write_bytes(b"\xff\xfe{\x00schema\x00: 1}")
    posture_store.invalidate_cache()

    # Act / Assert — both answer, and both refuse
    assert reader.read_record({}) is None
    assert reader.read_record_status({}) == (None, "unreadable")
    path = control_context._record_path_under(root)
    assert control_context.read_record_status(path=path)[1] == "unreadable"
    posture_store.invalidate_cache()


def test_the_degradation_hatch_writes_a_record_neither_side_honours(stamped_root):
    """A payload that is not a record at all: no schema, no identity fields."""
    write_payload(
        control_context._record_path_under(stamped_root), {"posture": {"live": "sandbox"}}
    )

    assert reader.read_record({}) is None
    assert reader.read_record_status({}) == (None, "unreadable")


# ---------------------------------------------------------------------------
# the switch predicate
# ---------------------------------------------------------------------------

_GW = {"gateways": {"read_only": {"address": "gw"}}}


@pytest.mark.parametrize(
    "section",
    [
        pytest.param({"type": "epics", "connector": {"epics": {"gateways": {}}}}, id="live-only"),
        pytest.param(
            {"type": "virtual_accelerator", "connector": {"virtual_accelerator": {"port": 5064}}},
            id="va-only",
        ),
        pytest.param(
            {"type": "live_standin", "connector": {"live_standin": {"port": 5074}}},
            id="standin-only",
        ),
        pytest.param({}, id="no-type"),
        pytest.param(
            {"type": "epics", "connector": {"epics": _GW, "virtual_accelerator": {"port": 5064}}},
            id="live-and-va",
        ),
        pytest.param(
            {
                "type": "virtual_accelerator",
                "connector": {
                    "virtual_accelerator": {"writes_enabled": True},
                    "live_standin": {"port": 5074},
                },
            },
            id="va-and-standin",
        ),
        pytest.param(
            {
                "type": "live_standin",
                "connector": {
                    "epics": _GW,
                    "virtual_accelerator": {"writes_enabled": True},
                    "live_standin": {"port": 5074},
                },
            },
            id="standin-baseline-three-blocks",
        ),
        pytest.param(
            {
                "type": "virtual_accelerator",
                "connector": {"epics": _GW, "virtual_accelerator": {"serving": "in_process"}},
            },
            id="in-process-va-and-epics",
        ),
        pytest.param(None, id="none"),
        pytest.param([], id="list"),
        pytest.param("epics", id="string"),
    ],
)
def test_the_hook_decides_switch_capability_as_the_connector_module_does(section):
    """The stdlib mirror and the framework predicate answer every shape alike."""
    assert reader._switch_capable(section) == switch_capable(section)
