"""The test guards that hold without the framework installed.

The connectors suite runs where only ``osprey-connectors`` is installed, so the
guards both suites need live here and both conftests import them by name. Not
as a plugin: a conftest's autouse fixtures are ordered by name within that
conftest, and the "set up before" reasoning below depends on that order, while
a plugin's fixtures would all run first. In a run that collects both trees the
session-level hooks are called once per conftest, so each is idempotent.

Nothing here imports ``osprey`` or the framework's ``tests`` package.
"""

import logging
import os
import sys
import time
import uuid
from pathlib import Path

import pytest
from rich.logging import RichHandler

from osprey_connectors.logger import _QUIET_THIRD_PARTY_LOGGERS
from osprey_connectors_tests._harness import (
    _env_scope_guard,
    _live_threads,
    _repo_cleanliness,
    ci_diagnostics,
)
from osprey_connectors_tests._harness._env_scope_guard import restore_module_environment
from osprey_connectors_tests._paths import REPO_ROOT as _REPO_ROOT

# ``_REPO_ROOT`` is the checkout this package sits in: the fallback when a test
# leaves the process in a deleted cwd, and the root whose ``var/`` the guards
# below watch.

#: Root-logging state as it was before collection — see ``restore_root_logging``.
#: Captured at conftest import, which is the only moment guaranteed to precede
#: every test *and* every higher-scoped fixture.
_PRISTINE_LOGGING = {
    "root_level": logging.getLogger().level,
    "root_handlers": list(logging.getLogger().handlers),
    "third_party": {name: logging.getLogger(name).level for name in _QUIET_THIRD_PARTY_LOGGERS},
}


# ===================================================================
# Collection-time environment pollution guard
# ===================================================================
#
# Importing a test module can import third-party code that loads a .env at
# import time — litellm calls dotenv.load_dotenv() on import, and python-dotenv
# walks UP from its package directory, so on a developer machine it can find an
# ancestor checkout's .env and inject real credentials, PROJECT_ROOT, and TZ
# into os.environ before the first test runs. CI has no ancestor .env, so
# whatever those variables change locally is invisible there.
#
# Only that injection is undone. A blanket restore would also strip the
# coordination variables some test modules legitimately export at import time
# (tests/va/* set EPICS_CA_SERVER_PORT et al. so the soft-IOC subprocesses they
# spawn share their ports), so a key is removed only when it BOTH appeared
# during collection AND carries exactly the value an ancestor .env defines.

# import-time required because `osprey.cli.styles` builds its module-level
# Rich Console at import time and Rich reads FORCE_COLOR in `Console.__init__`,
# so a fixture runs too late to un-force a console that already exists: the
# color-forcing variables must be gone BEFORE collection imports any osprey
# module. A developer terminal that
# exports FORCE_COLOR otherwise turns CliRunner captures into ANSI-laced text
# and fails seven CLI tests on pristine main — CI is green only because its
# runners never export it. `NO_COLOR` is left alone: unset is the same default
# CI runs under.
os.environ.pop("FORCE_COLOR", None)
os.environ.pop("CLICOLOR_FORCE", None)

# Rich deliberately clamps ``TERM=dumb`` consoles to 80 columns and suppresses
# cursor control even when a test explicitly requests ``force_terminal=True``.
# CI presents a capable TERM, so normalize only the dumb/unknown host values;
# tests for a restricted terminal can still set one explicitly after collection.
if os.environ.get("TERM", "").lower() in {"dumb", "unknown"}:
    os.environ["TERM"] = "xterm-256color"

# LiteLLM's model map decides which model ids it can route. osprey._litellm_env
# defaults it to the copy LiteLLM ships; here it is assigned, not defaulted, so
# no developer export can make a run fetch the live map from GitHub and the
# routing every test sees is the same on a laptop, on CI, and offline.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

_PRE_COLLECTION_ENV = dict(os.environ)


def _ancestor_dotenv_values() -> dict[str, str]:
    """Every assignment an ancestor .env could have injected, keyed by var."""
    from dotenv import dotenv_values

    injected: dict[str, str] = {}
    for directory in (_REPO_ROOT, *_REPO_ROOT.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            injected.update({k: v for k, v in dotenv_values(candidate).items() if v is not None})
    return injected


def pytest_collection_finish():
    """Undo ancestor-.env injection performed by imports during collection."""
    import time

    added = set(os.environ) - set(_PRE_COLLECTION_ENV)
    if added:
        injected = _ancestor_dotenv_values()
        for key in added:
            if key in injected and os.environ[key] == injected[key]:
                del os.environ[key]
    # Re-sync the C library's cached zone with the (possibly restored) TZ.
    # Collection can leave the two disagreeing (TZ injected, cache not), and
    # then the first test to call tzset() flips tzname mid-suite and reads as
    # a leaker to _no_host_timezone_leak below.
    time.tzset()


# ===================================================================
# Environment guard
# ===================================================================
#
# Ordering note: autouse fixtures of equal scope are set up in *alphabetical*
# order, not declaration order — `pytest --setup-plan <test>` prints the real
# sequence. So `_no_host_timezone_leak` is the outermost function-scoped fixture
# here (leading underscore), not this one, and anything that must sit outside
# that guard's snapshot window has to be session-scoped rather than merely
# declared earlier.


@pytest.fixture(autouse=True, scope="function")
def restore_environ():
    """Snapshot and restore the whole of ``os.environ`` around every test.

    Leak guarded: plenty of code writes ``os.environ`` directly rather than
    through ``monkeypatch`` — ``.env`` loading in the health CLI and the config
    loader uses override semantics, and the web terminal lifespan assigns
    ``OSPREY_CONFIG`` by hand. Without a full snapshot those writes survive into
    sibling tests, which under xdist makes results depend on how the files
    happened to be distributed.

    ``OSPREY_CONFIG`` and ``CONFIG_FILE`` are additionally cleared on the way in,
    so a developer who exports either in their shell still gets a pristine run.
    ``TZ`` is handled once per session instead -- see ``_clear_dotenv_timezone``.

    This snapshot cannot see writes made by a *module*-scoped fixture, which is
    set up before it — ``restore_environ_per_module`` below covers those.
    """
    saved = dict(os.environ)

    os.environ.pop("OSPREY_CONFIG", None)
    os.environ.pop("CONFIG_FILE", None)

    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


@pytest.fixture(autouse=True, scope="module")
def restore_environ_per_module():
    """Snapshot and restore ``os.environ`` around every test *module*.

    Module-scoped fixtures — a real ``osprey build`` over a seeded exemplar repo,
    which loads that repo's ``.env`` with override semantics — are set up before
    the function-scoped snapshot above is taken, so their writes land inside that
    snapshot's ``saved`` copy and are restored, not removed, by every later test.
    This restores the environment at module teardown so they cannot outlive the
    module that created them.

    A fixture that outlives the module keeps its writes: the plugin registered
    in :func:`pytest_configure` records them, and
    :func:`~osprey_connectors_tests._harness._env_scope_guard.restore_module_environment`
    replays them on top of the snapshot. Without that, a session-scoped fixture first
    requested by this module's second test would be left alive with its
    environment rolled back under it.
    """
    with restore_module_environment():
        yield


@pytest.fixture(autouse=True, scope="function")
def session_posture_leak_guard(monkeypatch):
    """Run every test outside a web-terminal session's write posture.

        Three variables carry that posture: ``OSPREY_EXECUTION_MODE`` (a read-only
        run), and the two anchors ``OSPREY_POSTURE_SESSION`` and
        ``OSPREY_AGENT_DATA_ROOT``, which the web server always stamps as a pair.
        A developer running the suite from inside a narrowed session would
        otherwise hand every record-touching test a session id and an
        agent-data root no test asked for — and, worse, a test that
        redirects the record by patching ``resolve_shared_data_root`` would be
        silently inert, because ``posture_store.agent_data_root()`` reads the
        variable FIRST and only falls back to the resolver.

        Two more carry the container's own control-state binds:
        ``OSPREY_CONTROL_CONTEXT_DIR``, the one identity's directory, and
        ``OSPREY_CONTROL_CONTEXT_TREE``, every identity's. They are the rung
        ABOVE the agent-data root on all four resolvers, so a suite run inside a
        bound container resolves every control-state path to that deployment —
        past the stamp a test set and past the resolver it patched, which is the
        same silent inertness one rung higher, and aimed at a real deployment's
        record rather than a throwaway.

    All five are CLEARED, not pointed elsewhere. Stamping a throwaway root here
        would look tidier — clearing the variable does not stop a write, it aims it
        at ``<repo>/var/agent_data`` via ``resolve_shared_data_root()`` — but the
        stamp is preferred over the resolver by both readers, so a suite-wide stamp
        silently overrides every test that redirects the root by patching
        ``resolve_shared_data_root``, which is most of them: measured, it turns 90
        tests across ``tests/mcp_server`` red and the whole hooks tree with it.
        The leak is therefore closed where it is caused — a test whose code writes
        under that root stamps its own — and kept closed by
        :func:`no_agent_data_in_the_repo` below, which fails the session if the run
        created the directory.

        Named without a leading underscore on purpose. Autouse fixtures of equal
        scope are set up in alphabetical order, so this sorts AFTER
        ``restore_environ`` and is therefore torn down before it — a name like
        ``_no_session_posture`` would delete the variables before that snapshot was
        taken, and the restore would then drop them for the rest of the session.
    """
    for anchor in (
        "OSPREY_EXECUTION_MODE",
        "OSPREY_POSTURE_SESSION",
        "OSPREY_AGENT_DATA_ROOT",
        "OSPREY_CONTROL_CONTEXT_DIR",
        "OSPREY_CONTROL_CONTEXT_TREE",
    ):
        monkeypatch.delenv(anchor, raising=False)
    yield


_REAL_DEPLOYMENT_LANES = ("tests/e2e/", "tests/va/e2e/")

#: ``<repo>/var/agent_data`` — the directory :func:`no_agent_data_in_the_repo`
#: watches. Spelled once, at import, because the creator hook below needs it
#: before any fixture has run.
_AGENT_DATA_MARKER = _REPO_ROOT / "var" / "agent_data"

#: ``<repo>/var/spool`` — the spool root the record writer falls back to when
#: ``OSPREY_SPOOL_DIR`` is unset, watched by :func:`no_spool_in_the_repo`. It
#: hangs off the same resolver as the agent-data root, so the two leak together
#: and are watched together.
_SPOOL_MARKER = _REPO_ROOT / "var" / "spool"

#: The directory as it was when this conftest was imported — the only moment
#: guaranteed to precede every test and every higher-scoped fixture. A
#: directory that was already there belongs to a real local deployment, and
#: what is IN it is that deployment's business; what this run adds to it is
#: this suite's, because the first leak would otherwise leave the guard
#: disarmed for every run after it.
_AGENT_DATA_BASELINE = _repo_cleanliness.snapshot(_AGENT_DATA_MARKER)

#: The same baseline for the spool root, taken at the same moment.
_SPOOL_BASELINE = _repo_cleanliness.snapshot(_SPOOL_MARKER)

#: The first test in THIS worker whose teardown found this run's mark on the
#: directory, ``None`` while none has. Recorded by
#: :func:`pytest_runtest_teardown` and read by the guard.
_AGENT_DATA_FIRST_SEEN: str | None = None

#: The same bound for ``<repo>/var/spool``.
_SPOOL_FIRST_SEEN: str | None = None

#: Whether the item now running is a real-deployment one. Written by
#: :func:`pytest_runtest_setup`, cleared by :func:`pytest_runtest_teardown`,
#: and read by the divert in :func:`agent_data_never_the_checkout`, which is
#: installed process-wide and so has no other way to tell whose write it is
#: about to place. Per ITEM, not per collection: a session that mixes one
#: real-deployment item with a unit lane must divert the unit lane.
_REAL_DEPLOYMENT_ITEM_RUNNING = False

#: Whether any real-deployment item has run in THIS worker. Once one has, both
#: repo guards stand down for the rest of the session: those lanes write under
#: ``var/`` in this checkout on purpose, and nothing distinguishes their
#: directory from a leaked one after the fact.
_REAL_DEPLOYMENT_LANE_RAN = False


def _is_real_deployment_item(item) -> bool:
    """Whether *item* runs against this checkout as a real deployment.

    The one predicate behind the divert, the repo guards and the spool zone,
    so a lane cannot be exempt from one of them and not the others.
    """
    return item.nodeid.startswith(_REAL_DEPLOYMENT_LANES)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Open the window in which resolution to this checkout is legitimate.

    ``tryfirst`` so the flag is already correct when the item's own fixtures
    are set up: a real-deployment fixture that resolves the project root during
    setup must see the checkout, not the divert.
    """
    global _REAL_DEPLOYMENT_ITEM_RUNNING, _REAL_DEPLOYMENT_LANE_RAN
    _REAL_DEPLOYMENT_ITEM_RUNNING = _is_real_deployment_item(item)
    _REAL_DEPLOYMENT_LANE_RAN = _REAL_DEPLOYMENT_LANE_RAN or _REAL_DEPLOYMENT_ITEM_RUNNING


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item):
    """Timestamp this run's mark on ``<repo>/var/{agent_data,spool}`` against a
    test, and close the real-deployment window the setup hook opened.

    The guards below run once per worker at session teardown and can only
    report *that* the directory appeared, which is the least useful half of the
    finding: the run is over, and the reader is left grepping a whole tree of
    app fixtures for whichever one resolved the agent-data root to the
    checkout. This narrows it to a window instead — the first test that ran
    with this run's mark on the directory, whether or not the directory itself
    was there before the run started.

    Deliberately reported as an upper bound rather than as blame. The writer is
    typically not the test named: the last one measured was a uvicorn daemon
    thread that ``ServerLauncher`` auto-launched out of an interface app's
    lifespan and that reached its ``mkdir`` some tests later, in a worker that
    had moved on. Under ``-n`` the naming worker may not even be the one that
    created it. What the name is good for is bounding the search — nothing in
    this worker before it can be the cause.

    One hook for every marker, because pytest calls it once per test and each
    marker stops being read after its own first hit — two comparisons at the
    start of a run, none once both findings are in hand. A comparison is one
    ``stat`` and one directory listing, cheap enough to leave armed for every
    lane rather than gated behind a flag nobody would set before the leak had
    already cost them an afternoon.

    A wrapper, because the two halves want opposite ends of the teardown. The
    comparison runs BEFORE every other implementation: pytest's own runs the
    fixture finalizers, and for the last item of a session those include the
    two guards below, so a comparison placed after them would never record a
    bound for the item that most needs one. The window it gives up in exchange
    is the item's own fixture teardown, which the next item's comparison covers
    — a bound is an upper bound, and the docstring above already says so. The
    flag is cleared AFTER them, so a real-deployment item's fixtures still
    resolve to the checkout while they finalize.
    """
    global _AGENT_DATA_FIRST_SEEN, _SPOOL_FIRST_SEEN
    global _REAL_DEPLOYMENT_ITEM_RUNNING
    if not _REAL_DEPLOYMENT_LANE_RAN:
        worker = os.environ.get("PYTEST_XDIST_WORKER", "master")
        if _AGENT_DATA_FIRST_SEEN is None and _repo_cleanliness.what_this_run_did(
            _AGENT_DATA_BASELINE, _AGENT_DATA_MARKER
        ):
            _AGENT_DATA_FIRST_SEEN = f"{item.nodeid} (worker {worker})"
        if _SPOOL_FIRST_SEEN is None and _repo_cleanliness.what_this_run_did(
            _SPOOL_BASELINE, _SPOOL_MARKER
        ):
            _SPOOL_FIRST_SEEN = f"{item.nodeid} (worker {worker})"
    try:
        yield
    finally:
        _REAL_DEPLOYMENT_ITEM_RUNNING = False


def _first_seen_clause(first_seen: str | None) -> str:
    """The "already present by the end of ..." half of a guard's failure."""
    if not first_seen:
        return ""
    return (
        f"\nalready present by the end of {first_seen} — the writer is at or "
        "before that point, and may be a background thread an earlier test started"
    )


#: The sentence both guards end on. Neither seam crosses a process boundary,
#: and neither reaches a module that bound its resolver at import time, so a
#: failure here bounds the search rather than naming the writer.
_GUARD_KNOWN_LIMIT = (
    "\nThis worker is the only one that can be quoted here, in both directions: "
    "a writer in another worker, in a subprocess, or in a module that bound its "
    "resolver at import time is not narrowed by the bound above — and under -n, "
    "a worker that drew only unit items can be failed by a directory an e2e item "
    "created on purpose in a worker of its own, in which case this failure is "
    "the wrong answer and the e2e lane is the explanation."
)


@pytest.fixture(autouse=True, scope="session")
def agent_data_never_the_checkout(tmp_path_factory):
    """Divert a resolution that lands on THIS checkout to a throwaway root.

    Every path to the agent-data root funnels through
    ``osprey_connectors.workspace.resolve_project_root`` — ``resolve_shared_data_root``
    and ``resolve_agent_data_root`` both anchor ``agent_data.base_dir`` on
    whatever it answers, and the record writer anchors its host spool on it too.
    It is also the only seam in that chain that is looked up by module global at
    call time, so binding it here reaches every caller regardless of how it
    imported its own resolver: the modules that did ``from ... import
    resolve_shared_data_root`` at import time still run the original function
    body, and that body reads this name.

    Session-scoped because the leak it closes is won by a RACE, not by a test.
    The directory is created inside a uvicorn daemon thread — the artifacts
    gallery that ``ServerLauncher`` auto-launches from an interface app's
    lifespan, whose ``store_watcher.start()`` mkdirs
    ``<root>/artifacts`` — and that thread outlives the test that started it by
    an unbounded margin. A function-scoped redirection is therefore not merely
    untidy but unsound: ``monkeypatch`` undoes it at teardown and the thread
    then resolves against the real checkout, which is exactly the intermittency
    that made this leak look like a different module every run. A module-scoped
    one loses the same race, one module later.

    A WRAPPER, not a constant. A blanket redirect would answer the same tmp
    directory for a test that configured its own ``project_root`` and asserts
    what a tool derived from it — so only the one case that constitutes the
    leak is diverted: resolution landing on the checkout the suite is running
    in. Everything else, including every test that patches a resolver of its
    own further up the stack, is untouched — this sits BELOW those patches
    rather than overriding them, which is what a suite-wide
    ``OSPREY_AGENT_DATA_ROOT`` stamp cannot say for itself (see
    :func:`session_posture_leak_guard`).

    Disarmed while a real-deployment item is running, on the same predicate as
    :func:`no_agent_data_in_the_repo`: those lanes run agents and servers with
    this checkout as the project root on purpose, and diverting it would be
    rewriting the thing under test.

    Disarmed per ITEM rather than per collection. The exemption asks whether
    the write happening now belongs to a real-deployment test, which is the
    only question its answer turns on; a collection-wide one answers for the
    whole session, so a single real-deployment item in a mixed selection would
    aim every unit test in the run at this checkout and the guards below would
    stand down over it.
    """
    from osprey_connectors import workspace as connector_workspace

    real_resolve_project_root = connector_workspace.resolve_project_root
    diverted = tmp_path_factory.mktemp("project-root-not-the-checkout")

    def _never_the_checkout(config=None):
        root = real_resolve_project_root(config)
        if _REAL_DEPLOYMENT_ITEM_RUNNING:
            return root
        try:
            landed_here = Path(root).resolve() == _REPO_ROOT
        except OSError:  # pragma: no cover - a root that vanished mid-run
            return root
        return diverted if landed_here else root

    # Installed for the lifetime of the process, with no undo. The thread this
    # exists to catch is a uvicorn daemon that outlives the fixture that started
    # it, so a redirect lifted at teardown leaves it resolving against the real
    # checkout for as long as the interpreter is still up -- the same unsoundness
    # a function-scoped redirect has, one scope later. Nothing runs after a
    # session fixture's teardown that needs the original back.
    connector_workspace.resolve_project_root = _never_the_checkout
    yield


@pytest.fixture(autouse=True, scope="session")
def no_agent_data_in_the_repo():
    """Fail the session if the suite created ``<repo>/var/agent_data``.

    The regression this exists for is silent by construction: ``var/`` is
    gitignored, the marker files are unlinked on the way out, and what is left
    behind is an empty directory that ``git status`` never mentions. It was
    found twice by hand; this is what finds it the third time.

    This is the guard that actually holds the line, rather than a suite-wide
    stamp of ``OSPREY_AGENT_DATA_ROOT`` — see
    :func:`session_posture_leak_guard`. A stamp would silence the symptom for
    every test at once, at the cost of overriding the resolver patch most
    store-touching tests use to redirect that root.

    Only what the RUN did is a failure. A directory that was already there is
    not one — that is what keeps this from firing on a developer who has
    actually run OSPREY in this checkout — but an entry this run put INTO it
    is, because a directory left behind by an interrupted run is
    indistinguishable from one a developer owns, and keying on existence hands
    the guard's silence to the first leak that happens here. The consequence is
    worth stating plainly: a live local deployment writing into that directory
    while the suite runs is now named as a failure. The message names the entry
    that appeared, so that case can be recognised for what it is.

    The real-deployment lanes (``tests/e2e/``, ``tests/va/e2e/``) are exempt:
    they run agents and servers with this checkout as the project root, so the
    executor's run folders land under ``var/agent_data`` by design, not by a
    fixture's mistake. Exempt once such an item has RUN in this worker, not
    once one has been collected — a mixed selection still guards its unit half,
    which a collection-wide exemption cannot do.
    """
    marker = _AGENT_DATA_MARKER
    yield
    if _REAL_DEPLOYMENT_LANE_RAN:
        return
    clause = _repo_cleanliness.what_this_run_did(_AGENT_DATA_BASELINE, marker)
    if clause:
        raise AssertionError(
            f"the test run {clause} — something resolved the agent-data root "
            "to the repository. A test that writes the posture store or a control-target "
            "state file must stamp OSPREY_AGENT_DATA_ROOT at a tmp path (see "
            "session_posture_leak_guard) rather than leave it to resolve_shared_data_root()."
            f"{_first_seen_clause(_AGENT_DATA_FIRST_SEEN)}"
            f"{_GUARD_KNOWN_LIMIT}"
        )


def _spool_leak(baseline, marker: Path, first_seen: str | None) -> str | None:
    """The failure :func:`no_spool_in_the_repo` raises for *marker*, or ``None``.

    Separate from the fixture so the message can be checked against a marker
    of a test's own, without writing into the checkout's spool.
    """
    clause = _repo_cleanliness.what_this_run_did(baseline, marker)
    if not clause:
        return None
    return (
        f"the test run {clause} — something wrote a record line with the spool "
        "resolved to the repository, where the agent-record service would ingest it "
        "as a fact that really happened. Every test gets a spool root of its own from "
        "the root conftest's _isolate_spool_zone, which sets OSPREY_SPOOL_DIR; a leak "
        "means something bypassed it: a test that unset the variable, a subprocess "
        "started with a scrubbed environment, or a thread that outlived its test."
        f"{_first_seen_clause(first_seen)}"
        f"{_GUARD_KNOWN_LIMIT}"
    )


@pytest.fixture(autouse=True, scope="session")
def no_spool_in_the_repo():
    """Fail the session if the suite wrote under ``<repo>/var/spool``.

    The twin of :func:`no_agent_data_in_the_repo`, watching the record
    writer's host fallback. A line left there is worse than an untidy
    directory: the next agent-record service started in this checkout ingests
    it into the record, where nobody can tell it from a fact that happened.

    Keyed on what the run did, the same way as its twin, and exempt once a
    real-deployment item has run in this worker, for their reason.
    """
    marker = _SPOOL_MARKER
    yield
    if _REAL_DEPLOYMENT_LANE_RAN:
        return
    message = _spool_leak(_SPOOL_BASELINE, marker, _SPOOL_FIRST_SEEN)
    if message:
        raise AssertionError(message)


# ===================================================================
# Host-timezone leak guard
# ===================================================================


@pytest.fixture(autouse=True, scope="session")
def _clear_dotenv_timezone():
    """Drop a ``.env``-injected ``TZ`` once, before any test runs.

    litellm calls a bare ``load_dotenv()`` at import, and the bare form searches
    *parent* directories -- so a worktree with no ``.env`` of its own still
    inherits the checkout's. That publishes ``TZ`` into ``os.environ`` during
    collection without a matching ``time.tzset()``, leaving the variable and the
    C library's cached zone disagreeing. The first test to call ``tzset()`` then
    reconciles them mid-test, and ``_no_host_timezone_leak`` reports the change
    against whichever test happened to make the call. It only bites where the
    host zone differs from the ``.env`` one, which is why CI -- which has no
    ``.env`` at all -- never sees it.

    Session scope is load-bearing: the clear has to land outside every
    function-scoped fixture's window, including the guard's own. Autouse
    fixtures of equal scope are ordered alphabetically, so ``_no_host_timezone``
    sets up first and tears down last; doing this per test would fall inside
    that window and read as a leak.
    """
    if os.environ.pop("TZ", None) is not None:
        time.tzset()
    yield


@pytest.fixture(autouse=True, scope="function")
def _no_host_timezone_leak():
    """Fail the test that leaves the process on a different host timezone.

    Finalizers run LIFO: a bare ``request.addfinalizer(time.tzset)`` runs
    *before* monkeypatch restores ``TZ``, leaving every later test on the
    wrong host zone. This fixture's teardown runs after monkeypatch's own,
    which is exactly the ordering the bug turns on.
    """
    import os as _os
    import time as _time

    before = (_os.environ.get("TZ"), _time.tzname)

    yield

    after = (_os.environ.get("TZ"), _time.tzname)
    assert after == before, (
        f"test leaked the host timezone: TZ/tzname was {before} before the test "
        f"and {after} after. Undo the monkeypatch *before* calling time.tzset() "
        f"so the C library re-reads the restored TZ, not the patched one."
    )


# ===================================================================
# Working-directory guard
# ===================================================================


@pytest.fixture(autouse=True, scope="function")
def restore_cwd():
    """Restore the process working directory after every test.

    Leak guarded: raw ``os.chdir`` calls scattered through the suite, plus any
    exception path that skips their ``finally``, leave the process parked in a
    foreign (often deleted ``tmp_path``) directory. The compose generator
    resolves ``SERVICES_DIR`` relative to the cwd, so a leaked cwd silently
    changes what later tests render.

    This is a backstop, not a licence: individual ``os.chdir`` sites still own
    their own restore. Deliberately cheap — it snapshots the path string and
    nothing else.
    """
    try:
        original = os.getcwd()
    except OSError:
        # Already parked in a deleted directory when this test started.
        original = str(_REPO_ROOT)

    yield

    try:
        os.chdir(original)
    except OSError:
        # The test deleted the directory it started in.
        os.chdir(_REPO_ROOT)


# ===================================================================
# Root-logging guard
# ===================================================================


@pytest.fixture(autouse=True, scope="function")
def restore_root_logging():
    """Return root logging to its pre-collection state before and after every test.

    Leak guarded: ``configure_logging()`` (``osprey_connectors/logger.py``) makes
    three process-global changes — it sets the root logger level, installs an
    Osprey ``RichHandler``, and raises six third-party loggers to WARNING. Any
    test that reaches an entry point triggers all three, and they would then
    decide what ``caplog`` sees in every test that ran after it on the same
    worker, including whether httpx/urllib3 records are captured at all.

    The baseline is ``_PRISTINE_LOGGING``, captured at conftest import, rather
    than a per-test snapshot. A per-test snapshot cannot see far enough back:
    pytest sets higher-scoped fixtures up *before* this function-scoped one, so
    a module-scoped fixture that configures logging is already reflected in the
    snapshot and would be preserved rather than undone, for every later test on
    the worker. ``tests/cli/test_dockerfile_template.py`` runs the CLI from a
    module-scoped fixture and does exactly that.

    Only ``RichHandler`` instances are ever removed. ``caplog`` installs its
    ``LogCaptureHandler`` around each test, and anything a test or fixture owns
    itself, are left alone.

    The CLI adds a fourth process-global change: it installs its altitude gate
    (``osprey.cli.altitude``) as a filter on that ``RichHandler``, which decides
    what a run renders. A handler a test survives with — one that was already
    pristine, or one a fixture owns — would carry that gate into every later
    test on the worker, so any gate is stripped as well.
    """
    root = logging.getLogger()
    # `in` on handlers is an identity check — they define no __eq__.
    pristine_handlers = _PRISTINE_LOGGING["root_handlers"]

    def _strip_altitude_gate(handler: logging.Handler) -> None:
        # A RichHandler carries no filters until the CLI gates it, and the CLI
        # can only have gated one if it was imported. Reading it from
        # sys.modules keeps this guard framework-free: a run with only the
        # connectors distribution installed has no gate to lift.
        if not handler.filters:
            return
        altitude = sys.modules.get("osprey.cli.altitude")
        if altitude is not None:
            altitude.lift_gate(handler)

    def _reset() -> None:
        root.setLevel(_PRISTINE_LOGGING["root_level"])
        for handler in list(root.handlers):
            if isinstance(handler, RichHandler):
                _strip_altitude_gate(handler)
                if handler not in pristine_handlers:
                    root.removeHandler(handler)
        for name, level in _PRISTINE_LOGGING["third_party"].items():
            logging.getLogger(name).setLevel(level)

    _reset()

    yield

    _reset()


# ===================================================================
# One-real-machine warning guard
# ===================================================================


@pytest.fixture(autouse=True, scope="function")
def reset_second_real_block_warning():
    """Forget which config shapes have already drawn the one-real-machine warning.

    Leak guarded: ``osprey_connectors.types`` says once per process that a
    config carries a second real connector block no control target reaches —
    once, because the resolver behind it answers every roster render and a line
    repeated at that rate is a line nobody reads. Held for the life of the
    process, that set makes any test asserting the warning depend on whether an
    earlier test on the same worker happened to resolve the same shape first,
    which under xdist depends on how the files were distributed.
    """
    from osprey_connectors import types as connector_types

    connector_types._SECOND_REAL_BLOCK_WARNED.clear()

    yield

    connector_types._SECOND_REAL_BLOCK_WARNED.clear()


# ===================================================================
# CI diagnostics: what survives a kill
# ===================================================================
#
# A test that FAILS reports itself. A lane that is KILLED — job timeout, runner
# OOM, a wedged worker held until the step cap — reports nothing at all: the
# signal lands while pytest's output is still in a buffer. These hooks keep a
# continuously flushed record on disk instead, so the post-mortem has something
# to read. See ci_diagnostics.py for the file formats and for why
# faulthandler must target a file rather than a worker's stderr.
#
# Entirely gated on OSPREY_CI_DIAG_DIR, which only the CI lanes set, as is the
# snapshot of a failing test's containers that joins the event log there: with
# the variable unset this installs nothing and costs nothing.


def pytest_configure(config):
    """Register the harness plugins, the CI recorder among them."""
    # Registered as a plugin rather than left as a hook in this file: a
    # session-scoped fixture's ``pytest_fixture_setup`` is dispatched on the
    # Session node, whose hook proxy carries only the conftests above it — this
    # one is not among them. See _env_scope_guard.py.
    if not config.pluginmanager.is_registered(_env_scope_guard):
        config.pluginmanager.register(_env_scope_guard, "osprey-env-scope-guard")

    # Reports what keeps the process alive after its last test; on for every
    # run because it prints nothing when nothing is left. See _live_threads.py.
    if not config.pluginmanager.is_registered(_live_threads):
        config.pluginmanager.register(_live_threads, _live_threads.PLUGIN_NAME)

    # The event log and stack sampler, and a failing test's containers
    # snapshotted before its module's fixtures remove them. Both are plugins,
    # so a run that loads both conftests still records each event once. See
    # ci_diagnostics.py.
    ci_diagnostics.register_recorder(config.pluginmanager)
    ci_diagnostics.register_container_snapshots(config.pluginmanager)


# ===================================================================
# xdist distribution: file-or-group scheduling
# ===================================================================
#
# The unit lane runs `-n 4 --dist loadgroup`. Stock LoadGroupScheduling gives
# every nodeid without an `@group` suffix its own scope, so unmarked tests get
# no file-level pinning at all and module-level state can straddle workers.
# FileOrGroupScheduling keeps upstream's behaviour for `xdist_group`-marked
# nodeids and gives everything else exact `--dist loadfile` semantics.
#
# `_split_scope` is an underscore method of xdist's documented scheduler
# extension point; tests/infrastructure/test_xdist_scheduler.py pins both the
# override's behaviour and the fact that upstream still calls it.

try:
    from xdist.scheduler import LoadGroupScheduling as _LoadGroupScheduling
except ImportError:  # pragma: no cover - xdist is a dev/CI-only dependency
    FileOrGroupScheduling = None
else:

    class FileOrGroupScheduling(_LoadGroupScheduling):
        """LoadGroupScheduling that falls back to file scope, not per-test scope."""

        def _split_scope(self, nodeid: str) -> str:
            """Group by `@group` suffix when present, else by test file path.

            The `@`-after-`]` check mirrors upstream's, so a parametrize value
            containing `@` is not mistaken for a group suffix.
            """
            if nodeid.rfind("@") > nodeid.rfind("]"):
                return super()._split_scope(nodeid)
            return nodeid.split("::", 1)[0]


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_make_scheduler(config, log):
    """Use FileOrGroupScheduling for `--dist loadgroup`, stock xdist otherwise.

    Returning None hands the choice back to xdist, leaving the e2e lane's
    `--dist loadfile` and ad-hoc `--dist load`/`worksteal` runs untouched.

    Declared optional because only xdist provides this hook's specification:
    pytest ends a session over a conftest hook it cannot match to a loaded
    plugin, so without the declaration a run with xdist disabled or absent
    would collect nothing at all.
    """
    if FileOrGroupScheduling is None:
        return None
    if config.getoption("dist", None) != "loadgroup":
        return None
    return FileOrGroupScheduling(config, log)


# ===================================================================
# Control context — the record and the per-server reports
# ===================================================================
#
# Re-exported from ``osprey_connectors_tests._harness._control_context_fixtures``
# so that every suite downstream of the control target writes the two files the same way, and a
# schema change fails all of them at once instead of one at a time. The
# writers are plain functions there for suites that would rather import them
# at module level; these are the factory fixtures that hand the same functions
# to a test that asks for them by name.


@pytest.fixture
def write_control_context():
    """Write a control-context record under an explicit agent-data root.

    ``write_control_context(root, target="va", generation=3,
    posture={"va": "sandbox"})`` — see
    :func:`osprey_connectors_tests._harness._control_context_fixtures.write_control_context` for the
    owner and ``last_switch`` arguments. Defaults to a record this process
    owns, so a test gets a live owner without asking for one.
    """
    from osprey_connectors_tests._harness._control_context_fixtures import (
        write_control_context as _write,
    )

    return _write


@pytest.fixture
def write_server_report():
    """Write one controls server's report under an explicit agent-data root.

    ``write_server_report(root, 4321, session="s", applied_target="va",
    applied_generation=3)`` — see
    :func:`osprey_connectors_tests._harness._control_context_fixtures.write_server_report`.
    Every field but the PID defaults to "this server has not answered yet".
    """
    from osprey_connectors_tests._harness._control_context_fixtures import (
        write_server_report as _write,
    )

    return _write


@pytest.fixture
def control_context_root(tmp_path, monkeypatch):
    """A scratch agent-data root, stamped, with the reader cache dropped.

    The three lines every control-context suite opens with: a root that exists,
    ``OSPREY_AGENT_DATA_ROOT`` pointing at it, and no record parsed from a
    previous test still held in the module cache. The ``control_target/``
    directory is created, so a test can write a raw file into it without
    reaching for ``mkdir`` first.
    """
    from osprey_connectors import control_context, posture_store

    root = tmp_path / "agent_data"
    (root / posture_store.STATE_DIR_NAME).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(root))
    control_context.invalidate_cache()
    yield root
    control_context.invalidate_cache()


# ===================================================================
# Connector state the framework suite resets on its own
# ===================================================================


@pytest.fixture(autouse=True, scope="function")
def reset_connector_config_cache():
    """Drop the connectors' parsed-config cache before and after every test.

    The connectors half of the framework conftest's ``reset_state_between_tests``,
    which also resets the framework registry and so cannot be imported here. Not
    imported by that conftest: its own fixture already clears this cache.
    """
    from osprey_connectors.workspace import reset_config_cache

    reset_config_cache()
    yield
    reset_config_cache()


@pytest.fixture
def ioc_prefix() -> str:
    """One record prefix per test, so an IOC this test did not start — another
    xdist worker's, or one that fell back to a different port — serves none of
    the names this test reads, and a mix-up fails as "cannot connect" instead
    of returning somebody else's plausible value."""
    return f"IOC{uuid.uuid4().hex[:8].upper()}"
