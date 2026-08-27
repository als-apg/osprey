"""Tests for the PTY-side resolvers — ``session_target_for_pid`` and its meta twin.

``resolve_session_target`` answers "which target is the session *I* am in on"
by matching a record's ``owner_ppid`` against this process's own parent. The
web terminal cannot ask it that way: the process that wants the answer is the
web server, and the session it is asking about is a PTY child of it. It knows
that PTY's pid, and the controls server's ``owner_ppid`` is the Claude Code
process *inside* that PTY — the PTY pid itself when the shell command is
``claude``, a descendant of it when ``claude_code.cli_version`` pins the CLI
and the shell command becomes ``npx …``.

So the match runs the other way round: walk the ancestors of the record's
``owner_ppid`` and see whether the PTY pid is on that chain. Everything else
is the same fail-closed contract the hook reader and
:func:`resolve_session_target` already keep — zero matches and two matches
both mean "no answer", and no failure mode raises.

``session_target_meta_for_pid`` answers the same question with the matched
record's display metadata attached — the label an operator reads on the badge.
It runs off the same matcher, so the two can never name different records, and
the tests at the bottom of this file pin exactly that.

The process tree is never real here: :func:`target_banner._parent_pid` is
replaced by a synthetic parent map, exactly the seam
``tests/hooks/test_target_state_reader.py`` uses for the hook-side walk.
"""

from __future__ import annotations

import json

import pytest

from osprey.mcp_server.control_system import target_banner, target_state

#: The PTY process the web terminal knows about.
PTY_PID = 7000

#: The Claude Code process inside that PTY when the CLI is pinned and the PTY
#: child is ``npx`` — a descendant of PTY_PID, not PTY_PID itself.
CLAUDE_PID = 7001

#: A parent chain nobody in these tests owns.
STRANGER_PID = 9001

#: A pid that cannot name a running process: the largest a 32-bit ``pid_t``
#: holds, which no kernel hands out.
DEAD_PID = 2_147_483_646

#: Synthetic process tree: CLAUDE_PID -> PTY_PID -> 6000 -> init.
PARENT_MAP = {CLAUDE_PID: PTY_PID, PTY_PID: 6000, 6000: 1, STRANGER_PID: 1}


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    """Point the target-state directory at ``tmp_path``."""
    root = tmp_path / "agent_data"
    monkeypatch.setattr(target_state, "resolve_shared_data_root", lambda: root)
    return root / target_state.STATE_DIR_NAME


@pytest.fixture
def synthetic_tree(monkeypatch):
    """Replace the ancestor walk's one syscall seam with a fixed parent map."""
    monkeypatch.setattr(target_banner, "_parent_pid", lambda pid: PARENT_MAP.get(int(pid)))
    return PARENT_MAP


DEFAULT_TARGETS = {
    "live": {"label": "live machine", "endpoint": "gw:5064", "real_machine": True},
    "va": {"label": "virtual accelerator", "endpoint": "localhost:5074"},
}


def write_state(
    state_dir, *, target, owner_ppid, server_pid, name=None, raw=None, targets=DEFAULT_TARGETS
):
    """Write one state file; ``raw`` replaces the record wholesale.

    *targets* is the per-target display metadata block, so a test can publish
    the label a particular deployment's writer would have minted — or, passing
    ``None``, a record that carries no metadata at all.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / (name or f"{target_state.STATE_FILE_PREFIX}{server_pid}.json")
    if raw is not None:
        path.write_text(raw, encoding="utf-8")
        return path
    path.write_text(
        json.dumps(
            {
                "target": target,
                "generation": 1,
                "server_pid": server_pid,
                "owner_ppid": owner_ppid,
                "targets": targets,
                "children": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def alive(monkeypatch, *pids):
    """Report exactly *pids* as running processes."""
    wanted = {int(p) for p in pids}
    monkeypatch.setattr(target_state, "is_process_alive", lambda pid: int(pid) in wanted)


# ── the two shapes that must match ──────────────────────────────────────────


def test_owner_ppid_equal_to_the_pty_pid_matches(state_dir, synthetic_tree, monkeypatch):
    """The default shape: the PTY child *is* ``claude``, so it owns the server."""
    write_state(state_dir, target="va", owner_ppid=PTY_PID, server_pid=5150)
    alive(monkeypatch, 5150)

    assert target_banner.session_target_for_pid(PTY_PID) == "va"


def test_owner_ppid_descended_from_the_pty_pid_matches(state_dir, synthetic_tree, monkeypatch):
    """The pinned-CLI shape: the PTY child is ``npx`` and ``claude`` is under it.

    Equality-only matching would silently answer "no session target" on every
    deployment that pins ``claude_code.cli_version`` — and the badge would go
    on showing the baseline there while the operator was switched away.
    """
    write_state(state_dir, target="va", owner_ppid=CLAUDE_PID, server_pid=5150)
    alive(monkeypatch, 5150)

    assert target_banner.session_target_for_pid(PTY_PID) == "va"


def test_the_live_target_is_reported_as_itself(state_dir, synthetic_tree, monkeypatch):
    write_state(state_dir, target="live", owner_ppid=PTY_PID, server_pid=5150)
    alive(monkeypatch, 5150)

    assert target_banner.session_target_for_pid(PTY_PID) == "live"


# ── every failure mode is "no answer" ───────────────────────────────────────


def test_no_state_files_yield_no_answer(state_dir, synthetic_tree, monkeypatch):
    state_dir.mkdir(parents=True, exist_ok=True)
    alive(monkeypatch)

    assert target_banner.session_target_for_pid(PTY_PID) is None


def test_a_missing_state_directory_yields_no_answer(state_dir, synthetic_tree, monkeypatch):
    alive(monkeypatch)

    assert target_banner.session_target_for_pid(PTY_PID) is None


def test_another_sessions_record_is_not_this_ptys(state_dir, synthetic_tree, monkeypatch):
    """A second session in the same checkout must not answer for this one."""
    write_state(state_dir, target="va", owner_ppid=STRANGER_PID, server_pid=5150)
    alive(monkeypatch, 5150)

    assert target_banner.session_target_for_pid(PTY_PID) is None


def test_two_matching_live_records_are_ambiguous(state_dir, synthetic_tree, monkeypatch):
    """Ambiguity is fail-closed: two answers is the same as none."""
    write_state(state_dir, target="va", owner_ppid=PTY_PID, server_pid=5150)
    write_state(state_dir, target="live", owner_ppid=CLAUDE_PID, server_pid=5151)
    alive(monkeypatch, 5150, 5151)

    assert target_banner.session_target_for_pid(PTY_PID) is None


def test_a_dead_servers_record_is_ignored(state_dir, synthetic_tree, monkeypatch):
    """A killed server leaves its file behind; it does not still speak."""
    write_state(state_dir, target="va", owner_ppid=PTY_PID, server_pid=DEAD_PID)
    alive(monkeypatch)

    assert target_banner.session_target_for_pid(PTY_PID) is None


def test_a_live_record_wins_over_a_dead_sibling(state_dir, synthetic_tree, monkeypatch):
    write_state(state_dir, target="live", owner_ppid=PTY_PID, server_pid=DEAD_PID)
    write_state(state_dir, target="va", owner_ppid=PTY_PID, server_pid=5150)
    alive(monkeypatch, 5150)

    assert target_banner.session_target_for_pid(PTY_PID) == "va"


def test_a_corrupt_record_does_not_hide_a_good_sibling(state_dir, synthetic_tree, monkeypatch):
    write_state(
        state_dir,
        target=None,
        owner_ppid=None,
        server_pid=0,
        name="target_state_1.json",
        raw="{oops",
    )
    write_state(state_dir, target="va", owner_ppid=PTY_PID, server_pid=5150)
    alive(monkeypatch, 5150)

    assert target_banner.session_target_for_pid(PTY_PID) == "va"


def test_an_unknown_target_name_yields_no_answer(state_dir, synthetic_tree, monkeypatch):
    write_state(state_dir, target="somewhere-else", owner_ppid=PTY_PID, server_pid=5150)
    alive(monkeypatch, 5150)

    assert target_banner.session_target_for_pid(PTY_PID) is None


def test_an_unparseable_owner_ppid_is_skipped(state_dir, synthetic_tree, monkeypatch):
    write_state(state_dir, target="va", owner_ppid="not-a-pid", server_pid=5150)
    alive(monkeypatch, 5150)

    assert target_banner.session_target_for_pid(PTY_PID) is None


@pytest.mark.parametrize("pty_pid", [None, 0, -1, "nonsense"])
def test_a_pid_that_is_not_a_pid_yields_no_answer(state_dir, synthetic_tree, monkeypatch, pty_pid):
    """The route passes whatever ``PtySession.pid`` gave it, ``None`` included."""
    write_state(state_dir, target="va", owner_ppid=PTY_PID, server_pid=5150)
    alive(monkeypatch, 5150)

    assert target_banner.session_target_for_pid(pty_pid) is None


# ── the ancestor walk itself ────────────────────────────────────────────────


def test_the_walk_is_bounded_and_survives_a_cycle(monkeypatch):
    """A lying process table must end the walk, not spin in it."""
    monkeypatch.setattr(target_banner, "_parent_pid", lambda pid: {10: 11, 11: 10}.get(int(pid)))

    chain = target_banner._ancestor_pids(10)

    assert chain == [10, 11]


def test_the_walk_stops_at_the_hop_limit(monkeypatch):
    """An unbounded chain is truncated rather than followed forever."""
    monkeypatch.setattr(target_banner, "_parent_pid", lambda pid: int(pid) + 1)

    chain = target_banner._ancestor_pids(1000)

    assert len(chain) == target_banner.MAX_ANCESTOR_HOPS


def test_the_walk_ends_where_the_parent_is_unknowable(monkeypatch):
    """A process that has already exited simply ends the chain."""
    monkeypatch.setattr(target_banner, "_parent_pid", lambda pid: None)

    assert target_banner._ancestor_pids(PTY_PID) == [PTY_PID]


def test_the_walk_stops_at_the_pid_it_was_asked_about(monkeypatch):
    """Every hop past the answer is a syscall spent learning nothing.

    On a platform without ``/proc`` it is a fork of ``ps``, on a route the
    badge polls every few seconds per open card.
    """
    asked = []

    def _parent(pid):
        asked.append(int(pid))
        return PARENT_MAP.get(int(pid))

    monkeypatch.setattr(target_banner, "_parent_pid", _parent)

    chain = target_banner._ancestor_pids(CLAUDE_PID, stop_at=PTY_PID)

    assert chain == [CLAUDE_PID, PTY_PID]
    # PTY_PID's own parent is never asked for: the question was already answered.
    assert asked == [CLAUDE_PID]


def test_a_stop_pid_that_is_never_reached_still_ends_the_walk(monkeypatch):
    """`stop_at` shortens the walk; it does not change what a miss looks like."""
    monkeypatch.setattr(target_banner, "_parent_pid", lambda pid: PARENT_MAP.get(int(pid)))

    assert STRANGER_PID not in target_banner._ancestor_pids(CLAUDE_PID, stop_at=STRANGER_PID)


def test_one_memo_is_shared_across_every_records_walk(state_dir, monkeypatch):
    """Several sessions converge on the same upper ancestors within a hop or two.

    Without a shared memo the walk would re-ask for those upper pids once per
    record — N forks of `ps` per request where one will do.
    """
    asked = []

    def _parent(pid):
        asked.append(int(pid))
        return {CLAUDE_PID: 7002, 7002: PTY_PID, 7050: 7002, PTY_PID: 6000, 6000: 1}.get(int(pid))

    monkeypatch.setattr(target_banner, "_parent_pid", _parent)
    # Two live records under two different Claude processes, both inside the
    # same PTY — the ambiguous case, chosen because it forces BOTH walks to run.
    write_state(state_dir, target="va", owner_ppid=CLAUDE_PID, server_pid=5150)
    write_state(state_dir, target="va", owner_ppid=7050, server_pid=5151)
    alive(monkeypatch, 5150, 5151)

    target_banner.session_target_for_pid(PTY_PID)

    assert len(asked) == len(set(asked)), f"a pid was looked up twice: {asked}"
    # 7002 is on both chains and must be resolved exactly once.
    assert asked.count(7002) == 1


def test_ps_timeout_is_short_enough_for_a_polled_route():
    """The badge polls this; the budget is "give up", not "wait for a slow answer"."""
    assert target_banner.PS_TIMEOUT_S <= 2


def test_parent_pid_of_a_pid_that_does_not_exist_is_none():
    """Both readers — ``/proc`` and ``ps`` — answer "no parent", never raise."""
    assert target_banner._parent_pid(DEAD_PID) is None


# ── the metadata twin ───────────────────────────────────────────────────────
#
# The badge needs a NAME for the target, not just its key: a deployment whose
# live target is a stand-in publishes "LIVE MACHINE (stand-in)", and a reader
# that re-derived that from config would be a second opinion about which
# machine an operator is pointed at. So the metadata comes off the record, and
# off the SAME match the name resolver uses.

STANDIN_TARGETS = {
    "live": {
        "label": "LIVE MACHINE (stand-in)",
        "endpoint": "localhost:5074",
        "real_machine": True,
        "probe_channel": "SR:BEAM",
    },
    "va": {"label": "virtual accelerator (simulation)", "endpoint": "localhost:5064"},
}


def test_the_matched_records_metadata_comes_back_under_its_own_target(
    state_dir, synthetic_tree, monkeypatch
):
    """Everything the writer recorded for the target the session is on."""
    write_state(
        state_dir, target="live", owner_ppid=PTY_PID, server_pid=5150, targets=STANDIN_TARGETS
    )
    alive(monkeypatch, 5150)

    assert target_banner.session_target_meta_for_pid(PTY_PID) == {
        "target": "live",
        "label": "LIVE MACHINE (stand-in)",
        "endpoint": "localhost:5074",
        "real_machine": True,
        "probe_channel": "SR:BEAM",
    }


def test_the_two_resolvers_answer_off_the_same_record(state_dir, synthetic_tree, monkeypatch):
    """The bare name and the label are the same record's, or the badge lies."""
    write_state(
        state_dir, target="live", owner_ppid=CLAUDE_PID, server_pid=5150, targets=STANDIN_TARGETS
    )
    alive(monkeypatch, 5150)

    assert target_banner.session_target_for_pid(PTY_PID) == "live"
    assert target_banner.session_target_meta_for_pid(PTY_PID)["target"] == "live"


def test_no_matching_record_yields_no_metadata(state_dir, synthetic_tree, monkeypatch):
    """Zero matches is no answer here too — the caller falls back to the render."""
    write_state(state_dir, target="va", owner_ppid=STRANGER_PID, server_pid=5150)
    alive(monkeypatch, 5150)

    assert target_banner.session_target_meta_for_pid(PTY_PID) is None


def test_two_matching_records_yield_no_metadata(state_dir, synthetic_tree, monkeypatch):
    """Ambiguity is fail-closed on this side of the pair as well."""
    write_state(state_dir, target="va", owner_ppid=PTY_PID, server_pid=5150)
    write_state(state_dir, target="live", owner_ppid=CLAUDE_PID, server_pid=5151)
    alive(monkeypatch, 5150, 5151)

    assert target_banner.session_target_meta_for_pid(PTY_PID) is None
    assert target_banner.session_target_for_pid(PTY_PID) is None


def test_an_unknown_target_name_yields_no_metadata(state_dir, synthetic_tree, monkeypatch):
    write_state(state_dir, target="somewhere-else", owner_ppid=PTY_PID, server_pid=5150)
    alive(monkeypatch, 5150)

    assert target_banner.session_target_meta_for_pid(PTY_PID) is None


def test_metadata_cannot_rename_the_target_it_was_matched_on(
    state_dir, synthetic_tree, monkeypatch
):
    """The validated name wins over anything the metadata block claims.

    The name comes from the record's own ``target`` field, which was checked
    against the known targets; a ``target`` key inside the metadata has been
    through no such check. Letting it through would let a corrupt — or
    hand-edited — file tell an operator they are somewhere they are not.
    """
    write_state(
        state_dir,
        target="va",
        owner_ppid=PTY_PID,
        server_pid=5150,
        targets={"va": {"target": "live", "label": "virtual accelerator (simulation)"}},
    )
    alive(monkeypatch, 5150)

    meta = target_banner.session_target_meta_for_pid(PTY_PID)

    assert meta["target"] == "va"
    assert meta["label"] == "virtual accelerator (simulation)"


def test_a_record_with_no_metadata_block_still_names_its_target(
    state_dir, synthetic_tree, monkeypatch
):
    """One dict shape for the caller: an absent key reads as "not recorded".

    A hand-edited or half-written record must not make the badge branch on a
    missing key — it gets the target it can trust and nothing it cannot.
    """
    write_state(state_dir, target="va", owner_ppid=PTY_PID, server_pid=5150, targets=None)
    alive(monkeypatch, 5150)

    assert target_banner.session_target_meta_for_pid(PTY_PID) == {"target": "va"}


@pytest.mark.parametrize("pty_pid", [None, 0, -1, "nonsense"])
def test_a_pid_that_is_not_a_pid_yields_no_metadata(
    state_dir, synthetic_tree, monkeypatch, pty_pid
):
    write_state(state_dir, target="va", owner_ppid=PTY_PID, server_pid=5150)
    alive(monkeypatch, 5150)

    assert target_banner.session_target_meta_for_pid(pty_pid) is None
